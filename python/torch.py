"""mini-torch: a NumPy-backed stand-in for the slice of PyTorch the browser lessons use.

PyTorch doesn't run in Pyodide, so browser challenges import this module as `torch`.
The point is that the code you write here is the code you run on your GPU later.
Only the API listed in TORCH_SUBSET.md exists. Anything else raises MiniTorchUnsupported
with a message that says so, instead of a confusing AttributeError.

Layout: dtypes and promotion, the autograd engine, Tensor, creation and random functions,
math, shape ops, then the submodules torch.nn, torch.nn.functional, torch.nn.init,
torch.nn.utils, torch.optim and torch.cuda (registered in sys.modules so
`import torch.nn.functional as F` works).
"""

import builtins as _builtins
import collections as _collections
import copy as _copy
import functools as _functools
import math as _math
import re as _re
import sys as _sys
import types as _types

import numpy as _np


class MiniTorchUnsupported(AttributeError):
    """Raised for torch API that exists in real PyTorch but not in the browser shim."""


def _unsupported(name):
    return MiniTorchUnsupported(
        f"`{name}` is real PyTorch, but the browser's mini-torch doesn't include it. "
        "Browser lessons use the subset in TORCH_SUBSET.md. "
        "It will work on your GPU."
    )


# ---------------------------------------------------------------- dtypes
class dtype:
    __slots__ = ("name", "np", "is_floating_point", "itemsize", "_cname", "_short")

    def __init__(self, name, np_dtype, cname, short):
        self.name = name
        self.np = _np.dtype(np_dtype)
        self.is_floating_point = self.np.kind == "f"
        self.itemsize = self.np.itemsize
        self._cname = cname  # torch's C++ name, used in its error messages ("long int")
        self._short = short  # torch's short name ("Long")

    @property
    def is_signed(self):
        return self.np.kind in "if"

    def __repr__(self):
        return f"torch.{self.name}"

    def __reduce__(self):
        return self.name


_dtype_cls = dtype
_bool_dt = dtype("bool", _np.bool_, "bool", "Bool")
uint8 = dtype("uint8", _np.uint8, "unsigned char", "Byte")
int8 = dtype("int8", _np.int8, "signed char", "Char")
int16 = dtype("int16", _np.int16, "short int", "Short")
int32 = dtype("int32", _np.int32, "int", "Int")
int64 = dtype("int64", _np.int64, "long int", "Long")
float16 = dtype("float16", _np.float16, "c10::Half", "Half")
float32 = dtype("float32", _np.float32, "float", "Float")
float64 = dtype("float64", _np.float64, "double", "Double")
_BY_NP = {d.np: d for d in (_bool_dt, uint8, int8, int16, int32, int64, float16, float32, float64)}

# Names that would shadow Python builtins inside this module are served by __getattr__.
_ALIASES = {"bool": _bool_dt, "half": float16, "float": float32, "double": float64,
            "short": int16, "int": int32, "long": int64}

_F32 = _np.dtype(_np.float32)
_F64 = _np.dtype(_np.float64)
_I64 = _np.dtype(_np.int64)
_BOOL = _np.dtype(_np.bool_)


def _dt(a):
    return _BY_NP[a.dtype]


def _cat(d):
    """Promotion category of a NumPy dtype: 0 bool, 1 integer, 2 floating."""
    k = d.kind
    return 0 if k == "b" else (2 if k == "f" else 1)


def _promote(d1, d2):
    if d1 is None:
        return d2
    if d2 is None:
        return d1
    c1, c2 = _cat(d1), _cat(d2)
    if c1 != c2:
        return d1 if c1 > c2 else d2  # torch: int + float16 is float16, not NumPy's float64
    return _np.promote_types(d1, d2)


def _combine(hi, lo):
    """torch's combine_categories: a lower-priority operand only wins with a higher category."""
    if hi is None:
        return lo
    if lo is None:
        return hi
    if _cat(hi) == 2:
        return hi
    if _cat(hi) == 0 or _cat(lo) == 2:
        return _promote(hi, lo)
    return hi


def _result_type(*vals):
    """torch.result_type: dimensioned tensors beat 0-d tensors beat Python numbers."""
    dim = zero = wrapped = None
    for v in vals:
        if isinstance(v, Tensor):
            if v._a.ndim:
                dim = _promote(dim, v._a.dtype)
            else:
                zero = _promote(zero, v._a.dtype)
        elif isinstance(v, bool):
            wrapped = _promote(wrapped, _BOOL)
        elif isinstance(v, int):
            wrapped = _promote(wrapped, _I64)
        elif v is not None:
            wrapped = _promote(wrapped, _F32)  # Python floats use the default dtype
    return _combine(dim, _combine(zero, wrapped))


def _to_dtype(a, d):
    """Cast like torch: float -> int truncates toward zero, then wraps (mod 256 for int8).

    Real PyTorch leaves out-of-range float -> int casts undefined. The shim wraps on purpose,
    so `128.0 -> -128` happens every time and the lesson can explain it.
    """
    if d.is_floating_point:
        return a.astype(d.np)
    if d is _bool_dt:
        return a != 0
    if a.dtype.kind == "f":
        # torch leaves NaN/inf -> int undefined too; the shim picks 0 instead of crashing.
        with _np.errstate(all="ignore"):
            a = _np.trunc(_np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)).astype(_np.int64)
    return a.astype(_np.int64).astype(d.np)


class Size(tuple):
    def __repr__(self):
        return f"torch.Size({list(self)})"

    def numel(self):
        n = 1
        for v in self:
            n *= v
        return n


class finfo:
    def __init__(self, type=None):  # noqa: A002 - mirrors torch
        d = type or float32
        if not d.is_floating_point:
            raise TypeError("torch.finfo() requires a floating point input type. Use torch.iinfo to handle 'torch.finfo'")
        f = _np.finfo(d.np)
        self.bits, self.eps, self.max, self.min = f.bits, float(f.eps), float(f.max), float(f.min)
        self.tiny = self.smallest_normal = float(f.tiny)
        self.resolution = float(f.resolution)
        self.dtype = d.name


class iinfo:
    def __init__(self, type):  # noqa: A002
        if type.is_floating_point or type is _bool_dt:
            raise TypeError("torch.iinfo() requires an integer input type. Use torch.finfo to handle 'torch.iinfo'")
        f = _np.iinfo(type.np)
        self.bits, self.min, self.max = f.bits, int(f.min), int(f.max)
        self.dtype = type.name


def get_default_dtype():
    return float32


# ---------------------------------------------------------------- devices
class device:
    __slots__ = ("type", "index")

    def __init__(self, type="cpu", index=None):  # noqa: A002
        if isinstance(type, device):
            type, index = type.type, type.index
        kind, _, idx = str(type).partition(":")
        self.type = kind
        self.index = int(idx) if idx else index

    def __repr__(self):
        return f"device(type='{self.type}')" if self.index is None else f"device(type='{self.type}', index={self.index})"

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"

    def __eq__(self, other):
        return str(self) == str(other) if isinstance(other, (device, str)) else NotImplemented

    def __hash__(self):
        return hash(str(self))


_device_cls = device
_CPU = device("cpu")


def _check_device(d):
    if d is None:
        return
    if isinstance(d, (str, device)) and device(d).type == "cpu":
        return
    raise MiniTorchUnsupported(
        f"`device={str(d)!r}`: the browser has no GPU, so mini-torch only runs on the CPU. "
        "Pick the device with `'cuda' if torch.cuda.is_available() else 'cpu'`, which gives 'cpu' here and 'cuda' on your GPU."
    )


# ---------------------------------------------------------------- grad mode
_GRAD = [True]


class _GradMode:
    """Context manager and decorator: `with torch.no_grad():`, `@torch.no_grad()` and bare `@torch.no_grad`."""
    _enabled = True

    def __new__(cls, orig_func=None):
        if callable(orig_func):  # bare decorator: return the wrapped function itself
            return cls()(orig_func)
        return super().__new__(cls)

    def __init__(self, orig_func=None):
        self._prev = []

    def __enter__(self):
        self._prev.append(_GRAD[0])
        _GRAD[0] = self._enabled

    def __exit__(self, *exc):
        _GRAD[0] = self._prev.pop()
        return False

    def __call__(self, fn):
        if not callable(fn):
            raise TypeError(f"{type(self).__name__}() takes a function to decorate")
        mode = type(self)

        @_functools.wraps(fn)
        def wrapper(*a, **k):
            with mode():
                return fn(*a, **k)
        return wrapper


class no_grad(_GradMode):
    _enabled = False


class enable_grad(_GradMode):
    _enabled = True


class inference_mode(no_grad):
    """Same as no_grad in the shim."""

    def __new__(cls, mode=True):
        return super().__new__(cls, mode if callable(mode) else None)

    def __init__(self, mode=True):
        super().__init__()


class set_grad_enabled:
    def __init__(self, mode):
        self._prev = _GRAD[0]
        _GRAD[0] = _builtins.bool(mode)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        _GRAD[0] = self._prev
        return False


def is_grad_enabled():
    return _GRAD[0]


# ---------------------------------------------------------------- autograd engine
class _Leaf:
    """Where a leaf tensor's gradient lands (torch's AccumulateGrad)."""
    __slots__ = ("tensor",)
    inputs = ()

    def __init__(self, t):
        self.tensor = t

    def name(self):
        return "AccumulateGrad"


class _Node:
    """One recorded op: its input nodes and a function from output grad to input grads."""
    __slots__ = ("_name", "inputs", "fn", "saved", "retain", "freed")

    def __init__(self, name, inputs, fn, saved):
        self._name = name
        self.inputs = inputs
        self.fn = fn
        self.saved = saved  # [(version box, version at save time)]
        self.retain = None
        self.freed = False

    def name(self):
        return self._name

    def __repr__(self):
        return f"<{self._name} object>"


_FREED = ("Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). "
          "Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). "
          "Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.")


def _track(*xs):
    if not _GRAD[0]:
        return False
    for x in xs:
        if isinstance(x, Tensor) and x._node is not None:
            return True
    return False


def _new(a, ver=None, view=False):
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    t = object.__new__(Tensor)
    t._a = a
    t._ver = ver if ver is not None else [0]
    t._node = None
    t.grad = None
    t._view = view
    return t


def _result(a, inputs, backward, name, saved=(), save_out=False):
    """Wrap an op's output and, when autograd is recording, attach its backward."""
    out = _new(a)
    if _track(*inputs):
        sv = [(t._ver, t._ver[0]) for t in saved if isinstance(t, Tensor)]
        if save_out:
            sv.append((out._ver, 0))
        edges = tuple(x._node if isinstance(x, Tensor) else None for x in inputs)
        out._node = _Node(name, edges, backward, sv)
    return out


def _topo(root):
    order, seen, stack = [], set(), [(root, False)]
    while stack:
        n, done = stack.pop()
        if done:
            order.append(n)
            continue
        if id(n) in seen:
            continue
        seen.add(id(n))
        stack.append((n, True))
        for i in n.inputs:
            if i is not None and id(i) not in seen:
                stack.append((i, False))
    order.reverse()
    return order


def _accumulate(t, g):
    g = _np.asarray(g)
    if g.shape != t._a.shape:
        g = _np.broadcast_to(g, t._a.shape)
    if t.grad is None or t.grad._a.shape != t._a.shape or not t.grad._a.flags.writeable:
        t.grad = _new(g.astype(t._a.dtype))
    else:
        t.grad._a += g.astype(t.grad._a.dtype)
        t.grad._ver[0] += 1


def _run_backward(root, gradient, retain_graph):
    if root._node is None:
        raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
    if gradient is None:
        if root._a.size != 1:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        g0 = _np.ones_like(root._a)
    else:
        g0 = gradient._a if isinstance(gradient, Tensor) else _np.asarray(gradient)
        if g0.shape != root._a.shape:
            raise RuntimeError(f"Mismatch in shape: grad_output[0] has a shape of {Size(g0.shape)} and output[0] has a shape of {Size(root._a.shape)}.")
        g0 = g0.astype(root._a.dtype)
    grads = {id(root._node): g0}
    with no_grad():
        for node in _topo(root._node):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if isinstance(node, _Leaf):
                _accumulate(node.tensor, g)
                continue
            if node.freed:
                raise RuntimeError(_FREED)
            for box, v in node.saved:
                if box[0] != v:
                    raise RuntimeError(
                        "one of the variables needed for gradient computation has been modified by an inplace operation: "
                        f"the input of {node._name} is at version {box[0]}; expected version {v} instead. "
                        "Use an out-of-place op (`x = x + 1`) or `.clone()` before changing it.")
            if node.retain is not None:
                _accumulate(node.retain, g)
            in_grads = node.fn(g)
            if node.saved and not retain_graph:
                node.freed = True
                node.fn = None
            for inp, ig in zip(node.inputs, in_grads):
                if inp is None or ig is None:
                    continue
                k = id(inp)
                grads[k] = ig if k not in grads else grads[k] + ig


def _unbroadcast(g, shape):
    shape = tuple(shape)
    if g.shape == shape:
        return g
    extra = g.ndim - len(shape)
    if extra > 0:
        g = g.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and g.shape[i] != 1)
    if axes:
        g = g.sum(axis=axes, keepdims=True)
    return g.reshape(shape)


def _expand_back(g, shape, axes, keepdim):
    """Grad of a reduction over `axes`, broadcast back to the input's shape."""
    if axes is None:
        g = _np.reshape(g, (1,) * len(shape))
    elif not keepdim:
        g = _np.expand_dims(g, axes)
    return _np.broadcast_to(g, shape)


# ---------------------------------------------------------------- argument helpers
def _norm_dim(ndim, dim):
    if not isinstance(dim, (int, _np.integer)) or isinstance(dim, _builtins.bool):
        raise TypeError(f"dim must be an int, not {type(dim).__name__}")
    dim = int(dim)
    lo, hi = (-1, 0) if ndim == 0 else (-ndim, ndim - 1)
    if not lo <= dim <= hi:
        raise IndexError(f"Dimension out of range (expected to be in range of [{lo}, {hi}], but got {dim})")
    return dim % max(ndim, 1)


def _dim(a, dim):
    """Normalise a dim argument like torch, with torch's error for one that doesn't exist."""
    return _norm_dim(a.ndim, dim)


def _axes(a, dim):
    """Reduction axes as a sorted tuple, or None for every dimension."""
    if dim is None:
        return None
    dims = list(dim) if isinstance(dim, (tuple, list)) else [dim]
    if not dims:
        return None
    axes = []
    for d in dims:
        nd = _dim(a, d)
        if nd in axes:
            raise RuntimeError(f"dim {nd} appears multiple times in the list of dims")
        axes.append(nd)
    if a.ndim == 0:
        return None
    return tuple(sorted(axes))


def _shape_arg(shape, fn):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    out = []
    for v in shape:
        if v is Ellipsis:
            raise TypeError(f"{fn}(): argument 'shape' must be tuple of ints, but found element of type ellipsis")
        if isinstance(v, Tensor) and v._a.size == 1 and _cat(v._a.dtype) == 1:
            v = int(v.item())
        if isinstance(v, _np.integer):
            v = int(v)
        if not isinstance(v, int) or isinstance(v, _builtins.bool):
            raise TypeError(f"{fn}(): argument 'shape' must be tuple of ints, but found element of type {type(v).__name__}")
        out.append(v)
    return tuple(out)


def _size_arg(size, fn):
    shape = _shape_arg(size, fn)
    if any(v < 0 for v in shape):
        raise RuntimeError(f"Trying to create tensor with negative dimension {min(shape)}: {list(shape)}")
    return shape


def _tensor_arg(t, fn, arg="input"):
    if isinstance(t, Tensor):
        return t
    if t is Ellipsis:
        raise TypeError(f"{fn}(): argument '{arg}' must be Tensor, not ellipsis")
    raise TypeError(f"{fn}(): argument '{arg}' must be Tensor, not {type(t).__name__}")


def _coerce(v):
    if isinstance(v, Tensor):
        return v
    if isinstance(v, _np.generic):
        v = v.item()
    if isinstance(v, (_builtins.bool, int, float)):
        return v
    if v is Ellipsis:
        raise TypeError("unsupported operand type(s): 'Tensor' and 'ellipsis'")
    raise TypeError(f"unsupported operand type(s): 'Tensor' and '{type(v).__name__}'")


def _raw(v):
    return v._a if isinstance(v, Tensor) else v


def _check_broadcast(a, b):
    """Raise torch's error for shapes that don't broadcast (NumPy's wording names neither tensor)."""
    sa, sb = _np.shape(a), _np.shape(b)
    n = max(len(sa), len(sb))
    for i in range(n - 1, -1, -1):
        off = n - 1 - i
        da = sa[len(sa) - 1 - off] if off < len(sa) else 1
        db = sb[len(sb) - 1 - off] if off < len(sb) else 1
        if da != db and da != 1 and db != 1:
            raise RuntimeError(f"The size of tensor a ({da}) must match the size of tensor b ({db}) at non-singleton dimension {i}")


def _float_input(t):
    """Inputs to exp/log/softmax/...: ints are promoted to float32 like torch."""
    a = t._a
    return a if _cat(a.dtype) == 2 else a.astype(_F32)


# torch.return_types.*: tuple subclasses whose names show up in errors like
# "unsupported operand type(s) for /: 'torch.return_types.max' and 'int'".
def _return_type(name, fields=("values", "indices")):
    cls = _collections.namedtuple(name, fields)
    cls.__name__ = cls.__qualname__ = f"torch.return_types.{name}"
    return cls


_MaxResult = _return_type("max")
_MinResult = _return_type("min")
_SortResult = _return_type("sort")
_TopkResult = _return_type("topk")


# ---------------------------------------------------------------- elementwise binary ops
def _fwd_pow(a, b):
    return _np.power(a, b)


_FWD = {"add": _np.add, "sub": _np.subtract, "mul": _np.multiply, "div": _np.divide,
        "pow": _fwd_pow, "floordiv": _np.floor_divide, "mod": _np.mod,
        "maximum": _np.maximum, "minimum": _np.minimum}
_NAMES = {"add": "AddBackward0", "sub": "SubBackward0", "mul": "MulBackward0", "div": "DivBackward0",
          "pow": "PowBackward0", "floordiv": "FloorDivideBackward0", "mod": "RemainderBackward0",
          "maximum": "MaximumBackward0", "minimum": "MinimumBackward0"}


def _binary(x, y, kind):
    x, y = _coerce(x), _coerce(y)
    xa, ya = _raw(x), _raw(y)
    _check_broadcast(xa, ya)
    rt = _result_type(x, y)
    if kind == "div" and _cat(rt) < 2:
        rt = _F32
    if kind == "pow" and _cat(rt) < 2 and _np.any(_np.asarray(ya) < 0):
        raise RuntimeError("Integers to negative integer powers are not allowed.")
    with _np.errstate(all="ignore"):
        if _cat(rt) == 2:
            a = _np.asarray(xa, dtype=rt)
            b = _np.asarray(ya, dtype=rt)
            out = _np.asarray(_FWD[kind](a, b), dtype=rt)
        else:
            # Compute in int64, then wrap into the result dtype (int8 overflows like torch: 127 * 2 -> -2).
            a = _np.asarray(xa, dtype=_I64)
            b = _np.asarray(ya, dtype=_I64)
            if kind in ("floordiv", "mod") and _np.any(b == 0):
                raise ZeroDivisionError("ZeroDivisionError")  # torch's integer division by zero
            res = _np.asarray(_FWD[kind](a, b))
            out = res != 0 if rt == _BOOL else res.astype(rt)
    if _cat(rt) < 2 or not _track(x, y):
        return _new(out)
    xs, ys = _np.shape(xa), _np.shape(ya)

    def bw(g):
        with _np.errstate(all="ignore"):
            if kind == "add":
                return _unbroadcast(g, xs), _unbroadcast(g, ys)
            if kind == "sub":
                return _unbroadcast(g, xs), _unbroadcast(-g, ys)
            if kind == "mul":
                return _unbroadcast(g * b, xs), _unbroadcast(g * a, ys)
            if kind == "div":
                return _unbroadcast(g / b, xs), _unbroadcast(-g * a / (b * b), ys)
            if kind == "pow":
                gx = _np.where(b == 0, 0.0, g * b * _np.power(a, b - 1)).astype(rt)
                gy = _np.where((a == 0) & (b >= 0), 0.0, g * out * _np.log(a)).astype(rt)
                return _unbroadcast(gx, xs), _unbroadcast(gy, ys)
            if kind == "mod":
                return _unbroadcast(g, xs), _unbroadcast(-g * _np.floor_divide(a, b), ys)
            if kind in ("maximum", "minimum"):
                win = (a > b) if kind == "maximum" else (a < b)
                tie = a == b
                gx = _np.where(tie, g / 2, _np.where(win, g, 0)).astype(rt)
                gy = _np.where(tie, g / 2, _np.where(win, 0, g)).astype(rt)
                return _unbroadcast(gx, xs), _unbroadcast(gy, ys)
            raise RuntimeError(f"the derivative for '{'floor_divide' if kind == 'floordiv' else kind}' is not implemented.")
    # Save only what torch saves, so in-place changes and repeated backward() fail exactly when torch's do.
    xt = isinstance(x, Tensor) and x._node is not None
    yt = isinstance(y, Tensor) and y._node is not None
    if kind in ("add", "sub"):
        saved = ()
    elif kind == "mul":
        saved = ((y,) if xt else ()) + ((x,) if yt else ())
    elif kind == "div":
        saved = (x, y) if yt else (y,)
    else:
        saved = (x, y)
    return _result(out, (x, y), bw, _NAMES[kind], saved=saved, save_out=kind == "pow" and yt)


def _compare(x, y, op):
    x, y = _coerce(x), _coerce(y)
    xa, ya = _raw(x), _raw(y)
    _check_broadcast(xa, ya)
    rt = _result_type(x, y)
    with _np.errstate(all="ignore"):
        return _new(_np.asarray(op(_np.asarray(xa, dtype=rt), _np.asarray(ya, dtype=rt))))


def _bitwise(x, y, op, name):
    x, y = _coerce(x), _coerce(y)
    rt = _result_type(x, y)
    if _cat(rt) == 2:
        raise RuntimeError(f"\"{name}_cpu\" not implemented for 'Float'")
    xa, ya = _raw(x), _raw(y)
    _check_broadcast(xa, ya)
    return _new(_np.asarray(op(_np.asarray(xa, dtype=rt), _np.asarray(ya, dtype=rt))).astype(rt))


# ---------------------------------------------------------------- elementwise unary ops
def _unary(x, name, fwd, bwd, keep_int=False, save=None):
    """fwd(a) -> out; bwd(g, a, out) -> grad. save: None, 'in' or 'out' (for version checks)."""
    x = _tensor_arg(x, f"torch.{name.lower()}")
    a = x._a if keep_int else _float_input(x)
    with _np.errstate(all="ignore"):
        out = _np.asarray(fwd(a))
        if out.dtype != a.dtype and a.dtype.kind == "f":
            out = out.astype(a.dtype)
    if _cat(out.dtype) < 2 or not _track(x):
        return _new(out)

    def bw(g):
        with _np.errstate(all="ignore"):
            return (_np.asarray(bwd(g, a, out), dtype=out.dtype),)
    return _result(out, (x,), bw, name, saved=(x,) if save == "in" else (), save_out=save == "out")


_erf_vec = _np.vectorize(_math.erf, otypes=[_np.float64])


def exp(x): return _unary(x, "ExpBackward0", _np.exp, lambda g, a, o: g * o, save="out")
def log(x): return _unary(x, "LogBackward0", _np.log, lambda g, a, o: g / a, save="in")
def log2(x): return _unary(x, "Log2Backward0", _np.log2, lambda g, a, o: g / (a * _math.log(2.0)), save="in")
def sqrt(x): return _unary(x, "SqrtBackward0", _np.sqrt, lambda g, a, o: g / (2 * o), save="out")
def rsqrt(x): return _unary(x, "RsqrtBackward0", lambda a: 1 / _np.sqrt(a), lambda g, a, o: -0.5 * g * o * o * o, save="out")
def tanh(x): return _unary(x, "TanhBackward0", _np.tanh, lambda g, a, o: g * (1 - o * o), save="out")
def sigmoid(x): return _unary(x, "SigmoidBackward0", lambda a: 1 / (1 + _np.exp(-a)), lambda g, a, o: g * o * (1 - o), save="out")
def sin(x): return _unary(x, "SinBackward0", _np.sin, lambda g, a, o: g * _np.cos(a), save="in")
def cos(x): return _unary(x, "CosBackward0", _np.cos, lambda g, a, o: -g * _np.sin(a), save="in")
def erf(x): return _unary(x, "ErfBackward0", lambda a: _erf_vec(a).astype(a.dtype), lambda g, a, o: g * (2 / _math.sqrt(_math.pi)) * _np.exp(-a * a), save="in")
def reciprocal(x): return _unary(x, "ReciprocalBackward0", lambda a: 1 / a, lambda g, a, o: -g * o * o, save="out")
def relu(x): return _unary(x, "ReluBackward0", lambda a: _np.maximum(a, a.dtype.type(0)), lambda g, a, o: g * (o > 0), keep_int=True, save="out")
def neg(x): return _unary(x, "NegBackward0", _np.negative, lambda g, a, o: -g, keep_int=True)
def sign(x): return _unary(x, "SignBackward0", _np.sign, lambda g, a, o: _np.zeros_like(g), keep_int=True)
def floor(x): return _unary(x, "FloorBackward0", _np.floor, lambda g, a, o: _np.zeros_like(g), keep_int=True)
def ceil(x): return _unary(x, "CeilBackward0", _np.ceil, lambda g, a, o: _np.zeros_like(g), keep_int=True)
def trunc(x): return _unary(x, "TruncBackward0", _np.trunc, lambda g, a, o: _np.zeros_like(g), keep_int=True)


def _abs(x):
    return _unary(_tensor_arg(x, "torch.abs"), "AbsBackward0", _np.abs, lambda g, a, o: g * _np.sign(a), keep_int=True, save="in")


def _round(x, decimals=0):
    x = _tensor_arg(x, "torch.round")
    if _cat(x._a.dtype) < 2:
        return _new(x._a.copy())
    # Rounds half to even, like torch (0.5 -> 0, 1.5 -> 2).
    return _unary(x, "RoundBackward0", lambda a: _np.round(a, decimals), lambda g, a, o: _np.zeros_like(g))


def _pow(x, y):
    if not isinstance(x, Tensor):
        return _binary(x, _tensor_arg(y, "torch.pow", "exponent"), "pow")
    return _binary(x, y, "pow")


def clamp(t, min=None, max=None):  # noqa: A002
    t = _tensor_arg(t, "torch.clamp")
    a = t._a
    if min is None and max is None:
        raise RuntimeError("torch.clamp: At least one of 'min' or 'max' must not be None")
    for v in (min, max):
        if v is Ellipsis:
            raise TypeError("clamp(): argument 'min'/'max' must be a Number, not ellipsis")
    lo = min._a if isinstance(min, Tensor) else min
    hi = max._a if isinstance(max, Tensor) else max
    out = a.astype(_np.int64) if a.dtype.kind != "f" else a  # wide enough for any Python int bound
    if lo is not None:
        out = _np.maximum(out, lo)
    if hi is not None:
        out = _np.minimum(out, hi)
    bound_float = any(isinstance(v, float) or (isinstance(v, _np.ndarray) and v.dtype.kind == "f") for v in (lo, hi))
    if a.dtype.kind != "f" and bound_float:
        out = out.astype(_np.float32)
    else:
        out = out.astype(a.dtype)
    if _cat(out.dtype) < 2 or not _track(t):
        return _new(out)

    def bw(g):
        keep = _np.ones(a.shape, dtype=bool)
        if lo is not None:
            keep &= a >= lo
        if hi is not None:
            keep &= a <= hi
        return (g * keep,)
    return _result(out, (t,), bw, "ClampBackward1", saved=(t,))


clip = clamp


def maximum(x, y):
    return _binary(_tensor_arg(x, "torch.maximum"), _tensor_arg(y, "torch.maximum", "other"), "maximum")


def minimum(x, y):
    return _binary(_tensor_arg(x, "torch.minimum"), _tensor_arg(y, "torch.minimum", "other"), "minimum")


# ---------------------------------------------------------------- matmul
def _matmul_error(a, b):
    if a.ndim == 1 and b.ndim == 1:
        return RuntimeError(f"inconsistent tensor size, expected tensor [{a.shape[0]}] and src [{b.shape[0]}] to have the same number of elements, but got {a.shape[0]} and {b.shape[0]} elements respectively")
    rows = int(_np.prod(a.shape[:-1])) if a.ndim > 1 else 1
    if b.ndim <= 2:
        bshape = f"{b.shape[0]}x{b.shape[1]}" if b.ndim == 2 else f"{b.shape[0]}x1"
        return RuntimeError(f"mat1 and mat2 shapes cannot be multiplied ({rows}x{a.shape[-1]} and {bshape})")
    return RuntimeError(f"Expected size for first two dimensions of batch2 tensor to be: [{b.shape[0]}, {a.shape[-1]}] but got: [{b.shape[0]}, {b.shape[-2]}].")


def matmul(x, y):
    x = _tensor_arg(x, "torch.matmul")
    y = _tensor_arg(y, "torch.matmul", "other")
    a, b = x._a, y._a
    if a.ndim == 0 or b.ndim == 0:
        raise RuntimeError(f"both arguments to matmul need to be at least 1D, but they are {a.ndim}D and {b.ndim}D")
    if a.dtype != b.dtype:
        raise RuntimeError(f"expected m1 and m2 to have the same dtype, but got: {_dt(a)._cname} != {_dt(b)._cname}")
    k_b = b.shape[0] if b.ndim == 1 else b.shape[-2]
    if a.shape[-1] != k_b:
        raise _matmul_error(a, b)
    try:
        _np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    except ValueError:
        raise _matmul_error(a, b) from None
    with _np.errstate(all="ignore"):
        if _cat(a.dtype) < 2:
            out = _np.matmul(a.astype(_I64), b.astype(_I64)).astype(a.dtype)
        else:
            out = _np.asarray(_np.matmul(a, b), dtype=a.dtype)
    if _cat(a.dtype) < 2 or not _track(x, y):
        return _new(out)

    def bw(g):
        if a.ndim == 1 and b.ndim == 1:
            return g * b, g * a
        a2 = a[None, :] if a.ndim == 1 else a
        b2 = b[:, None] if b.ndim == 1 else b
        g2 = g
        if a.ndim == 1:
            g2 = _np.expand_dims(g2, -2)
        if b.ndim == 1:
            g2 = _np.expand_dims(g2, -1)
        ga = _np.matmul(g2, _np.swapaxes(b2, -1, -2))
        gb = _np.matmul(_np.swapaxes(a2, -1, -2), g2)
        return _unbroadcast(ga, a2.shape).reshape(a.shape), _unbroadcast(gb, b2.shape).reshape(b.shape)
    name = "MmBackward0" if a.ndim == 2 and b.ndim == 2 else "MatmulBackward0"
    saved = ((y,) if x._node is not None else ()) + ((x,) if y._node is not None else ())
    return _result(out, (x, y), bw, name, saved=saved)


def mm(x, y):
    for t, arg in ((x, "input"), (y, "mat2")):
        if _tensor_arg(t, "torch.mm", arg).ndim != 2:
            raise RuntimeError(f"{'self' if arg == 'input' else 'mat2'} must be a matrix")
    return matmul(x, y)


def bmm(x, y):
    for t, arg in ((x, "input"), (y, "mat2")):
        if _tensor_arg(t, "torch.bmm", arg).ndim != 3:
            raise RuntimeError(f"batch1 must be a 3D tensor" if arg == "input" else "batch2 must be a 3D tensor")
    return matmul(x, y)


def outer(x, y):
    x = _tensor_arg(x, "torch.outer")
    y = _tensor_arg(y, "torch.outer", "vec2")
    return x.reshape(-1, 1) * y.reshape(1, -1)


def square(x):
    return _binary(_tensor_arg(x, "torch.square"), 2, "pow")


# ---------------------------------------------------------------- reductions
def _sum(x, dim=None, keepdim=False, dtype=None, keepdims=None):
    x = _tensor_arg(x, "torch.sum")
    if keepdims is not None:
        keepdim = keepdims
    a = x._a
    axes = _axes(a, dim)
    out_dt = dtype.np if dtype is not None else (a.dtype if _cat(a.dtype) == 2 else _I64)
    acc = _F32 if out_dt == _np.float16 else out_dt  # torch accumulates half in float
    out = _np.asarray(_np.sum(a, axis=axes, keepdims=keepdim and axes is not None, dtype=acc)).astype(out_dt)
    if keepdim and axes is None and dim is not None:
        out = out.reshape(a.shape if a.ndim == 0 else (1,) * a.ndim)
    if _cat(out_dt) < 2 or not _track(x):
        return _new(out)
    shape = a.shape
    return _result(out, (x,), lambda g: (_expand_back(g, shape, axes, keepdim),), "SumBackward1" if dim is not None else "SumBackward0")


def _mean(x, dim=None, keepdim=False, dtype=None, keepdims=None):
    x = _tensor_arg(x, "torch.mean")
    if keepdims is not None:
        keepdim = keepdims
    a = x._a
    if dtype is not None:
        a = a.astype(dtype.np)
    if a.dtype.kind != "f":
        raise RuntimeError("mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: " + _dt(a)._short)
    axes = _axes(a, dim)
    n = a.size if axes is None else int(_np.prod([a.shape[i] for i in axes]))
    with _np.errstate(all="ignore"):
        acc = _np.sum(a, axis=axes, keepdims=keepdim and axes is not None, dtype=_F32 if a.dtype == _np.float16 else a.dtype)
        out = _np.asarray(acc / a.dtype.type(n) if n else acc * _np.nan).astype(a.dtype)
    if keepdim and axes is None and dim is not None:
        out = out.reshape((1,) * a.ndim)
    if not _track(x):
        return _new(out)
    shape = a.shape
    return _result(out, (x,), lambda g: (_expand_back(g, shape, axes, keepdim) / a.dtype.type(n),), "MeanBackward1" if dim is not None else "MeanBackward0")


def _var(x, dim=None, unbiased=None, keepdim=False, correction=None, keepdims=None):
    x = _tensor_arg(x, "torch.var")
    if keepdims is not None:
        keepdim = keepdims
    if isinstance(dim, _builtins.bool) and unbiased is None:  # legacy x.var(False)
        dim, unbiased = None, dim
    c = correction if correction is not None else (0 if unbiased is False else 1)
    if _cat(x._a.dtype) < 2:
        raise RuntimeError("std and var only support floating point and complex dtypes")
    axes = _axes(x._a, dim)
    n = x._a.size if axes is None else int(_np.prod([x._a.shape[i] for i in axes]))
    d = x - _mean(x, dim, keepdim=True)
    return _sum(d * d, dim, keepdim=keepdim) / _builtins.max(n - c, 0)


def _std(x, dim=None, unbiased=None, keepdim=False, correction=None, keepdims=None):
    return sqrt(_var(x, dim, unbiased, keepdim, correction, keepdims))


def _check_nonempty(a, fn):
    if a.size == 0:
        raise RuntimeError(f"{fn}(): Expected reduction over a non-empty tensor")


def _even_extreme(x, out, axes, keepdim, name):
    """Grad of max()/amax(): split evenly between tied extremes, like torch."""
    a = x._a
    shape = a.shape

    def bw(g):
        full = _expand_back(out, shape, axes, keepdim)
        mask = (a == full)
        count = mask.sum(axis=axes, keepdims=True) if axes is not None else mask.sum()
        return (_expand_back(g, shape, axes, keepdim) * mask / count,)
    return _result(out, (x,), bw, name, saved=(x,))


def _extreme(x, dim, keepdim, is_max, fn):
    x = _tensor_arg(x, f"torch.{fn}")
    a = x._a
    _check_nonempty(a, fn)
    if isinstance(dim, Tensor):
        return (maximum if is_max else minimum)(x, dim)
    if dim is None:
        if keepdim:
            raise TypeError(f"{fn}() received an invalid combination of arguments: keepdim needs dim")
        out = _np.asarray(a.max() if is_max else a.min())
        if _cat(a.dtype) < 2 or not _track(x):
            return _new(out)
        return _even_extreme(x, out, None, False, "MaxBackward1" if is_max else "MinBackward1")
    d = _dim(a, dim)
    a1 = a.reshape(1) if a.ndim == 0 else a
    values = _np.asarray(_np.max(a1, axis=d, keepdims=keepdim) if is_max else _np.min(a1, axis=d, keepdims=keepdim))
    idx = _np.asarray(_np.argmax(a1, axis=d) if is_max else _np.argmin(a1, axis=d), dtype=_np.int64)
    if keepdim:
        idx = _np.expand_dims(idx, d)
    if a.ndim == 0:
        values, idx = values.reshape(()), idx.reshape(())
    cls = _MaxResult if is_max else _MinResult
    if _cat(a.dtype) < 2 or not _track(x):
        return cls(_new(values), _new(idx))
    shape = a.shape

    def bw(g):
        z = _np.zeros(shape, dtype=g.dtype)
        if a.ndim == 0:
            return (g.reshape(()),)
        ii = idx if keepdim else _np.expand_dims(idx, d)
        gg = g if keepdim else _np.expand_dims(g, d)
        _np.put_along_axis(z, ii, gg, axis=d)
        return (z,)
    return cls(_result(values, (x,), bw, "MaxBackward0" if is_max else "MinBackward0"), _new(idx))


def _max(x, dim=None, keepdim=False, other=None):
    if other is not None:
        return maximum(x, other)
    return _extreme(x, dim, keepdim, True, "max")


def _min(x, dim=None, keepdim=False, other=None):
    if other is not None:
        return minimum(x, other)
    return _extreme(x, dim, keepdim, False, "min")


def _aminmax(x, dim, keepdim, is_max):
    fn = "amax" if is_max else "amin"
    x = _tensor_arg(x, f"torch.{fn}")
    a = x._a
    _check_nonempty(a, fn)
    if isinstance(dim, (tuple, list)) and not dim:
        dim = None
    axes = _axes(a, dim)
    red = _np.max if is_max else _np.min
    if axes is None:
        out = _np.asarray(red(a, keepdims=keepdim and dim is not None and a.ndim > 0))
        if keepdim and dim is None and a.ndim:
            out = out.reshape((1,) * a.ndim)
    else:
        out = _np.asarray(red(a, axis=axes, keepdims=keepdim))
    if _cat(a.dtype) < 2 or not _track(x):
        return _new(out)
    return _even_extreme(x, out, axes, keepdim and axes is not None, "AmaxBackward0" if is_max else "AminBackward0")


def amax(x, dim=(), keepdim=False):
    return _aminmax(x, None if dim == () else dim, keepdim, True)


def amin(x, dim=(), keepdim=False):
    return _aminmax(x, None if dim == () else dim, keepdim, False)


def _argext(x, dim, keepdim, is_max):
    fn = "argmax" if is_max else "argmin"
    x = _tensor_arg(x, f"torch.{fn}")
    a = x._a
    _check_nonempty(a, fn)
    f = _np.argmax if is_max else _np.argmin
    if dim is None:
        out = _np.asarray(f(a), dtype=_np.int64)
        if keepdim:
            out = out.reshape((1,) * a.ndim)
        return _new(out)
    d = _dim(a, dim)
    if a.ndim == 0:
        return _new(_np.asarray(0, dtype=_np.int64))
    out = _np.asarray(f(a, axis=d), dtype=_np.int64)
    if keepdim:
        out = _np.expand_dims(out, d)
    return _new(out)


def argmax(x, dim=None, keepdim=False):
    return _argext(x, dim, keepdim, True)


def argmin(x, dim=None, keepdim=False):
    return _argext(x, dim, keepdim, False)


def _any(x, dim=None, keepdim=False):
    x = _tensor_arg(x, "torch.any")
    axes = _axes(x._a, dim)
    return _new(_np.asarray(_np.any(x._a, axis=axes, keepdims=keepdim and axes is not None)))


def _all(x, dim=None, keepdim=False):
    x = _tensor_arg(x, "torch.all")
    axes = _axes(x._a, dim)
    return _new(_np.asarray(_np.all(x._a, axis=axes, keepdims=keepdim and axes is not None)))


def norm(x, p="fro", dim=None, keepdim=False):
    x = _tensor_arg(x, "torch.norm")
    if p in ("fro", 2, 2.0):
        a = x._a
        if _cat(a.dtype) < 2:
            raise RuntimeError(f"linalg.vector_norm: Expected a floating point or complex tensor as input. Got {_dt(a)._short}")
        axes = _axes(a, dim)
        sq = _np.sum(a.astype(_F64) ** 2, axis=axes, keepdims=keepdim and axes is not None)
        out = _np.asarray(_np.sqrt(sq)).astype(a.dtype)
        if not _track(x):
            return _new(out)

        def bw(g):
            full = _expand_back(out, a.shape, axes, keepdim)
            with _np.errstate(all="ignore"):
                ratio = _np.where(full == 0, 0, a / full)  # torch's subgradient: 0 at the origin
            return (_expand_back(g, a.shape, axes, keepdim) * ratio,)
        return _result(out, (x,), bw, "LinalgVectorNormBackward0", saved=(x,), save_out=True)
    if p in (1, 1.0):
        return _sum(_abs(x), dim, keepdim=keepdim)
    raise _unsupported(f"torch.norm(p={p!r})")


def cumsum(x, dim, dtype=None):
    x = _tensor_arg(x, "torch.cumsum")
    a = x._a
    d = _dim(a, dim)
    out_dt = dtype.np if dtype is not None else (a.dtype if _cat(a.dtype) == 2 else _I64)
    a1 = a.reshape(1) if a.ndim == 0 else a
    out = _np.cumsum(a1, axis=d, dtype=out_dt).reshape(a.shape)
    if _cat(out_dt) < 2 or not _track(x):
        return _new(out)

    def bw(g):
        g1 = g.reshape(1) if g.ndim == 0 else g
        return (_np.flip(_np.cumsum(_np.flip(g1, d), axis=d), d).reshape(a.shape),)
    return _result(out, (x,), bw, "CumsumBackward0")


# ---------------------------------------------------------------- softmax family
def _softmax(x, dim, dtype=None, fn="softmax"):
    x = _tensor_arg(x, f"torch.{fn}")
    a = x._a
    if dtype is not None:
        a = a.astype(dtype.np)
    if _cat(a.dtype) < 2:
        raise RuntimeError(f"\"{fn}_lastdim_kernel_impl\" not implemented for '{_dt(a)._short}'")
    if a.ndim == 0:
        _dim(a, dim)
        out = _np.ones_like(a) if fn == "softmax" else _np.zeros_like(a)
        return _result(out, (x,), lambda g: (_np.zeros_like(g),), "SoftmaxBackward0")
    d = _dim(a, dim)
    with _np.errstate(all="ignore"):
        m = _np.max(a, axis=d, keepdims=True)
        m = _np.where(_np.isfinite(m), m, 0)
        z = a - m
        e = _np.exp(z)
        s = _np.sum(e, axis=d, keepdims=True)
        if fn == "softmax":
            out = (e / s).astype(a.dtype)
        else:
            out = (z - _np.log(s)).astype(a.dtype)
    if not _track(x):
        return _new(out)
    if fn == "softmax":
        def bw(g):
            return (out * (g - _np.sum(g * out, axis=d, keepdims=True)),)
        return _result(out, (x,), bw, "SoftmaxBackward0", save_out=True)

    def bw_log(g):
        return (g - _np.exp(out) * _np.sum(g, axis=d, keepdims=True),)
    return _result(out, (x,), bw_log, "LogSoftmaxBackward0", save_out=True)


def softmax(x, dim, dtype=None):
    return _softmax(x, dim, dtype, "softmax")


def log_softmax(x, dim, dtype=None):
    return _softmax(x, dim, dtype, "log_softmax")


# ---------------------------------------------------------------- sorting and sampling
def _stable_argsort(a, d, descending):
    if descending:
        key = -(a.astype(_F64) if a.dtype.kind == "f" else a.astype(_I64))
    else:
        key = a
    return _np.argsort(key, axis=d, kind="stable")


def _take_result(x, idx, d, name):
    """Values picked along dim d by idx (sort/topk), with the matching backward."""
    a = x._a
    vals = _np.take_along_axis(a, idx, axis=d)
    if _cat(a.dtype) < 2 or not _track(x):
        return _new(vals)
    shape = a.shape

    def bw(g):
        z = _np.zeros(shape, dtype=g.dtype)
        _np.put_along_axis(z, idx, g, axis=d)
        return (z,)
    return _result(vals, (x,), bw, name)


def sort(x, dim=-1, descending=False, stable=False):
    x = _tensor_arg(x, "torch.sort")
    a = x._a
    if a.ndim == 0:
        _dim(a, dim)
        return _SortResult(x.clone(), _new(_np.asarray(0, dtype=_np.int64)))
    d = _dim(a, dim)
    idx = _stable_argsort(a, d, descending)
    return _SortResult(_take_result(x, idx, d, "SortBackward0"), _new(idx.astype(_np.int64)))


def argsort(x, dim=-1, descending=False, stable=False):
    return sort(x, dim, descending, stable).indices


def topk(x, k, dim=-1, largest=True, sorted=True):  # noqa: A002
    x = _tensor_arg(x, "torch.topk")
    a = x._a
    a1 = x if a.ndim else x.reshape(1)
    d = _dim(a1._a, dim)
    if not 0 <= k <= a1._a.shape[d]:
        raise RuntimeError("selected index k out of range")
    idx = _np.take(_stable_argsort(a1._a, d, largest), _np.arange(k), axis=d)
    vals = _take_result(a1, idx, d, "TopkBackward0")
    if a.ndim == 0:
        return _TopkResult(vals.reshape(()), _new(idx.reshape(()).astype(_np.int64)))
    return _TopkResult(vals, _new(idx.astype(_np.int64)))


def multinomial(input, num_samples, replacement=False, generator=None):  # noqa: A002
    x = _tensor_arg(input, "torch.multinomial")
    a = x._a
    if a.ndim not in (1, 2):
        raise RuntimeError("prob_dist must be 1 or 2 dim")
    if _cat(a.dtype) < 2:
        raise RuntimeError("multinomial only supports floating-point dtypes for input, got: " + _dt(a)._short)
    p = a.astype(_F64).reshape(-1, a.shape[-1])
    n_cat = p.shape[1]
    if not _np.all(_np.isfinite(p)) or _np.any(p < 0):
        raise RuntimeError("probability tensor contains either `inf`, `nan` or element < 0")
    if not replacement and num_samples > n_cat:
        raise RuntimeError("cannot sample n_sample > prob_dist.size(-1) samples without replacement")
    if num_samples <= 0:
        raise RuntimeError("cannot sample n_sample <= 0 samples")
    rng = _gen(generator)._rng
    out = _np.zeros((p.shape[0], num_samples), dtype=_np.int64)
    for r in range(p.shape[0]):
        w = p[r].copy()
        if w.sum() <= 0:
            raise RuntimeError("invalid multinomial distribution (sum of probabilities <= 0)")
        if not replacement and _np.count_nonzero(w) < num_samples:
            raise RuntimeError("invalid multinomial distribution (with replacement=False, not enough non-negative category to sample)")
        if replacement:
            cdf = _np.cumsum(w)
            cdf /= cdf[-1]
            out[r] = _np.minimum(_np.searchsorted(cdf, rng.random(num_samples), side="right"), n_cat - 1)
        else:
            for s in range(num_samples):
                cdf = _np.cumsum(w)
                cdf /= cdf[-1]
                k = int(_np.minimum(_np.searchsorted(cdf, rng.random(), side="right"), n_cat - 1))
                out[r, s] = k
                w[k] = 0.0
    return _new(out.reshape(num_samples) if a.ndim == 1 else out)


# ---------------------------------------------------------------- comparisons and selection
def isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):  # noqa: A002
    x = _tensor_arg(input, "torch.isclose")
    y = _tensor_arg(other, "torch.isclose", "other")
    if x._a.dtype != y._a.dtype:
        raise RuntimeError(f"{_dt(x._a)._short} did not match {_dt(y._a)._short}")
    _check_broadcast(x._a, y._a)
    return _new(_np.isclose(x._a, y._a, rtol=rtol, atol=atol, equal_nan=equal_nan))


def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):  # noqa: A002
    return _builtins.bool(isclose(input, other, rtol, atol, equal_nan)._a.all())


def equal(input, other):  # noqa: A002
    x = _tensor_arg(input, "torch.equal")
    y = _tensor_arg(other, "torch.equal", "other")
    return x._a.shape == y._a.shape and _builtins.bool(_np.all(x._a == y._a))


def where(condition, input=None, other=None):  # noqa: A002
    c = _tensor_arg(condition, "torch.where", "condition")
    if input is None and other is None:
        return nonzero(c, as_tuple=True)
    if c._a.dtype != _BOOL:
        raise RuntimeError(f"where expected condition to be a boolean tensor, but got a tensor with dtype {_dt(c._a)._short}")
    x, y = _coerce(input), _coerce(other)
    xa, ya = _raw(x), _raw(y)
    _check_broadcast(c._a, xa)
    _check_broadcast(c._a, ya)
    _check_broadcast(xa, ya)
    rt = _result_type(x, y)
    cond = c._a
    out = _np.where(cond, _np.asarray(xa, dtype=rt), _np.asarray(ya, dtype=rt))
    if _cat(rt) < 2 or not _track(x, y):
        return _new(out)
    xs, ys = _np.shape(xa), _np.shape(ya)

    def bw(g):
        zero = _np.zeros_like(g)
        return _unbroadcast(_np.where(cond, g, zero), xs), _unbroadcast(_np.where(cond, zero, g), ys)
    return _result(out, (x, y), bw, "WhereBackward0")


def masked_fill(x, mask, value):
    x = _tensor_arg(x, "torch.masked_fill")
    m = _tensor_arg(mask, "masked_fill", "mask")
    if m._a.dtype != _BOOL:
        raise RuntimeError(f"masked_fill_ only supports boolean masks, but got mask with dtype {_dt(m._a)._cname}")
    _check_broadcast(x._a, m._a)
    if _np.broadcast_shapes(x._a.shape, m._a.shape) != x._a.shape:
        raise RuntimeError(f"The expanded size of the mask {list(m._a.shape)} doesn't fit the tensor {list(x._a.shape)}")
    v = value.item() if isinstance(value, Tensor) else value
    a = x._a
    fill = _to_dtype(_np.asarray(v), _dt(a)) if _cat(a.dtype) < 2 else _np.asarray(v, dtype=a.dtype)
    mk = m._a
    out = _np.where(mk, fill, a)
    if _cat(a.dtype) < 2 or not _track(x):
        return _new(out)
    return _result(out, (x,), lambda g: (_np.where(mk, 0, g).astype(g.dtype),), "MaskedFillBackward0")


def _tri(x, diagonal, upper):
    fn = "triu" if upper else "tril"
    x = _tensor_arg(x, f"torch.{fn}")
    a = x._a
    if a.ndim < 2:
        raise RuntimeError(f"{fn}: input tensor must have at least 2 dimensions")
    f = _np.triu if upper else _np.tril
    out = f(a, diagonal)
    if _cat(a.dtype) < 2 or not _track(x):
        return _new(out)
    return _result(out, (x,), lambda g: (f(g, diagonal),), "TriuBackward0" if upper else "TrilBackward0")


def tril(x, diagonal=0):
    return _tri(x, diagonal, False)


def triu(x, diagonal=0):
    return _tri(x, diagonal, True)


def nonzero(x, as_tuple=False):
    x = _tensor_arg(x, "torch.nonzero")
    idx = _np.nonzero(x._a.reshape(1) if x._a.ndim == 0 else x._a)
    if as_tuple:
        return tuple(_new(i.astype(_np.int64)) for i in idx)
    return _new(_np.stack(idx, axis=1).astype(_np.int64) if idx else _np.zeros((0, 0), dtype=_np.int64))


def logical_not(x):
    return _new(_np.logical_not(_tensor_arg(x, "torch.logical_not")._a))


def logical_and(x, y):
    return _new(_np.logical_and(_tensor_arg(x, "torch.logical_and")._a, _raw(_coerce(y))))


def logical_or(x, y):
    return _new(_np.logical_or(_tensor_arg(x, "torch.logical_or")._a, _raw(_coerce(y))))


# ---------------------------------------------------------------- shape ops
def _shaped(x, out, bw, name):
    """Result of a shape op: a view sharing memory when untracked, a tracked copy otherwise."""
    if _track(x):
        return _result(out.copy() if _np.may_share_memory(out, x._a) else out, (x,), bw, name)
    if _np.may_share_memory(out, x._a):
        return _new(out, ver=x._ver, view=True)
    return _new(out)


def _reshape(x, shape, fn="reshape"):
    shape = _shape_arg(shape, fn)
    n = x._a.size
    if shape.count(-1) > 1:
        raise RuntimeError("only one dimension can be inferred")
    known = 1
    for v in shape:
        if v != -1:
            known *= v
    if any(v < -1 for v in shape) or (-1 in shape and (known == 0 or n % known)) or (-1 not in shape and known != n):
        raise RuntimeError(f"shape '{list(shape)}' is invalid for input of size {n}")
    src = x._a.shape
    return _shaped(x, x._a.reshape(shape), lambda g: (g.reshape(src),), "ViewBackward0")


def reshape(x, shape):
    return _reshape(_tensor_arg(x, "torch.reshape"), (shape,))


def transpose(x, dim0, dim1):
    x = _tensor_arg(x, "torch.transpose")
    a = x._a
    if a.ndim == 0:
        _dim(a, dim0)
        _dim(a, dim1)
        return _shaped(x, a, lambda g: (g,), "TransposeBackward0")
    d0, d1 = _dim(a, dim0), _dim(a, dim1)
    return _shaped(x, _np.swapaxes(a, d0, d1), lambda g: (_np.swapaxes(g, d0, d1),), "TransposeBackward0")


swapaxes = transpose


def permute(x, *dims):
    x = _tensor_arg(x, "torch.permute")
    dims = _shape_arg(dims, "permute")
    a = x._a
    if len(dims) != a.ndim:
        raise RuntimeError(f"permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = {a.ndim} is not equal to len(dims) = {len(dims)}")
    nd = [_dim(a, d) for d in dims]
    if len(set(nd)) != len(nd):
        raise RuntimeError("permute(): duplicate dims are not allowed.")
    inv = tuple(int(i) for i in _np.argsort(nd))
    return _shaped(x, _np.transpose(a, nd), lambda g: (_np.transpose(g, inv),), "PermuteBackward0")


def unsqueeze(x, dim):
    x = _tensor_arg(x, "torch.unsqueeze")
    a = x._a
    d = _norm_dim(a.ndim + 1, dim)
    src = a.shape
    return _shaped(x, _np.expand_dims(a, d), lambda g: (g.reshape(src),), "UnsqueezeBackward0")


def squeeze(x, dim=None):
    x = _tensor_arg(x, "torch.squeeze")
    a = x._a
    if dim is None:
        axes = tuple(i for i, s in enumerate(a.shape) if s == 1)
    else:
        dims = dim if isinstance(dim, (tuple, list)) else (dim,)
        axes = tuple(sorted({_dim(a, d) for d in dims if a.ndim and a.shape[_dim(a, d)] == 1}))
        if a.ndim == 0:
            for d in dims:
                _dim(a, d)
    src = a.shape
    return _shaped(x, _np.squeeze(a, axis=axes) if axes else a, lambda g: (g.reshape(src),), "SqueezeBackward0")


def flatten(x, start_dim=0, end_dim=-1):
    x = _tensor_arg(x, "torch.flatten")
    a = x._a
    if a.ndim == 0:
        return _reshape(x, (1,))
    s, e = _dim(a, start_dim), _dim(a, end_dim)
    if s > e:
        raise RuntimeError("flatten() has invalid args: start_dim cannot come after end_dim")
    shape = a.shape[:s] + (int(_np.prod(a.shape[s:e + 1])),) + a.shape[e + 1:]
    return _reshape(x, shape)


def _expand(x, sizes):
    sizes = _shape_arg(sizes, "expand")
    a = x._a
    if len(sizes) < a.ndim:
        raise RuntimeError(f"expand(torch.FloatTensor{{{list(a.shape)}}}, size={list(sizes)}): the number of sizes provided ({len(sizes)}) must be greater or equal to the number of dimensions in the tensor ({a.ndim})")
    lead = len(sizes) - a.ndim
    target = []
    for i, s in enumerate(sizes):
        cur = a.shape[i - lead] if i >= lead else None
        if s == -1:
            if cur is None:
                raise RuntimeError(f"The expanded size of the tensor (-1) isn't allowed in a leading, non-existing dimension {i}")
            s = cur
        elif cur is not None and cur != s and cur != 1:
            raise RuntimeError(f"The expanded size of the tensor ({s}) must match the existing size ({cur}) at non-singleton dimension {i}.  Target sizes: {list(sizes)}.  Tensor sizes: {list(a.shape)}")
        target.append(s)
    src = a.shape
    return _shaped(x, _np.broadcast_to(a, tuple(target)), lambda g: (_unbroadcast(g, src),), "ExpandBackward0")


def _concat_check(tensors, fn):
    if isinstance(tensors, Tensor):
        raise TypeError(f"{fn}(): argument 'tensors' (position 1) must be tuple of Tensors, not Tensor")
    ts = list(tensors)
    if not ts:
        raise RuntimeError(f"{fn}(): expected a non-empty list of Tensors")
    for t in ts:
        _tensor_arg(t, fn, "tensors")
    return ts


def cat(tensors, dim=0):
    ts = _concat_check(tensors, "torch.cat")
    ref = next((t for t in ts if t._a.ndim), None)
    if ref is None:
        raise RuntimeError("zero-dimensional tensor (at position 0) cannot be concatenated")
    d = _dim(ref._a, dim)
    for i, t in enumerate(ts):
        if t._a.ndim == 0:
            raise RuntimeError(f"zero-dimensional tensor (at position {i}) cannot be concatenated")
        if t._a.ndim != ref._a.ndim:
            raise RuntimeError(f"Tensors must have same number of dimensions: got {ref._a.ndim} and {t._a.ndim}")
        for k in range(ref._a.ndim):
            if k != d and t._a.shape[k] != ref._a.shape[k]:
                raise RuntimeError(f"Sizes of tensors must match except in dimension {d}. Expected size {ref._a.shape[k]} but got size {t._a.shape[k]} for tensor number {i} in the list.")
    rt = _result_type(*[_new(t._a.reshape(-1)) for t in ts])
    out = _np.concatenate([t._a.astype(rt, copy=False) for t in ts], axis=d)
    if _cat(rt) < 2 or not _track(*ts):
        return _new(out)
    bounds = _np.cumsum([t._a.shape[d] for t in ts])[:-1]

    def bw(g):
        return tuple(_np.split(g, bounds, axis=d))
    return _result(out, tuple(ts), bw, "CatBackward0")


concat = concatenate = cat


def stack(tensors, dim=0):
    ts = _concat_check(tensors, "torch.stack")
    for i, t in enumerate(ts):
        if t._a.shape != ts[0]._a.shape:
            raise RuntimeError(f"stack expects each tensor to be equal size, but got {list(ts[0]._a.shape)} at entry 0 and {list(t._a.shape)} at entry {i}")
    d = _norm_dim(ts[0]._a.ndim + 1, dim)
    rt = _result_type(*[_new(t._a.reshape(-1)) for t in ts])
    out = _np.stack([t._a.astype(rt, copy=False) for t in ts], axis=d)
    if _cat(rt) < 2 or not _track(*ts):
        return _new(out)

    def bw(g):
        return tuple(_np.take(g, i, axis=d) for i in range(len(ts)))
    return _result(out, tuple(ts), bw, "StackBackward0")


def _narrow(x, d, start, length):
    idx = [slice(None)] * x._a.ndim
    idx[d] = slice(start, start + length)
    return x[tuple(idx)]


def split(x, split_size_or_sections, dim=0):
    x = _tensor_arg(x, "torch.split")
    d = _dim(x._a, dim)
    n = x._a.shape[d]
    if isinstance(split_size_or_sections, int):
        size = split_size_or_sections
        if size <= 0 and n:
            raise RuntimeError(f"split_size can only be 0 if dimension size is 0, but got dimension size of {n}")
        sizes = [size] * (n // size) + ([n % size] if n % size else []) if size else [0]
    else:
        sizes = list(split_size_or_sections)
        if _builtins.sum(sizes) != n:
            raise RuntimeError(f"split_with_sizes expects split_sizes to sum exactly to {n} (input tensor's size at dimension {d}), but got split_sizes={sizes}")
    out, start = [], 0
    for s in sizes:
        out.append(_narrow(x, d, start, s))
        start += s
    return tuple(out)


def chunk(x, chunks, dim=0):
    x = _tensor_arg(x, "torch.chunk")
    if chunks <= 0:
        raise RuntimeError("chunk expects `chunks` to be greater than 0, got: " + str(chunks))
    n = x._a.shape[_dim(x._a, dim)]
    size = -(-n // chunks) if n else 1
    return split(x, size, dim)


def unbind(x, dim=0):
    x = _tensor_arg(x, "torch.unbind")
    d = _dim(x._a, dim)
    idx = [slice(None)] * x._a.ndim
    out = []
    for i in range(x._a.shape[d]):
        idx[d] = i
        out.append(x[tuple(idx)])
    return tuple(out)


def gather(x, dim, index):
    x = _tensor_arg(x, "torch.gather")
    ix = _tensor_arg(index, "torch.gather", "index")
    a = x._a
    if ix._a.dtype != _I64:
        raise RuntimeError("gather(): Expected dtype int64 for index")
    if ix._a.ndim != a.ndim:
        raise RuntimeError("Index tensor must have the same number of dimensions as input tensor")
    d = _dim(a, dim)
    for k in range(a.ndim):
        if k != d and ix._a.shape[k] > a.shape[k]:
            raise RuntimeError(f"Size does not match at dimension {k} expected index {list(ix._a.shape)} to be smaller than self {list(a.shape)} apart from dimension {d}")
    bad = (ix._a < 0) | (ix._a >= a.shape[d])
    if bad.any():
        raise RuntimeError(f"index {int(ix._a[bad][0])} is out of bounds for dimension {d} with size {a.shape[d]}")
    full = list(_np.indices(ix._a.shape, sparse=True))
    full[d] = ix._a
    full = tuple(full)
    out = a[full]
    if _cat(a.dtype) < 2 or not _track(x):
        return _new(out)
    shape = a.shape

    def bw(g):
        z = _np.zeros(shape, dtype=g.dtype)
        _np.add.at(z, full, g)
        return (z,)
    return _result(out, (x,), bw, "GatherBackward0")


def index_select(x, dim, index):
    x = _tensor_arg(x, "torch.index_select")
    ix = _tensor_arg(index, "torch.index_select", "index")
    if ix._a.ndim > 1:
        raise IndexError("index_select(): Index is supposed to be a vector")
    d = _dim(x._a, dim)
    idx = [slice(None)] * x._a.ndim
    idx[d] = ix.reshape(-1)
    return x[tuple(idx)]


def _scatter_index(a, d, ix, fn):
    if ix._a.dtype != _I64:
        raise RuntimeError(f"{fn}(): Expected dtype int64 for index")
    if ix._a.ndim != a.ndim:
        raise RuntimeError("Index tensor must have the same number of dimensions as self tensor")
    bad = (ix._a < 0) | (ix._a >= a.shape[d])
    if bad.any():
        raise RuntimeError(f"index {int(ix._a[bad][0])} is out of bounds for dimension {d} with size {a.shape[d]}")
    full = list(_np.indices(ix._a.shape, sparse=True))
    full[d] = ix._a
    return tuple(full)


def scatter(x, dim, index, src=None, value=None, reduce=None):
    """out = x.clone(); out[index[i][j]][j] = src[i][j] (for dim=0), like torch.scatter."""
    x = _tensor_arg(x, "torch.scatter")
    ix = _tensor_arg(index, "torch.scatter", "index")
    if reduce is not None:
        raise _unsupported("scatter(reduce=...)")
    if src is None:
        src = value
    a = x._a
    d = _dim(a, dim)
    full = _scatter_index(a, d, ix, "scatter")
    out = a.copy()
    s = _coerce(src)
    if isinstance(s, Tensor):
        sa = s._a[tuple(slice(0, n) for n in ix._a.shape)]
        out[full] = sa if sa.dtype == a.dtype else _to_dtype(sa, _dt(a))
    else:
        out[full] = _to_dtype(_np.asarray(s), _dt(a))
    if _cat(a.dtype) < 2 or not _track(x, s):
        return _new(out)
    src_shape = s._a.shape if isinstance(s, Tensor) else ()

    def bw(g):
        gx = g.copy()
        gx[full] = 0
        gs = None
        if isinstance(s, Tensor):
            gs = _np.zeros(src_shape, dtype=g.dtype)
            gs[tuple(slice(0, n) for n in ix._a.shape)] = g[full]
        return gx, gs
    return _result(out, (x, s), bw, "ScatterBackward0")


def scatter_add(x, dim, index, src):
    x = _tensor_arg(x, "torch.scatter_add")
    ix = _tensor_arg(index, "torch.scatter_add", "index")
    s = _tensor_arg(src, "torch.scatter_add", "src")
    a = x._a
    d = _dim(a, dim)
    full = _scatter_index(a, d, ix, "scatter_add")
    out = a.copy()
    sa = s._a[tuple(slice(0, n) for n in ix._a.shape)]
    _np.add.at(out, full, sa if sa.dtype == a.dtype else _to_dtype(sa, _dt(a)))
    if _cat(a.dtype) < 2 or not _track(x, s):
        return _new(out)
    src_shape = s._a.shape

    def bw(g):
        gs = _np.zeros(src_shape, dtype=g.dtype)
        gs[tuple(slice(0, n) for n in ix._a.shape)] = g[full]
        return g, gs
    return _result(out, (x, s), bw, "ScatterAddBackward0")


def flip(x, dims):
    x = _tensor_arg(x, "torch.flip")
    dims = (dims,) if isinstance(dims, int) else tuple(dims)
    axes = tuple(_dim(x._a, d) for d in dims)
    return _shaped(x, _np.flip(x._a, axes).copy(), lambda g: (_np.flip(g, axes),), "FlipBackward0")


def _repeat(x, sizes):
    sizes = _shape_arg(sizes, "repeat")
    a = x._a
    if len(sizes) < a.ndim:
        raise RuntimeError("Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor")
    a2 = a.reshape((1,) * (len(sizes) - a.ndim) + a.shape)
    out = _np.tile(a2, sizes)
    src = a.shape

    def bw(g):
        # Fold each repeated dim into (repeats, size) and sum the repeats away.
        shape = []
        for r, n in zip(sizes, a2.shape):
            shape += [r, n]
        return (g.reshape(shape).sum(axis=tuple(range(0, 2 * len(sizes), 2))).reshape(src),)
    return _shaped(x, out, bw, "RepeatBackward0")


def repeat_interleave(x, repeats, dim=None):
    x = _tensor_arg(x, "torch.repeat_interleave")
    if not isinstance(repeats, int):
        raise _unsupported("repeat_interleave(repeats=<tensor>)")
    a = x._a
    if dim is None:
        x, a, d = x.reshape(-1), a.reshape(-1), 0
    else:
        d = _dim(a, dim)
    out = _np.repeat(a, repeats, axis=d)
    src = a.shape

    def bw(g):
        shape = src[:d] + (src[d], repeats) + src[d + 1:]
        return (g.reshape(shape).sum(axis=d + 1),)
    return _shaped(x, out, bw, "RepeatInterleaveBackward0")


# ---------------------------------------------------------------- indexing
def _index_part(i):
    if isinstance(i, Tensor):
        a = i._a
        if a.dtype == _BOOL or a.dtype == _np.uint8:
            return a.astype(bool)
        if a.dtype in (_I64, _np.dtype(_np.int32)):
            return int(a) if a.ndim == 0 else a  # a 0-d index selects, like an int
        raise IndexError("tensors used as indices must be long, int, byte or bool tensors")
    if isinstance(i, _np.ndarray):
        return i
    if isinstance(i, range):
        return _np.arange(i.start, i.stop, i.step, dtype=_np.int64)
    if isinstance(i, (list, tuple)):
        parts = [p.item() if isinstance(p, Tensor) else p for p in i]
        arr = _np.asarray(parts)
        if arr.size == 0:
            return arr.astype(_np.int64)
        if arr.dtype.kind == "f":
            raise IndexError("tensors used as indices must be long, int, byte or bool tensors")
        return arr
    if isinstance(i, slice):
        if i.step is not None and int(i.step) <= 0:
            raise ValueError("step must be greater than zero")
        return i
    if isinstance(i, _np.integer):
        return int(i)
    if isinstance(i, float):
        raise IndexError("only integers, slices (`:`), ellipsis (`...`), None and long or byte Variables are valid indices (got float)")
    if i is Ellipsis or i is None or isinstance(i, (int, _builtins.bool)):
        return i
    raise IndexError(f"only integers, slices (`:`), ellipsis (`...`), None and long or byte Variables are valid indices (got {type(i).__name__})")


def _index(idx):
    if isinstance(idx, tuple):
        return tuple(_index_part(i) for i in idx)
    return (_index_part(idx),)


def _at_index(idx):
    """The same index for np.add.at: boolean masks become their nonzero() index arrays."""
    out = []
    for i in idx:
        if isinstance(i, _np.ndarray) and i.dtype == _BOOL:
            out.extend(_np.nonzero(i))
        else:
            out.append(i)
    return tuple(out)


def _index_error(e):
    msg = str(e).replace("for axis", "for dimension")
    if msg.startswith("too many indices"):
        m = _re.search(r"is (\d+)-dimensional", msg)
        msg = f"too many indices for tensor of dimension {m.group(1)}" if m else msg
    return IndexError(msg)


def _getitem(x, idx):
    nidx = _index(idx)
    if all(isinstance(i, int) and not isinstance(i, _builtins.bool) for i in nidx):
        nidx = nidx + (Ellipsis,)  # NumPy returns a 0-d view instead of a detached scalar
    try:
        out = x._a[nidx]
    except IndexError as e:
        raise _index_error(e) from None
    out = _np.asarray(out)
    if _cat(x._a.dtype) < 2 or not _track(x):
        if _np.may_share_memory(out, x._a):
            return _new(out, ver=x._ver, view=True)
        return _new(out)
    shape = x._a.shape
    aidx = _at_index(nidx)

    def bw(g):
        z = _np.zeros(shape, dtype=g.dtype)
        _np.add.at(z, aidx, g)
        return (z,)
    return _result(out.copy() if _np.may_share_memory(out, x._a) else out, (x,), bw, "IndexBackward0")


def _index_put(src, idx, value):
    """Out-of-place `src[idx] = value`, differentiable in both."""
    nidx = _index(idx)
    a = src._a
    v = _coerce(value)
    va = v._a if isinstance(v, Tensor) else _np.asarray(v)
    out = a.copy()
    try:
        target_shape = out[nidx].shape
    except IndexError as e:
        raise _index_error(e) from None
    try:
        _np.broadcast_shapes(va.shape, target_shape)
    except ValueError:
        raise RuntimeError(f"shape mismatch: value tensor of shape {list(va.shape)} cannot be broadcast to indexing result of shape {list(target_shape)}") from None
    out[nidx] = _to_dtype(va, _dt(a)) if va.dtype != a.dtype else va
    if _cat(a.dtype) < 2 or not _track(src, v):
        return _new(out)
    vs = va.shape

    def bw(g):
        gs = g.copy()
        gs[nidx] = 0
        return gs, _unbroadcast(_np.asarray(g[nidx]), vs)
    return _result(out, (src, v), bw, "IndexPutBackward0")


# ---------------------------------------------------------------- in-place machinery
def _constant_like(s, value, name):
    """zero_()/fill_() as an op: the result is constant, so the old values get zero gradient."""
    out = _np.full_like(s._a, value)
    if _cat(out.dtype) < 2:
        return _new(out)
    return _result(out, (s,), lambda g: (_np.zeros_like(g),), name)


def _inplace(self, compute, involved=(), cast_ok=False, fast=None, view_ok=True):
    """Run an in-place op the way torch does.

    compute(src) returns the out-of-place result for a tensor holding self's current values.
    When autograd is involved, self takes over that result's graph node; earlier ops that used
    self keep pointing at its old node, and ops that saved self's values notice the version bump.

    Views (untracked slices of untracked tensors) still get the new values written through to
    their base, but the base doesn't learn the gradient path. That's right for `x[i] += v`,
    which Python finishes with `x.__setitem__(i, view)` (recorded on x), so arithmetic is allowed
    on views (view_ok); item assignment into a view (`x[i][j] = v`) has no such follow-up and raises.
    """
    grad_on = _GRAD[0]
    if grad_on and isinstance(self._node, _Leaf):
        raise RuntimeError("a leaf Variable that requires grad is being used in an in-place operation.")
    if not self._a.flags.writeable:
        raise RuntimeError("unsupported operation: more than one element of the written-to tensor refers to a single memory location. "
                           "Please clone() the tensor before performing the operation.")
    graph = grad_on and (self._node is not None or _track(*involved))
    if graph and self._view and not view_ok:
        raise MiniTorchUnsupported(
            "Writing a value that requires grad into a view (a slice, reshape or transpose) isn't in the browser's mini-torch. "
            "Index the original tensor directly (`buf[0, 1] = w`, not `buf[0][1] = w`) or build the result out of place (`x = x + y`). "
            "It will work on your GPU.")
    if not graph and fast is not None:
        fast()
        self._ver[0] += 1
        return self
    if graph:
        src = _new(self._a.copy())
        src._node = self._node
    else:
        src = self
    with set_grad_enabled(graph):
        r = compute(src)
    ra = r._a
    if ra.shape != self._a.shape:
        raise RuntimeError(f"output with shape {list(self._a.shape)} doesn't match the broadcast shape {list(ra.shape)}")
    if not cast_ok and _cat(ra.dtype) > _cat(self._a.dtype):
        raise RuntimeError(f"result type {_dt(ra)._short} can't be cast to the desired output type {_dt(self._a)._short}")
    self._a[...] = ra if ra.dtype == self._a.dtype else _to_dtype(ra, _dt(self._a))
    self._ver[0] += 1
    if graph:
        self._node = r._node
    return self


# ---------------------------------------------------------------- repr
def _fmt_floats(a):
    finite = a[_np.isfinite(a)].astype(_F64)
    nz = _np.abs(finite[finite != 0])
    int_mode = finite.size == 0 or _builtins.bool(_np.all(finite == _np.round(finite)))
    sci = False
    if nz.size:
        mx, mn = float(nz.max()), float(nz.min())
        sci = mx / mn > 1000.0 or mx > 1e8 or (not int_mode and mn < 1e-4)

    def fmt(v):
        if v != v:
            return "nan"
        if v in (float("inf"), float("-inf")):
            return "inf" if v > 0 else "-inf"
        if sci:
            return f"{v:.4e}"
        if int_mode:
            return f"{v:.0f}."
        return f"{v:.4f}"
    return fmt


def _summarize(a, edge=3):
    """The values torch would show for a big tensor: the first and last `edge` along each dim."""
    if a.ndim == 0:
        return a
    for d in range(a.ndim):
        if a.shape[d] > 2 * edge:
            a = _np.concatenate([_np.take(a, range(edge), axis=d), _np.take(a, range(a.shape[d] - edge, a.shape[d]), axis=d)], axis=d)
    return a


def _fmt_rec(a, fmt, depth, summarize, edge=3):
    if a.ndim == 0:
        return fmt(a.item())
    n = a.shape[0]
    if summarize and n > 2 * edge:
        head = [_fmt_rec(a[i], fmt, depth + 1, summarize) for i in range(edge)]
        tail = [_fmt_rec(a[i], fmt, depth + 1, summarize) for i in range(n - edge, n)]
        parts = head + ["..."] + tail
    else:
        parts = [_fmt_rec(a[i], fmt, depth + 1, summarize) for i in range(n)]
    if a.ndim == 1:
        return "[" + ", ".join(parts) + "]"
    sep = "," + "\n" * (a.ndim - 1) + " " * (8 + depth)
    return "[" + sep.join(parts) + "]"


def _tensor_repr(t, prefix="tensor("):
    a = t._a
    suffix = []
    if a.dtype not in (_F32, _I64, _BOOL):
        suffix.append(f"dtype={_dt(a)!r}")
    if a.size == 0:
        body = "[]"
        if a.ndim > 1:
            suffix.insert(0, f"size={tuple(a.shape)}")
        if a.dtype == _I64:
            suffix.append("dtype=torch.int64")
    else:
        summarize = a.size > 1000
        shown = _summarize(a) if summarize else a
        if a.dtype == _BOOL:
            fmt = str
        elif a.dtype.kind == "f":
            fmt = _fmt_floats(shown)
        else:
            fmt = str
        body = _fmt_rec(a, fmt, 0, summarize)
    if t._node is not None:
        suffix.append("requires_grad=True" if isinstance(t._node, _Leaf) else f"grad_fn=<{t._node._name}>")
    return prefix + body + "".join(", " + s for s in suffix) + ")"


# ---------------------------------------------------------------- Tensor
class Tensor:
    __slots__ = ("_a", "_ver", "_node", "grad", "_view")
    __array_ufunc__ = None  # NumPy defers to Tensor's own operators (np.float32(2) * t -> Tensor)

    def __new__(cls, *args, **kwargs):
        """Legacy constructor: torch.Tensor([1.0, 2.0]) is float32 data; torch.Tensor(2, 3) is a zero-filled shape."""
        if cls is not Tensor or kwargs:
            raise _unsupported("torch.Tensor(...) with keyword arguments")
        if len(args) == 1 and isinstance(args[0], (list, tuple, _np.ndarray, Tensor)) and not isinstance(args[0], Size):
            return _new(_to_dtype(_from_data(args[0]), float32))
        return _new(_np.zeros(_size_arg(args, "Tensor"), dtype=_np.float32))

    def __init__(self, *args, **kwargs):
        pass

    # ----- metadata -----
    @property
    def dtype(self):
        return _dt(self._a)

    @property
    def shape(self):
        return Size(self._a.shape)

    def size(self, dim=None):
        if dim is None:
            return Size(self._a.shape)
        return self._a.shape[_dim(self._a, dim)] if self._a.ndim else 1

    def dim(self):
        return self._a.ndim

    @property
    def ndim(self):
        return self._a.ndim

    def numel(self):
        return int(self._a.size)

    nelement = numel

    def element_size(self):
        return self._a.dtype.itemsize

    @property
    def itemsize(self):
        return self._a.dtype.itemsize

    @property
    def nbytes(self):
        return int(self._a.nbytes)

    @property
    def device(self):
        return _CPU

    @property
    def is_cuda(self):
        return False

    def is_floating_point(self):
        return self._a.dtype.kind == "f"

    def is_contiguous(self):
        return True

    # ----- autograd -----
    @property
    def requires_grad(self):
        return self._node is not None

    @requires_grad.setter
    def requires_grad(self, flag):
        self.requires_grad_(flag)

    def requires_grad_(self, requires_grad=True):
        if self._node is not None and not isinstance(self._node, _Leaf):
            if not requires_grad:
                raise RuntimeError("you can only change requires_grad flags of leaf variables. If you want to use a computed variable "
                                   "in a subgraph that doesn't require differentiation use var_no_grad = var.detach().")
            return self
        if requires_grad:
            if self._a.dtype.kind != "f":
                raise RuntimeError("only Tensors of floating point dtype can require gradients")
            if self._node is None:
                self._node = _Leaf(self)
        else:
            self._node = None
        return self

    @property
    def is_leaf(self):
        return self._node is None or isinstance(self._node, _Leaf)

    @property
    def grad_fn(self):
        return None if self.is_leaf else self._node

    @property
    def retains_grad(self):
        return not self.is_leaf and self._node.retain is self

    def retain_grad(self):
        if self._node is None:
            raise RuntimeError("can't retain_grad on Tensor that has requires_grad=False")
        if not isinstance(self._node, _Leaf):
            self._node.retain = self

    def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
        if create_graph:
            raise _unsupported("backward(create_graph=True)")
        if inputs is not None:
            raise _unsupported("backward(inputs=...)")
        _run_backward(self, gradient, _builtins.bool(retain_graph))

    def detach(self):
        return _new(self._a, ver=self._ver, view=self._view)

    def detach_(self):
        self._node = None
        return self

    @property
    def data(self):
        return _new(self._a)

    @data.setter
    def data(self, value):
        self._a = _tensor_arg(value, "data", "value")._a

    def clone(self):
        return _result(self._a.copy(), (self,), lambda g: (g,), "CloneBackward0")

    def contiguous(self):
        return self

    # ----- conversion -----
    def _cast(self, d):
        if self._a.dtype == d.np:
            return self
        out = _to_dtype(self._a, d)
        if not d.is_floating_point or not _track(self):
            return _new(out)
        src = self._a.dtype
        return _result(out, (self,), lambda g: (g.astype(src),), "ToCopyBackward0")

    def to(self, *args, dtype=None, device=None, non_blocking=False, copy=False, **kwargs):
        target = dtype
        for v in args:
            if isinstance(v, _dtype_cls):
                target = v
            elif isinstance(v, (str, _device_cls)):
                _check_device(v)
            elif isinstance(v, Tensor):
                target = v.dtype
            else:
                raise _unsupported(f"Tensor.to({v!r})")
        _check_device(device)
        if kwargs:
            raise _unsupported(f"Tensor.to({', '.join(kwargs)}=...)")
        if target is not None and not isinstance(target, _dtype_cls):
            raise _unsupported(f"Tensor.to({target!r})")
        out = self if target is None else self._cast(target)
        return out.clone() if copy and out is self else out

    def cpu(self):
        return self

    def cuda(self, *args, **kwargs):
        _check_device("cuda")

    def type(self, dtype=None):
        if dtype is None:
            return f"torch.{_dt(self._a)._short}Tensor"
        return self._cast(dtype)

    def type_as(self, other):
        return self._cast(other.dtype)

    def float(self): return self._cast(float32)
    def double(self): return self._cast(float64)
    def half(self): return self._cast(float16)
    def long(self): return self._cast(int64)
    def int(self): return self._cast(int32)
    def short(self): return self._cast(int16)
    def char(self): return self._cast(int8)
    def byte(self): return self._cast(uint8)
    def bool(self): return self._cast(_bool_dt)

    def tolist(self):
        return self._a.tolist()

    def item(self):
        if self._a.size != 1:
            raise RuntimeError(f"a Tensor with {self._a.size} elements cannot be converted to Scalar")
        return self._a.reshape(()).item()

    def numpy(self, force=False):
        if self._node is not None and not force:
            raise RuntimeError("Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.")
        return self._a

    def __array__(self, dtype=None, copy=None):
        a = self.numpy()
        return a if dtype is None else a.astype(dtype)

    # ----- math methods -----
    def abs(self): return _abs(self)
    def exp(self): return exp(self)
    def log(self): return log(self)
    def log2(self): return log2(self)
    def sqrt(self): return sqrt(self)
    def rsqrt(self): return rsqrt(self)
    def tanh(self): return tanh(self)
    def sigmoid(self): return sigmoid(self)
    def relu(self): return relu(self)
    def sin(self): return sin(self)
    def cos(self): return cos(self)
    def erf(self): return erf(self)
    def neg(self): return neg(self)
    def sign(self): return sign(self)
    def floor(self): return floor(self)
    def ceil(self): return ceil(self)
    def trunc(self): return trunc(self)
    def reciprocal(self): return reciprocal(self)
    def round(self, decimals=0): return _round(self, decimals)
    def clamp(self, min=None, max=None): return clamp(self, min, max)  # noqa: A002
    clip = clamp
    def pow(self, exponent): return _binary(self, exponent, "pow")
    def add(self, other, alpha=1): return _binary(self, other if alpha == 1 else other * alpha, "add")
    def sub(self, other, alpha=1): return _binary(self, other if alpha == 1 else other * alpha, "sub")
    def mul(self, other): return _binary(self, other, "mul")
    def div(self, other): return _binary(self, other, "div")
    def matmul(self, other): return matmul(self, other)
    def maximum(self, other): return maximum(self, other)
    def minimum(self, other): return minimum(self, other)

    def sum(self, dim=None, keepdim=False, dtype=None, **kw): return _sum(self, dim, keepdim, dtype, **kw)
    def mean(self, dim=None, keepdim=False, dtype=None, **kw): return _mean(self, dim, keepdim, dtype, **kw)
    def var(self, dim=None, unbiased=None, keepdim=False, *, correction=None, keepdims=None): return _var(self, dim, unbiased, keepdim, correction, keepdims)
    def std(self, dim=None, unbiased=None, keepdim=False, *, correction=None, keepdims=None): return _std(self, dim, unbiased, keepdim, correction, keepdims)
    def max(self, dim=None, keepdim=False): return _max(self, dim, keepdim)
    def min(self, dim=None, keepdim=False): return _min(self, dim, keepdim)
    def amax(self, dim=(), keepdim=False): return amax(self, dim, keepdim)
    def amin(self, dim=(), keepdim=False): return amin(self, dim, keepdim)
    def argmax(self, dim=None, keepdim=False): return argmax(self, dim, keepdim)
    def argmin(self, dim=None, keepdim=False): return argmin(self, dim, keepdim)
    def any(self, dim=None, keepdim=False): return _any(self, dim, keepdim)
    def all(self, dim=None, keepdim=False): return _all(self, dim, keepdim)
    def norm(self, p="fro", dim=None, keepdim=False): return norm(self, p, dim, keepdim)
    def cumsum(self, dim, dtype=None): return cumsum(self, dim, dtype)
    def softmax(self, dim, dtype=None): return softmax(self, dim, dtype)
    def log_softmax(self, dim, dtype=None): return log_softmax(self, dim, dtype)
    def sort(self, dim=-1, descending=False, stable=False): return sort(self, dim, descending, stable)
    def argsort(self, dim=-1, descending=False, stable=False): return argsort(self, dim, descending, stable)
    def topk(self, k, dim=-1, largest=True, sorted=True): return topk(self, k, dim, largest, sorted)  # noqa: A002
    def multinomial(self, num_samples, replacement=False, generator=None): return multinomial(self, num_samples, replacement, generator)
    def tril(self, diagonal=0): return tril(self, diagonal)
    def triu(self, diagonal=0): return triu(self, diagonal)
    def masked_fill(self, mask, value): return masked_fill(self, mask, value)
    def where(self, condition, other): return where(condition, self, other)
    def isclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False): return isclose(self, other, rtol, atol, equal_nan)
    def allclose(self, other, rtol=1e-05, atol=1e-08, equal_nan=False): return allclose(self, other, rtol, atol, equal_nan)
    def equal(self, other): return equal(self, other)
    def nonzero(self, as_tuple=False): return nonzero(self, as_tuple)
    def logical_not(self): return logical_not(self)
    def eq(self, other): return self == other
    def ne(self, other): return self != other
    def lt(self, other): return self < other
    def le(self, other): return self <= other
    def gt(self, other): return self > other
    def ge(self, other): return self >= other

    # ----- shape methods -----
    def reshape(self, *shape): return _reshape(self, shape)
    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], _dtype_cls):
            raise _unsupported("Tensor.view(dtype)")
        return _reshape(self, shape, "view")
    def view_as(self, other): return _reshape(self, tuple(other.shape), "view")
    def reshape_as(self, other): return _reshape(self, tuple(other.shape))
    def flatten(self, start_dim=0, end_dim=-1): return flatten(self, start_dim, end_dim)
    def transpose(self, dim0, dim1): return transpose(self, dim0, dim1)
    swapaxes = transpose
    def permute(self, *dims): return permute(self, *dims)
    def unsqueeze(self, dim): return unsqueeze(self, dim)
    def squeeze(self, dim=None): return squeeze(self, dim)
    def expand(self, *sizes): return _expand(self, sizes)
    def expand_as(self, other): return _expand(self, tuple(other.shape))
    def split(self, split_size, dim=0): return split(self, split_size, dim)
    def chunk(self, chunks, dim=0): return chunk(self, chunks, dim)
    def unbind(self, dim=0): return unbind(self, dim)
    def gather(self, dim, index): return gather(self, dim, index)
    def index_select(self, dim, index): return index_select(self, dim, index)
    def scatter(self, dim, index, src=None, value=None, reduce=None): return scatter(self, dim, index, src, value, reduce)
    def scatter_add(self, dim, index, src): return scatter_add(self, dim, index, src)
    def flip(self, dims): return flip(self, dims)
    def repeat(self, *sizes): return _repeat(self, sizes)
    def repeat_interleave(self, repeats, dim=None): return repeat_interleave(self, repeats, dim)
    def outer(self, vec2): return outer(self, vec2)
    def square(self): return square(self)
    def mm(self, mat2): return mm(self, mat2)
    def bmm(self, mat2): return bmm(self, mat2)

    def scatter_(self, dim, index, src=None, value=None, reduce=None):
        s = _coerce(src if src is not None else value)
        return _inplace(self, lambda me: scatter(me, dim, index, s, reduce=reduce), involved=(s,), cast_ok=True)

    def scatter_add_(self, dim, index, src):
        return _inplace(self, lambda me: scatter_add(me, dim, index, src), involved=(src,), cast_ok=True)

    def t(self):
        if self._a.ndim > 2:
            raise RuntimeError(f"t() expects a tensor with <= 2 dimensions, but self is {self._a.ndim}D")
        return transpose(self, 0, 1) if self._a.ndim == 2 else self

    @property
    def T(self):
        return permute(self, *range(self._a.ndim - 1, -1, -1)) if self._a.ndim else self

    @property
    def mT(self):
        if self._a.ndim < 2:
            raise RuntimeError("tensor.mT is only supported on matrices or batches of matrices. Got 1-D tensor.")
        return transpose(self, -2, -1)

    # ----- in-place ops -----
    def _iop(self, other, kind):
        o = _coerce(other)
        return _inplace(self, lambda s: _binary(s, o, kind), involved=(o,))

    def add_(self, other, alpha=1): return self._iop(other if alpha == 1 else other * alpha, "add")
    def sub_(self, other, alpha=1): return self._iop(other if alpha == 1 else other * alpha, "sub")
    def mul_(self, other): return self._iop(other, "mul")
    def div_(self, other): return self._iop(other, "div")
    def pow_(self, other): return self._iop(other, "pow")
    __iadd__ = add_
    __isub__ = sub_
    __imul__ = mul_
    __itruediv__ = div_

    def __ipow__(self, other):
        return self._iop(other, "pow")

    def clamp_(self, min=None, max=None):  # noqa: A002
        return _inplace(self, lambda s: clamp(s, min, max))

    def masked_fill_(self, mask, value):
        return _inplace(self, lambda s: masked_fill(s, mask, value))

    def zero_(self):
        return _inplace(self, lambda s: _constant_like(s, 0, "ZeroBackward0"), fast=lambda: self._a.fill(0))

    def fill_(self, value):
        v = value.item() if isinstance(value, Tensor) else value
        return _inplace(self, lambda s: _constant_like(s, v, "FillBackward0"), cast_ok=True, fast=lambda: self._a.fill(v))

    def copy_(self, src):
        s = _tensor_arg(src, "copy_", "src")
        _np.broadcast_shapes(s._a.shape, self._a.shape)
        return _inplace(self, lambda me: _index_put(me, Ellipsis, s), involved=(s,), cast_ok=True)

    def normal_(self, mean=0.0, std=1.0, generator=None):
        vals = _gen(generator)._rng.standard_normal(self._a.shape) * std + mean
        return _inplace(self, lambda s: _new(vals.astype(s._a.dtype)), cast_ok=True)

    def uniform_(self, a=0.0, b=1.0, generator=None):
        vals = _gen(generator)._rng.random(self._a.shape) * (b - a) + a
        return _inplace(self, lambda s: _new(vals.astype(s._a.dtype)), cast_ok=True)

    def __setitem__(self, idx, value):
        v = _coerce(value)

        def fast():
            nidx = _index(idx)
            va = v._a if isinstance(v, Tensor) else _np.asarray(v)
            try:
                target = self._a[nidx]
            except IndexError as e:
                raise _index_error(e) from None
            try:
                _np.broadcast_shapes(va.shape, _np.shape(target))
            except ValueError:
                raise RuntimeError(f"shape mismatch: value tensor of shape {list(va.shape)} cannot be broadcast to indexing result of shape {list(_np.shape(target))}") from None
            self._a[nidx] = va if va.dtype == self._a.dtype else _to_dtype(va, _dt(self._a))
        _inplace(self, lambda s: _index_put(s, idx, v), involved=(v,), cast_ok=True, fast=fast, view_ok=False)

    # ----- indexing and iteration -----
    def __getitem__(self, idx):
        return _getitem(self, idx)

    def __len__(self):
        if self._a.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return len(self._a)

    def __iter__(self):
        if self._a.ndim == 0:
            raise TypeError("iteration over a 0-d tensor")
        return (self[i] for i in range(len(self._a)))

    def __contains__(self, value):
        return _builtins.bool((self == value)._a.any())

    # ----- arithmetic -----
    def __add__(self, o): return _binary(self, o, "add")
    def __radd__(self, o): return _binary(o, self, "add")
    def __sub__(self, o): return _binary(self, o, "sub")
    def __rsub__(self, o): return _binary(o, self, "sub")
    def __mul__(self, o): return _binary(self, o, "mul")
    def __rmul__(self, o): return _binary(o, self, "mul")
    def __truediv__(self, o): return _binary(self, o, "div")
    def __rtruediv__(self, o): return _binary(o, self, "div")
    def __floordiv__(self, o): return _binary(self, o, "floordiv")
    def __rfloordiv__(self, o): return _binary(o, self, "floordiv")
    def __mod__(self, o): return _binary(self, o, "mod")
    def __rmod__(self, o): return _binary(o, self, "mod")
    def __pow__(self, o): return _binary(self, o, "pow")
    def __rpow__(self, o): return _binary(o, self, "pow")
    def __matmul__(self, o): return matmul(self, o)
    def __rmatmul__(self, o): return matmul(o, self)
    def __neg__(self): return neg(self)
    def __pos__(self): return self
    def __abs__(self): return _abs(self)
    def __and__(self, o): return _bitwise(self, o, _np.bitwise_and, "bitwise_and")
    def __rand__(self, o): return _bitwise(o, self, _np.bitwise_and, "bitwise_and")
    def __or__(self, o): return _bitwise(self, o, _np.bitwise_or, "bitwise_or")
    def __ror__(self, o): return _bitwise(o, self, _np.bitwise_or, "bitwise_or")
    def __xor__(self, o): return _bitwise(self, o, _np.bitwise_xor, "bitwise_xor")
    def __rxor__(self, o): return _bitwise(o, self, _np.bitwise_xor, "bitwise_xor")

    def __invert__(self):
        if self._a.dtype.kind == "f":
            raise RuntimeError("\"bitwise_not_cpu\" not implemented for 'Float'")
        return _new(_np.invert(self._a))

    def __lt__(self, o): return _compare(self, o, _np.less)
    def __le__(self, o): return _compare(self, o, _np.less_equal)
    def __gt__(self, o): return _compare(self, o, _np.greater)
    def __ge__(self, o): return _compare(self, o, _np.greater_equal)
    # `t == None` is False and `t in [None, ...]` works, like torch; numbers and tensors compare elementwise.
    def __eq__(self, o):
        return NotImplemented if o is None or isinstance(o, (str, list, tuple, dict)) else _compare(self, o, _np.equal)

    def __ne__(self, o):
        return NotImplemented if o is None or isinstance(o, (str, list, tuple, dict)) else _compare(self, o, _np.not_equal)
    __hash__ = object.__hash__

    def __bool__(self):
        if self._a.size != 1:
            raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")
        return _builtins.bool(self._a.reshape(()))

    def __float__(self): return float(self.item())
    def __int__(self): return int(self.item())

    def __index__(self):
        if self._a.size != 1 or _cat(self._a.dtype) != 1:
            raise TypeError("only integer tensors of a single element can be converted to an index")
        return int(self.item())

    def __format__(self, spec):
        if not spec:
            return repr(self)
        if self._a.ndim == 0:
            return format(self.item(), spec)
        raise TypeError("unsupported format string passed to Tensor.__format__")

    def __repr__(self):
        return _tensor_repr(self)

    __str__ = __repr__

    def __deepcopy__(self, memo):
        if not self.is_leaf:
            raise RuntimeError("Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment.")
        t = object.__new__(type(self))
        memo[id(self)] = t
        t._a = self._a.copy()
        t._ver = [0]
        t._view = False
        t.grad = _copy.deepcopy(self.grad, memo)
        t._node = _Leaf(t) if self._node is not None else None
        if hasattr(self, "__dict__"):
            t.__dict__.update(_copy.deepcopy(self.__dict__, memo))
        return t

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise _unsupported(f"Tensor.{name}")


# ---------------------------------------------------------------- creation
def _from_data(data):
    if isinstance(data, Tensor):
        return data._a.copy()
    if isinstance(data, _np.ndarray):
        a = data.copy()
        if a.dtype == _np.int32:
            # Pyodide's NumPy defaults to 32-bit ints (np.array([1, 2]) is int32 there, int64 on a
            # desktop), so treat int32 arrays as torch would see them on the learner's GPU machine.
            return a.astype(_np.int64)
        if a.dtype in _BY_NP:
            return a
        if a.dtype.kind == "f":
            return a.astype(_np.float64)
        if a.dtype.kind in "iu":
            return a.astype(_np.int64)
        raise TypeError(f"can't convert np.ndarray of type numpy.{a.dtype}.")
    else:
        def plain(v):
            if isinstance(v, Tensor):
                if v._a.size != 1:
                    raise ValueError("only one element tensors can be converted to Python scalars")
                return v.item()
            if isinstance(v, (list, tuple)):
                return [plain(x) for x in v]
            if v is Ellipsis:
                raise TypeError("Could not infer dtype of ellipsis")
            return v
        try:
            a = _np.asarray(plain(data))
        except ValueError as e:
            if "inhomogeneous" in str(e) or "sequence" in str(e):
                raise ValueError("expected sequence of equal length at each level: the nested lists are ragged") from None
            raise
    # Python floats become float32 and ints int64, like torch (NumPy's default int is 32-bit in Pyodide).
    if a.dtype == _BOOL:
        return a
    if a.dtype.kind == "f":
        return a.astype(_np.float32)
    if a.dtype.kind in "iu":
        return a.astype(_np.int64)
    raise TypeError(f"Could not infer dtype of {type(data).__name__}" if a.dtype.kind == "O" else f"mini-torch can only build tensors from numbers, got {a.dtype}")


def _finish(a, dtype=None, requires_grad=False, device=None):
    _check_device(device)
    if dtype is not None:
        a = _to_dtype(a, dtype) if a.dtype != dtype.np else a
    t = _new(a)
    if requires_grad:
        t.requires_grad_(True)
    return t


def tensor(data, dtype=None, device=None, requires_grad=False):
    return _finish(_from_data(data), dtype, requires_grad, device)


def as_tensor(data, dtype=None, device=None):
    if isinstance(data, Tensor) and (dtype is None or data.dtype is dtype):
        return data
    return tensor(data, dtype, device)


def from_numpy(arr):
    if arr.dtype not in _BY_NP:
        raise TypeError(f"can't convert np.ndarray of type numpy.{arr.dtype}.")
    if arr.dtype == _np.int32:  # see _from_data: a copy, so it no longer shares memory
        return _new(arr.astype(_np.int64))
    return _new(arr)


def is_tensor(x):
    return isinstance(x, Tensor)


def numel(x):
    return _tensor_arg(x, "torch.numel").numel()


def _filled(size, value, dtype, requires_grad, device, fn):
    shape = _size_arg(size, fn)
    d = dtype or float32
    return _finish(_np.full(shape, value, dtype=d.np), None, requires_grad, device)


def zeros(*size, dtype=None, device=None, requires_grad=False):
    return _filled(size, 0, dtype, requires_grad, device, "zeros")


def ones(*size, dtype=None, device=None, requires_grad=False):
    return _filled(size, 1, dtype, requires_grad, device, "ones")


def empty(*size, dtype=None, device=None, requires_grad=False):
    return _filled(size, 0, dtype, requires_grad, device, "empty")


def full(size, fill_value, dtype=None, device=None, requires_grad=False):
    if isinstance(fill_value, Tensor):
        fill_value = fill_value.item()
    if dtype is None:
        dtype = _bool_dt if isinstance(fill_value, _builtins.bool) else (int64 if isinstance(fill_value, int) else float32)
    return _filled((size,) if isinstance(size, int) else size, fill_value, dtype, requires_grad, device, "full")


def _like(t, dtype):
    t = _tensor_arg(t, "like")
    return t._a.shape, dtype or t.dtype


def zeros_like(input, dtype=None, device=None, requires_grad=False):  # noqa: A002
    shape, d = _like(input, dtype)
    return _filled(shape, 0, d, requires_grad, device, "zeros_like")


def ones_like(input, dtype=None, device=None, requires_grad=False):  # noqa: A002
    shape, d = _like(input, dtype)
    return _filled(shape, 1, d, requires_grad, device, "ones_like")


empty_like = zeros_like


def full_like(input, fill_value, dtype=None, device=None, requires_grad=False):  # noqa: A002
    shape, d = _like(input, dtype)
    return _filled(shape, fill_value, d, requires_grad, device, "full_like")


def arange(start, end=None, step=1, dtype=None, device=None, requires_grad=False):
    if end is None:
        start, end = 0, start
    vals = [v.item() if isinstance(v, Tensor) else v for v in (start, end, step)]
    if step == 0:
        raise RuntimeError("step must be nonzero")
    is_float = any(isinstance(v, float) for v in vals)
    if (vals[1] - vals[0]) * vals[2] < 0:
        raise RuntimeError("upper bound and larger bound inconsistent with step sign")
    if dtype is None:
        dtype = float32 if is_float else int64
    if is_float:
        n = int(_math.ceil((vals[1] - vals[0]) / vals[2]))
        a = (vals[0] + _np.arange(n, dtype=_np.float64) * vals[2])
    else:
        a = _np.arange(vals[0], vals[1], vals[2], dtype=_np.int64)
    return _finish(a.astype(dtype.np), None, requires_grad, device)


def linspace(start, end, steps, dtype=None, device=None, requires_grad=False):
    return _finish(_np.linspace(float(start), float(end), int(steps)).astype((dtype or float32).np), None, requires_grad, device)


def eye(n, m=None, dtype=None, device=None, requires_grad=False):
    return _finish(_np.eye(n, m, dtype=(dtype or float32).np), None, requires_grad, device)


# ---------------------------------------------------------------- random
class Generator:
    """Seedable random source. Values are NumPy PCG64 draws: deterministic, but not torch's numbers."""

    _DEFAULT_SEED = 67280421310721  # what torch reports for a fresh generator

    def __init__(self, device="cpu"):
        _check_device(device)
        self.manual_seed(self._DEFAULT_SEED)

    def manual_seed(self, seed):
        if isinstance(seed, Tensor):
            seed = seed.item()
        if not isinstance(seed, (int, _np.integer)) or isinstance(seed, _builtins.bool):
            raise TypeError(f"manual_seed expected a long, but got {type(seed).__name__}")
        self._seed = int(seed)
        self._rng = _np.random.Generator(_np.random.PCG64(int(seed) % (1 << 64)))
        return self

    def initial_seed(self):
        return self._seed

    @property
    def device(self):
        return _CPU

    def __repr__(self):
        return "<torch._C.Generator object>"


default_generator = Generator()


def _gen(generator):
    if generator is None:
        return default_generator
    if not isinstance(generator, Generator):
        raise TypeError(f"generator must be a torch.Generator, not {type(generator).__name__}")
    return generator


def manual_seed(seed):
    return default_generator.manual_seed(seed)


def initial_seed():
    return default_generator.initial_seed()


def _float_dtype(dtype, fn):
    d = dtype or float32
    if not d.is_floating_point:
        raise RuntimeError(f"\"{fn}\" not implemented for '{d._short}'")
    return d


def randn(*size, generator=None, dtype=None, device=None, requires_grad=False):
    d = _float_dtype(dtype, "normal_kernel_cpu")
    a = _gen(generator)._rng.standard_normal(_size_arg(size, "randn"))
    return _finish(_np.asarray(a).astype(d.np), None, requires_grad, device)


def rand(*size, generator=None, dtype=None, device=None, requires_grad=False):
    d = _float_dtype(dtype, "check_uniform_bounds")
    a = _gen(generator)._rng.random(_size_arg(size, "rand"))
    return _finish(_np.asarray(a).astype(d.np), None, requires_grad, device)


def randint(*args, size=None, generator=None, dtype=None, device=None, requires_grad=False, low=None, high=None):
    args = list(args)
    if size is None:
        if not args or not isinstance(args[-1], (tuple, list, Size)):
            raise TypeError("randint() received an invalid combination of arguments: it needs a size tuple, like torch.randint(0, 10, (3,))")
        size = args.pop()
    if high is None:
        if len(args) == 2:
            low, high = args
        elif len(args) == 1:
            high = args[0]
        else:
            raise TypeError("randint() missing required argument 'high'")
    low = 0 if low is None else low
    if low >= high:
        raise RuntimeError(f"random_ expects 'from' to be less than 'to', but got from={low} >= to={high}")
    a = _gen(generator)._rng.integers(int(low), int(high), _size_arg((size,), "randint"))
    return _finish(_np.asarray(a, dtype=_np.int64).astype((dtype or int64).np), None, requires_grad, device)


def randperm(n, generator=None, dtype=None, device=None, requires_grad=False):
    return _finish(_gen(generator)._rng.permutation(int(n)).astype((dtype or int64).np), None, requires_grad, device)


def randn_like(input, dtype=None, generator=None):  # noqa: A002
    shape, d = _like(input, dtype)
    return randn(shape, generator=generator, dtype=d)


def rand_like(input, dtype=None, generator=None):  # noqa: A002
    shape, d = _like(input, dtype)
    return rand(shape, generator=generator, dtype=d)


def bernoulli(input, p=None, generator=None):  # noqa: A002
    x = _tensor_arg(input, "torch.bernoulli")
    prob = x._a if p is None else _np.full(x._a.shape, p)
    draws = _gen(generator)._rng.random(x._a.shape) < prob
    return _new(draws.astype(x._a.dtype if x._a.dtype.kind == "f" else _F32))


# ---------------------------------------------------------------- functions that shadow builtins
_ALIASES.update(abs=_abs, round=_round, max=_max, min=_min, sum=_sum, all=_all, any=_any, pow=_pow)
mean = _mean
var = _var
std = _std


def add(input, other, alpha=1): return _binary(_tensor_arg(input, "torch.add"), other if alpha == 1 else other * alpha, "add")  # noqa: A002
def sub(input, other, alpha=1): return _binary(_tensor_arg(input, "torch.sub"), other if alpha == 1 else other * alpha, "sub")  # noqa: A002
def mul(input, other): return _binary(_tensor_arg(input, "torch.mul"), other, "mul")  # noqa: A002
def div(input, other): return _binary(_tensor_arg(input, "torch.div"), other, "div")  # noqa: A002


def __getattr__(name):
    if name in _ALIASES:
        return _ALIASES[name]
    if name.startswith("__"):
        raise AttributeError(name)
    raise _unsupported(f"torch.{name}")


__path__ = []  # lets `import torch.<missing>` reach _TorchSubmoduleFinder instead of "torch is not a package"


# ---------------------------------------------------------------- submodules
def _submodule(name, doc):
    mod = _types.ModuleType(f"torch.{name}", doc)

    def missing(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        raise _unsupported(f"torch.{name}.{attr}")
    mod.__getattr__ = missing
    _sys.modules[f"torch.{name}"] = mod
    return mod


class _TorchSubmoduleFinder:
    """`import torch.<something we don't have>` says so plainly instead of 'torch is not a package'."""

    @staticmethod
    def find_spec(fullname, path=None, target=None):
        if fullname.startswith("torch.") and fullname not in _sys.modules:
            raise _unsupported(fullname)
        return None


if not any(type(f).__name__ == "_TorchSubmoduleFinder" for f in _sys.meta_path):
    _sys.meta_path.append(_TorchSubmoduleFinder())

cuda = _submodule("cuda", "No GPU in the browser.")
cuda.is_available = lambda: False
cuda.device_count = lambda: 0

# ----- torch.nn.functional -----
_functional = _submodule("nn.functional", "torch.nn.functional for the browser.")


def _named(t, name):
    """Give a composed op's final node torch's grad_fn name (layer_norm is built from mean/var/mul)."""
    if t._node is not None and not isinstance(t._node, _Leaf):
        t._node._name = name
    return t


def _linear(input, weight, bias=None):  # noqa: A002
    out = matmul(input, transpose(weight, 0, 1) if weight.ndim == 2 else weight)
    return out if bias is None else out + bias


def _embedding(input, weight):  # noqa: A002
    ix = _tensor_arg(input, "embedding", "indices")
    if _cat(ix._a.dtype) != 1:
        raise RuntimeError(f"Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.{_dt(ix._a)._short}Tensor instead (while checking arguments for embedding)")
    n = weight.shape[0]
    if ix._a.size and (ix._a.min() < 0 or ix._a.max() >= n):
        raise IndexError("index out of range in self")
    return _named(weight[ix], "EmbeddingBackward0")


def _one_hot(tensor, num_classes=-1):
    t = _tensor_arg(tensor, "one_hot")
    if t._a.dtype != _I64:
        raise RuntimeError("one_hot is only applicable to index tensor of type LongTensor.")
    a = t._a
    if a.size and a.min() < 0:
        raise RuntimeError("Class values must be non-negative.")
    n = int(a.max()) + 1 if num_classes == -1 else num_classes
    if a.size and a.max() >= n:
        raise RuntimeError("Class values must be smaller than num_classes.")
    return _new((a[..., None] == _np.arange(n)).astype(_np.int64))


def _gelu(input, approximate="none"):  # noqa: A002
    x = input
    if approximate == "tanh":
        return 0.5 * x * (1 + tanh(_math.sqrt(2 / _math.pi) * (x + 0.044715 * x * x * x)))
    if approximate != "none":
        raise RuntimeError(f"approximate argument must be either none or tanh, got {approximate!r}")
    return 0.5 * x * (1 + erf(x * (1 / _math.sqrt(2))))


def _silu(input):  # noqa: A002
    return input * sigmoid(input)


def _dropout(input, p=0.5, training=True, inplace=False):  # noqa: A002
    if p < 0 or p > 1:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    if not training or p == 0:
        return input
    if p == 1:
        return input * 0
    keep = default_generator._rng.random(input._a.shape) >= p
    return input * _new(keep.astype(input._a.dtype) / input._a.dtype.type(1 - p))


def _layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):  # noqa: A002
    ns = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    x = input
    if tuple(x.shape[x.ndim - len(ns):]) != ns or x.ndim < len(ns):
        raise RuntimeError(f"Given normalized_shape={list(ns)}, expected input with shape [*, {', '.join(str(v) for v in ns)}], but got input of size{list(x.shape)}")
    dims = tuple(range(-len(ns), 0))
    d = x - _mean(x, dims, keepdim=True)
    y = d * rsqrt(_mean(d * d, dims, keepdim=True) + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return _named(y, "NativeLayerNormBackward0")


def _rms_norm(input, normalized_shape, weight=None, eps=None):  # noqa: A002
    ns = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    x = input
    if tuple(x.shape[x.ndim - len(ns):]) != ns:
        raise RuntimeError(f"Given normalized_shape={list(ns)}, expected input with shape [*, {', '.join(str(v) for v in ns)}], but got input of size{list(x.shape)}")
    eps = finfo(x.dtype).eps if eps is None else eps
    y = x * rsqrt(_mean(x * x, tuple(range(-len(ns), 0)), keepdim=True) + eps)
    return y if weight is None else y * weight


def _reduce_loss(loss, reduction):
    if reduction == "mean":
        return _mean(loss)
    if reduction == "sum":
        return _sum(loss)
    if reduction == "none":
        return loss
    raise ValueError(f"{reduction} is not a valid value for reduction")


def _nll_loss(input, target, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0, fn="nll_loss"):  # noqa: A002
    logp = _tensor_arg(input, fn)
    tgt = _tensor_arg(target, fn, "target")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"{reduction} is not a valid value for reduction")
    if _cat(tgt._a.dtype) != 1:
        raise RuntimeError(f"expected scalar type Long but found {_dt(tgt._a)._short}")
    if tgt._a.dtype != _I64:
        raise RuntimeError(f"expected scalar type Long but found {_dt(tgt._a)._short}")
    if logp.ndim == 0:
        raise ValueError("Expected 1 or more dimensions (got 0)")
    if logp.ndim == 1:
        if tgt.ndim != 0:
            raise RuntimeError(f"Expected input batch_size (1) to match target batch_size ({tgt.numel()}).")
        lp2, t1, out_shape = logp.unsqueeze(0), tgt.reshape(1), ()
    elif logp.ndim == 2:
        lp2, t1, out_shape = logp, tgt, tuple(tgt.shape)
        if tgt.ndim != 1 or tgt.shape[0] != logp.shape[0]:
            raise ValueError(f"Expected input batch_size ({logp.shape[0]}) to match target batch_size ({tgt.shape[0] if tgt.ndim else 1}).")
    else:
        c = logp.shape[1]
        expect = (logp.shape[0],) + tuple(logp.shape[2:])
        if tuple(tgt.shape) != expect:
            raise RuntimeError(f"Expected target size {list(expect)}, got {list(tgt.shape)}")
        lp2 = permute(logp, 0, *range(2, logp.ndim), 1).reshape(-1, c)
        t1, out_shape = tgt.reshape(-1), tuple(tgt.shape)
    n, c = lp2.shape
    ta = t1._a
    valid = ta != ignore_index
    bad = valid & ((ta < 0) | (ta >= c))
    if bad.any():
        raise IndexError(f"Target {int(ta[bad][0])} is out of bounds.")
    safe = _np.where(valid, ta, 0)
    picked = lp2[_np.arange(n), safe] if n else _sum(lp2, 1)
    w = _np.ones(n, dtype=lp2._a.dtype) if weight is None else weight._a[safe].astype(lp2._a.dtype)
    w = w * valid
    loss = -picked * _new(w)
    if label_smoothing:
        if weight is not None:
            raise _unsupported("cross_entropy(weight=..., label_smoothing=...)")
        smooth = -_sum(lp2, 1) * _new(valid.astype(lp2._a.dtype)) / c
        loss = loss * (1 - label_smoothing) + smooth * label_smoothing
    if reduction == "none":
        return loss.reshape(out_shape)
    total = _sum(loss)
    if reduction == "sum":
        out = total
    else:
        out = total / float(w.sum()) if w.sum() else total * float("nan")
    if out._node is not None:
        out._node._name = "NllLossBackward0"
    return out


def _cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean", label_smoothing=0.0):  # noqa: A002
    x = _tensor_arg(input, "cross_entropy")
    t = _tensor_arg(target, "cross_entropy", "target")
    if x.ndim == 0:
        raise ValueError("Expected 1 or more dimensions (got 0)")
    cdim = 1 if x.ndim > 1 else 0
    if _cat(t._a.dtype) == 2:  # class probabilities
        if tuple(t.shape) != tuple(x.shape):
            raise RuntimeError(f"Expected target size {list(x.shape)}, got {list(t.shape)}")
        c = x.shape[cdim]
        logp = log_softmax(x, cdim)
        if label_smoothing:
            t = t * (1 - label_smoothing) + label_smoothing / c
        per = logp * t
        if weight is not None:
            per = per * weight.reshape(c, *([1] * (x.ndim - cdim - 1)))
        loss = -_sum(per, cdim)
        if reduction == "mean":
            return _sum(loss) / (x.numel() // c)  # torch averages over every position, N·d1·…
        return _reduce_loss(loss, reduction)
    return _nll_loss(log_softmax(x, cdim), t, weight, ignore_index, reduction, label_smoothing, "cross_entropy")


def _mse_loss(input, target, size_average=None, reduce=None, reduction="mean"):  # noqa: A002
    d = input - target
    return _reduce_loss(d * d, reduction)


def _l1_loss(input, target, size_average=None, reduce=None, reduction="mean"):  # noqa: A002
    return _reduce_loss(_abs(input - target), reduction)


def _f_softmax(input, dim=None, _stacklevel=3, dtype=None):  # noqa: A002
    if dim is None:  # torch's legacy implicit dim
        dim = 0 if input.ndim in (0, 1, 3) else 1
    return softmax(input, dim, dtype)


def _f_log_softmax(input, dim=None, _stacklevel=3, dtype=None):  # noqa: A002
    if dim is None:
        dim = 0 if input.ndim in (0, 1, 3) else 1
    return log_softmax(input, dim, dtype)


def _pad(input, pad, mode="constant", value=None):  # noqa: A002
    """F.pad, constant mode: pad = (last_left, last_right, 2nd_last_left, 2nd_last_right, ...)."""
    x = _tensor_arg(input, "pad")
    if mode != "constant":
        raise _unsupported(f"F.pad(mode={mode!r})")
    pad = tuple(int(p) for p in pad)
    if len(pad) % 2 or len(pad) // 2 > x.ndim:
        raise RuntimeError(f"Padding length must be divisible by 2 and at most twice the number of dimensions of the input, got {len(pad)}")
    if any(p < 0 for p in pad):
        raise _unsupported("F.pad with negative padding")
    a = x._a
    widths = [(0, 0)] * a.ndim
    for k in range(len(pad) // 2):
        widths[a.ndim - 1 - k] = (pad[2 * k], pad[2 * k + 1])
    fill = 0 if value is None else value
    out = _np.pad(a, widths, constant_values=_to_dtype(_np.asarray(fill), _dt(a)))
    inner = tuple(slice(lo, lo + n) for (lo, _), n in zip(widths, a.shape))
    return _shaped(x, out, lambda g: (g[inner],), "ConstantPadNdBackward0")


def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    if enable_gqa:
        raise _unsupported("scaled_dot_product_attention(enable_gqa=True)")
    q, k, v = query, key, value
    L, S = q.shape[-2], k.shape[-2]
    s = 1 / _math.sqrt(q.shape[-1]) if scale is None else scale
    att = matmul(q, transpose(k, -2, -1)) * s
    if is_causal:
        if attn_mask is not None:
            raise RuntimeError("_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True")
        att = masked_fill(att, _new(~_np.tril(_np.ones((L, S), dtype=bool))), float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype is _bool_dt:
            att = masked_fill(att, logical_not(attn_mask), float("-inf"))
        else:
            att = att + attn_mask
    att = softmax(att, -1)
    if dropout_p:
        att = _dropout(att, dropout_p, True)
    return matmul(att, v)


for _name, _fn in dict(
    linear=_linear, embedding=_embedding, one_hot=_one_hot, relu=relu, gelu=_gelu, silu=_silu, sigmoid=sigmoid,
    tanh=tanh, softmax=_f_softmax, log_softmax=_f_log_softmax, dropout=_dropout, layer_norm=_layer_norm,
    rms_norm=_rms_norm, cross_entropy=_cross_entropy, nll_loss=_nll_loss, mse_loss=_mse_loss, l1_loss=_l1_loss,
    scaled_dot_product_attention=_sdpa, pad=_pad,
).items():
    setattr(_functional, _name, _fn)

# ----- torch.nn -----
nn = _submodule("nn", "torch.nn for the browser.")
nn.functional = _functional


class Parameter(Tensor):
    """A Tensor that a Module registers as a parameter. Always a leaf; requires grad by default."""

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = _new(_np.zeros(0, dtype=_np.float32))
        data = _tensor_arg(data, "Parameter", "data")
        p = object.__new__(cls)
        p._a = data._a
        p._ver = data._ver
        p._node = None
        p.grad = None
        p._view = False
        if requires_grad:
            if data._a.dtype.kind != "f":
                raise RuntimeError("Only Tensors of floating point and complex dtype can require gradients")
            p._node = _Leaf(p)
        return p

    def __init__(self, data=None, requires_grad=True):
        pass

    def __repr__(self):
        return "Parameter containing:\n" + _tensor_repr(self)

    __str__ = __repr__


_LoadResult = _collections.namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])


class Module:
    def __init__(self, *args, **kwargs):
        d = self.__dict__
        d["training"] = True
        d["_parameters"] = {}
        d["_buffers"] = {}
        d["_non_persistent"] = set()
        d["_modules"] = {}

    def forward(self, *args, **kwargs):
        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "forward" function')

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        d = self.__dict__
        if isinstance(value, nn.Parameter):
            if "_parameters" not in d:
                raise AttributeError("cannot assign parameters before Module.__init__() call")
            d["_modules"].pop(name, None)
            d["_buffers"].pop(name, None)
            d["_parameters"][name] = value
        elif isinstance(value, nn.Module):
            if "_modules" not in d:
                raise AttributeError("cannot assign module before Module.__init__() call")
            d["_parameters"].pop(name, None)
            d["_buffers"].pop(name, None)
            d["_modules"][name] = value
        elif "_parameters" in d and name in d["_parameters"]:
            if value is not None:
                kind = f"torch.{_dt(value._a)._short}Tensor" if isinstance(value, Tensor) else type(value).__name__
                raise TypeError(f"cannot assign '{kind}' as parameter '{name}' (torch.nn.Parameter or None expected)")
            d["_parameters"][name] = None
        elif "_modules" in d and name in d["_modules"]:
            if value is not None:
                raise TypeError(f"cannot assign '{type(value).__name__}' as child module '{name}' (torch.nn.Module or None expected)")
            d["_modules"][name] = None
        elif "_buffers" in d and name in d["_buffers"]:
            d["_buffers"][name] = value
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        for k in ("_parameters", "_buffers", "_modules"):
            self.__dict__.get(k, {}).pop(name, None)
        object.__delattr__(self, name)

    def register_buffer(self, name, tensor, persistent=True):
        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__() call")
        self.__dict__["_buffers"][name] = tensor
        if not persistent:
            self.__dict__["_non_persistent"].add(name)
        object.__setattr__(self, name, tensor)

    def register_parameter(self, name, param):
        setattr(self, name, param)

    def add_module(self, name, module):
        setattr(self, name, module)

    def named_modules(self, memo=None, prefix=""):
        memo = set() if memo is None else memo
        if id(self) in memo:
            return
        memo.add(id(self))
        yield prefix, self
        for name, m in self._modules.items():
            if m is not None:
                yield from m.named_modules(memo, f"{prefix}.{name}" if prefix else name)

    def modules(self):
        for _, m in self.named_modules():
            yield m

    def named_children(self):
        for name, m in self._modules.items():
            if m is not None:
                yield name, m

    def children(self):
        for _, m in self.named_children():
            yield m

    def named_parameters(self, prefix="", recurse=True):
        seen = set()
        mods = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for mp, m in mods:
            for n, p in m._parameters.items():
                if p is None or id(p) in seen:
                    continue
                seen.add(id(p))
                yield (f"{mp}.{n}" if mp else n), p

    def parameters(self, recurse=True):
        for _, p in self.named_parameters(recurse=recurse):
            yield p

    def named_buffers(self, prefix="", recurse=True):
        mods = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for mp, m in mods:
            for n, b in m._buffers.items():
                if b is not None:
                    yield (f"{mp}.{n}" if mp else n), b

    def buffers(self, recurse=True):
        for _, b in self.named_buffers(recurse=recurse):
            yield b

    def train(self, mode=True):
        for m in self.modules():
            m.__dict__["training"] = _builtins.bool(mode)
        return self

    def eval(self):
        return self.train(False)

    def requires_grad_(self, requires_grad=True):
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none=True):
        for p in self.parameters():
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad._a[...] = 0

    def apply(self, fn):
        for m in self.children():
            m.apply(fn)
        fn(self)
        return self

    def to(self, *args, **kwargs):
        target = kwargs.get("dtype")
        for v in args:
            if isinstance(v, _dtype_cls):
                target = v
            else:
                _check_device(v)
        _check_device(kwargs.get("device"))
        if target is not None:
            for t in list(self.parameters()) + list(self.buffers()):
                if t.is_floating_point():
                    t._a = _to_dtype(t._a, target)
        return self

    def cpu(self):
        return self

    def cuda(self, *args, **kwargs):
        _check_device("cuda")

    def state_dict(self):
        out = _collections.OrderedDict()
        for mp, m in self.named_modules():
            pre = f"{mp}." if mp else ""
            for n, p in m._parameters.items():
                if p is not None:
                    out[pre + n] = p.detach()
            for n, b in m._buffers.items():
                if b is not None and n not in m._non_persistent:
                    out[pre + n] = b.detach()
        return out

    def load_state_dict(self, state_dict, strict=True):
        own = self.state_dict()
        missing = [k for k in own if k not in state_dict]
        unexpected = [k for k in state_dict if k not in own]
        errors = []
        for k, v in state_dict.items():
            if k in own and tuple(own[k].shape) != tuple(v.shape):
                errors.append(f"size mismatch for {k}: copying a param with shape {Size(v.shape)} from checkpoint, the shape in current model is {Size(own[k].shape)}.")
        if strict:
            if missing:
                errors.insert(0, "Missing key(s) in state_dict: " + ", ".join(f'"{k}"' for k in missing) + ".")
            if unexpected:
                errors.insert(0, "Unexpected key(s) in state_dict: " + ", ".join(f'"{k}"' for k in unexpected) + ".")
        if errors:
            raise RuntimeError(f"Error(s) in loading state_dict for {type(self).__name__}:\n\t" + "\n\t".join(errors))
        for k, v in state_dict.items():
            if k in own:
                own[k]._a[...] = _to_dtype(v._a, own[k].dtype)
                own[k]._ver[0] += 1
        return _LoadResult(missing, unexpected)

    def extra_repr(self):
        return ""

    def __repr__(self):
        lines = []
        for name, m in self._modules.items():
            child = repr(m).replace("\n", "\n  ")
            lines.append(f"  ({name}): {child}")
        head = f"{type(self).__name__}({self.extra_repr()}"
        if not lines:
            return head + ")"
        return head + "\n" + "\n".join(lines) + "\n)"


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        _check_device(device)
        self.in_features, self.out_features = in_features, out_features
        bound = 1 / _math.sqrt(in_features) if in_features else 0.0
        d = (dtype or float32).np
        rng = default_generator._rng
        self.weight = nn.Parameter(_new(rng.uniform(-bound, bound, (out_features, in_features)).astype(d)))
        self.bias = nn.Parameter(_new(rng.uniform(-bound, bound, (out_features,)).astype(d))) if bias else None

    def forward(self, input):  # noqa: A002
        return _linear(input, self.weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, device=None, dtype=None, _weight=None):
        super().__init__()
        _check_device(device)
        if padding_idx is not None:
            raise _unsupported("nn.Embedding(padding_idx=...)")
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        w = _weight._a if _weight is not None else default_generator._rng.standard_normal((num_embeddings, embedding_dim))
        self.weight = nn.Parameter(_new(_np.asarray(w).astype((dtype or float32).np)))

    def forward(self, input):  # noqa: A002
        return _embedding(input, self.weight)

    def extra_repr(self):
        return f"{self.num_embeddings}, {self.embedding_dim}"


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        d = (dtype or float32).np
        if elementwise_affine:
            self.weight = nn.Parameter(_new(_np.ones(self.normalized_shape, dtype=d)))
            self.bias = nn.Parameter(_new(_np.zeros(self.normalized_shape, dtype=d))) if bias else None
        else:
            self.weight = self.bias = None

    def forward(self, input):  # noqa: A002
        return _layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.weight is not None}"


class RMSNorm(Module):
    def __init__(self, normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(_new(_np.ones(self.normalized_shape, dtype=(dtype or float32).np))) if elementwise_affine else None

    def forward(self, input):  # noqa: A002
        return _rms_norm(input, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.weight is not None}"


class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__()
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        d = (dtype or float32).np
        self.weight = nn.Parameter(_new(_np.ones(num_features, dtype=d))) if affine else None
        self.bias = nn.Parameter(_new(_np.zeros(num_features, dtype=d))) if affine else None
        if track_running_stats:
            self.register_buffer("running_mean", _new(_np.zeros(num_features, dtype=d)))
            self.register_buffer("running_var", _new(_np.ones(num_features, dtype=d)))
            self.register_buffer("num_batches_tracked", _new(_np.asarray(0, dtype=_np.int64)))
        else:
            self.running_mean = self.running_var = None

    def forward(self, input):  # noqa: A002
        x = input
        if x.ndim not in (2, 3):
            raise ValueError(f"expected 2D or 3D input (got {x.ndim}D input)")
        if x.shape[1] != self.num_features:
            raise RuntimeError(f"running_mean should contain {x.shape[1]} elements not {self.num_features}")
        dims = (0,) if x.ndim == 2 else (0, 2)
        shape = (1, -1) if x.ndim == 2 else (1, -1, 1)
        if self.training or self.running_mean is None:
            n = x.numel() // x.shape[1]
            if self.training and n <= 1:
                raise ValueError(f"Expected more than 1 value per channel when training, got input size {x.shape}")
            mean_ = _mean(x, dims, keepdim=True)
            var_ = _var(x, dims, unbiased=False, keepdim=True)
            if self.training and self.running_mean is not None:
                with no_grad():
                    m = self.momentum
                    self.running_mean._a[...] = (1 - m) * self.running_mean._a + m * mean_._a.reshape(-1)
                    self.running_var._a[...] = (1 - m) * self.running_var._a + m * var_._a.reshape(-1) * n / _builtins.max(n - 1, 1)
                    self.num_batches_tracked._a[...] += 1
        else:
            mean_ = self.running_mean.reshape(*shape)
            var_ = self.running_var.reshape(*shape)
        y = (x - mean_) * rsqrt(var_ + self.eps)
        if self.weight is not None:
            y = y * self.weight.reshape(*shape) + self.bias.reshape(*shape)
        return y

    def extra_repr(self):
        return f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.weight is not None}, track_running_stats={self.running_mean is not None}"


class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p

    def forward(self, input):  # noqa: A002
        return _dropout(input, self.p, self.training)

    def extra_repr(self):
        return f"p={self.p}, inplace=False"


def _activation(name, fn, extra=""):
    def forward(self, input):  # noqa: A002
        return fn(input)
    return type(name, (Module,), {"forward": forward, "extra_repr": lambda self: extra})


ReLU = _activation("ReLU", relu)
Tanh = _activation("Tanh", tanh)
Sigmoid = _activation("Sigmoid", sigmoid)
SiLU = _activation("SiLU", _silu)
Identity = _activation("Identity", lambda x: x)


class GELU(Module):
    def __init__(self, approximate="none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, input):  # noqa: A002
        return _gelu(input, self.approximate)

    def extra_repr(self):
        return f"approximate='{self.approximate}'"


class Softmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, input):  # noqa: A002
        return _f_softmax(input, self.dim)

    def extra_repr(self):
        return f"dim={self.dim}"


class CrossEntropyLoss(Module):
    def __init__(self, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.weight, self.ignore_index, self.reduction, self.label_smoothing = weight, ignore_index, reduction, label_smoothing

    def forward(self, input, target):  # noqa: A002
        return _cross_entropy(input, target, self.weight, ignore_index=self.ignore_index, reduction=self.reduction, label_smoothing=self.label_smoothing)


class MSELoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):  # noqa: A002
        return _mse_loss(input, target, reduction=self.reduction)


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], _collections.OrderedDict):
            for k, m in args[0].items():
                self.add_module(k, m)
        else:
            for i, m in enumerate(args):
                self.add_module(str(i), m)

    def forward(self, input):  # noqa: A002
        for m in self._modules.values():
            input = m(input)
        return input

    def __getitem__(self, idx):
        mods = list(self._modules.values())
        if isinstance(idx, slice):
            return nn.Sequential(_collections.OrderedDict(list(self._modules.items())[idx]))  # keeps the keys
        return mods[idx]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def append(self, module):
        self.add_module(str(len(self)), module)
        return self


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        for m in modules or []:
            self.append(m)

    def append(self, module):
        self.add_module(str(len(self._modules)), module)
        return self

    def extend(self, modules):
        for m in modules:
            self.append(m)
        return self

    def __getitem__(self, idx):
        mods = list(self._modules.values())
        if isinstance(idx, slice):
            return nn.ModuleList(mods[idx])
        return mods[idx]

    def __setitem__(self, idx, module):
        key = list(self._modules)[idx]
        self.add_module(key, module)

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())


class ModuleDict(Module):
    def __init__(self, modules=None):
        super().__init__()
        for k, m in (modules or {}).items():
            self.add_module(k, m)

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self.add_module(key, module)

    def __contains__(self, key):
        return key in self._modules

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()


for _name, _obj in dict(
    Parameter=Parameter, Module=Module, Linear=Linear, Embedding=Embedding, LayerNorm=LayerNorm, RMSNorm=RMSNorm,
    BatchNorm1d=BatchNorm1d, Dropout=Dropout, ReLU=ReLU, GELU=GELU, Tanh=Tanh, Sigmoid=Sigmoid, SiLU=SiLU,
    Identity=Identity, Softmax=Softmax, CrossEntropyLoss=CrossEntropyLoss, MSELoss=MSELoss, Sequential=Sequential,
    ModuleList=ModuleList, ModuleDict=ModuleDict,
).items():
    _obj.__module__ = "torch.nn"
    setattr(nn, _name, _obj)

# ----- torch.nn.init -----
_init = _submodule("nn.init", "torch.nn.init for the browser.")
nn.init = _init


def _init_fill(t, values):
    with no_grad():
        t._a[...] = _np.asarray(values).astype(t._a.dtype)
        t._ver[0] += 1
    return t


_init.normal_ = lambda tensor, mean=0.0, std=1.0, generator=None: _init_fill(tensor, _gen(generator)._rng.standard_normal(tensor._a.shape) * std + mean)
_init.uniform_ = lambda tensor, a=0.0, b=1.0, generator=None: _init_fill(tensor, _gen(generator)._rng.random(tensor._a.shape) * (b - a) + a)
_init.zeros_ = lambda tensor: _init_fill(tensor, 0)
_init.ones_ = lambda tensor: _init_fill(tensor, 1)
_init.constant_ = lambda tensor, val: _init_fill(tensor, val)


def _fans(t):
    if t.ndim < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    receptive = int(_np.prod(t.shape[2:])) if t.ndim > 2 else 1
    return t.shape[1] * receptive, t.shape[0] * receptive


def _kaiming_std(t, a, mode, nonlinearity):
    fan_in, fan_out = _fans(t)
    gains = {"linear": 1.0, "sigmoid": 1.0, "tanh": 5.0 / 3, "relu": _math.sqrt(2.0),
             "leaky_relu": _math.sqrt(2.0 / (1 + a ** 2))}
    if nonlinearity not in gains:
        raise _unsupported(f"nn.init kaiming with nonlinearity={nonlinearity!r}")
    return gains[nonlinearity] / _math.sqrt(fan_in if mode == "fan_in" else fan_out)


def _kaiming_uniform_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", generator=None):
    bound = _math.sqrt(3.0) * _kaiming_std(tensor, a, mode, nonlinearity)
    return _init_fill(tensor, _gen(generator)._rng.uniform(-bound, bound, tensor._a.shape))


def _kaiming_normal_(tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", generator=None):
    return _init_fill(tensor, _gen(generator)._rng.standard_normal(tensor._a.shape) * _kaiming_std(tensor, a, mode, nonlinearity))


def _xavier_uniform_(tensor, gain=1.0, generator=None):
    fan_in, fan_out = _fans(tensor)
    bound = gain * _math.sqrt(6.0 / (fan_in + fan_out))
    return _init_fill(tensor, _gen(generator)._rng.uniform(-bound, bound, tensor._a.shape))


def _xavier_normal_(tensor, gain=1.0, generator=None):
    fan_in, fan_out = _fans(tensor)
    return _init_fill(tensor, _gen(generator)._rng.standard_normal(tensor._a.shape) * gain * _math.sqrt(2.0 / (fan_in + fan_out)))


_init.kaiming_uniform_ = _kaiming_uniform_
_init.kaiming_normal_ = _kaiming_normal_
_init.xavier_uniform_ = _xavier_uniform_
_init.xavier_normal_ = _xavier_normal_

# ----- torch.nn.utils -----
_nn_utils = _submodule("nn.utils", "torch.nn.utils for the browser.")
nn.utils = _nn_utils


def _clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    params = [parameters] if isinstance(parameters, Tensor) else list(parameters)
    grads = [p.grad._a for p in params if p.grad is not None]
    if norm_type not in (2, 2.0):
        raise _unsupported(f"clip_grad_norm_(norm_type={norm_type})")
    if not grads:
        return _new(_np.asarray(0.0, dtype=_np.float32))
    total = _np.float32(_np.sqrt(_builtins.sum(float((g.astype(_np.float64) ** 2).sum()) for g in grads)))
    if error_if_nonfinite and not _np.isfinite(total):
        raise RuntimeError(f"The total norm of order {float(norm_type)} for gradients from `parameters` is non-finite, so it cannot be clipped.")
    coef = _builtins.min(float(max_norm) / (float(total) + 1e-6), 1.0)
    for p in params:
        if p.grad is not None:
            p.grad._a *= p.grad._a.dtype.type(coef)
    return _new(_np.asarray(total, dtype=_np.float32))


_nn_utils.clip_grad_norm_ = _clip_grad_norm_

# ----- torch.optim -----
optim = _submodule("optim", "torch.optim for the browser.")


class Optimizer:
    def __init__(self, params, defaults):
        self.defaults = defaults
        params = [params] if isinstance(params, Tensor) else list(params)
        if not params:
            raise ValueError("optimizer got an empty parameter list")
        groups = params if isinstance(params[0], dict) else [{"params": params}]
        self.param_groups = []
        for g in groups:
            g = dict(g)
            g["params"] = [g["params"]] if isinstance(g["params"], Tensor) else list(g["params"])
            for p in g["params"]:
                if not p.is_leaf:
                    raise ValueError("can't optimize a non-leaf Tensor")
            for k, v in defaults.items():
                g.setdefault(k, v)
            self.param_groups.append(g)
        self.state = {}

    def zero_grad(self, set_to_none=True):
        for g in self.param_groups:
            for p in g["params"]:
                if set_to_none:
                    p.grad = None
                elif p.grad is not None:
                    p.grad._a[...] = 0

    def _params(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    yield g, p, p.grad._a

    @staticmethod
    def _write(p, new):
        p._a[...] = new.astype(p._a.dtype)
        p._ver[0] += 1


class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov))

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g, p, d_p in self._params():
            if g["weight_decay"]:
                d_p = d_p + g["weight_decay"] * p._a
            if g["momentum"]:
                st = self.state.setdefault(p, {})
                buf = st.get("momentum_buffer")
                buf = d_p.copy() if buf is None else g["momentum"] * buf + (1 - g["dampening"]) * d_p
                st["momentum_buffer"] = buf
                d_p = d_p + g["momentum"] * buf if g["nesterov"] else buf
            self._write(p, p._a - g["lr"] * d_p)
        return loss


class Adam(Optimizer):
    _decoupled = False

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        if amsgrad:
            raise _unsupported("optim.Adam(amsgrad=True)")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g, p, grad in self._params():
            lr, (b1, b2), eps, wd = g["lr"], g["betas"], g["eps"], g["weight_decay"]
            st = self.state.setdefault(p, {})
            if not st:
                st["step"] = 0
                st["exp_avg"] = _np.zeros_like(p._a)
                st["exp_avg_sq"] = _np.zeros_like(p._a)
            st["step"] += 1
            t = st["step"]
            w = p._a
            if wd and self._decoupled:
                w = w * (1 - lr * wd)
            elif wd:
                grad = grad + wd * w
            st["exp_avg"] = st["exp_avg"] + (grad - st["exp_avg"]) * (1 - b1)
            st["exp_avg_sq"] = st["exp_avg_sq"] * b2 + grad * grad * (1 - b2)
            bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
            denom = _np.sqrt(st["exp_avg_sq"]) / _math.sqrt(bc2) + eps
            self._write(p, w - (lr / bc1) * st["exp_avg"] / denom)
        return loss


class AdamW(Adam):
    _decoupled = True

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)


for _name, _obj in dict(Optimizer=Optimizer, SGD=SGD, Adam=Adam, AdamW=AdamW).items():
    _obj.__module__ = "torch.optim"
    setattr(optim, _name, _obj)

# The nn and optim classes live only under torch.nn / torch.optim, as in torch (no `torch.Linear`).
for _name in ("Module", "Linear", "Embedding", "LayerNorm", "RMSNorm", "BatchNorm1d", "Dropout", "ReLU", "GELU", "Tanh",
              "Sigmoid", "SiLU", "Identity", "Softmax", "CrossEntropyLoss", "MSELoss", "Sequential", "ModuleList",
              "ModuleDict", "Parameter", "Optimizer", "SGD", "Adam", "AdamW"):
    del globals()[_name]
del _name, _fn, _obj
