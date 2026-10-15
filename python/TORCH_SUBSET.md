# mini-torch: the browser's `torch`

PyTorch doesn't run in Pyodide. Browser challenges import `python/torch.py` as `torch`, a NumPy-backed shim with its own small autograd. The code a learner writes in the browser is the same code they run on their GPU later, so the shim mirrors torch's names, signatures, dtypes, broadcasting, return types and error messages for the subset below.

**This file is the contract.** Lesson code, solutions, mistakes and `tests.py` may use anything listed here and nothing else. Anything outside it raises `MiniTorchUnsupported`, and the tests panel says so plainly: "`torch.einsum` is real PyTorch, but the browser's mini-torch doesn't include it. Browser lessons use the subset in TORCH_SUBSET.md. It will work on your GPU." The same goes for missing `Tensor` methods, `torch.nn`/`torch.nn.functional`/`torch.optim` members and `import torch.<submodule>`.

`import torch`, `import torch.nn as nn`, `import torch.nn.functional as F`, `from torch import nn, optim` and `import torch.optim as optim` all work.

Every group below is checked in real Pyodide by `pipeline/test/shim*.test.ts` (forward values, dtypes, errors; every differentiable op against finite differences).

## Tensors and dtypes

| API | Notes |
|---|---|
| `torch.tensor(data, dtype=None, device=None, requires_grad=False)` | Nested lists (any depth), Python scalars, 0-d tensors inside lists, NumPy arrays. Python floats → `float32`, ints → `int64`, bools → `bool` (as torch). NumPy arrays keep their dtype (`float64` stays `float64`), except that int32 arrays become `int64`: Pyodide's NumPy makes `np.array([1, 2])` int32 where a desktop makes it int64, and int64 is what `one_hot`, `cross_entropy` and `embedding` need. Ragged lists raise `ValueError`. |
| `torch.Tensor(data)`, `torch.Tensor(2, 3)` | Legacy constructor: data becomes `float32`; ints are a zero-filled shape. Used by micrograd's torch comparison. |
| `torch.as_tensor`, `torch.from_numpy`, `torch.is_tensor`, `torch.numel` | `from_numpy` shares memory (except for int32 arrays, which are copied to int64, see above). |
| dtypes | `torch.float32`/`float`, `float64`/`double`, `float16`/`half`, `int64`/`long`, `int32`/`int`, `int16`/`short`, `int8`, `uint8`, `bool`. `repr` is `torch.int8`; `.is_floating_point`, `.itemsize`, `.is_signed`. **No `bfloat16`** (NumPy has none). |
| `torch.finfo(dtype)`, `torch.iinfo(dtype)` | `.bits .eps .max .min .tiny` / `.bits .min .max`. |
| `torch.get_default_dtype()` | Always `float32`. |
| `t.dtype .shape .ndim .device .is_cuda .itemsize .nbytes` | `shape` is a `torch.Size` (a tuple). `device` is always `cpu`. |
| `t.size(dim=None) t.dim() t.numel() t.element_size() t.is_floating_point() t.is_contiguous()` | `is_contiguous()` is always `True`. |
| `t.to(dtype)`, `t.to(device)`, `t.to(other)`, `t.to(dtype=...)` | Returns `t` itself when nothing changes, like torch. |
| `t.float() .double() .half() .long() .int() .short() .char() .byte() .bool() .type(dtype) .type_as(other)` | float → int truncates toward zero, then **wraps** out-of-range values (`128.0 → -128` for int8). Real torch leaves out-of-range float→int casts undefined; the shim wraps every time so the overflow lesson is reproducible. NaN and ±inf become 0. float → float casts are differentiable. |
| `t.item() t.tolist() t.numpy() np.asarray(t)` | Floats show their float32 value (`0.41999998688697815`). `numpy()` shares memory and refuses a tensor that requires grad, like torch. |
| `t.cpu()`, `t.contiguous()`, `t.clone()`, `t.detach()` | |
| `bool(t) float(t) int(t)`, `operator.index(t)`, f-string formats | One-element tensors where torch requires it; format specs (`f"{t:.4f}"`) on 0-d tensors only. A 1-element int tensor works as a list index. |

### Type promotion

Exactly torch's `result_type`: dimensioned tensors outrank 0-d tensors, which outrank Python numbers; a lower-priority operand only changes the dtype when it brings a higher category (bool < int < float).

- `int8 + 1 → int8` (and wraps: `127 * 2 → -2`, `100 + 200 → 44`), `int8 + 1.5 → float32`, `int8 + int64 tensor → int64`, `int8 * tensor(2) → int8`.
- `float16 * 2.0 → float16`, `float16 + float32 tensor → float32`, `int64 tensor + float16 tensor → float16` (NumPy would say float64).
- `/` always gives a float (`float32` for ints). `//` is floor division, `%` is Python-style remainder; integer division by zero raises `ZeroDivisionError`, as in torch.
- `int ** negative int` raises torch's "Integers to negative integer powers are not allowed."

### Printing

`repr` follows torch's formats: `tensor([1., 2.])` for whole-number floats, `tensor([0.4200, -1.3000])` (4 decimals), `tensor([1.0000e-05, 1.0000e+00])` when nonzero values span more than 1000× (`tensor([1.0000e+00, 2.0000e+03])`) or a non-integer one goes under 1e-4, `dtype=` for anything but float32/int64/bool, `requires_grad=True` on leaves, `grad_fn=<MulBackward0>` on results, the 2-D/3-D bracket layout, and `...` summarising tensors over 1000 elements. Column padding is not replicated (`tensor([1.0000, -0.5000])`, torch pads `[ 1.0000, -0.5000]`).

## Creation

| API | Notes |
|---|---|
| `torch.zeros/ones/empty(*size, dtype=None, device=None, requires_grad=False)` | Sizes as ints or one tuple. `empty` is zero-filled. Default `float32`. |
| `torch.full(size, fill_value, dtype=None)` | dtype follows the fill value: `7 → int64`, `7.0 → float32`, `True → bool`. |
| `torch.zeros_like/ones_like/empty_like/full_like(t, ..., dtype=None)` | Keep `t`'s dtype unless given. |
| `torch.arange(end)`, `arange(start, end, step=1, dtype=None)` | All-int arguments → `int64`, any float → `float32`. |
| `torch.linspace(start, end, steps)`, `torch.eye(n, m=None)` | `float32`. |

## Random numbers

| API | Notes |
|---|---|
| `torch.Generator()`, `g.manual_seed(n)` (returns `g`), `g.initial_seed()` | `torch.Generator().manual_seed(2147483647)` works as in makemore. |
| `torch.manual_seed(n)`, `torch.initial_seed()`, `torch.default_generator` | Seeds the default generator. A fresh generator starts from torch's default seed, 67280421310721, so unseeded runs are repeatable too. |
| `torch.randn/rand(*size, generator=None, dtype=None, requires_grad=False)` | Float dtypes only (int raises torch's `"normal_kernel_cpu" not implemented for 'Long'`). |
| `torch.randint(high, size)`, `randint(low, high, size, generator=None, dtype=None)` | `size` must be a tuple, as in torch. `int64`. |
| `torch.randperm(n)`, `torch.randn_like`, `torch.rand_like`, `torch.bernoulli(p)` | |
| `torch.multinomial(probs, num_samples, replacement=False, generator=None)`, `t.multinomial(...)` | 1-D or 2-D (one row per distribution). Weights needn't sum to 1. Zero-weight categories are never drawn. Without replacement, draws are distinct. torch's errors for negative/NaN weights, all-zero rows, and too many samples without replacement. |
| `t.normal_(mean, std, generator=None)`, `t.uniform_(a, b, generator=None)` | In place. |

**Same seed, same numbers, but not torch's numbers.** Draws come from NumPy's PCG64, so `torch.manual_seed(42); torch.randn(3)` is repeatable in the browser but differs from real torch. Lesson tests must not hard-code values from a real torch run: compute expected values from the learner's own seeded draws, use a reference implementation inside `tests.py`, or check statistics. Draw sequences also differ (one `randn(2, 3)` is not two `randn(3)`).

## Shapes, joining and indexing

| API | Notes |
|---|---|
| `t.reshape(*shape)`, `t.view(*shape)`, `torch.reshape(t, shape)`, `t.view_as/reshape_as(other)` | Ints or one tuple, one `-1`. A bad shape raises torch's `shape '[-1, 5]' is invalid for input of size 16`. `view` never fails on contiguity. |
| `t.transpose(d0, d1)`/`swapaxes`, `t.permute(*dims)`, `t.T`, `t.mT`, `t.t()` | `T` reverses all dims (torch's deprecated behaviour for ≠2-D). |
| `t.unsqueeze(d)`, `t.squeeze(dim=None)`, `t.flatten(start_dim=0, end_dim=-1)` | `squeeze(d)` on a non-1 dim is a no-op, like torch. |
| `t.expand(*sizes)` (`-1` keeps a size), `t.expand_as(other)` | Broadcast view, read-only. |
| `t.repeat(*sizes)`, `t.repeat_interleave(int, dim=None)`, `t.flip(dims)` | `repeat_interleave` takes an int count only. |
| `torch.cat/concat/concatenate(tensors, dim=0)`, `torch.stack(tensors, dim=0)` | torch's errors for mismatched sizes. `cat` promotes dtypes. |
| `t.split(int or list, dim=0)`, `t.chunk(n, dim=0)`, `t.unbind(dim=0)` | Tuples of tensors. |
| `t.gather(dim, index)`, `t.index_select(dim, index)`, `t.scatter(dim, index, src or value=)`, `t.scatter_add(dim, index, src)`, `scatter_`/`scatter_add_` | int64 indices, torch's out-of-bounds errors. Duplicate scatter targets: last write wins (torch leaves it unspecified). |
| `torch.outer(a, b)` | |
| `t[...]` | Ints, negative ints, slices with positive steps, `None`, `...`, lists, `range`, int32/int64 tensors of any shape (`C[Xb]` with `Xb` of shape `[32, 3]` gives `[32, 3, d]`), bool masks (`t[t > 0]`), and tuples mixing them (`logprobs[range(n), Yb]`). Errors use torch's wording (`index 3 is out of bounds for dimension 0 with size 3`). |
| `t[...] = value`, `t[i, j] += 1` | Value is a number or a broadcastable tensor, cast to `t`'s dtype (`int_t[0] = 2.7` stores 2). Bigram counting `N[ix1, ix2] += 1` works. |
| `len(t)`, iteration, `x in t` | Iterating gives rows (differentiable). |

**Views share storage.** Slices, `view`/`reshape`, `transpose`, `permute`, `T`, `unsqueeze`, `squeeze`, `expand` and `detach()` return tensors that share memory with the original when autograd isn't recording through them, so `row = x[1]; row += 1` and `x[0, 1].add_(5)` change `x`, as in torch (indexing down to one element gives a 0-d view; a 0-d int tensor index acts like an int). **Difference:** when the source requires grad (a model's activations in the forward pass), these results are copies, so writing into them doesn't change the source. Index the source directly instead (`h[0, 1] = …`, not `h[0][1] = …`).

## Math

| API | Notes |
|---|---|
| `+ - * / // % **`, unary `-`, `abs()`, `@` | Broadcasting follows torch, and shapes that don't fit raise torch's "The size of tensor a (8) must match the size of tensor b (3) at non-singleton dimension 1". |
| `torch.add/sub(x, y, alpha=1)`, `torch.mul/div`, `t.add/sub/mul/div/pow` | |
| `torch.matmul(a, b)`, `a @ b`, `torch.mm`, `torch.bmm` | 1-D, 2-D and batched (broadcast batch dims). Both operands need the same dtype (torch: "expected m1 and m2 to have the same dtype, but got: long int != float"). Wrong shapes raise "mat1 and mat2 shapes cannot be multiplied (6x4 and 5x2)", leading dims folded as torch does. |
| `exp log log2 sqrt rsqrt tanh sigmoid relu sin cos erf reciprocal square neg sign floor ceil trunc` | As `torch.f(t)` and `t.f()`. Ints become float32 for the float functions. |
| `torch.round(t, decimals=0)`, `t.round()` | Half to even, like torch (`0.5 → 0`, `1.5 → 2`, `63.5 → 64`). |
| `torch.clamp/clip(t, min=None, max=None)`, `t.clamp/clip` | Either bound may be `None`, not both. Number or tensor bounds. int8 with an out-of-range int bound stays int8. |
| `torch.maximum/minimum(a, b)` | Elementwise. |
| `t.sum/mean(dim=None, keepdim=False, dtype=None)` | `dim` is an int or tuple; `keepdims=` is accepted too (backprop-ninja code uses it). `sum` of bool/int gives `int64`. `mean` of ints raises torch's "mean(): could not infer output dtype…". |
| `t.var/std(dim=None, unbiased=True, keepdim=False, correction=None)` | Bessel-corrected by default, like torch. |
| `t.max()`, `t.max(dim, keepdim=False)`, `torch.max(t, dim)`, `torch.max(a, b)`; same for `min` | With `dim`, returns `(values, indices)` as `torch.return_types.max`, a tuple with `.values`/`.indices`. Dividing it by a number raises torch's TypeError, which the harness explains. Ties: first index. |
| `t.amax/amin(dim=(), keepdim=False)` | Values only; `dim` int or tuple. |
| `t.argmax/argmin(dim=None, keepdim=False)` | No `dim`: index into the flattened tensor. |
| `t.any/all(dim=None)`, `t.norm(p='fro' or 1, dim=None, keepdim=False)`, `t.cumsum(dim)` | |
| `softmax(dim)`, `log_softmax(dim)` (`torch.`, `t.` and `F.`) | `dim` is required except in `F.softmax`/`F.log_softmax`, which use torch's legacy implicit dim. Max-subtracted; a row of all `-inf` gives `nan`, like torch. Int input raises torch's error. |
| `t.sort(dim=-1, descending=False)` → `torch.return_types.sort`, `t.argsort(...)`, `t.topk(k, dim=-1, largest=True)` → `torch.return_types.topk` | Stable: ties keep their original order. |
| `torch.tril/triu(t, diagonal=0)` | Last two dims. |
| `t.masked_fill(mask, value)` | Bool masks only (torch's error otherwise), broadcast to `t`'s shape. `float('-inf')` for causal attention. |
| `torch.where(cond, x, y)` (numbers allowed), `torch.where(cond)`, `t.nonzero(as_tuple=False)` | `cond` must be bool. |
| `== != < <= > >=`, `& \| ^ ~`, `t.eq/ne/lt/le/gt/ge`, `torch.logical_not/and/or` | Bitwise ops on bool/int only. `t == None` is `False`, as in torch. |
| `torch.isclose(a, b, rtol=1e-5, atol=1e-8)`, `torch.allclose(...)`, `torch.equal(a, b)` | Both tensors, same dtype (torch: "Long did not match Float"). |

## torch.nn.functional

| API | Notes |
|---|---|
| `F.cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)` | Class indices (`int64`, shape `[N]`, or `[]` for a single `[C]` row, or `[N, d…]` for `[N, C, d…]`) or class probabilities (same shape as input). `grad_fn` is `NllLossBackward0`. torch's errors: "Target 3 is out of bounds.", "Expected input batch_size (2) to match target batch_size (3).", "expected scalar type Long but found Int". `label_smoothing` with `weight` is not supported. |
| `F.nll_loss`, `F.mse_loss`, `F.l1_loss` | `reduction='mean' \| 'sum' \| 'none'`. |
| `F.softmax`, `F.log_softmax`, `F.relu`, `F.gelu(approximate='none' \| 'tanh')`, `F.silu`, `F.sigmoid`, `F.tanh` | `gelu` default is the exact erf form. |
| `F.one_hot(t, num_classes=-1)` | int64 in, int64 out (cast with `.float()`). |
| `F.embedding(idx, weight)`, `F.linear(x, weight, bias=None)` | `weight` is `[out, in]` like torch. |
| `F.layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5)`, `F.rms_norm(x, normalized_shape, weight=None, eps=None)` | Biased variance, like torch. |
| `F.dropout(x, p=0.5, training=True)` | Uses the default generator. |
| `F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)` | Computed as explicit masked softmax attention. Bool masks mean "may attend"; float masks are added. |
| `F.pad(x, pad, value=0)` | Constant mode, non-negative pads, last dim first. |

## Autograd

| API | Notes |
|---|---|
| `requires_grad=True` (factories and `torch.tensor`), `t.requires_grad = …`, `t.requires_grad_()` | Float tensors only ("only Tensors of floating point dtype can require gradients"). Changing a non-leaf's flag raises torch's error. |
| `loss.backward(gradient=None, retain_graph=False)` | Implicit gradient only for one-element outputs ("grad can be implicitly created only for scalar outputs"). A second backward through ops that saved tensors raises torch's "Trying to backward through the graph a second time…". |
| `t.grad`, `t.grad = None`, `t.grad.zero_()` | Gradients accumulate across `backward()` calls. Non-leaves have `grad` `None` unless they called `retain_grad()`. |
| `t.retain_grad()`, `t.retains_grad`, `t.is_leaf`, `t.grad_fn` (`.name()`), `t.detach()`, `t.data` | `.data` shares memory and is invisible to autograd (`p.data += -lr * p.grad`). |
| `torch.no_grad()`, `torch.enable_grad()`, `torch.inference_mode()`, `torch.set_grad_enabled(b)`, `torch.is_grad_enabled()` | Context managers and decorators: `@torch.no_grad()` and bare `@torch.no_grad`, on functions and methods. `inference_mode` behaves like `no_grad`. |

Differentiable: everything in Math and torch.nn.functional that returns floats, including the arithmetic operators, matmul, every reduction (max/amax split ties evenly like torch; `max(dim)` sends the grad to the returned index), softmax/log_softmax, the losses, the shape ops, `cat`/`stack`/`split`/`chunk`/`unbind`, indexing with every index kind above, `gather`/`scatter`/`index_select`, `masked_fill`, `where`, `tril`/`triu`, `cumsum`, `sort`/`topk` values, `clamp`, `F.pad`, and float casts. Rounding ops (`round`, `floor`, `ceil`, `trunc`, `sign`) have zero gradient, as in torch. `//` has no gradient (backward raises). `grad_fn` names match torch for single ops (`MulBackward0`, `SoftmaxBackward0`, `NllLossBackward0`, `NativeLayerNormBackward0`, `EmbeddingBackward0`, `ViewBackward0`); a few composites differ (`F.linear` shows `AddBackward0`, not `AddmmBackward0`; batched `@` shows `MatmulBackward0`; every index shows `IndexBackward0`), so don't test those names.

**In-place ops** (`+= -= *= /=`, `add_ sub_ mul_ div_ pow_ clamp_ masked_fill_ zero_ fill_ copy_ scatter_`, item assignment) follow torch's rules:
- On a leaf that requires grad while grad mode is on: torch's "a leaf Variable that requires grad is being used in an in-place operation." Update parameters inside `torch.no_grad()` or through `.data`.
- On a non-leaf, or with a value that requires grad: recorded in the graph. Karpathy's manual-backprop line `dlogits = F.softmax(logits, 1); dlogits[range(n), Yb] -= 1` works.
- If an op saved the old value for its backward, `backward()` raises torch's "one of the variables needed for gradient computation has been modified by an inplace operation…" (version counters, as in torch).
- The result must fit the tensor's dtype ("result type Float can't be cast to the desired output type Long") and shape ("output with shape [3] doesn't match the broadcast shape [2, 3]").

Not supported: `create_graph=True` (no double backward), `backward(inputs=…)`, `torch.autograd.*` (`grad`, `Function`, `gradcheck`), hooks, sparse gradients, and item assignment of a value that requires grad into a view (`buf[0][1] = w` raises `MiniTorchUnsupported`; write `buf[0, 1] = w`). `buf[ix] += v` with `v` requiring grad works (the manual-backprop `dC[ix] += demb[k, j]` loop), because Python finishes it with `buf[ix] = …` on `buf`. A standalone alias does not: after `row = buf[0]; row += v`, `buf`'s values change but its gradient doesn't reach `v`. Views of tensors that require grad are copies (see Shapes), so writing into one doesn't reach the original.

## torch.nn

| API | Notes |
|---|---|
| `nn.Module` | `super().__init__()` first (torch's "cannot assign parameters/module before Module.__init__() call" otherwise). Assigning a `Parameter`, `Module` or registered buffer registers it. `__call__` → `forward` (torch's "missing the required "forward" function" otherwise). `parameters() named_parameters() buffers() named_buffers() children() named_children() modules() named_modules() register_buffer(name, t, persistent=True) register_parameter add_module train(mode=True) eval() training requires_grad_(b) zero_grad(set_to_none=True) apply(fn) to(dtype or 'cpu') cpu() state_dict() load_state_dict(sd, strict=True) extra_repr()`. `repr` prints torch's module tree. `copy.deepcopy(model)` works. |
| `nn.Parameter(t, requires_grad=True)` | A leaf `Tensor` subclass; prints `Parameter containing:`. Assigning a plain tensor to a parameter's name raises torch's TypeError. |
| `nn.Linear(in, out, bias=True)` | `weight [out, in]`, `bias [out]`, both U(−1/√in, 1/√in) like torch (values differ, see Random numbers). Attributes `in_features`, `out_features`. |
| `nn.Embedding(num, dim)` | `weight` ~ N(0, 1). torch's "index out of range in self". No `padding_idx`. |
| `nn.LayerNorm(shape, eps=1e-5, elementwise_affine=True, bias=True)`, `nn.RMSNorm(shape, eps=None)` | |
| `nn.BatchNorm1d(C, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)` | `[N, C]` or `[N, C, L]`. Training normalises with the biased batch variance and updates `running_mean`/`running_var` (unbiased) and `num_batches_tracked`; `eval()` uses the running stats. torch's "Expected more than 1 value per channel when training". |
| `nn.Dropout(p)`, `nn.ReLU()`, `nn.GELU(approximate='none')`, `nn.Tanh()`, `nn.Sigmoid()`, `nn.SiLU()`, `nn.Softmax(dim)`, `nn.Identity()` | |
| `nn.CrossEntropyLoss(...)`, `nn.MSELoss(reduction)` | Same arguments as the functional forms. |
| `nn.Sequential(*mods or OrderedDict)`, `nn.ModuleList(mods)`, `nn.ModuleDict(dict)` | Indexing, slicing (Sequential/ModuleList), `len`, iteration, `append`/`extend`; ModuleDict entries are also attributes (`self.transformer.wte`). |
| `nn.init.normal_ uniform_ zeros_ ones_ constant_ kaiming_uniform_ kaiming_normal_ xavier_uniform_ xavier_normal_` | Fill in place without recording, as torch. |
| `nn.utils.clip_grad_norm_(params, max_norm)` | L2 norm only; returns the total norm. |

## torch.optim

| API | Notes |
|---|---|
| `optim.SGD(params, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False)` | |
| `optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)`, `optim.AdamW(..., weight_decay=1e-2)` | torch's update rules (AdamW's decoupled decay, bias correction). No `amsgrad`. |
| `opt.step()`, `opt.zero_grad(set_to_none=True)`, `opt.param_groups`, `opt.state` | `params` is an iterable of tensors or of param-group dicts. Change `group['lr']` for a schedule. torch's "optimizer got an empty parameter list" and "can't optimize a non-leaf Tensor". |

## Devices

`torch.device('cpu')`, `device='cpu'` arguments, `.to('cpu')`, `.cpu()`, `torch.cuda.is_available()` (`False`) and `torch.cuda.device_count()` (`0`) work, so `device = 'cuda' if torch.cuda.is_available() else 'cpu'` runs unchanged. Asking for `cuda` (or any other device) raises `MiniTorchUnsupported` explaining that the browser has no GPU.

## Not supported

These raise `MiniTorchUnsupported`. Keep them in GPU-side code (`run` replays, "Run it yourself").

- dtypes: `bfloat16`, complex, `float8`, quantized tensors, `Tensor.view(dtype)`.
- Ops: `einsum`, `roll`, `prod`/`cumprod`, `unique`, `bincount`, `histc`, `quantile`, `diag`, `kron`, `log1p`/`expm1`, `polar`, FFT, `torch.linalg.*`, convolutions and pooling (`nn.Conv*`, `F.conv*`), `nn.RNN/LSTM/GRU`, `nn.MultiheadAttention`, `nn.Transformer*`, `F.normalize`, `F.cosine_similarity`, non-constant `F.pad`, `scatter(reduce=)`, `repeat_interleave` with tensor counts, `nn.Embedding(padding_idx=)`.
- Infrastructure: `torch.compile`, `torch.jit`, `torch.save/load`, `torch.utils.data`, `torch.optim.lr_scheduler`, `torch.cuda.*` beyond `is_available`/`device_count`, `torch.distributed`, AMP (`torch.autocast`, `GradScaler`), `torch.backends.*`, `torch.set_printoptions`, `torch.set_default_dtype`.
- Autograd: see Autograd.

## Performance

NumPy-backed and interpreted, sized for lesson tensors (up to a few thousand elements, small models). Measured in Pyodide under Node: about 27k scalar index updates per second (`N[a, b] += 1`, so count bigrams over a few thousand names, not all 32k), about 2 ms per training step of a makemore MLP (batch 32, 27×10 embedding, 200 hidden), and about 1 ms per forward plus backward of single-head attention at B=4, T=16, C=32. Keep browser checks to a few hundred steps.

## Writing lesson tests

- Use `harness.arr(t)` for the NumPy view of a learner's tensor, then compare with NumPy. `t.detach().numpy()` works too.
- Don't hard-code values from a real torch run (see Random numbers). Seed with `torch.Generator().manual_seed(...)` inside `tests.py` and compute the reference there.
- Error text follows torch, so a test can diagnose from `str(exc)` the same way the learner would read it on their GPU.

## Adding to the subset

Add only what a lesson's browser code needs, mirror torch's semantics (dtype, shape, broadcasting, return types, error wording), add a check to `pipeline/test/shim*.test.ts` (and a finite-difference check in `shim-autograd.test.ts` for anything differentiable), and update this file.
