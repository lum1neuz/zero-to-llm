"""Test harness for browser challenges.

Runs the learner's file, then the lesson's tests, and returns JSON the UI renders.
Diagnostics come from running the learner's code. When the learner's code raises,
`explain()` turns the exception into plain words that point at their line.

Lesson tests look like:

    from harness import test, fail, warn, learner_fn

    @test("largest_is_127", "The largest |w| lands on ±127")
    def _(L):
        quantize = learner_fn(L, "quantize_absmax")
        ...
        if wrong:
            fail("Your scale comes from 1.0, not −3.0. Which weight has the largest *magnitude*?")
"""

import contextlib
import io
import json
import re
import sys
import traceback

import torch as _torch

_TESTS = []
_WARNINGS = []
_CTX = {}


class TestFailure(Exception):
    pass


class _Skip(Exception):
    pass


class LearnerError(Exception):
    """The learner's code raised; message is already explained in plain words."""


def test(test_id, name):
    def deco(fn):
        _TESTS.append((test_id, name, fn))
        return fn
    return deco


def fail(message):
    raise TestFailure(message)


def skip(message=""):
    raise _Skip(message)


def warn(message):
    if message not in _WARNINGS:
        _WARNINGS.append(message)


def script_error():
    """The explained error from running the learner's file top to bottom, or None."""
    return _CTX.get("script_error")


def stdout():
    return _CTX.get("stdout", "")


def learner_fn(L, name):
    fn = L.get(name)
    if fn is None:
        err = script_error()
        if err:
            fail(f"`{name}` never got defined because your script stopped first: {err['diagnosis']}")
        fail(f"There's no function called `{name}`. The tests call it by that name. Was it renamed or deleted?")
    if not callable(fn):
        fail(f"`{name}` should be a function, but it's {type(fn).__name__}.")

    def call(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (TestFailure, _Skip):
            raise
        except BaseException as exc:  # noqa: BLE001 - anything the learner's code raises
            info = explain(exc, _CTX.get("code", ""), _CTX.get("filename", "main.py"))
            if info["plain"]:
                raise LearnerError(info["diagnosis"]) from None
            where = f" (line {info['line']})" if info.get("line") else ""
            raise LearnerError(f"Calling `{name}` raised an error{where}. {info['diagnosis']}") from None
    return call


def arr(x):
    """NumPy view of a mini-torch tensor, or None if x isn't one."""
    return getattr(x, "_a", None) if isinstance(x, _torch.Tensor) else None


def as_float(x):
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return None


# ---------- explaining exceptions ----------
_PLACEHOLDER_RE = re.compile(r"^\s*(?:(\w+)\s*=|return)\s*\.\.\.\s*(?:#.*)?$")


def _placeholders(code):
    out = []
    func = None
    for n, line in enumerate(code.splitlines(), 1):
        m = re.match(r"^def\s+(\w+)", line)
        if m:
            func = m.group(1)
        pm = _PLACEHOLDER_RE.match(line)
        if pm:
            out.append({"line": n, "name": pm.group(1), "func": func})
    return out


def _placeholder_text(p):
    if p["name"]:
        return f"`{p['name']}` is still `...`, the placeholder (line {p['line']}). Fill it in first."
    return f"`{p['func']}` still returns `...`, the placeholder (line {p['line']})."


def _learner_line(exc, filename):
    line = None
    for fr in traceback.extract_tb(exc.__traceback__):
        if fr.filename == filename:
            line = fr.lineno
    if line is None and isinstance(exc, SyntaxError) and exc.filename == filename:
        line = exc.lineno
    return line


def _trimmed_traceback(exc, filename, code):
    frames = [fr for fr in traceback.extract_tb(exc.__traceback__) if fr.filename == filename]
    src = code.splitlines()
    lines = ["Traceback (most recent call last):"]
    for fr in frames:
        lines.append(f'  File "{fr.filename}", line {fr.lineno}, in {fr.name}')
        text = fr.line or (src[fr.lineno - 1].strip() if fr.lineno and 0 < fr.lineno <= len(src) else "")
        if text:
            lines.append(f"    {text}")
    if isinstance(exc, SyntaxError):
        lines.append(f'  File "{filename}", line {exc.lineno}')
        if exc.text:
            lines.append("    " + exc.text.rstrip())
    lines.append(f"{type(exc).__name__}: {exc}")
    return "\n".join(lines)


def explain(exc, code, filename):
    line = _learner_line(exc, filename)
    msg = str(exc)
    kind = type(exc).__name__
    diag = None
    plain = False  # True when the diagnosis already says everything (no "Calling X raised" prefix)
    if isinstance(exc, _torch.MiniTorchUnsupported):
        diag = msg
        plain = True
    elif isinstance(exc, SyntaxError):
        detail = (exc.msg or "").rstrip(".")
        if isinstance(exc, IndentationError):
            diag = f"Python couldn't read line {exc.lineno}: {detail}. Check the spaces at the start of the line. Each level is 4 spaces."
        else:
            diag = f"Python couldn't read line {exc.lineno}: {detail}."
    elif isinstance(exc, NameError):
        m = re.search(r"name '(\w+)' is not defined", msg)
        name = m.group(1) if m else "?"
        if name == "torch":
            diag = "`torch` isn't defined. The `import torch` line at the top is gone."
        else:
            diag = f"`{name}` isn't defined{f' at line {line}' if line else ''}. Did the line that creates it get deleted or renamed?"
    elif isinstance(exc, TypeError) and "torch.return_types.max" in msg:
        diag = ("`.max(dim=…)` returns two things, `(values, indices)`, not a tensor. "
                "Use `.amax(dim=…)` to get only the values, or take `.values`.")
    elif isinstance(exc, IndexError) and msg.startswith("Dimension out of range"):
        m = re.search(r"\[(-?\d+), (-?\d+)\], but got (-?\d+)", msg)
        if m:
            ndim = int(m.group(2)) + 1
            diag = (f"This tensor has {ndim} dimension{'s' if ndim != 1 else ''}, so `dim={m.group(3)}` doesn't exist. "
                    "Print its `.shape` to see what you're reducing over.")
    elif isinstance(exc, (TypeError, AttributeError)) and "ellipsis" in msg:
        ph = _placeholders(code)
        src = code.splitlines()
        failing = src[line - 1] if line and 0 < line <= len(src) else ""
        # Prefer the placeholder the failing line uses, then a `return ...`, then the first one.
        pick = next((p for p in ph if p["name"] and re.search(rf"\b{re.escape(p['name'])}\b", failing)), None)
        pick = pick or next((p for p in ph if not p["name"]), None) or (ph[0] if ph else None)
        diag = _placeholder_text(pick) if pick else "Something is still `...`, a placeholder. Replace it with real code."
        plain = True
    elif isinstance(exc, RecursionError):
        diag = "A function keeps calling itself and never stops."
    if diag is None:
        diag = f"{kind}: {msg}" + (f" (line {line})" if line else "")
    return {"type": kind, "message": msg, "line": line, "diagnosis": diag, "plain": plain,
            "traceback": _trimmed_traceback(exc, filename, code)}


# ---------- running ----------
def run(code, tests_src, filename="main.py"):
    _TESTS.clear()
    _WARNINGS.clear()
    _CTX.clear()
    _CTX.update(code=code, filename=filename)

    ns = {"__name__": "__main__"}
    out = io.StringIO()
    try:
        compiled = compile(code, filename, "exec")
        with contextlib.redirect_stdout(out):
            exec(compiled, ns)
    except BaseException as exc:  # noqa: BLE001
        _CTX["script_error"] = explain(exc, code, filename)
    _CTX["stdout"] = out.getvalue()
    L = {k: v for k, v in ns.items() if not k.startswith("__")}

    test_ns = {"__name__": "lesson_tests"}
    exec(compile(tests_src, "tests.py", "exec"), test_ns)

    results = []
    for test_id, name, fn in _TESTS:
        r = {"id": test_id, "name": name, "status": "pass", "message": ""}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fn(L)
        except TestFailure as e:
            r.update(status="fail", message=str(e))
        except LearnerError as e:
            r.update(status="fail", message=str(e))
        except _Skip as e:
            r.update(status="skip", message=str(e))
        except BaseException as e:  # noqa: BLE001 - a bug in the lesson's test, not the learner's code
            r.update(status="error", message=f"This test crashed. That's a bug in the lesson, not your code. ({type(e).__name__}: {e})")
        results.append(r)

    return {
        "stdout": _CTX["stdout"],
        "scriptError": _CTX.get("script_error"),
        "tests": results,
        "warnings": list(_WARNINGS),
    }


def run_json(code, tests_src, filename="main.py"):
    return json.dumps(run(code, tests_src, filename))


if __name__ == "__main__":  # python harness.py learner.py tests.py
    with open(sys.argv[1], encoding="utf-8") as f1, open(sys.argv[2], encoding="utf-8") as f2:
        print(json.dumps(run(f1.read(), f2.read(), sys.argv[1]), indent=2))
