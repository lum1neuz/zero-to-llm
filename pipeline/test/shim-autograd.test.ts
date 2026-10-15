// mini-torch autograd: every differentiable op is checked against central finite differences in
// float64, then the engine's rules (accumulation, retain_grad, no_grad, in-place and version checks).
import assert from "node:assert/strict";
import { describe, test } from "node:test";
import { runChallenge } from "../pyrun.ts";

const TESTS = `from harness import test\n@test("t", "t")\ndef _(L):\n    pass\n`;

// gradcheck(name, f, *shapes): backprop a random projection of f's output and compare each input's
// .grad with central differences. Prints only failures, so a passing group prints nothing.
const GRADCHECK = `
import numpy as np
import torch
import torch.nn.functional as F

def gradcheck(name, f, *shapes, positive=(), tol=1e-5):
    rng = np.random.default_rng(0)
    arrays = [rng.uniform(0.5, 2.0, s) if k in positive else rng.standard_normal(s) for k, s in enumerate(shapes)]
    xs = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in arrays]
    out = f(*xs)
    w = rng.standard_normal(tuple(out.shape))
    (out * torch.tensor(w)).sum().backward()
    for k, a in enumerate(arrays):
        num = np.zeros_like(a)
        for i in np.ndindex(*a.shape):
            old = a[i]
            a[i] = old + 1e-6
            fp = float((f(*[torch.tensor(v) for v in arrays]).detach().numpy() * w).sum())
            a[i] = old - 1e-6
            fm = float((f(*[torch.tensor(v) for v in arrays]).detach().numpy() * w).sum())
            a[i] = old
            num[i] = (fp - fm) / 2e-6
        got = xs[k].grad.numpy() if xs[k].grad is not None else np.zeros_like(a)
        err = float(np.abs(got - num).max()) if a.size else 0.0
        if not err < tol:
            print(f"{name}: input {k} grad off by {err:.2e}")
`;

async function run(code: string) {
  const r = await runChallenge(`${GRADCHECK}\n${code}`, TESTS, "probe.py");
  if (r.scriptError) throw new Error(r.scriptError.traceback);
  return r.stdout.trim();
}

const gradchecks = (lines: string[]) => run(lines.join("\n"));

describe("mini-torch gradients match finite differences", () => {
  test("arithmetic with broadcasting, pow, matmul", async () => {
    assert.equal(await gradchecks([
      "gradcheck('add', lambda a, b: a + b, (3, 4), (4,))",
      "gradcheck('sub', lambda a, b: a - b, (3, 1), (1, 4))",
      "gradcheck('mul_scalar', lambda a: a * 3.0 - 2, (3,))",
      "gradcheck('div', lambda a, b: a / b, (2, 3), (2, 3), positive=(1,))",
      "gradcheck('rdiv', lambda a: 1.0 / a, (4,), positive=(0,))",
      "gradcheck('pow', lambda a: a ** 3, (5,))",
      "gradcheck('pow_neg', lambda a: a ** -0.5, (5,), positive=(0,))",
      "gradcheck('pow_tensor', lambda a, b: a ** b, (4,), (4,), positive=(0,))",
      "gradcheck('rpow', lambda a: 2.0 ** a, (4,))",
      "gradcheck('square', lambda a: a.square(), (3,))",
      "gradcheck('mod', lambda a: a % 0.7, (5,))",
      "gradcheck('reused', lambda a: (a * a + a.exp()) * a, (4,))",
      "gradcheck('mm', lambda a, b: a @ b, (3, 4), (4, 2))",
      "gradcheck('mm_3d_2d', lambda a, b: a @ b, (2, 3, 4), (4, 2))",
      "gradcheck('bmm', lambda a, b: torch.bmm(a, b), (2, 3, 4), (2, 4, 5))",
      "gradcheck('vec_mat', lambda a, b: a @ b, (4,), (4, 3))",
      "gradcheck('mat_vec', lambda a, b: torch.matmul(a, b), (3, 4), (4,))",
      "gradcheck('dot', lambda a, b: a @ b, (4,), (4,))",
      "gradcheck('outer', lambda a, b: torch.outer(a, b), (3,), (4,))",
    ]), "");
  });
  test("reductions", async () => {
    assert.equal(await gradchecks([
      "gradcheck('sum', lambda a: a.sum(), (3, 4))",
      "gradcheck('sum_keepdim', lambda a: a.sum(1, keepdim=True), (3, 4))",
      "gradcheck('sum_tuple', lambda a: a.sum((0, 2)), (2, 3, 4))",
      "gradcheck('mean', lambda a: a.mean(0), (3, 4))",
      "gradcheck('var', lambda a: a.var(0, keepdim=True), (5, 3))",
      "gradcheck('var_biased', lambda a: a.var(1, unbiased=False), (3, 5))",
      "gradcheck('std', lambda a: a.std(), (6,))",
      "gradcheck('max', lambda a: a.max(), (3, 4))",
      "gradcheck('max_dim', lambda a: a.max(1).values, (3, 4))",
      "gradcheck('max_keepdim', lambda a: a.max(1, keepdim=True)[0], (3, 4))",
      "gradcheck('min_dim', lambda a: a.min(0).values, (3, 4))",
      "gradcheck('amax', lambda a: a.amax(dim=1, keepdim=True), (3, 4))",
      "gradcheck('amin', lambda a: a.amin((0, 1)), (3, 4))",
      "gradcheck('maximum', lambda a, b: torch.maximum(a, b), (6,), (6,))",
      "gradcheck('cumsum', lambda a: a.cumsum(1), (2, 4))",
      "gradcheck('norm', lambda a: a.norm(), (5,))",
      "gradcheck('sort', lambda a: a.sort(descending=True).values, (2, 5))",
      "gradcheck('topk', lambda a: torch.topk(a, 2).values, (3, 5))",
    ]), "");
  });
  test("elementwise functions", async () => {
    assert.equal(await gradchecks([
      "gradcheck('exp', lambda a: a.exp(), (5,))",
      "gradcheck('log', lambda a: a.log(), (5,), positive=(0,))",
      "gradcheck('sqrt', lambda a: torch.sqrt(a), (5,), positive=(0,))",
      "gradcheck('rsqrt', lambda a: torch.rsqrt(a), (5,), positive=(0,))",
      "gradcheck('tanh', lambda a: torch.tanh(a), (5,))",
      "gradcheck('sigmoid', lambda a: a.sigmoid(), (5,))",
      "gradcheck('relu', lambda a: F.relu(a), (7,))",
      "gradcheck('gelu', lambda a: F.gelu(a), (7,))",
      "gradcheck('gelu_tanh', lambda a: F.gelu(a, approximate='tanh'), (7,))",
      "gradcheck('silu', lambda a: F.silu(a), (7,))",
      "gradcheck('abs', lambda a: a.abs(), (7,))",
      "gradcheck('clamp', lambda a: a.clamp(-0.5, 0.5), (9,))",
      "gradcheck('sin_cos', lambda a: a.sin() * a.cos(), (5,))",
    ]), "");
  });
  test("softmax, log_softmax and the losses", async () => {
    assert.equal(await gradchecks([
      "gradcheck('softmax', lambda a: F.softmax(a, dim=-1), (3, 5))",
      "gradcheck('softmax_dim0', lambda a: torch.softmax(a, 0), (3, 5))",
      "gradcheck('log_softmax', lambda a: F.log_softmax(a, dim=1), (3, 5))",
      "gradcheck('causal_softmax', lambda a: a.masked_fill(torch.tril(torch.ones(4, 4)) == 0, float('-inf')).softmax(-1), (4, 4))",
      "gradcheck('cross_entropy', lambda a: F.cross_entropy(a, torch.tensor([0, 4, 2])), (3, 5))",
      "gradcheck('ce_sum', lambda a: F.cross_entropy(a, torch.tensor([1, 1, 3]), reduction='sum'), (3, 5))",
      "gradcheck('ce_none', lambda a: F.cross_entropy(a, torch.tensor([1, 0, 3]), reduction='none'), (3, 5))",
      "gradcheck('ce_ignore', lambda a: F.cross_entropy(a, torch.tensor([1, -100, 3])), (3, 5))",
      "gradcheck('ce_smoothing', lambda a: F.cross_entropy(a, torch.tensor([1, 2, 3]), label_smoothing=0.1), (3, 5))",
      "gradcheck('ce_soft_targets', lambda a, t: F.cross_entropy(a, t.softmax(1)), (3, 5), (3, 5))",
      "gradcheck('ce_3d', lambda a: F.cross_entropy(a, torch.tensor([[0, 1], [2, 3]])), (2, 4, 2))",
      "gradcheck('ce_1d', lambda a: F.cross_entropy(a, torch.tensor(2)), (5,))",
      "gradcheck('nll', lambda a: F.nll_loss(a.log_softmax(1), torch.tensor([0, 2])), (2, 3))",
      "gradcheck('mse', lambda a, b: F.mse_loss(a, b), (4, 3), (4, 3))",
    ]), "");
  });
  test("shape ops, indexing, concatenation and selection", async () => {
    assert.equal(await gradchecks([
      "gradcheck('slice', lambda a: a[1:, ::2], (4, 5))",
      "gradcheck('int', lambda a: a[2], (4, 5))",
      "gradcheck('list', lambda a: a[[0, 2, 2]], (4, 3))",
      "gradcheck('pairs', lambda a: a[torch.arange(3), torch.tensor([1, 0, 1])], (3, 2))",
      "gradcheck('range', lambda a: a[range(3), [2, 0, 1]], (3, 4))",
      "gradcheck('mask', lambda a: a[a > 0], (4, 3))",
      "gradcheck('index_2d', lambda a: a[torch.tensor([[0, 1, 1], [2, 0, 3]])], (4, 3))",
      "gradcheck('none_ellipsis', lambda a: a[None, ..., 1], (3, 4))",
      "gradcheck('reshape_view', lambda a: a.reshape(4, 3).view(-1, 2), (3, 4))",
      "gradcheck('transpose', lambda a: a.transpose(0, 2), (2, 3, 4))",
      "gradcheck('permute', lambda a: a.permute(2, 0, 1), (2, 3, 4))",
      "gradcheck('T', lambda a: a.T @ a, (3, 4))",
      "gradcheck('unsqueeze_squeeze', lambda a: a.unsqueeze(1).squeeze(), (3, 4))",
      "gradcheck('flatten', lambda a: a.flatten(1), (2, 3, 4))",
      "gradcheck('expand', lambda a: a.expand(3, 4) * 2, (1, 4))",
      "gradcheck('cat', lambda a, b: torch.cat([a, b, a], dim=1), (2, 3), (2, 2))",
      "gradcheck('stack', lambda a, b: torch.stack([a, b], dim=1), (2, 3), (2, 3))",
      "gradcheck('split', lambda a: a.split(2, dim=1)[1] * 3, (3, 5))",
      "gradcheck('chunk', lambda a: torch.chunk(a, 3)[2], (5, 2))",
      "gradcheck('unbind', lambda a: a.unbind(1)[1], (3, 2))",
      "gradcheck('iterate', lambda a: sum(row.sum() * i for i, row in enumerate(a)), (3, 2))",
      "gradcheck('gather', lambda a: a.gather(1, torch.tensor([[0, 0], [2, 1]])), (2, 3))",
      "gradcheck('index_select', lambda a: a.index_select(0, torch.tensor([2, 0, 2])), (3, 2))",
      "gradcheck('scatter', lambda a, s: a.scatter(1, torch.tensor([[2, 0], [1, 0]]), s), (2, 3), (2, 2))",
      "gradcheck('scatter_add', lambda a, s: a.scatter_add(1, torch.tensor([[0, 0, 2]]), s), (1, 3), (1, 3))",
      "gradcheck('masked_fill', lambda a: a.masked_fill(torch.tensor([True, False, True]), 0.0), (2, 3))",
      "gradcheck('where', lambda a, b: torch.where(a > 0, a, b), (3, 4), (4,))",
      "gradcheck('triu', lambda a: a.triu(1), (3, 3))",
      "gradcheck('flip', lambda a: a.flip(1) * torch.arange(4.0, dtype=torch.float64), (2, 4))",
      "gradcheck('repeat', lambda a: a.repeat(2, 1, 2), (2, 3))",
      "gradcheck('repeat_interleave', lambda a: a.repeat_interleave(2, dim=1) * torch.arange(12.0, dtype=torch.float64).reshape(2, 6), (2, 3))",
      "gradcheck('pad', lambda a: F.pad(a, (1, 2, 0, 1), value=3.0), (2, 3))",
      "gradcheck('clone', lambda a: a.clone() * a, (3,))",
    ]), "");
  });
  test("layers: embedding, linear, layer_norm, rms_norm, attention, BatchNorm1d", async () => {
    assert.equal(await gradchecks([
      "gradcheck('embedding', lambda w: F.embedding(torch.tensor([[1, 2], [0, 1]]), w), (3, 4))",
      "gradcheck('linear', lambda x, w, b: F.linear(x, w, b), (2, 3), (4, 3), (4,))",
      "gradcheck('one_hot_matmul', lambda w: F.one_hot(torch.tensor([0, 2]), 3).double() @ w, (3, 2))",
      "gradcheck('layer_norm', lambda x, w, b: F.layer_norm(x, (4,), w, b), (3, 4), (4,), (4,))",
      "gradcheck('rms_norm', lambda x, w: F.rms_norm(x, (4,), w), (3, 4), (4,))",
      "gradcheck('sdpa_causal', lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True), (2, 3, 4), (2, 3, 4), (2, 3, 4))",
      "gradcheck('sdpa_mask', lambda q, k, v: F.scaled_dot_product_attention(q, k, v, attn_mask=torch.tensor([[True, False], [True, True]])), (2, 4), (2, 4), (2, 3))",
      "gradcheck('batchnorm_train', lambda x: torch.nn.BatchNorm1d(3).to(torch.float64)(x), (5, 3))",
      "gradcheck('layernorm_module', lambda x: torch.nn.LayerNorm(4).to(torch.float64)(x), (2, 4))",
    ]), "");
  });
  test("in-place ops inside the graph: setitem and -= on a non-leaf, += into a constant buffer", async () => {
    assert.equal(await gradchecks([
      "def manual_backprop_style(a):",
      "    d = a * 2 + 1",
      "    d[range(2), [0, 2]] -= 1",
      "    d /= 2",
      "    return d",
      "gradcheck('setitem_nonleaf', manual_backprop_style, (2, 3))",
      "def accumulate(a, b):",
      "    buf = torch.zeros(2, 3, dtype=torch.float64)",
      "    buf += a",
      "    buf[0] = b",
      "    buf *= 2",
      "    return buf",
      "gradcheck('inplace_constant', accumulate, (2, 3), (3,))",
      "def scatter_inplace(a):",
      "    b = a * 1",
      "    b.scatter_(1, torch.tensor([[0], [2]]), 5.0)",
      "    return b",
      "gradcheck('scatter_', scatter_inplace, (2, 3))",
      "def accumulate_rows(a):",
      "    buf = torch.zeros(3, 2, dtype=torch.float64)",
      "    for k, ix in enumerate(torch.tensor([2, 0, 2])):",
      "        buf[ix] += a[k]",
      "    return buf",
      "gradcheck('row_accumulate', accumulate_rows, (3, 2))",
    ]), "");
  });
});

describe("mini-torch autograd rules", () => {
  test("micrograd's torch comparison gives Karpathy's numbers", async () => {
    assert.equal(await run([
      "x = torch.Tensor([-4.0]).double()",
      "x.requires_grad = True",
      "z = 2 * x + 2 + x",
      "q = z.relu() + z * x",
      "h = (z * z).relu()",
      "y = h + q + q * x",
      "y.backward()",
      "print(y.data.item(), x.grad.item())",
      "a = torch.Tensor([-4.0]).double()",
      "b = torch.Tensor([2.0]).double()",
      "a.requires_grad = True",
      "b.requires_grad = True",
      "c = a + b",
      "d = a * b + b**3",
      "c = c + c + 1",
      "c = c + 1 + c + (-a)",
      "d = d + d * 2 + (b + a).relu()",
      "d = d + 3 * d + (b - a).relu()",
      "e = c - d",
      "f = e**2",
      "g = f / 2.0",
      "g = g + 10.0 / f",
      "g.backward()",
      "print(round(g.data.item(), 6), round(a.grad.item(), 6), round(b.grad.item(), 6))",
    ].join("\n")), "-20.0 46.0\n24.704082 138.833819 645.577259");
  });
  test("grads accumulate, retain_grad keeps a non-leaf's grad, grad can be reset to None", async () => {
    assert.equal(await run([
      "a = torch.tensor([1.0, 2.0], requires_grad=True)",
      "(a * 3).sum().backward()",
      "(a * 3).sum().backward()",
      "print(a.grad.tolist(), a.is_leaf, (a * 3).is_leaf, (a * 3).grad_fn, (a * 3).grad_fn.name(), a.grad_fn)",
      "a.grad = None",
      "h = a * a",
      "h.retain_grad()",
      "h.sum().backward()",
      "print(a.grad.tolist(), h.grad.tolist(), h.retains_grad, (a * a).grad)",
      "a.grad.zero_()",
      "print(a.grad.tolist(), torch.tensor([1.0]).requires_grad, torch.ones(2, requires_grad=True).requires_grad)",
    ].join("\n")), [
      "[6.0, 6.0] True False <MulBackward0 object> MulBackward0 None",
      "[2.0, 4.0] [1.0, 1.0] True None",
      "[0.0, 0.0] False True",
    ].join("\n"));
  });
  test("no_grad, enable_grad, inference_mode and the decorator forms", async () => {
    assert.equal(await run([
      "a = torch.tensor([1.0], requires_grad=True)",
      "with torch.no_grad():",
      "    z = a * 2",
      "    with torch.enable_grad():",
      "        e = a * 2",
      "print(z.requires_grad, e.requires_grad, torch.is_grad_enabled())",
      "@torch.no_grad()",
      "def f(t):",
      "    return t * 2",
      "@torch.no_grad",
      "def g(t):",
      "    return t * 2",
      "class Model:",
      "    @torch.no_grad()",
      "    def generate(self, t):",
      "        return t * 3",
      "with torch.inference_mode():",
      "    i = a * 2",
      "print(f(a).requires_grad, g(a).requires_grad, Model().generate(a).requires_grad, i.requires_grad, torch.is_grad_enabled())",
      "with torch.set_grad_enabled(False):",
      "    s = a * 2",
      "print(s.requires_grad, torch.is_grad_enabled())",
    ].join("\n")), "False True True\nFalse False False False True\nFalse True");
  });
  test("parameter updates: in place under no_grad, through .data, and detach", async () => {
    assert.equal(await run([
      "w = torch.tensor([1.0, 2.0], requires_grad=True)",
      "(w * w).sum().backward()",
      "with torch.no_grad():",
      "    w -= 0.1 * w.grad",
      "print(w.tolist(), w.requires_grad, w.is_leaf)",
      "w.data += -0.5 * w.grad",
      "print([round(v, 4) for v in w.tolist()], w.is_leaf)",
      "d = w.detach()",
      "d[0] = 9.0",
      "print(d.requires_grad, w[0].item())",
      "params = [torch.ones(2, requires_grad=True), torch.zeros(1, requires_grad=True)]",
      "for p in params:",
      "    p.requires_grad = False",
      "print([p.requires_grad for p in params], torch.ones(1).requires_grad_().requires_grad)",
    ].join("\n")), "[0.800000011920929, 1.600000023841858] True True\n[-0.2, -0.4] True\nFalse 9.0\n[False, False] True");
  });
  test("torch's errors: in-place on a leaf, second backward, non-scalar backward, modified saved tensors, int grads", async () => {
    const r = await run([
      "def err(fn):",
      "    try:",
      "        fn()",
      "        return 'no error'",
      "    except Exception as e:",
      "        return f'{type(e).__name__}: {e}'",
      "a = torch.tensor([1.0, 2.0], requires_grad=True)",
      "print(err(lambda: a.add_(1)))",
      "print(err(lambda: a.__setitem__(0, 5.0)))",
      "y = (a * a).sum()",
      "y.backward(retain_graph=True)",
      "y.backward()",
      "print(a.grad.tolist(), err(y.backward))",
      "print(err(lambda: (a + 1).sum().backward()), (a + 1).sum().backward() is None)",
      "print(err(lambda: (a * 2).backward()))",
      "print(err(lambda: torch.tensor([1.0]).backward()))",
      "u = torch.tensor([3.0], requires_grad=True)",
      "s = u.exp()",
      "s += 1",
      "print(err(s.backward))",
      "print(err(lambda: torch.tensor([1, 2]).requires_grad_()))",
      "print(err(lambda: setattr(a * 2, 'requires_grad', False)))",
      "print(err(lambda: (a * 2).numpy()), a.detach().numpy().tolist())",
      "(a * 2).sum().backward(torch.tensor(1.0))",
      "print(err(lambda: (a * 2).backward(torch.ones(3))))",
    ].join("\n"));
    assert.equal(r, [
      "RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.",
      "RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.",
      "[4.0, 8.0] RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.",
      "no error True",
      "RuntimeError: grad can be implicitly created only for scalar outputs",
      "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn",
      "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: the input of ExpBackward0 is at version 1; expected version 0 instead. Use an out-of-place op (`x = x + 1`) or `.clone()` before changing it.",
      "RuntimeError: only Tensors of floating point dtype can require gradients",
      "RuntimeError: you can only change requires_grad flags of leaf variables. If you want to use a computed variable in a subgraph that doesn't require differentiation use var_no_grad = var.detach().",
      "RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead. [1.0, 2.0]",
      "RuntimeError: Mismatch in shape: grad_output[0] has a shape of torch.Size([3]) and output[0] has a shape of torch.Size([2]).",
    ].join("\n"));
  });
  test("ops save only what torch saves: in-place after add/mul-by-constant and repeated backward are fine", async () => {
    assert.equal(await run([
      "a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)",
      "results = []",
      "for f in (lambda h: h + 1, lambda h: h * 2, lambda h: h / 2, lambda h: h @ torch.ones(3)):",
      "    a.grad = None",
      "    h = a * 1.0",
      "    y = f(h)",
      "    h.add_(1)",
      "    y.sum().backward()",
      "    results.append(a.grad.tolist())",
      "print(results)",
      "b = torch.tensor([3.0], requires_grad=True)",
      "c = b + b",
      "c.backward()",
      "c.backward()",
      "print(b.grad.tolist())",
      "h = b * 2",
      "h.zero_()",
      "h2 = b * 2",
      "h2.fill_(5.0)",
      "print(h.requires_grad, h.grad_fn.name(), h2.grad_fn.name())",
      "(h + h2 + b).sum().backward()",
      "print(b.grad.tolist())",
      "z = torch.zeros(3, requires_grad=True)",
      "z.norm().backward()",
      "print(z.grad.tolist(), F.embedding(torch.tensor([0]), torch.ones(2, 2, requires_grad=True)).grad_fn.name(), F.layer_norm(a, (3,)).grad_fn.name(), a.reshape(3, 1).grad_fn.name())",
    ].join("\n")), [
      "[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]",
      "[4.0]",
      "True ZeroBackward0 FillBackward0",
      "[5.0]",
      "[0.0, 0.0, 0.0] EmbeddingBackward0 NativeLayerNormBackward0 ViewBackward0",
    ].join("\n"));
  });
  test("writing a value that requires grad: direct indexing works, chained indexing says how to fix it", async () => {
    const r = await runChallenge([
      "import torch",
      "w = torch.tensor([2.0], requires_grad=True)",
      "buf = torch.zeros(2, 3)",
      "buf[0, 1] = w[0]",
      "buf.sum().backward()",
      "print(buf.requires_grad, w.grad.tolist())",
      "torch.zeros(2, 3)[0][1] = w[0]",
    ].join("\n"), TESTS, "probe.py");
    assert.equal(r.stdout.trim(), "True [1.0]");
    assert.equal(r.scriptError?.type, "MiniTorchUnsupported");
    assert.match(r.scriptError!.diagnosis, /Index the original tensor directly \(`buf\[0, 1\] = w`, not `buf\[0\]\[1\] = w`\)/);
  });
  test("manual backprop through a makemore MLP matches autograd", async () => {
    // The 0.4 exercise: cross_entropy's gradient wrt logits is (softmax - one_hot) / n,
    // built with the in-place setitem Karpathy uses on a tensor that's still in the graph.
    assert.equal(await run([
      "g = torch.Generator().manual_seed(2147483647)",
      "n = 8",
      "C = torch.randn((27, 4), generator=g, requires_grad=True)",
      "W1 = torch.randn((12, 16), generator=g, requires_grad=True)",
      "W2 = torch.randn((16, 27), generator=g, requires_grad=True)",
      "Xb = torch.randint(0, 27, (n, 3), generator=g)",
      "Yb = torch.randint(0, 27, (n,), generator=g)",
      "emb = C[Xb]",
      "h = torch.tanh(emb.view(n, -1) @ W1)",
      "logits = h @ W2",
      "logit_maxes = logits.max(1, keepdim=True).values",
      "norm_logits = logits - logit_maxes",
      "counts = norm_logits.exp()",
      "counts_sum_inv = counts.sum(1, keepdims=True) ** -1",
      "probs = counts * counts_sum_inv",
      "logprobs = probs.log()",
      "loss = -logprobs[range(n), Yb].mean()",
      "for t in [logits, h, emb]:",
      "    t.retain_grad()",
      "loss.backward()",
      "print(abs(loss.item() - F.cross_entropy(logits, Yb).item()) < 1e-5)",
      "dlogits = F.softmax(logits, 1)",
      "dlogits[range(n), Yb] -= 1",
      "dlogits /= n",
      "dh = dlogits @ W2.T",
      "dhpre = (1.0 - h**2) * dh",
      "demb = (dhpre @ W1.T).view(emb.shape)",
      "dC = torch.zeros_like(C)",
      "for k in range(n):",
      "    for j in range(3):",
      "        dC[Xb[k, j]] += demb[k, j]",
      "for name, mine, t in [('logits', dlogits, logits), ('h', dh, h), ('emb', demb, emb), ('C', dC, C)]:",
      "    print(name, torch.allclose(mine, t.grad, atol=1e-6), (mine - t.grad).abs().max().item() < 1e-5)",
    ].join("\n")), "True\nlogits True True\nh True True\nemb True True\nC True True");
  });
});
