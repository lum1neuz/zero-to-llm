// mini-torch API groups from TORCH_SUBSET.md, checked in real Pyodide: forward values, dtypes and errors.
// Gradients live in shim-autograd.test.ts, torch.nn and torch.optim in shim-nn.test.ts.
import assert from "node:assert/strict";
import { describe, test } from "node:test";
import { runChallenge } from "../pyrun.ts";

const TESTS = `from harness import test\n@test("t", "t")\ndef _(L):\n    pass\n`;
const PRELUDE = [
  "import torch",
  "import torch.nn.functional as F",
  "def err(fn):",
  "    try:",
  "        fn()",
  "        return 'no error'",
  "    except Exception as e:",
  "        return f'{type(e).__name__}: {e}'",
].join("\n");

async function out(...lines: string[]) {
  const r = await runChallenge(`${PRELUDE}\n${lines.join("\n")}`, TESTS, "probe.py");
  if (r.scriptError) throw new Error(r.scriptError.traceback);
  return r.stdout.trim();
}

describe("mini-torch creation and dtypes", () => {
  test("factories and their default dtypes", async () => {
    assert.equal(await out(
      "print(torch.zeros(2, 3).shape, torch.ones((2,)).dtype, torch.zeros(2, dtype=torch.int32).dtype, torch.empty(1).shape)",
      "print(torch.full((2,), 7).dtype, torch.full((2,), 7.0).dtype, torch.full((1,), True).dtype)",
      "print(torch.arange(4).tolist(), torch.arange(4).dtype, torch.arange(0, 1, 0.25).tolist(), torch.arange(1, 7, 2).tolist())",
      "print(torch.linspace(0, 1, 5).tolist(), torch.eye(2).tolist())",
      "x = torch.tensor([[1, 2], [3, 4]])",
      "print(torch.zeros_like(x).dtype, torch.ones_like(x, dtype=torch.float32).dtype, torch.full_like(x, 2).tolist())",
    ), [
      "torch.Size([2, 3]) torch.float32 torch.int32 torch.Size([1])",
      "torch.int64 torch.float32 torch.bool",
      "[0, 1, 2, 3] torch.int64 [0.0, 0.25, 0.5, 0.75] [1, 3, 5]",
      "[0.0, 0.25, 0.5, 0.75, 1.0] [[1.0, 0.0], [0.0, 1.0]]",
      "torch.int64 torch.float32 [[2, 2], [2, 2]]",
    ].join("\n"));
  });
  test("tensor() from nested lists, 0-d values and NumPy arrays", async () => {
    assert.equal(await out(
      "import numpy as np",
      "print(torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]]).shape, torch.tensor(5).dtype, torch.tensor(2.5).dtype, torch.tensor([True, False]).dtype)",
      "print(torch.tensor(np.array([1.0, 2.0])).dtype, torch.tensor([torch.tensor(1.0), torch.tensor(2.0)]).tolist(), torch.tensor([]).shape)",
      "print(torch.Tensor([1, 2]).dtype, torch.Tensor(2, 3).shape, isinstance(torch.ones(1), torch.Tensor))",
      "print(err(lambda: torch.tensor([[1, 2], [3]])))",
    ), [
      "torch.Size([2, 2, 1]) torch.int64 torch.float32 torch.bool",
      "torch.float64 [1.0, 2.0] torch.Size([0])",
      "torch.float32 torch.Size([2, 3]) True",
      "ValueError: expected sequence of equal length at each level: the nested lists are ragged",
    ].join("\n"));
  });
  test("dtypes, aliases, casts, finfo and iinfo", async () => {
    assert.equal(await out(
      "print(torch.float is torch.float32, torch.long, torch.int, torch.half, torch.double, torch.bool, torch.uint8)",
      "x = torch.tensor([1.7, -2.5, 0.0])",
      "print(x.long().tolist(), x.int().dtype, x.half().dtype, x.double().dtype, x.bool().tolist(), x.to(torch.int16).tolist())",
      "print(x.float() is x, x.element_size(), x.half().element_size(), x.nbytes, x.is_floating_point(), torch.float16.itemsize)",
      "print(torch.finfo(torch.float16).max, torch.finfo().eps == torch.finfo(torch.float32).eps, torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max)",
    ), [
      "True torch.int64 torch.int32 torch.float16 torch.float64 torch.bool torch.uint8",
      "[1, -2, 0] torch.int32 torch.float16 torch.float64 [True, True, False] [1, -2, 0]",
      "True 4 2 12 True 2",
      "65504.0 True -128 127",
    ].join("\n"));
  });
  test("type promotion follows torch", async () => {
    assert.equal(await out(
      "i8, i64 = torch.tensor([1, 2], dtype=torch.int8), torch.tensor([1, 2])",
      "h = torch.tensor([1.0], dtype=torch.float16)",
      "print((i8 + 1).dtype, (i8 + 1.5).dtype, (i8 + i64).dtype, (i8 + torch.tensor(2)).dtype, (i8 + torch.tensor(2.5, dtype=torch.float64)).dtype)",
      "print((h * 2.0).dtype, (h * torch.tensor([2.0])).dtype, (h + i64).dtype, (torch.tensor([True]) + 1).dtype, (i64 / 2).dtype)",
      "print((torch.tensor([7, -7]) // 2).tolist(), (torch.tensor([7, -7]) % 3).tolist(), (torch.tensor([2, 3]) ** 2).tolist(), (2 ** torch.tensor([1.0, 3.0])).tolist())",
      "print(err(lambda: torch.tensor([2]) ** -1))",
    ), [
      "torch.int8 torch.float32 torch.int64 torch.int8 torch.float64",
      "torch.float16 torch.float32 torch.float16 torch.int64 torch.float32",
      "[3, -4] [1, 2] [4, 9] [2.0, 8.0]",
      "RuntimeError: Integers to negative integer powers are not allowed.",
    ].join("\n"));
  });
  test("printing follows torch: integer floats, scientific notation, 3-D layout, grad info, summarising", async () => {
    assert.equal(await out(
      "print(torch.tensor([1.0, 2.0]), torch.tensor(3.14159), torch.tensor([1e-5, 1.0]), torch.tensor([1, 2], dtype=torch.int8))",
      "print(torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]))",
      "w = torch.tensor([1.0, 2.0], requires_grad=True)",
      "print(w, w * 2, torch.tensor([1.0], dtype=torch.float16), torch.tensor([True, False]))",
      "print(torch.arange(2000.0))",
      "print(torch.tensor([1.0, 2000.0]), torch.tensor([1.0, 999.0]))",
    ), [
      "tensor([1., 2.]) tensor(3.1416) tensor([1.0000e-05, 1.0000e+00]) tensor([1, 2], dtype=torch.int8)",
      "tensor([[[1., 2.],",
      "         [3., 4.]],",
      "",
      "        [[5., 6.],",
      "         [7., 8.]]])",
      "tensor([1., 2.], requires_grad=True) tensor([2., 4.], grad_fn=<MulBackward0>) tensor([1.], dtype=torch.float16) tensor([True, False])",
      "tensor([0.0000e+00, 1.0000e+00, 2.0000e+00, ..., 1.9970e+03, 1.9980e+03, 1.9990e+03])",
      "tensor([1.0000e+00, 2.0000e+03]) tensor([1., 999.])",
    ].join("\n"));
  });
});

describe("mini-torch shape ops and indexing", () => {
  test("reshape, view, transpose, permute, T, unsqueeze, squeeze, flatten, expand, size(dim)", async () => {
    assert.equal(await out(
      "x = torch.arange(24).reshape(2, 3, 4)",
      "print(x.view(6, -1).shape, x.transpose(0, 2).shape, x.permute(2, 0, 1).shape, x.flatten(1).shape, x.flatten().shape)",
      "print(x[0].T.shape, x.mT.shape, x.unsqueeze(1).shape, x.unsqueeze(-1).shape, x[None].shape, torch.zeros(1, 3, 1).squeeze().shape, torch.zeros(1, 3, 1).squeeze(0).shape)",
      "print(x.size(-1), x.size(), x.numel(), x.ndim, x.dim(), torch.ones(3, 1).expand(3, 4).shape, torch.ones(4).expand(2, -1).shape)",
      "print(err(lambda: x.view(5, -1)), '|', err(lambda: x.permute(1, 0)), '|', err(lambda: torch.ones(3).expand(2, 4)))",
    ), [
      "torch.Size([6, 4]) torch.Size([4, 3, 2]) torch.Size([4, 2, 3]) torch.Size([2, 12]) torch.Size([24])",
      "torch.Size([4, 3]) torch.Size([2, 4, 3]) torch.Size([2, 1, 3, 4]) torch.Size([2, 3, 4, 1]) torch.Size([1, 2, 3, 4]) torch.Size([3]) torch.Size([3, 1])",
      "4 torch.Size([2, 3, 4]) 24 3 3 torch.Size([3, 4]) torch.Size([2, 4])",
      "RuntimeError: shape '[5, -1]' is invalid for input of size 24 | RuntimeError: permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = 3 is not equal to len(dims) = 2 | RuntimeError: The expanded size of the tensor (4) must match the existing size (3) at non-singleton dimension 1.  Target sizes: [2, 4].  Tensor sizes: [3]",
    ].join("\n"));
  });
  test("cat, stack, split, chunk, unbind, gather, index_select, scatter, flip, repeat", async () => {
    assert.equal(await out(
      "x = torch.arange(12).reshape(3, 4)",
      "print(torch.cat([x, x]).shape, torch.cat((x, x), dim=1).shape, torch.stack([x, x], dim=1).shape)",
      "print([t.shape[1] for t in x.split(3, dim=1)], [t.shape[1] for t in x.split([1, 3], dim=1)], [t.shape[0] for t in x.chunk(2)], len(x.unbind(1)))",
      "g = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])",
      "print(g.gather(1, torch.tensor([[2], [0]])).tolist(), g.index_select(1, torch.tensor([0, 2])).tolist(), torch.zeros(2, 3).scatter(1, torch.tensor([[1], [2]]), 1.0).tolist())",
      "print(g.flip(1).tolist(), g.repeat(1, 2).shape, g.repeat_interleave(2, dim=0).shape, torch.outer(torch.tensor([1, 2]), torch.tensor([1, 10])).tolist())",
      "print(err(lambda: torch.cat([torch.zeros(2, 3), torch.zeros(2, 4)])))",
      "print(err(lambda: torch.stack([torch.zeros(2), torch.zeros(3)])))",
    ), [
      "torch.Size([6, 4]) torch.Size([3, 8]) torch.Size([3, 2, 4])",
      "[3, 1] [1, 3] [2, 1] 4",
      "[[3.0], [4.0]] [[1.0, 3.0], [4.0, 6.0]] [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]",
      "[[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]] torch.Size([2, 6]) torch.Size([4, 3]) [[1, 10], [2, 20]]",
      "RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 3 but got size 4 for tensor number 1 in the list.",
      "RuntimeError: stack expects each tensor to be equal size, but got [2] at entry 0 and [3] at entry 1",
    ].join("\n"));
  });
  test("indexing with ints, slices, None, lists, ranges, int tensors and boolean masks", async () => {
    assert.equal(await out(
      "x = torch.arange(12).reshape(3, 4)",
      "print(x[1].tolist(), x[:, 1].tolist(), x[-1, -1].item(), x[1:, ::2].tolist(), x[..., 0].tolist(), x[None, 0].shape)",
      "print(x[[0, 2]].tolist(), x[range(3), [3, 2, 1]].tolist(), x[torch.tensor([0, 2]), torch.tensor([1, 3])].tolist(), x[x > 8].tolist())",
      "C = torch.arange(10.0).reshape(5, 2)",
      "print(C[torch.tensor([[0, 4], [1, 1]])].shape, C[torch.tensor(3)].tolist())",
      "print(err(lambda: x[3]), '|', err(lambda: x[0, 0, 0]), '|', err(lambda: x[torch.tensor([0.0])]))",
    ), [
      "[4, 5, 6, 7] [1, 5, 9] 11 [[4, 6], [8, 10]] [0, 4, 8] torch.Size([1, 4])",
      "[[0, 1, 2, 3], [8, 9, 10, 11]] [3, 6, 9] [1, 11] [9, 10, 11]",
      "torch.Size([2, 2, 2]) [6.0, 7.0]",
      "IndexError: index 3 is out of bounds for dimension 0 with size 3 | IndexError: too many indices for tensor of dimension 2 | IndexError: tensors used as indices must be long, int, byte or bool tensors",
    ].join("\n"));
  });
  test("item assignment: bigram counts, masks, broadcasting values, views that share storage", async () => {
    assert.equal(await out(
      "N = torch.zeros((3, 3), dtype=torch.int32)",
      "for a, b in [(0, 1), (0, 1), (2, 2)]:",
      "    N[a, b] += 1",
      "print(N.tolist(), N.dtype)",
      "y = torch.zeros(5)",
      "y[torch.tensor([1, 3])] = torch.tensor([7.0, 9.0])",
      "y[y == 0] = -1",
      "y[0] = 2.5",
      "print(y.tolist())",
      "m = torch.zeros(2, 3)",
      "m[:, 1] = torch.tensor([1.0, 2.0])",
      "m[1] = 5",
      "print(m.tolist())",
      "x = torch.arange(6).reshape(2, 3)",
      "x.view(6)[0] = 100",
      "row = x[1]",
      "row += 10",
      "print(x.tolist())",
      "print(err(lambda: m.__setitem__(0, torch.ones(2))))",
    ), [
      "[[0, 2, 0], [0, 0, 0], [0, 0, 1]] torch.int32",
      "[2.5, 7.0, -1.0, 9.0, -1.0]",
      "[[0.0, 1.0, 0.0], [5.0, 5.0, 5.0]]",
      "[[100, 1, 2], [13, 14, 15]]",
      "RuntimeError: shape mismatch: value tensor of shape [2] cannot be broadcast to indexing result of shape [3]",
    ].join("\n"));
  });
  test("integer indexing returns views too: 0-d elements and 0-d int tensor indices write through", async () => {
    assert.equal(await out(
      "x = torch.zeros(2, 3)",
      "v = x[0, 1]",
      "v += 5",
      "x[1][2].fill_(7)",
      "x[1, 0].add_(3)",
      "r = x[torch.tensor(1)]",
      "r += 1",
      "print(x.tolist(), x[torch.tensor(0)].shape, x[0, 1].shape)",
    ), "[[0.0, 5.0, 0.0], [4.0, 1.0, 8.0]] torch.Size([3]) torch.Size([])");
  });
  test("NumPy int arrays become int64 (Pyodide's NumPy defaults to int32); integer division by zero raises", async () => {
    assert.equal(await out(
      "import numpy as np",
      "a = np.array([1, 2])",
      "print(torch.tensor(a).dtype, torch.from_numpy(a).dtype, F.one_hot(torch.tensor(a), 3).shape, torch.from_numpy(np.array([1.0])).dtype)",
      "print(err(lambda: torch.tensor([4, 2]) // torch.tensor([2, 0])), '|', err(lambda: torch.tensor([4]) % 0), (torch.tensor([1.0]) // 0).tolist())",
    ), "torch.int64 torch.int64 torch.Size([2, 3]) torch.float64\nZeroDivisionError: ZeroDivisionError | ZeroDivisionError: ZeroDivisionError [inf]");
  });
  test("in-place ops keep their dtype and refuse a result they can't hold", async () => {
    assert.equal(await out(
      "x = torch.zeros(3)",
      "x += 1",
      "x *= torch.tensor([1.0, 2.0, 3.0])",
      "x.sub_(1).div_(2)",
      "x.clamp_(max=0.75)",
      "print(x.tolist(), x.zero_().tolist(), x.fill_(2).tolist(), x.masked_fill_(torch.tensor([True, False, False]), 0.0).tolist())",
      "i = torch.tensor([1, 2])",
      "print(err(lambda: i.add_(0.5)), '|', err(lambda: x.add_(torch.ones(2, 3))))",
      "j = torch.tensor([1, 2])",
      "j[0] = 2.7",
      "print(j.tolist(), torch.zeros(2).copy_(torch.tensor([3, 4])).tolist())",
    ), [
      "[0.0, 0.5, 0.75] [0.0, 0.0, 0.0] [2.0, 2.0, 2.0] [0.0, 2.0, 2.0]",
      "RuntimeError: result type Float can't be cast to the desired output type Long | RuntimeError: output with shape [3] doesn't match the broadcast shape [2, 3]",
      "[2, 2] [3.0, 4.0]",
    ].join("\n"));
  });
});

describe("mini-torch math", () => {
  test("reductions with dim, keepdim and tuples of dims", async () => {
    assert.equal(await out(
      "m = torch.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])",
      "print(m.sum().item(), m.sum(0).tolist(), m.sum(dim=1, keepdim=True).shape, m.sum(1, keepdims=True).shape, m.sum((0, 1)).item(), torch.sum(m, 1).tolist())",
      "print(m.mean(1).tolist(), m.mean((0, 1), keepdim=True).shape, torch.mean(m).item(), m.var().item(), m.var(0).tolist(), m.std(1, unbiased=False).tolist()[0] == m.std(1, correction=0).tolist()[0])",
      "print(m.max().item(), m.max(1).values.tolist(), m.max(1).indices.tolist(), m.min(dim=0).indices.tolist(), torch.max(m, 1)[0].tolist(), torch.max(m, torch.tensor(3.0)).tolist())",
      "print(m.argmax().item(), m.argmax(1).tolist(), m.argmin(dim=1, keepdim=True).shape, m.amax(1).tolist(), m.amin().item(), torch.amax(m, dim=(0, 1)).item())",
      "print(torch.tensor([True, True, False]).sum().item(), torch.tensor([True, True, False]).sum().dtype, (m > 2).float().mean().item(), (m > 5).any().item(), (m > 0).all().item())",
      "print(m.cumsum(1).tolist(), torch.tensor([1, 2, 3]).cumsum(0).dtype, torch.tensor([3.0, 4.0]).norm().item())",
      "print(err(lambda: torch.tensor([1, 2]).mean()), '|', err(lambda: m.max(dim=2)), '|', err(lambda: torch.tensor([]).max()))",
    ), [
      "21.0 [5.0, 7.0, 9.0] torch.Size([2, 1]) torch.Size([2, 1]) 21.0 [9.0, 12.0]",
      "[3.0, 4.0] torch.Size([1, 1]) 3.5 3.5 [4.5, 4.5, 4.5] True",
      "6.0 [5.0, 6.0] [1, 2] [0, 1, 0] [5.0, 6.0] [[3.0, 5.0, 3.0], [4.0, 3.0, 6.0]]",
      "5 [1, 2] torch.Size([2, 1]) [5.0, 6.0] 1.0 6.0",
      "2 torch.int64 0.6666666865348816 True True",
      "[[1.0, 6.0, 9.0], [4.0, 6.0, 12.0]] torch.int64 5.0",
      "RuntimeError: mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long | IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2) | RuntimeError: max(): Expected reduction over a non-empty tensor",
    ].join("\n"));
  });
  test("matmul and @, with torch's errors for mismatched shapes and dtypes", async () => {
    assert.equal(await out(
      "a, b = torch.ones(2, 3), torch.arange(12.0).reshape(3, 4)",
      "print((a @ b).tolist(), torch.matmul(torch.ones(5, 2, 3), b).shape, (torch.ones(3) @ b).shape, (b.T @ torch.ones(3)).shape, (torch.ones(3) @ torch.ones(3)).item())",
      "print(torch.mm(a, b).shape, torch.bmm(torch.ones(2, 2, 3), torch.ones(2, 3, 1)).shape, (torch.tensor([[1, 2]]) @ torch.tensor([[3], [4]])).tolist())",
      "print(err(lambda: a @ a))",
      "print(err(lambda: torch.ones(2, 3, 4) @ torch.ones(5, 2)))",
      "print(err(lambda: torch.tensor([1, 2]) @ torch.tensor([1.0, 2.0])))",
    ), [
      "[[12.0, 15.0, 18.0, 21.0], [12.0, 15.0, 18.0, 21.0]] torch.Size([5, 2, 4]) torch.Size([4]) torch.Size([4]) 3.0",
      "torch.Size([2, 4]) torch.Size([2, 2, 1]) [[11]]",
      "RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)",
      "RuntimeError: mat1 and mat2 shapes cannot be multiplied (6x4 and 5x2)",
      "RuntimeError: expected m1 and m2 to have the same dtype, but got: long int != float",
    ].join("\n"));
  });
  test("elementwise functions", async () => {
    assert.equal(await out(
      "x = torch.tensor([-1.0, 0.0, 2.0])",
      "r = lambda t: [round(v, 4) for v in t.tolist()]",
      "print(r(x.exp()), r(torch.log(torch.tensor([1.0, 4.0]))), r(torch.sqrt(torch.tensor([4.0]))), r(x.tanh()), r(torch.sigmoid(x)), r(F.relu(x)), r(x.abs()))",
      "print(r(F.gelu(x)), r(F.gelu(x, approximate='tanh')), r(F.silu(x)), r(torch.rsqrt(torch.tensor([4.0]))), r(x.sin()), r(x.cos()), r(x.square()), r(x.sign()))",
      "print(torch.round(torch.tensor([0.5, 1.5, -2.5])).tolist(), torch.round(torch.tensor([1.2345]), decimals=2).tolist(), x.clamp(min=0).tolist(), x.clip(-0.5, 0.5).tolist(), torch.maximum(x, torch.zeros(3)).tolist())",
      "print(torch.exp(torch.tensor([1, 2])).dtype, x.floor().tolist(), torch.tensor([1.5]).ceil().tolist())",
    ), [
      "[0.3679, 1.0, 7.3891] [0.0, 1.3863] [2.0] [-0.7616, 0.0, 0.964] [0.2689, 0.5, 0.8808] [0.0, 0.0, 2.0] [1.0, 0.0, 2.0]",
      "[-0.1587, 0.0, 1.9545] [-0.1588, 0.0, 1.9546] [-0.2689, 0.0, 1.7616] [0.5] [-0.8415, 0.0, 0.9093] [0.5403, 1.0, -0.4161] [1.0, 0.0, 4.0] [-1.0, 0.0, 1.0]",
      "[0.0, 2.0, -2.0] [1.2300000190734863] [0.0, 0.0, 2.0] [-0.5, 0.0, 0.5] [0.0, 0.0, 2.0]",
      "torch.float32 [-1.0, 0.0, 2.0] [2.0]",
    ].join("\n"));
  });
  test("softmax and log_softmax, including fully masked rows", async () => {
    assert.equal(await out(
      "m = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])",
      "r = lambda t: [[round(v, 4) for v in row] for row in t.tolist()]",
      "print(r(F.softmax(m, dim=1)), r(m.softmax(-1)), r(torch.softmax(m, 0)), r(F.log_softmax(m, dim=1).exp()), r(F.softmax(m)))",
      "w = torch.zeros(3, 3).masked_fill(torch.tril(torch.ones(3, 3)) == 0, float('-inf'))",
      "print(r(w.softmax(-1)), torch.tensor([float('-inf'), float('-inf')]).softmax(0).tolist())",
      "print(err(lambda: m.softmax()), '|', err(lambda: torch.tensor([1, 2]).softmax(0)))",
    ), [
      "[[0.09, 0.2447, 0.6652], [0.3333, 0.3333, 0.3333]] [[0.09, 0.2447, 0.6652], [0.3333, 0.3333, 0.3333]] [[0.5, 0.7311, 0.8808], [0.5, 0.2689, 0.1192]] [[0.09, 0.2447, 0.6652], [0.3333, 0.3333, 0.3333]] [[0.09, 0.2447, 0.6652], [0.3333, 0.3333, 0.3333]]",
      "[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.3333, 0.3333, 0.3333]] [nan, nan]",
      "TypeError: Tensor.softmax() missing 1 required positional argument: 'dim' | RuntimeError: \"softmax_lastdim_kernel_impl\" not implemented for 'Long'",
    ].join("\n"));
  });
  test("sort, argsort, topk, tril, triu, masked_fill, where, comparisons", async () => {
    assert.equal(await out(
      "m = torch.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])",
      "s = m.sort(dim=1, descending=True)",
      "print(s.values.tolist(), s.indices.tolist(), type(s).__name__, m.argsort(1).tolist(), torch.sort(torch.tensor([2, 1, 2, 1])).indices.tolist())",
      "v, i = torch.topk(m, 2)",
      "print(v.tolist(), i.tolist(), m.topk(1, dim=0, largest=False).values.tolist(), err(lambda: m.topk(4)))",
      "print(torch.tril(torch.ones(3, 3)).tolist(), torch.ones(2, 3).triu(1).tolist(), torch.ones(3, 3).tril(-1).sum().item())",
      "print(m.masked_fill(m > 4, float('-inf')).tolist(), torch.where(m > 3, m, 0.0).tolist(), torch.where(m > 3, 1, 0).dtype, torch.where(m > 4)[1].tolist())",
      "print(err(lambda: m.masked_fill(torch.tensor([1, 0, 1]), 0.0)), '|', err(lambda: torch.where(m, m, m)))",
      "print((m == 5).tolist(), (m != 5).sum().item(), (m >= 4).nonzero().tolist(), torch.tensor([3]) in torch.tensor([1, 3]), m == None)",
      "print(((m > 1) & (m < 5)).tolist(), ((m < 2) | (m > 5)).sum().item(), (~(m > 3)).tolist(), torch.logical_not(m > 3).tolist() == (~(m > 3)).tolist())",
      "print(torch.isclose(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.1])).tolist(), torch.allclose(m, m + 1e-7), torch.equal(m, m.clone()), err(lambda: torch.allclose(torch.tensor([1]), torch.tensor([1.0]))))",
    ), [
      "[[5.0, 3.0, 1.0], [6.0, 4.0, 2.0]] [[1, 2, 0], [2, 0, 1]] torch.return_types.sort [[0, 2, 1], [1, 0, 2]] [1, 3, 0, 2]",
      "[[5.0, 3.0], [6.0, 4.0]] [[1, 2], [2, 0]] [[1.0, 2.0, 3.0]] RuntimeError: selected index k out of range",
      "[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]] [[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]] 3.0",
      "[[1.0, -inf, 3.0], [4.0, 2.0, -inf]] [[0.0, 5.0, 0.0], [4.0, 0.0, 6.0]] torch.int64 [1, 2]",
      "RuntimeError: masked_fill_ only supports boolean masks, but got mask with dtype long int | RuntimeError: where expected condition to be a boolean tensor, but got a tensor with dtype Float",
      "[[False, True, False], [False, False, False]] 5 [[0, 1], [1, 0], [1, 2]] True False",
      "[[False, False, True], [True, True, False]] 2 [[True, False, True], [False, True, False]] True",
      "[True, False] True True RuntimeError: Long did not match Float",
    ].join("\n"));
  });
});

describe("mini-torch random", () => {
  test("Generator().manual_seed and torch.manual_seed make draws reproducible", async () => {
    assert.equal(await out(
      "g1, g2 = torch.Generator().manual_seed(2147483647), torch.Generator().manual_seed(2147483647)",
      "print(torch.randn(3, generator=g1).tolist() == torch.randn(3, generator=g2).tolist(), torch.rand(2, 2, generator=g1).tolist() == torch.rand(2, 2, generator=g2).tolist())",
      "torch.manual_seed(42)",
      "a = torch.randn(2, 3)",
      "torch.manual_seed(42)",
      "print(torch.equal(a, torch.randn(2, 3)), torch.initial_seed(), a.dtype, torch.randn(2, dtype=torch.float64).dtype)",
      "r = torch.rand(1000, generator=g1)",
      "i = torch.randint(0, 10, (1000,), generator=g1)",
      "print(0 <= r.min().item(), r.max().item() < 1, i.dtype, i.min().item(), i.max().item(), torch.randint(5, (2, 2)).shape, sorted(torch.randperm(5, generator=g1).tolist()))",
      "print(abs(torch.randn(4000, generator=g1).mean().item()) < 0.05, abs(torch.randn(4000, generator=g1).std().item() - 1) < 0.05)",
      "print(err(lambda: torch.randint(5, 5, (2,))), '|', err(lambda: torch.randn(2, dtype=torch.int64)))",
    ), [
      "True True",
      "True 42 torch.float32 torch.float64",
      "True True torch.int64 0 9 torch.Size([2, 2]) [0, 1, 2, 3, 4]",
      "True True",
      "RuntimeError: random_ expects 'from' to be less than 'to', but got from=5 >= to=5 | RuntimeError: \"normal_kernel_cpu\" not implemented for 'Long'",
    ].join("\n"));
  });
  test("multinomial: with and without replacement, rows, zero-probability categories", async () => {
    assert.equal(await out(
      "p = torch.tensor([0.1, 0.0, 0.6, 0.3])",
      "s = torch.multinomial(p, num_samples=4000, replacement=True, generator=torch.Generator().manual_seed(1))",
      "print(s.shape, s.dtype, (s == 1).sum().item(), abs((s == 2).float().mean().item() - 0.6) < 0.03)",
      "g = torch.Generator().manual_seed(3)",
      "w = torch.multinomial(p, 3, generator=g)",
      "print(sorted(w.tolist()), torch.multinomial(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), 1).tolist(), torch.multinomial(torch.tensor([5.0, 0.0]), 2, replacement=True).tolist())",
      "print(p.multinomial(1, generator=g).shape, err(lambda: torch.multinomial(p, 4)))",
      "print(err(lambda: torch.multinomial(p, 5)), '|', err(lambda: torch.multinomial(torch.tensor([-1.0, 2.0]), 1)))",
    ), [
      "torch.Size([4000]) torch.int64 0 True",
      "[0, 2, 3] [[1], [0]] [0, 0]",
      "torch.Size([1]) RuntimeError: invalid multinomial distribution (with replacement=False, not enough non-negative category to sample)",
      "RuntimeError: cannot sample n_sample > prob_dist.size(-1) samples without replacement | RuntimeError: probability tensor contains either `inf`, `nan` or element < 0",
    ].join("\n"));
  });
});

describe("mini-torch functional", () => {
  test("cross_entropy matches the manual log-softmax formula, with reductions, ignore_index, smoothing and soft targets", async () => {
    assert.equal(await out(
      "logits = torch.tensor([[2.0, 0.5, -1.0], [0.0, 1.0, 3.0]])",
      "y = torch.tensor([0, 2])",
      "manual = -logits.log_softmax(1)[torch.arange(2), y].mean()",
      "r = lambda t: round(t.item(), 5)",
      "print(r(F.cross_entropy(logits, y)) == r(manual), r(F.cross_entropy(logits, y, reduction='sum')) == r(manual * 2), F.cross_entropy(logits, y, reduction='none').shape)",
      "print(r(F.cross_entropy(logits, torch.tensor([0, -100]))) == r(-logits.log_softmax(1)[0, 0]), r(F.nll_loss(logits.log_softmax(1), y)) == r(manual))",
      "print(r(F.cross_entropy(logits[0], torch.tensor(0))) == r(-logits[0].log_softmax(0)[0]), F.cross_entropy(torch.zeros(2, 4, 3), torch.zeros(2, 3, dtype=torch.long)).item() == torch.tensor(4.0).log().item())",
      "smooth = 0.9 * -logits.log_softmax(1)[torch.arange(2), y] + 0.1 * -logits.log_softmax(1).mean(1)",
      "print(r(F.cross_entropy(logits, y, label_smoothing=0.1)) == r(smooth.mean()), r(F.cross_entropy(logits, F.one_hot(y, 3).float())) == r(manual))",
      "x3, t3 = torch.randn(2, 3, 4), torch.rand(2, 3, 4)",
      "print(r(F.cross_entropy(x3, t3)) == r(-(x3.log_softmax(1) * t3).sum() / 8))",
      "print(err(lambda: F.cross_entropy(logits, torch.tensor([0, 3]))), '|', err(lambda: F.cross_entropy(logits, torch.tensor([0, 1, 2]))))",
      "print(err(lambda: F.cross_entropy(logits, y.int())))",
    ), [
      "True True torch.Size([2])",
      "True True",
      "True True",
      "True True",
      "True",
      "IndexError: Target 3 is out of bounds. | ValueError: Expected input batch_size (2) to match target batch_size (3).",
      "RuntimeError: expected scalar type Long but found Int",
    ].join("\n"));
  });
  test("one_hot, embedding, linear, layer_norm, rms_norm, mse_loss, dropout, pad", async () => {
    assert.equal(await out(
      "print(F.one_hot(torch.tensor([0, 2, 1]), 3).tolist(), F.one_hot(torch.tensor([3])).shape, F.one_hot(torch.tensor([1])).dtype, err(lambda: F.one_hot(torch.tensor([1.0]), 3)))",
      "W = torch.arange(6.0).reshape(3, 2)",
      "print(F.embedding(torch.tensor([[2, 0]]), W).tolist(), err(lambda: F.embedding(torch.tensor([3]), W)), F.linear(torch.ones(1, 2), W, torch.ones(3)).tolist())",
      "x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])",
      "ln = F.layer_norm(x, (4,))",
      "print(round(ln.mean().item(), 6), round(ln.var(unbiased=False).item(), 3), err(lambda: F.layer_norm(x, (3,))))",
      "print([round(v, 4) for v in F.rms_norm(x, (4,)).tolist()[0]], F.mse_loss(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0])).item())",
      "torch.manual_seed(0)",
      "d = F.dropout(torch.ones(1000), p=0.25)",
      "print(F.dropout(x, 0.5, training=False) is x, set(d.tolist()) == {0.0, torch.tensor(4 / 3).item()}, abs((d == 0).float().mean().item() - 0.25) < 0.05)",
      "print(F.pad(torch.ones(2, 2), (1, 0, 0, 1)).tolist(), F.pad(torch.ones(1, 1), (1, 1), value=-1.0).tolist())",
    ), [
      "[[1, 0, 0], [0, 0, 1], [0, 1, 0]] torch.Size([1, 4]) torch.int64 RuntimeError: one_hot is only applicable to index tensor of type LongTensor.",
      "[[[4.0, 5.0], [0.0, 1.0]]] IndexError: index out of range in self [[2.0, 6.0, 10.0]]",
      "0.0 1.0 RuntimeError: Given normalized_shape=[3], expected input with shape [*, 3], but got input of size[1, 4]",
      "[0.3651, 0.7303, 1.0954, 1.4606] 2.5",
      "True True True",
      "[[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]] [[-1.0, 1.0, -1.0]]",
    ].join("\n"));
  });
  test("scaled_dot_product_attention equals the explicit masked-softmax attention", async () => {
    assert.equal(await out(
      "torch.manual_seed(0)",
      "q, k, v = torch.randn(2, 4, 8), torch.randn(2, 4, 8), torch.randn(2, 4, 8)",
      "att = (q @ k.transpose(-2, -1)) * 8 ** -0.5",
      "att = att.masked_fill(torch.tril(torch.ones(4, 4)) == 0, float('-inf')).softmax(-1)",
      "print(torch.allclose(F.scaled_dot_product_attention(q, k, v, is_causal=True), att @ v, atol=1e-6))",
      "mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))",
      "print(torch.allclose(F.scaled_dot_product_attention(q, k, v, attn_mask=mask), att @ v, atol=1e-6))",
    ), "True\nTrue");
  });
});

describe("mini-torch devices", () => {
  test("cpu code runs; asking for cuda says the browser has no GPU", async () => {
    assert.equal(await out(
      "dev = 'cuda' if torch.cuda.is_available() else 'cpu'",
      "print(dev, torch.ones(2).to(dev).device, torch.ones(2, device='cpu').device, torch.device('cpu'), torch.ones(1).cpu().is_cuda, torch.cuda.device_count())",
      "print(err(lambda: torch.ones(2).to('cuda')))",
    ), [
      "cpu cpu cpu cpu False 0",
      "MiniTorchUnsupported: `device='cuda'`: the browser has no GPU, so mini-torch only runs on the CPU. Pick the device with `'cuda' if torch.cuda.is_available() else 'cpu'`, which gives 'cpu' here and 'cuda' on your GPU.",
    ].join("\n"));
  });
});
