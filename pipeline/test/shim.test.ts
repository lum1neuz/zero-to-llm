// mini-torch semantics the lessons rely on, checked in real Pyodide.
import assert from "node:assert/strict";
import { describe, test } from "node:test";
import { runChallenge } from "../pyrun.ts";

const probe = (code: string) => `import torch\n${code}`;
const TESTS = `from harness import test\n@test("t", "t")\ndef _(L):\n    pass\n`;

async function out(code: string) {
  const r = await runChallenge(probe(code), TESTS, "probe.py");
  if (r.scriptError) throw new Error(r.scriptError.traceback);
  return r.stdout.trim();
}

describe("mini-torch", () => {
  test("python floats become float32 and round half to even", async () => {
    assert.equal(await out("w = torch.tensor([0.5, 1.5, 2.5, -0.5])\nprint(w.dtype, torch.round(w).tolist())"), "torch.float32 [0.0, 2.0, 2.0, -0.0]");
  });
  test("to(int8) truncates, then wraps out-of-range values", async () => {
    assert.equal(await out("print(torch.tensor([127.9, 128.0, -129.0, -1.7]).to(torch.int8).tolist())"), "[127, -128, 127, -1]");
  });
  test("int8 times a float scale gives float32", async () => {
    assert.equal(await out("q = torch.tensor([2, -3]).to(torch.int8)\ns = torch.tensor([0.5, 1.0]).abs().max()\nprint((q * s).dtype, (q * s).tolist())"), "torch.float32 [2.0, -3.0]");
  });
  test("int8 arithmetic overflows like torch", async () => {
    assert.equal(await out("q = torch.tensor([127]).to(torch.int8)\nprint((q * 2).tolist(), (q * 2).dtype)"), "[-2] torch.int8");
  });
  test("int8 with an out-of-range Python int wraps instead of raising", async () => {
    assert.equal(await out("q = torch.tensor([100, -100]).to(torch.int8)\nprint((q * 1000).tolist(), (q + 200).tolist(), torch.clamp(q, -200, 200).tolist(), (q + 200).dtype)"),
      "[-96, 96] [44, 100] [100, -100] torch.int8");
  });
  test("to(dtype=...) works, and a 0-d int tensor keeps the int8 dtype", async () => {
    assert.equal(await out("q = torch.tensor([1.6, -2.4]).to(dtype=torch.int8)\nprint(q.tolist(), (q * torch.tensor(2)).dtype)"), "[1, -2] torch.int8");
  });
  test("NaN and infinity cast to int become 0 instead of crashing", async () => {
    assert.equal(await out("z = torch.tensor([0.0, 1.0])\nprint((z / 0.0).to(torch.int8).tolist())"), "[0, 0]");
  });
  test("format specs only apply to 0-d tensors, like torch", async () => {
    const r = await runChallenge(probe("t = torch.tensor([1.5])\nprint(f'{t:.2f}')"), TESTS, "probe.py");
    assert.equal(r.scriptError?.type, "TypeError");
    assert.equal(await out("print(f'{torch.tensor([1.5, 2.0]).max():.2f}')"), "2.00");
  });
  test("abs, max, mean, clamp, item, float and repr", async () => {
    const got = await out([
      "w = torch.tensor([0.42, -1.30, 2.10])",
      "print(w.abs().max().item() == torch.tensor(2.1).item(), round(w.mean().item(), 4))",
      "print(w.clamp(-1, 1).tolist() == torch.tensor([0.42, -1.0, 1.0]).tolist())",
      "print(torch.clamp(torch.tensor([3, -300]), -127, 127).to(torch.int8))",
      "print(torch.tensor([1.0, -0.5]))",
    ].join("\n"));
    assert.equal(got, "True 0.4067\nTrue\ntensor([3, -127], dtype=torch.int8)\ntensor([1.0000, -0.5000])");
  });
  test("reshape, view and flatten, with torch's error for a shape that doesn't fit", async () => {
    assert.equal(await out([
      "w = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])",
      "print(w.reshape(-1, 3).shape, w.view(2, 3).tolist(), w.reshape((3, 2)).shape)",
      "print(w.reshape(-1, 2).flatten().tolist() == w.tolist(), w.reshape(2, 3).reshape(-1).shape)",
    ].join("\n")), "torch.Size([2, 3]) [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] torch.Size([3, 2])\nTrue torch.Size([6])");
    const r = await runChallenge(probe("torch.tensor([1.0, 2.0, 3.0]).reshape(-1, 2)"), TESTS, "probe.py");
    assert.equal(r.scriptError?.message, "shape '[-1, 2]' is invalid for input of size 3");
  });
  test("amax(dim, keepdim) and max(dim) returning (values, indices)", async () => {
    assert.equal(await out([
      "wg = torch.tensor([[1.0, -4.0], [0.5, 2.0], [3.0, 1.0]])",
      "print(wg.abs().amax(dim=1, keepdim=True).tolist(), wg.amax(dim=1).shape, wg.amax(0).tolist(), wg.amax().item())",
      "m = wg.max(dim=1, keepdim=True)",
      "print(m.values.tolist(), m.indices.tolist(), m[0].shape, type(m).__name__)",
    ].join("\n")), "[[4.0], [2.0], [3.0]] torch.Size([3]) [3.0, 2.0] 3.0\n[[1.0], [2.0], [3.0]] [[0], [1], [0]] torch.Size([3, 1]) torch.return_types.max");
    const r = await runChallenge(probe("wg = torch.tensor([[1.0, -4.0]])\nscale = wg.abs().max(dim=1, keepdim=True) / 7"), TESTS, "probe.py");
    assert.equal(r.scriptError?.message, "unsupported operand type(s) for /: 'torch.return_types.max' and 'int'");
    assert.match(r.scriptError!.diagnosis, /returns two things, `\(values, indices\)`/);
  });
  test("broadcasting follows torch, including its error wording", async () => {
    assert.equal(await out("q = torch.tensor([[2.0, 4.0], [6.0, 8.0]])\nprint((q / torch.tensor([[2.0], [1.0]])).tolist(), (q / torch.tensor([2.0, 1.0])).tolist())"),
      "[[1.0, 2.0], [6.0, 8.0]] [[1.0, 4.0], [3.0, 8.0]]");
    const r = await runChallenge(probe("print(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) / torch.tensor([1.0, 2.0]))"), TESTS, "probe.py");
    assert.equal(r.scriptError?.message, "The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1");
  });
  test("a dim that doesn't exist is explained", async () => {
    const r = await runChallenge(probe("w = torch.tensor([1.0, -2.0])\ns = w.amax(dim=1)"), TESTS, "probe.py");
    assert.equal(r.scriptError?.message, "Dimension out of range (expected to be in range of [-1, 0], but got 1)");
    assert.match(r.scriptError!.diagnosis, /This tensor has 1 dimension, so `dim=1` doesn't exist. Print its `.shape`/);
    assert.equal(r.scriptError?.line, 3);
  });
  test("API outside the subset says so plainly", async () => {
    const r = await runChallenge(probe("torch.einsum('ij->ji', torch.ones(2, 2))"), TESTS, "probe.py");
    assert.match(r.scriptError!.diagnosis, /`torch\.einsum` is real PyTorch, but the browser's mini-torch doesn't include it\. Browser lessons use the subset in TORCH_SUBSET\.md/);
    const r2 = await runChallenge(probe("torch.tensor([1.0]).roll(1)"), TESTS, "probe.py");
    assert.match(r2.scriptError!.diagnosis, /Tensor\.roll/);
    const r3 = await runChallenge(probe("import torch.distributed"), TESTS, "probe.py");
    assert.match(r3.scriptError!.diagnosis, /`torch\.distributed` is real PyTorch/);
    const r4 = await runChallenge(probe("import torch.nn as nn\nnn.Conv2d(1, 1, 3)"), TESTS, "probe.py");
    assert.match(r4.scriptError!.diagnosis, /`torch\.nn\.Conv2d` is real PyTorch/);
  });
  test("placeholders and missing names are explained in plain words", async () => {
    const r = await runChallenge(probe("w = torch.tensor([1.0])\nscale = ...\nq = w / scale"), TESTS, "probe.py");
    assert.match(r.scriptError!.diagnosis, /`scale` is still `\.\.\.`, the placeholder \(line 3\)/);
    const r2 = await runChallenge("w = torch.tensor([1.0])", TESTS, "probe.py");
    assert.match(r2.scriptError!.diagnosis, /The `import torch` line at the top is gone/);
    const r3 = await runChallenge(probe("w = torch.tensor([1.0])\nwg = ...\nscale = wg.abs()"), TESTS, "probe.py");
    assert.match(r3.scriptError!.diagnosis, /`wg` is still `\.\.\.`, the placeholder \(line 3\)/);
  });
  test("the placeholder named is the one the failing line uses", async () => {
    const code = "import torch\nscale = ...\ndef f(q, s):\n    return ...\nw = torch.tensor([1.0])\nw_hat = f(w, 1)\nprint(w - w_hat)";
    const r = await runChallenge(code, TESTS, "probe.py");
    assert.match(r.scriptError!.diagnosis, /`f` still returns `\.\.\.`/);
    assert.match(r.scriptError!.traceback, /print\(w - w_hat\)/, "traceback shows the learner's source line");
  });
});
