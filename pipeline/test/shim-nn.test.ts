// mini-torch torch.nn and torch.optim, checked in real Pyodide.
import assert from "node:assert/strict";
import { describe, test } from "node:test";
import { runChallenge } from "../pyrun.ts";

const TESTS = `from harness import test\n@test("t", "t")\ndef _(L):\n    pass\n`;
const PRELUDE = [
  "import copy",
  "import torch",
  "import torch.nn as nn",
  "import torch.nn.functional as F",
  "from torch import optim",
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

const MLP = [
  "class MLP(nn.Module):",
  "    def __init__(self):",
  "        super().__init__()",
  "        self.emb = nn.Embedding(10, 4)",
  "        self.fc = nn.Linear(8, 5)",
  "        self.ln = nn.LayerNorm(5)",
  "        self.blocks = nn.ModuleList([nn.Linear(5, 5) for _ in range(2)])",
  "        self.drop = nn.Dropout(0.1)",
  "        self.register_buffer('mask', torch.tril(torch.ones(2, 2)))",
  "    def forward(self, idx):",
  "        x = self.emb(idx).view(idx.shape[0], -1)",
  "        x = self.ln(F.gelu(self.fc(x)))",
  "        for blk in self.blocks:",
  "            x = x + blk(x)",
  "        return self.drop(x)",
];

describe("mini-torch nn.Module", () => {
  test("registers parameters, submodules and buffers; prints like torch", async () => {
    assert.equal(await out(
      ...MLP,
      "torch.manual_seed(1337)",
      "net = MLP()",
      "print(net)",
      "print(sum(p.numel() for p in net.parameters()), [n for n, _ in net.named_parameters()][:4], list(net.state_dict())[:3], list(net.state_dict())[-3:])",
      "print(len(list(net.children())), len(list(net.modules())), [n for n, _ in net.named_buffers()], net.mask.shape)",
      "print(net.fc.weight.shape, net.fc.bias.shape, type(net.fc.weight).__name__, net.fc.weight.is_leaf, net.fc.weight.requires_grad)",
      "print(nn.Parameter(torch.ones(2)), isinstance(nn.Parameter(torch.ones(1)), torch.Tensor))",
    ), [
      "MLP(",
      "  (emb): Embedding(10, 4)",
      "  (fc): Linear(in_features=8, out_features=5, bias=True)",
      "  (ln): LayerNorm((5,), eps=1e-05, elementwise_affine=True)",
      "  (blocks): ModuleList(",
      "    (0): Linear(in_features=5, out_features=5, bias=True)",
      "    (1): Linear(in_features=5, out_features=5, bias=True)",
      "  )",
      "  (drop): Dropout(p=0.1, inplace=False)",
      ")",
      "155 ['emb.weight', 'fc.weight', 'fc.bias', 'ln.weight'] ['mask', 'emb.weight', 'fc.weight'] ['blocks.0.bias', 'blocks.1.weight', 'blocks.1.bias']",
      "5 8 ['mask'] torch.Size([2, 2])",
      "torch.Size([5, 8]) torch.Size([5]) Parameter True True",
      "Parameter containing:\ntensor([1., 1.], requires_grad=True) True",
    ].join("\n"));
  });
  test("layer initialisation and forward shapes", async () => {
    assert.equal(await out(
      "torch.manual_seed(0)",
      "lin = nn.Linear(16, 3)",
      "print(lin.weight.abs().max().item() <= 0.25, lin.bias.abs().max().item() <= 0.25, lin(torch.ones(2, 7, 16)).shape, nn.Linear(4, 2, bias=False).bias)",
      "emb = nn.Embedding(5, 3)",
      "print(emb(torch.tensor([[0, 4], [1, 1]])).shape, emb.weight.dtype, err(lambda: emb(torch.tensor([5]))), err(lambda: emb(torch.tensor([1.0]))))",
      "ln = nn.LayerNorm(4)",
      "y = ln(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))",
      "print(ln.weight.tolist(), ln.bias.tolist(), round(y.mean().item(), 6), round(y.var(unbiased=False).item(), 3), nn.RMSNorm(4)(torch.ones(1, 4)).tolist())",
      "seq = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1), nn.Tanh())",
      "print(seq(torch.ones(4, 2)).shape, len(seq), seq[0].weight.shape, type(seq[1:]).__name__, [type(m).__name__ for m in seq])",
      "print(nn.GELU()(torch.tensor([1.0])).tolist() == F.gelu(torch.tensor([1.0])).tolist(), nn.SiLU()(torch.zeros(1)).tolist(), nn.Sigmoid()(torch.zeros(1)).tolist(), nn.Softmax(dim=-1)(torch.zeros(2)).tolist(), nn.Identity()(torch.ones(1)).tolist())",
      "md = nn.ModuleDict(dict(wte=nn.Embedding(5, 2), h=nn.ModuleList([nn.Linear(2, 2)])))",
      "print(md.wte.weight.shape, md['h'][0].bias.shape, 'wte' in md, list(md.keys()), len(list(md.parameters())))",
      "print(nn.CrossEntropyLoss()(torch.zeros(2, 4), torch.tensor([0, 1])).item() == torch.tensor(4.0).log().item(), nn.MSELoss()(torch.ones(2), torch.zeros(2)).item())",
    ), [
      "True True torch.Size([2, 7, 3]) None",
      "torch.Size([2, 2, 3]) torch.float32 IndexError: index out of range in self RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.FloatTensor instead (while checking arguments for embedding)",
      "[1.0, 1.0, 1.0, 1.0] [0.0, 0.0, 0.0, 0.0] 0.0 1.0 [[1.0, 1.0, 1.0, 1.0]]",
      "torch.Size([4, 1]) 4 torch.Size([3, 2]) Sequential ['Linear', 'ReLU', 'Linear', 'Tanh']",
      "True [0.0] [0.5] [0.5, 0.5] [1.0]",
      "torch.Size([5, 2]) torch.Size([2]) True ['wte', 'h'] 3",
      "True 1.0",
    ].join("\n"));
  });
  test("train() and eval(): Dropout and BatchNorm1d switch behaviour", async () => {
    assert.equal(await out(
      "torch.manual_seed(0)",
      "drop = nn.Dropout(0.5)",
      "x = torch.ones(1000)",
      "print(drop.training, (drop(x) == 0).any().item(), drop.eval().training, torch.equal(drop(x), x))",
      "bn = nn.BatchNorm1d(3)",
      "xb = torch.randn(32, 3) * 2 + 5",
      "y = bn(xb)",
      "print(y.mean(0).abs().max().item() < 1e-5, abs(y.std(0, unbiased=False).mean().item() - 1) < 1e-3, bn.num_batches_tracked.item())",
      "print(torch.allclose(bn.running_mean, 0.1 * xb.mean(0)), torch.allclose(bn.running_var, 0.9 + 0.1 * xb.var(0)))",
      "bn.eval()",
      "print(torch.allclose(bn(xb), (xb - bn.running_mean) / (bn.running_var + 1e-5).sqrt(), atol=1e-6), bn.training, bn.num_batches_tracked.item())",
      "print(err(lambda: nn.BatchNorm1d(3).train()(torch.ones(1, 3))))",
    ), [
      "True True False True",
      "True True 1",
      "True True",
      "True False 1",
      "ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 3])",
    ].join("\n"));
  });
  test("freezing, state_dict round trips, deepcopy, to(dtype)", async () => {
    assert.equal(await out(
      "base = nn.Linear(4, 4)",
      "base.requires_grad_(False)",
      "A = nn.Parameter(torch.zeros(4, 2))",
      "print(sum(p.requires_grad for p in base.parameters()), A.requires_grad)",
      "seq = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))",
      "seq2 = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))",
      "print(seq2.load_state_dict(seq.state_dict()), torch.equal(seq2[0].weight, seq[0].weight), seq2[0].weight is not seq[0].weight)",
      "print(err(lambda: seq2.load_state_dict({'0.weight': torch.zeros(3, 3)})))",
      "c = copy.deepcopy(seq)",
      "print(c[0].weight is not seq[0].weight, torch.equal(c[0].weight, seq[0].weight), c[0].weight.requires_grad, type(c[0].weight).__name__)",
      "print(seq.to(torch.float64)[0].weight.dtype, seq.to('cpu') is seq)",
      "bn = nn.BatchNorm1d(2).to(torch.float64)",
      "print(bn.running_mean.dtype, bn.num_batches_tracked.dtype, list(seq[1:].state_dict()), list(seq[::2].state_dict())[-1])",
    ), [
      "0 True",
      "IncompatibleKeys(missing_keys=[], unexpected_keys=[]) True True",
      "RuntimeError: Error(s) in loading state_dict for Sequential:\n\tMissing key(s) in state_dict: \"0.bias\", \"2.weight\", \"2.bias\".\n\tsize mismatch for 0.weight: copying a param with shape torch.Size([3, 3]) from checkpoint, the shape in current model is torch.Size([3, 2]).",
      "True True True Parameter",
      "torch.float64 True",
      "torch.float64 torch.int64 ['2.weight', '2.bias'] 2.bias",
    ].join("\n"));
  });
  test("torch's errors for common Module mistakes", async () => {
    assert.equal(await out(
      "class NoInit(nn.Module):",
      "    def __init__(self):",
      "        self.w = nn.Parameter(torch.zeros(2))",
      "class NoForward(nn.Module):",
      "    pass",
      "lin = nn.Linear(3, 2)",
      "print(err(NoInit))",
      "print(err(lambda: NoForward()(torch.ones(1))))",
      "print(err(lambda: setattr(lin, 'weight', torch.zeros(2, 3))))",
      "print(err(lambda: lin(torch.ones(2, 4))))",
      "print(err(lambda: lin(torch.ones(2, 3, dtype=torch.long))))",
    ), [
      "AttributeError: cannot assign parameters before Module.__init__() call",
      "NotImplementedError: Module [NoForward] is missing the required \"forward\" function",
      "TypeError: cannot assign 'torch.FloatTensor' as parameter 'weight' (torch.nn.Parameter or None expected)",
      "RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x4 and 3x2)",
      "RuntimeError: expected m1 and m2 to have the same dtype, but got: long int != float",
    ].join("\n"));
  });
  test("nn.init functions fill in place and keep parameters as leaves", async () => {
    assert.equal(await out(
      "lin = nn.Linear(64, 32)",
      "nn.init.kaiming_uniform_(lin.weight, a=5 ** 0.5)",
      "nn.init.zeros_(lin.bias)",
      "w = nn.init.normal_(torch.empty(2000), mean=1.0, std=0.02)",
      "print(lin.weight.abs().max().item() <= 1 / 8, lin.bias.tolist() == [0.0] * 32, lin.weight.is_leaf, lin.weight.requires_grad, abs(w.mean().item() - 1) < 0.01, abs(w.std().item() - 0.02) < 0.002)",
      "u = nn.init.uniform_(torch.empty(1000), -0.1, 0.1)",
      "x = nn.init.xavier_uniform_(torch.empty(30, 20))",
      "print(u.abs().max().item() <= 0.1, x.abs().max().item() <= (6 / 50) ** 0.5, nn.init.ones_(torch.empty(2)).tolist(), nn.init.constant_(torch.empty(2), 3.0).tolist())",
    ), "True True True True True True\nTrue True [1.0, 1.0] [3.0, 3.0]");
  });
});

describe("mini-torch optim", () => {
  test("SGD with momentum, Adam and AdamW follow torch's update rules", async () => {
    // Hand-worked on f(p) = p², p0 = 1, lr 0.1, two steps (see TORCH_SUBSET.md for the formulas).
    assert.equal(await out(
      "def two_steps(make):",
      "    p = nn.Parameter(torch.ones(1))",
      "    opt = make([p])",
      "    for _ in range(2):",
      "        opt.zero_grad()",
      "        (p * p).sum().backward()",
      "        opt.step()",
      "    return round(p.item(), 4)",
      "print(two_steps(lambda ps: optim.SGD(ps, lr=0.1)), two_steps(lambda ps: optim.SGD(ps, lr=0.1, momentum=0.9)))",
      "print(two_steps(lambda ps: optim.Adam(ps, lr=0.1)), two_steps(lambda ps: optim.AdamW(ps, lr=0.1, weight_decay=0.1)))",
      "print(two_steps(lambda ps: optim.Adam([{'params': ps, 'lr': 0.1}])), two_steps(lambda ps: optim.SGD(ps, lr=0.1, weight_decay=0.5)))",
    ), "0.64 0.46\n0.8004 0.7816\n0.8004 0.5625");
  });
  test("zero_grad, param_groups lr changes, clip_grad_norm_", async () => {
    assert.equal(await out(
      "p = nn.Parameter(torch.tensor([3.0, 4.0]))",
      "opt = optim.SGD([p], lr=1.0)",
      "(p * p).sum().backward()",
      "opt.zero_grad(set_to_none=False)",
      "print(p.grad.tolist())",
      "opt.zero_grad()",
      "print(p.grad)",
      "p.grad = torch.tensor([3.0, 4.0])",
      "total = nn.utils.clip_grad_norm_([p], max_norm=1.0)",
      "print(total.item(), [round(v, 4) for v in p.grad.tolist()])",
      "for g in opt.param_groups:",
      "    g['lr'] = 0.5",
      "opt.step()",
      "print([round(v, 2) for v in p.tolist()], len(opt.param_groups), err(lambda: optim.SGD([], lr=0.1)), err(lambda: optim.SGD([p * 2], lr=0.1)))",
    ), [
      "[0.0, 0.0]",
      "None",
      "5.0 [0.6, 0.8]",
      "[2.7, 3.6] 1 ValueError: optimizer got an empty parameter list ValueError: can't optimize a non-leaf Tensor",
    ].join("\n"));
  });
  test("a small training loop drives the loss down; LoRA-style frozen base only trains the adapters", async () => {
    assert.equal(await out(
      ...MLP,
      "torch.manual_seed(1337)",
      "net = MLP()",
      "idx = torch.randint(0, 10, (6, 2))",
      "y = torch.tensor([0, 1, 2, 3, 4, 0])",
      "opt = optim.AdamW(net.parameters(), lr=1e-2)",
      "losses = []",
      "for step in range(30):",
      "    loss = F.cross_entropy(net(idx), y)",
      "    opt.zero_grad(set_to_none=True)",
      "    loss.backward()",
      "    nn.utils.clip_grad_norm_(net.parameters(), 1.0)",
      "    opt.step()",
      "    losses.append(loss.item())",
      "print(losses[-1] < losses[0] / 3)",
      "class LoRALinear(nn.Module):",
      "    def __init__(self, base, r=2, alpha=4):",
      "        super().__init__()",
      "        self.base = base",
      "        self.base.requires_grad_(False)",
      "        self.A = nn.Parameter(torch.randn(base.in_features, r) * 0.01)",
      "        self.B = nn.Parameter(torch.zeros(r, base.out_features))",
      "        self.scale = alpha / r",
      "    def forward(self, x):",
      "        return self.base(x) + (x @ self.A @ self.B) * self.scale",
      "layer = LoRALinear(nn.Linear(8, 8))",
      "before = layer.base.weight.clone()",
      "trainable = [p for p in layer.parameters() if p.requires_grad]",
      "opt = optim.SGD(trainable, lr=0.1)",
      "x = torch.randn(16, 8)",
      "for _ in range(5):",
      "    opt.zero_grad()",
      "    F.mse_loss(layer(x), x * 2).backward()",
      "    opt.step()",
      "print(len(trainable), sum(p.numel() for p in trainable), torch.equal(layer.base.weight, before), layer.base.weight.grad, layer.B.abs().sum().item() > 0)",
    ), "True\n2 32 True None True");
  });
});
