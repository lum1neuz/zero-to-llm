class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        # TODO: freeze every parameter of the base layer
        ...
        # TODO: A is [r, in] with small random values, B is [out, r] of zeros. Both train.
        self.A = ...
        self.B = ...
        # TODO: how much the update counts, from alpha and r
        self.scale = ...

    def forward(self, x):
        # TODO: the frozen layer's output plus the scaled low-rank update
        return ...


lora = LoRALinear(base, r=4, alpha=8)
print("same as base at step 0:", torch.equal(lora(x), base(x)))
print("trainable:", [name for name, p in lora.named_parameters() if p.requires_grad])
