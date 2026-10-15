import torch

# Target probabilities at 3 positions (k + 1 rows) and the draft's at the 2 drafted ones, over 4 tokens
p = torch.tensor([[0.10, 0.50, 0.30, 0.10],
                  [0.25, 0.25, 0.25, 0.25],
                  [0.70, 0.10, 0.10, 0.10]])
q = torch.tensor([[0.40, 0.40, 0.10, 0.10],
                  [0.10, 0.60, 0.20, 0.10]])

def verify(p, q, drafts, gen):
    out = []
    for i, x in enumerate(drafts):
        # TODO: keep x with probability min(1, p[i, x] / q[i, x]), using one torch.rand draw from gen
        keep = torch.rand(1, generator=gen) < p[i, x] / q[i, x]
        if keep:
            out.append(x)
            continue
        # TODO: rejected: sample a replacement from max(0, p - q) at this position, then stop
        fix = int(torch.multinomial((p[i] - q[i]).clamp(min=0), 1, generator=gen))
        out.append(fix)
        return out
    # TODO: every draft survived: one bonus token, sampled from the target's last row
    bonus = int(torch.multinomial(p[-1], 1, generator=gen))
    out.append(bonus)
    return out


gen = torch.Generator().manual_seed(0)
counts = [0, 0, 0, 0]
for _ in range(2000):
    drafts = [int(torch.multinomial(q[i], 1, generator=gen)) for i in range(2)]
    counts[int(verify(p, q, drafts, gen)[0])] += 1
print("first token:", [round(c / 2000, 2) for c in counts], "  target p: [0.1, 0.5, 0.3, 0.1]   draft q: [0.4, 0.4, 0.1, 0.1]")
