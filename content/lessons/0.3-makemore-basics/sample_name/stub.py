
def sample_name(P, g):
    out = []
    ix = 0  # start at "."
    for _ in range(50):  # a cap, so a bug can't loop forever
        # TODO: draw the next index from row ix of P, as a plain int
        ix = ...
        if ix == 0:
            break
        out.append(itos[ix])
    return "".join(out)


g = torch.Generator().manual_seed(2147483647)
for _ in range(5):
    print(sample_name(P, g))
