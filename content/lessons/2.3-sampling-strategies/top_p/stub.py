

def top_p(logits, p):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, order = probs.sort(descending=True)
    # TODO: the running total of the sorted probabilities
    cum = ...
    # TODO: True for every token to cut: the tokens ranked above it already cover p
    cut_sorted = ...
    cut = cut_sorted[order.argsort()]  # back to the vocabulary's order
    return logits.masked_fill(cut, float("-inf"))


for p in (0.5, 0.9, 0.99):
    kept = (top_p(row, p) > float("-inf")).sum().item()
    print(f"top_p(row, {p}) keeps {kept} of {len(row)} tokens")
