def greedy(logits):
    # TODO: the id of the largest logit in this one row
    return ...


def generate(model, idx, max_new_tokens, pick=greedy):
    for _ in range(max_new_tokens):
        # TODO: run the model on every token so far
        logits = ...
        # TODO: pick the next id from the LAST position's row of logits
        next_id = ...
        # TODO: append it to the end of idx
        idx = ...
    return idx


out = generate(model, prompt, 60)
print(repr(decode(out)))
print("positions processed:", model.positions)
