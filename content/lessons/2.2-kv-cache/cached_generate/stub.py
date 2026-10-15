

def generate_cached(model, idx, max_new_tokens):
    cache = {"k": torch.zeros(0, model.C), "v": torch.zeros(0, model.C)}
    # Prefill: run the prompt through step() one token at a time, filling the cache
    for pos in range(len(idx)):
        # TODO: logits for the prompt token at position pos
        logits = ...
    for _ in range(max_new_tokens):
        next_id = logits.argmax()
        idx = torch.cat([idx, next_id.view(1)])
        # TODO: run only the new token. Which position does it sit at?
        logits = ...
    return idx


slow, fast = TinyLM(), TinyLM()
a = generate(slow, prompt, 60)
b = generate_cached(fast, prompt, 60)
print(repr(decode(b)))
print("same text:", torch.equal(a, b))
print("positions processed  no cache:", slow.positions, "  cache:", fast.positions)
