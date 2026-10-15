
def pick_new_tokens(counts, base_vocab, k):
    # TODO: the tokens base_vocab doesn't have yet
    candidates = ...
    # TODO: for each candidate, the base tokens it saves across the corpus
    saved = ...
    # TODO: the k biggest savers, best first
    return ...

for k in (5, 10, 20):
    new = pick_new_tokens(COUNTS, BASE_VOCAB, k)
    print(f"base + {k:<2} new  {fertility(HELD_OUT, BASE_VOCAB | set(new)):.2f} tokens per word   last picks: {', '.join(new[-3:])}")
