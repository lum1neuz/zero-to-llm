def fertility(text, vocab):
    # TODO: the words of text, the same way the corpus is split
    ws = ...
    # TODO: how many tokens encode() makes across all of those words
    n_tokens = ...
    # TODO: tokens per word
    return ...

MN_VOCAB = train_bpe(CORPUS, 120)
print(f"base vocab ({len(BASE_VOCAB)} pieces)   {fertility(HELD_OUT, BASE_VOCAB):.2f} tokens per word")
print(f"Mongolian BPE ({len(MN_VOCAB)} pieces) {fertility(HELD_OUT, MN_VOCAB):.2f} tokens per word")
