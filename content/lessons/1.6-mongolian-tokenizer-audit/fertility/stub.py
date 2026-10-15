def fertility(sentences, merges):
    # TODO: tokens per word, over all the sentences together
    tokens = ...
    words = ...
    return tokens / words

for name, merges in (("bytes only", {}), ("English BPE", en_merges), ("Mongolian BPE", mn_merges)):
    print(f"{name:<14} {fertility(mongolian_test, merges):5.2f} tokens per Mongolian word")
