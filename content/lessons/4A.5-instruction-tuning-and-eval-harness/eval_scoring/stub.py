def normalize_answer(s):
    # TODO: lowercase, keep only letters, digits and spaces, and collapse runs of whitespace into single spaces
    return ...


def exact_match(pred, gold):
    return normalize_answer(pred) == normalize_answer(gold)


def pick_option(options, logprobs):
    # TODO: the index of the option with the highest log-probability per character
    return ...


em = sum(exact_match(pred, gold) for _, gold, pred in QA) / len(QA)
mc = sum(pick_option(opts, lps) == right for _, opts, lps, right in MC) / len(MC)
print(f"exact match {100 * em:.0f}%   multiple choice {100 * mc:.0f}%   score {50 * (em + mc):.1f}%")
