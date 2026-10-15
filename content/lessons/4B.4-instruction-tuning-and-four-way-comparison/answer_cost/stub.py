

def tokens_per_answer(rows):
    # TODO: each model's mean answer length in tokens, over every answer it gave
    return {model: ... for model, group in by(rows, "model").items()}


def tokens_per_right_answer(rows):
    # TODO: every token a model generated, divided by the answers it got right
    return {model: ... for model, group in by(rows, "model").items()}


scores = macro_scores(ROWS)
tpa = tokens_per_answer(ROWS)
tpr = tokens_per_right_answer(ROWS)
print(f"\n{'model':<12} {'score':>6} {'tok/answer':>11} {'tok/right':>10} {'cost vs control':>16}")
for m in scores:
    print(f"{m:<12} {scores[m]:5.1f}% {tpa[m]:11.1f} {tpr[m]:10.1f} {tpa[m] / tpa['control']:15.2f}×")
