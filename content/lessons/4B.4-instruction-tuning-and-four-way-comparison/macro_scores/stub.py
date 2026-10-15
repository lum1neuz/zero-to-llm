def macro_scores(rows):
    # TODO: the accuracy of each (model, task) pair, from 0 to 1
    acc = {key: ... for key, group in by(rows, "model", "task").items()}
    scores = {}
    for model in sorted({r.model for r in rows}):
        # TODO: this model's task accuracies
        mine = ...
        # TODO: their mean, in %
        scores[model] = ...
    return scores


for model, score in macro_scores(ROWS).items():
    print(f"{model:<12} {score:5.1f}%")
