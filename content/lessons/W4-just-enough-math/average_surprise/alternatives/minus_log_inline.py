import math

def surprise(p):
    # TODO: minus the log of p
    return -math.log(p)

print(surprise(0.5))

def average_surprise(probs):
    total = 0
    for p in probs:
        # TODO: add this guess's surprise
        total = total - math.log(p)
    return total / len(probs)

print(average_surprise([0.5, 0.25, 1.0]))
