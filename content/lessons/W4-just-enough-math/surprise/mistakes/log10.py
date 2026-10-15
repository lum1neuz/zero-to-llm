import math

def surprise(p):
    # TODO: minus the log of p
    return -math.log10(p)

print(surprise(0.5))
