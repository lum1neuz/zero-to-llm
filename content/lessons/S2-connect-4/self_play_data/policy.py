

def visit_policy(root, temperature=1.0):
    """S1's visit_policy, with 7 columns instead of 9 cells."""
    visits = [root.children[m].n if m in root.children else 0 for m in range(COLS)]
    if temperature == 0:
        best = visits.index(max(visits))
        return [1.0 if m == best else 0.0 for m in range(COLS)]
    powered = [n ** (1 / temperature) for n in visits]
    total = sum(powered)
    return [x / total for x in powered]
