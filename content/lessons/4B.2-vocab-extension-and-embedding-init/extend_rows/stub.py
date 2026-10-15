def extend(weight, pieces):
    # TODO: one new row per new token, the mean of its base pieces' rows
    new_rows = ...
    # TODO: every old row unchanged, the new rows after them
    return ...

# For each new token, the ids of the base pieces it splits into today.
PIECES = [[BASE.index(p) for p in encode(t, BASE_VOCAB)] for t in NEW_TOKENS]
E_ext = extend(E, PIECES)
H_ext = extend(H, PIECES)
print(f"E {list(E.shape)} → {list(E_ext.shape)}    H {list(H.shape)} → {list(H_ext.shape)}")
print(f"new token {len(BASE) + 1} \"{NEW_TOKENS[1]}\" starts as the mean of {[BASE[i] for i in PIECES[1]]}")
