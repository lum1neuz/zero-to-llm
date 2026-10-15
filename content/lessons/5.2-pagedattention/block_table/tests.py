import numpy as np
import torch
from harness import test, fail, learner_fn, arr, script_error, stdout

TABLE = [2, 7, 1, 4]   # a different request from the script's: 4 blocks, out of physical order
N_BLOCKS = 8


def block(L):
    b = L.get("BLOCK")
    return b if isinstance(b, int) and b > 0 else 4


def slots(L):
    """Call slot_of for every position of TABLE; explain float indices and per-token tables."""
    fn = L.get("slot_of")
    if not callable(fn):
        learner_fn(L, "slot_of")
    B = block(L)
    out = []
    for pos in range(len(TABLE) * B):
        try:
            r = fn(list(TABLE), pos)
        except TypeError as e:
            if "float" in str(e):
                fail(f"`slot_of` raised: {e}. `pos / BLOCK` gives a float ({5} / {B} = {5 / B}), and list positions must be "
                     "whole numbers. Which operator divides and drops the remainder?")
            r = learner_fn(L, "slot_of")(list(TABLE), pos)
        except IndexError:
            fail(f"`slot_of(block_table, {pos})` ran off the end of a {len(TABLE)}-entry block table. "
                 "The table has one entry per *block*, not per token. Which block holds that position?")
        except Exception:  # noqa: BLE001 - let the harness explain it
            r = learner_fn(L, "slot_of")(list(TABLE), pos)
        out.append(r)
    return B, out


@test("slot_of", "slot_of maps a position to (physical block, offset)")
def _(L):
    B, out = slots(L)
    first = out[0]
    if first is Ellipsis:
        fail("`slot_of` returns `...`. Keep `return physical, offset` and fill in the two placeholders above it.")
    if isinstance(first, tuple) and len(first) == 2 and first[0] is Ellipsis:
        both = first[1] is Ellipsis
        fail("`physical` and `offset` are still `...`, the placeholders." if both else "`physical` is still `...`, the placeholder.")
    if isinstance(first, tuple) and len(first) == 2 and first[1] is Ellipsis:
        fail("`offset` is still `...`, the placeholder.")
    if not all(isinstance(r, tuple) and len(r) == 2 for r in out):
        bad = next(r for r in out if not (isinstance(r, tuple) and len(r) == 2))
        fail(f"`slot_of` returns {bad!r}, one value. The loop unpacks two, `b, o = slot_of(...)`: return the physical block and the offset.")
    got = [(int(a), int(b)) for a, b in out]
    right = [(TABLE[p // B], p % B) for p in range(len(got))]
    if got == right:
        return
    p = B + 1   # a position in logical block 1, which lives in physical block 7
    if got == [(q // B, q % B) for q in range(len(got))]:
        fail(f"For block table {TABLE}, position {p} is in logical block 1, which lives in physical block {TABLE[1]} "
             f"(`block_table[1]`). Yours says block {got[p][0]}, the logical number. Look it up in the table.")
    if got == [(TABLE[q // B], q) for q in range(len(got))]:
        fail(f"Position {p} sits at offset {p % B} of its block, because positions 0–{B - 1} fill the block before it. "
             f"Yours says offset {got[p][1]}, but a block only has slots 0–{B - 1}.")
    k = next(i for i in range(len(got)) if got[i] != right[i])
    fail(f"With block table {TABLE}, position {k} should be (block {right[k][0]}, offset {right[k][1]}), but `slot_of` says {got[k]}.")


def pool(B, d):
    """Slot (b, o) holds [100·b + o, −(100·b + o), 0.5, …], so every key names where it came from."""
    p = np.zeros((N_BLOCKS, B, d), dtype=np.float32)
    for b in range(N_BLOCKS):
        for o in range(B):
            p[b, o, 0] = 100 * b + o
            p[b, o, 1] = -(100 * b + o)
            p[b, o, 2:] = 0.5
    return p


@test("gather_kv", "gather_kv returns the request's keys in order, cut to its length")
def _(L):
    B = block(L)
    d = 3
    length = 3 * B + 2   # the last block is only half full
    P = pool(B, d)
    gather = learner_fn(L, "gather_kv")
    out = gather(torch.tensor(P), list(TABLE), length)
    if out is None:
        fail("`gather_kv` returns nothing. It has to `return` the gathered keys.")
    if out is Ellipsis:
        fail("`gather_kv` still returns `...`, the placeholder.")
    a = arr(out)
    if a is None:
        fail(f"`gather_kv` should return a tensor, but it returns {type(out).__name__}.")
    right = P[TABLE].reshape(-1, d)[:length]
    if a.shape == right.shape and np.array_equal(a, right):
        return
    if a.shape == (len(TABLE), B, d):
        fail(f"You return shape [{len(TABLE)}, {B}, {d}], still one row per block. Lay the blocks end to end, one row per token.")
    if a.ndim == 1:
        fail(f"You return shape [{a.shape[0]}]: every number in one long row. Keep one row per token, each with its {d} numbers.")
    if a.ndim == 2 and a.shape[1] == 2 and d != 2:
        fail(f"Your keys come back with 2 numbers each, but this pool's keys have {d}. The script's pool has d = 2, the next one "
             "might not. Read d from the pool: `pool.shape[-1]`.")
    if a.shape == (len(TABLE) * B, d):
        fail(f"You return all {len(TABLE) * B} slots of the request's blocks, but it holds {length} tokens. The last block's "
             f"{len(TABLE) * B - length} empty slots hold stale keys from an earlier request. Cut to `length`.")
    if a.shape == right.shape and np.array_equal(a, P[:len(TABLE)].reshape(-1, d)[:length]):
        fail(f"You read physical blocks 0, 1, 2, 3 in order, but this request's blocks are {TABLE}. "
             "Other requests own the rest of the pool. Use the block table to pick the request's blocks.")
    if a.shape == right.shape:
        k = int(np.argmax(np.abs(a - right).max(axis=1) > 0))
        fail(f"Row {k} should be the key at position {k}, from physical block {TABLE[k // B]}, offset {k % B}. Yours comes from "
             f"somewhere else. Keep the blocks in block-table order.")
    fail(f"You return shape [{', '.join(map(str, a.shape))}], but {length} tokens of {d} numbers should be [{length}, {d}].")


@test("round_trip", "Keys written through slot_of come back through gather_kv")
def _(L):
    err = script_error()
    if err:
        fail(f"Your script stopped at line {err['line']}: {err['diagnosis']}" if err.get("line") else f"Your script stopped: {err['diagnosis']}")
    if "back" not in L:
        fail("The loop at the bottom that writes the keys and reads them back is missing. Put it back.")
    if not stdout().strip():
        fail("Nothing printed. Put the print line back.")
