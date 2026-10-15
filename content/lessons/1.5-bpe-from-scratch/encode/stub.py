
import time

def decode(ids):
    # TODO: look up the bytes each id stands for and join them into one bytes object
    data = ...
    return data.decode("utf-8", errors="replace")

def encode(text):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        # TODO: of the pairs present, the one training merged EARLIEST (unmerged pairs never win)
        pair = ...
        if pair not in merges:
            break   # nothing left that training ever merged
        ids = merge(ids, pair, merges[pair])
    return ids

test_set = [
    "Once upon a time, the little bird sang.",
    "The turtle and the bird were best friends.",
    "Сайн байна уу? Миний нэр Тэмүүжин.",
    "Монгол хэл, кирилл үсэг",
    "naïve café, 1,024 tokens",
    "你好，世界",
    "🙂🐢🐦",
    "def f(x):\n    return x * 2",
]
ok = sum(decode(encode(s)) == s for s in test_set)
print(f"round trip: {ok}/{len(test_set)} lines ok ({100 * ok / len(test_set):.0f}%)")
mn = "Сайн байна уу?"
print(f"{mn!r}: {len(mn)} characters, {len(mn.encode('utf-8'))} bytes -> {len(encode(mn))} tokens")

ids = encode(text)
print(f"story: {len(text.encode('utf-8'))} bytes -> {len(ids)} tokens")
t0 = time.perf_counter()
for _ in range(5):
    encode(text)
mb = 5 * len(text.encode("utf-8")) / 1e6
print(f"encoding speed {mb / (time.perf_counter() - t0):.3f} MB/s")
