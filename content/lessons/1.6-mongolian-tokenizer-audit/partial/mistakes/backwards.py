# 1.5's byte-level BPE, ready-made: train(text, vocab_size) -> merges, vocab; encode(text, merges) -> ids
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def train(text, vocab_size):
    ids = list(text.encode("utf-8"))
    merges, vocab = {}, {i: bytes([i]) for i in range(256)}
    for i in range(vocab_size - 256):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        ids = merge(ids, pair, 256 + i)
        merges[pair] = 256 + i
        vocab[256 + i] = vocab[pair[0]] + vocab[pair[1]]
    return merges, vocab

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

english = """Once upon a time, there was a little bird named Tumen. Tumen lived in a tall tree by a quiet river. Every morning, Tumen sang a happy song for the other birds in the tree.
One day, the wind was very strong. The wind blew the nest out of the tree and into the river. Tumen was sad. He did not know what to do.
A kind old turtle saw the nest in the water. The turtle swam to the nest and pushed it to the shore. From that day on, the little bird and the old turtle were best friends."""

mongolian_train = """Монгол Улс бол Төв Азид оршдог улс юм. Монгол Улсын нийслэл нь Улаанбаатар хот.
Монголчууд эрт дээр үеэс мал аж ахуй эрхэлж ирсэн. Нүүдэлчид гэр бүлээрээ малаа хариулж, улирал бүр нүүдэг.
Монгол гэр нь эсгий, мод, олсоор хийгдсэн байдаг. Говь цөлд тэмээ олон байдаг.
Өвөл маш хүйтэн, зун богино боловч сайхан. Хүүхдүүд намар сургуульдаа буцаж ирдэг.
Наадмын үеэр бөх, морин уралдаан, сур харваа болдог. Морь бол монголчуудын хамгийн үнэнч нөхөр.
Би өдөр бүр монгол хэлээр ном уншдаг. Ээж минь өглөө бүр сүүтэй цай чанадаг."""

# Held out: none of these sentences are in mongolian_train.
mongolian_test = [
    "Сайн байна уу? Миний нэр Тэмүүжин.",
    "Өнөөдөр цаг агаар сайхан байна.",
    "Бид маргааш Хархорин руу явна.",
    "Энэ ном маш сонирхолтой.",
    "Хүүхдүүд сургууль руу явж байна.",
    "Баярлалаа, баяртай!",
]

en_merges, en_vocab = train(english, 296)
mn_merges, mn_vocab = train(mongolian_train, 296)

def fertility(sentences, merges):
    # TODO: tokens per word, over all the sentences together
    tokens = sum(len(encode(s, merges)) for s in sentences)
    words = sum(len(s.split()) for s in sentences)
    return tokens / words

for name, merges in (("bytes only", {}), ("English BPE", en_merges), ("Mongolian BPE", mn_merges)):
    print(f"{name:<14} {fertility(mongolian_test, merges):5.2f} tokens per Mongolian word")

def is_partial(piece):
    # TODO: True when these bytes aren't whole characters by themselves. Can Python decode them?
    try:
        piece.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True

def partial_share(sentences, merges, vocab):
    pieces = [vocab[i] for s in sentences for i in encode(s, merges)]
    # TODO: the fraction of pieces that are only part of a character
    return sum(is_partial(p) for p in pieces) / len(pieces)

for name, merges, vocab in (("English BPE", en_merges, en_vocab), ("Mongolian BPE", mn_merges, mn_vocab)):
    print(f"{name:<14} {100 * partial_share(mongolian_test, merges, vocab):5.1f}% of tokens are pieces of a letter")
print("Өнөөдөр →", [mn_vocab[i].decode("utf-8", errors="replace") for i in encode("Өнөөдөр", mn_merges)])
