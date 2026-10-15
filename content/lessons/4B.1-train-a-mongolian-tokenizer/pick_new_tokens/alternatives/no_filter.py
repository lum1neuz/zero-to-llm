import re

# A small Mongolian corpus to train on. Your real one is the 4A.3 corpus, about 62M words.
CORPUS = """
Монгол Улс Төв Азид оршдог. Улаанбаатар бол Монгол Улсын нийслэл юм.
Би өглөө бүр цай уудаг. Манай гэр бүл хотод амьдардаг. Өвөл маш хүйтэн байдаг.
Хүүхдүүд сургууль руу явж байна. Тэр номын санд ном уншиж байна.
Малчид зун нүүдэллэдэг. Энэ морь их хурдан гүйдэг. Бид маргааш хөдөө явна.
Монгол хэл кирилл үсгээр бичигддэг. Наадмын үеэр бөх барилдаж, морь уралдана.
Ээж маань бууз хийж байна. Тэнгэр цэлмэг, нар хурц байна.
Оюутнууд их сургуульд суралцаж байна. Миний ах эмчээр ажилладаг.
Говьд тэмээ олон байдаг. Бид гэртээ хоол идэж байна.
Хөдөөгийн агаар цэвэр байдаг. Чи хаана амьдардаг вэ?
Багш хүүхдүүдэд түүх заадаг. Сэлэнгэ мөрөн хойд зүг рүү урсдаг.
Манай ангийн хүүхдүүд хөл бөмбөг тоглодог. Зун малчид нүүдэг.
Эмээ маань сүүтэй цай чанаж байна.
"""
# Sentences the trainer never sees.
HELD_OUT = """
Манай хүүхдүүд өглөө бүр сургуульдаа явдаг. Өвөл малчид гэртээ байдаг.
Тэр их сургуульд түүх заадаг. Бид зун хөдөө амьдардаг.
Улаанбаатар хотод олон хүн амьдардаг.
"""

LETTERS = "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
# A stand-in for Llama 3.1's vocab as far as Mongolian goes: every letter, plus short
# pieces it mostly learned from Russian.
BASE_VOCAB = set(LETTERS) | set(
    "ал ар ан ах ай ий ин ол он ор от ов ул ур эр ер ен ст та то на но ни ра ро ри ли ла ло "
    "да до ди ба ма ха хо га го де не те ре ле ам ат ад ак эн эл аа ээ уу оо ын ой "
    "сто тор ова ани анд ата ира ора ана ола ага".split())


def words(text):
    """Lowercased words, punctuation dropped. Lowercasing stretches a tiny corpus further."""
    return re.findall(r"\w+", text.lower())


def encode(word, vocab):
    """Greedy longest match: at each position, take the longest piece that's in vocab.
    A stand-in for BPE's merge order that lets you add tokens by adding strings."""
    toks, i = [], 0
    while i < len(word):
        j = len(word)
        while j > i + 1 and word[i:j] not in vocab:
            j -= 1
        toks.append(word[i:j])
        i = j
    return toks


def train_bpe(text, n_merges):
    """Your BPE trainer from 1.5, condensed. Returns the vocab: letters plus one token per merge."""
    counts = {}
    for w in words(text):
        counts[w] = counts.get(w, 0) + 1
    corpus = {tuple(w): c for w, c in counts.items()}
    vocab = {ch for w in corpus for ch in w}
    for _ in range(n_merges):
        pairs = {}
        for w, c in corpus.items():
            for a, b in zip(w, w[1:]):
                pairs[a, b] = pairs.get((a, b), 0) + c
        if not pairs:
            break
        a, b = max(pairs, key=pairs.get)
        vocab.add(a + b)
        merged = {}
        for w, c in corpus.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == (a, b):
                    out.append(a + b)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            merged[tuple(out)] = c
        corpus = merged
    return vocab
def fertility(text, vocab):
    # TODO: the words of text, the same way the corpus is split
    ws = words(text)
    # TODO: how many tokens encode() makes across all of those words
    n_tokens = sum(len(encode(w, vocab)) for w in ws)
    # TODO: tokens per word
    return n_tokens / len(ws)

MN_VOCAB = train_bpe(CORPUS, 120)
print(f"base vocab ({len(BASE_VOCAB)} pieces)   {fertility(HELD_OUT, BASE_VOCAB):.2f} tokens per word")
print(f"Mongolian BPE ({len(MN_VOCAB)} pieces) {fertility(HELD_OUT, MN_VOCAB):.2f} tokens per word")

def token_counts(text, vocab):
    """How often each token appears when every word of text is encoded with vocab."""
    counts = {}
    for w in words(text):
        for t in encode(w, vocab):
            counts[t] = counts.get(t, 0) + 1
    return counts

COUNTS = token_counts(CORPUS, MN_VOCAB)

def pick_new_tokens(counts, base_vocab, k):
    # TODO: the tokens base_vocab doesn't have yet
    candidates = list(counts)
    # TODO: for each candidate, the base tokens it saves across the corpus
    saved = {t: counts[t] * (len(encode(t, base_vocab)) - 1) for t in candidates}
    # TODO: the k biggest savers, best first
    return sorted(saved, key=saved.get, reverse=True)[:k]

for k in (5, 10, 20):
    new = pick_new_tokens(COUNTS, BASE_VOCAB, k)
    print(f"base + {k:<2} new  {fertility(HELD_OUT, BASE_VOCAB | set(new)):.2f} tokens per word   last picks: {', '.join(new[-3:])}")
