import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

# From 4B.1: the corpus, the stand-in for Llama 3.1's vocab, encode(), and your 20 picks.
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
HELD_OUT = """
Манай хүүхдүүд өглөө бүр сургуульдаа явдаг. Өвөл малчид гэртээ байдаг.
Тэр их сургуульд түүх заадаг. Бид зун хөдөө амьдардаг.
Улаанбаатар хотод олон хүн амьдардаг.
"""
LETTERS = "абвгдеёжзийклмноөпрстуүфхцчшщъыьэюя"
BASE_VOCAB = set(LETTERS) | set(
    "ал ар ан ах ай ий ин ол он ор от ов ул ур эр ер ен ст та то на но ни ра ро ри ли ла ло "
    "да до ди ба ма ха хо га го де не те ре ле ам ат ад ак эн эл аа ээ уу оо ын ой "
    "сто тор ова ани анд ата ира ора ана ола ага".split())
NEW_TOKENS = ["хүүхдүүд", "байна", "амьдардаг", "сургууль", "монгол", "байдаг", "малчид",
              "дэг", "улаанбаатар", "тэ", "нийслэл", "өглөө", "манай", "хүйтэн", "зун",
              "нүү", "морь", "бид", "хөд", "маань"]


def words(text):
    return re.findall(r"\w+", text.lower())


def encode(word, vocab):
    toks, i = [], 0
    while i < len(word):
        j = len(word)
        while j > i + 1 and word[i:j] not in vocab:
            j -= 1
        toks.append(word[i:j])
        i = j
    return toks


BASE = [" "] + sorted(BASE_VOCAB)  # id 0 is the gap between words
EXT = BASE + NEW_TOKENS            # new tokens take the next ids, after every old one


def to_ids(text, tokens):
    """Token ids for text, with id 0 before every word."""
    vocab, index = set(tokens), {t: i for i, t in enumerate(tokens)}
    ids = []
    for w in words(text):
        ids.append(0)
        ids += [index[t] for t in encode(w, vocab)]
    return torch.tensor(ids)


def pretrain(steps=150):
    """A stand-in for Llama 3.1: a bigram model, an embedding and an LM head with nothing in
    between, trained on the corpus with the base vocab. Takes about a second."""
    ids = to_ids(CORPUS, BASE)
    xs, ys = ids[:-1], ids[1:]
    torch.manual_seed(0)
    model = nn.Sequential(nn.Embedding(len(BASE), 16), nn.Linear(16, len(BASE), bias=False))
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(steps):
        loss = F.cross_entropy(model(xs), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
    # E is the embedding, [vocab, 16]. H is the LM head's weight, also [vocab, 16].
    return model[0].weight.detach(), model[1].weight.detach()


E, H = pretrain()
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
