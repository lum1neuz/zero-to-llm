MESSAGES = [
    {"role": "system", "content": "Чи монгол хэлээр хариулдаг туслах."},
    {"role": "user", "content": "Монгол Улсын нийслэл аль хот вэ?"},
    {"role": "assistant", "content": "Монгол Улсын нийслэл бол Улаанбаатар хот."},
]


def render(messages):
    """Llama 3.1's chat template as (piece, role) pairs, one toy token per word.
    Header pieces have role None. Each message's <|eot_id|> belongs to its speaker."""
    pieces = [("<|begin_of_text|>", None)]
    for m in messages:
        pieces += [("<|start_header_id|>", None), (m["role"], None), ("<|end_header_id|>", None)]
        pieces += [(word, m["role"]) for word in m["content"].split()]
        pieces += [("<|eot_id|>", m["role"])]
    return pieces


pieces = render(MESSAGES)
VOCAB = {}
ids = torch.tensor([VOCAB.setdefault(p, len(VOCAB)) for p, _ in pieces])
roles = [r for _, r in pieces]

