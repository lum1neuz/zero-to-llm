import torch
import torch.nn.functional as F

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

def build_labels(ids, roles):
    # TODO: a copy of ids where every position the assistant didn't say is -100, the label the loss skips
    labels = ids.clone()
    labels[torch.tensor([r != "assistant" for r in roles])] = -100
    return labels


def sft_loss(logits, labels):
    # TODO: the mean next-token loss, over only the positions whose label isn't -100
    return F.cross_entropy(logits, labels, ignore_index=-100)


labels = build_labels(ids, roles)
for (piece, role), label in zip(pieces, labels.tolist()):
    print(f"{piece:<22} {str(role):<10} {label}")
torch.manual_seed(0)
logits = torch.randn(len(ids), len(VOCAB))
print(f"loss {sft_loss(logits, labels).item():.4f} over {(labels[1:] != -100).sum().item()} assistant tokens")
