def build_labels(ids, roles):
    # TODO: a copy of ids where every position the assistant didn't say is -100, the label the loss skips
    labels = ...
    return labels


def sft_loss(logits, labels):
    # TODO: the mean next-token loss, over only the positions whose label isn't -100
    return ...


labels = build_labels(ids, roles)
for (piece, role), label in zip(pieces, labels.tolist()):
    print(f"{piece:<22} {str(role):<10} {label}")
torch.manual_seed(0)
logits = torch.randn(len(ids), len(VOCAB))
print(f"loss {sft_loss(logits, labels).item():.4f} over {(labels[1:] != -100).sum().item()} assistant tokens")
