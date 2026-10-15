import torch

# 81 names, enough to see the idea. On your machine you'll train on all 32,033 in names.txt.
words = """
emma olivia ava isabella sophia charlotte mia amelia harper evelyn abigail emily elizabeth mila
ella avery sofia camila aria scarlett victoria madison luna grace chloe penelope layla riley zoey
nora lily eleanor hannah lillian addison aubrey ellie stella natalie zoe liam noah oliver elijah
william james benjamin lucas henry alexander mason michael ethan daniel jacob logan jackson levi
sebastian mateo jack owen theodore aiden samuel joseph john david wyatt matthew luke asher carter
julian grayson leo jayden gabriel isaac lincoln anthony
""".split()

chars = "abcdefghijklmnopqrstuvwxyz"
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi["."] = 0
itos = {i: ch for ch, i in stoi.items()}

def count_bigrams(words):
    N = torch.zeros((27, 27), dtype=torch.int64)
    for w in words:
        # TODO: the characters of w, with "." added at both ends
        chs = "." + w + "."
        for ch1, ch2 in zip(chs, chs[1:]):
            # TODO: add one to the cell for this pair
            N[stoi[ch1], stoi[ch2]] += 1
    return N


N = count_bigrams(words)
print("pairs counted", N.sum().item())
print("most common first letter:", itos[N[0].argmax().item()])
