---
id: "5.3"
title: Speculative decoding
summary: Let a small draft model guess several tokens, check them all in one pass of the big model, and prove the output is unchanged, greedy and sampled.
phase: "5"
hours: 7
type: build
status: ready
protected: false
cut: 1
prerequisites:
  - "5.2"
papers:
  - title: Fast Inference from Transformers via Speculative Decoding
    authors: Leviathan et al.
    year: 2022
    url: https://arxiv.org/abs/2211.17192
quality_metric:
  id: output_match
  label: Outputs identical to the target model
  unit: "%"
  note: Greedy speculative decoding should reproduce the target alone exactly. The target scores k + 1 positions at once instead of one, which can round differently in fp16 and, rarely, flip a near-tie.
  inputs:
    - { id: match_pct, label: Greedy outputs identical to the target alone (%), hint: "50 prompts, 128 tokens, temperature 0" }
  compute: value
  reference: 100
  pass: { min: 96 }
systems_metric:
  id: speedup_x
  label: Decode speedup
  unit: ×
  note: One user, greedy, at the best k from your sweep.
  inputs:
    - { id: base_tps, label: Target alone (tok/s), hint: From spec_decode.py's baseline line }
    - { id: spec_tps, label: With speculative decoding (tok/s), hint: At your best k }
  compute: ratio
  reference: 1.41
  pass: { min: 1.15 }
quiz:
  - id: lossless
    q: You sample at temperature 0.8 with speculative decoding and the accept/reject rule. Compared with the target sampling alone, what changes in the output?
    answer: 0
    options:
      - text: Nothing. Each token follows the target's distribution exactly
        why: Keeping a draft with probability min(1, p/q) and replacing rejections from max(0, p − q) gives exactly p. That's the theorem in Leviathan et al. The draft only changes how fast you get there.
      - text: It drifts toward the draft, since most tokens come from the draft
        why: Kept drafts are thinned by p/q, and the replacements add back what the draft under-proposes. The two corrections cancel the draft's bias exactly.
      - text: It's exact at temperature 0 only. With sampling it's an approximation
        why: The rule is exact for sampling too. Greedy is the special case where p and q are one-hot, so "keep" means "the argmaxes agree".
  - id: bad_draft
    q: Your draft model's acceptance rate is α = 0.1. What happens?
    answer: 1
    options:
      - text: Quality drops, because bad drafts get into the output
        why: A draft only gets in when the target accepts it, and rejected ones are replaced from the target. The output is exactly the target's, whatever the draft does.
      - text: Outputs stay exact, but decoding is slower than without a draft
        why: Each round yields about 1.1 tokens but costs a target pass plus the draft's passes. At k = 1 and c = 0.15 that's 1.1 / 1.15, a slowdown.
      - text: Nothing. It falls back to plain decoding at the same speed
        why: Nothing falls back automatically. You still pay for every draft pass, and they hardly ever save a target pass.
  - id: why_cheap
    q: Why can the target check 5 drafted tokens in about the time it takes to generate 1?
    answer: 2
    options:
      - text: The target skips the positions the draft got right
        why: The target scores every drafted position. Agreement is decided after the pass, from its logits.
      - text: The check runs on the draft model, which is small
        why: The draft only proposes. The verdict has to come from the target, or the output wouldn't be the target's.
      - text: Decode is memory-bound, and one pass reads the weights once for all 6 positions
        why: As in 5.1, a step at batch 1 is dominated by streaming the weights. Scoring 6 positions instead of 1 adds a little math, which the GPU had spare.
---

## Goal

Make one user's decoding faster without changing a single output token. A small **draft** model guesses `k` tokens; the big **target** scores all of them in one forward pass; you keep the guesses it agrees with plus one token of its own. Build it greedy first and check it matches plain greedy, then write the accept/reject rule that keeps *sampled* outputs exact, and work out when it pays off. On the GPU, draft your Mongolian 8B with Llama 3.2 1B and find the best `k`.

## Learn

```card
type: concept
id: draft_verify
eyebrow: Draft, then verify
title: One target pass can check many guesses.
visual:
  kind: bars
  max: 4
  bars:
    - { label: k = 1, value: 1.75, text: "1.75" }
    - { label: k = 2, value: 2.31, text: "2.31" }
    - { label: k = 4, value: 3.05, text: "3.05" }
    - { label: k = 8, value: 3.70, text: "3.70" }
body: |
  At batch 1 the target is memory-bound: a pass costs about the same whether it scores 1 position or 5. So let a cheap **draft** model guess `k` tokens, run the target once over all of them, and keep the guesses up to the first one it disagrees with. Then add the target's own token: its correction there, or a free bonus if every guess survived. The bars show tokens per target pass when each guess survives with probability 0.75.
```

```card
type: predict
id: greedy_trace
eyebrow: Predict the output
title: The draft proposed three tokens. Which ones get appended?
code: |
  proposal = [5, 2, 7]
  target_choice = [5, 2, 1, 4]   # argmax of the target's rows n-1 … n+2
  out = []
  for i in range(4):
      out.append(target_choice[i])
      if i == 3 or target_choice[i] != proposal[i]:
          break
  print(out)
options:
  - text: "[5, 2, 1]"
    correct: true
    why: "The target agrees with 5 and 2, then picks 1 where the draft guessed 7. Append its 1 and stop: three tokens from one target pass."
  - text: "[5, 2, 7]"
    why: That keeps the draft's guess where the target disagreed. The output must be the target's choice, 1.
  - text: "[5, 2, 1, 4]"
    why: Row 3 was computed after the draft's 7, which never made it into the output, so its 4 answers the wrong question. Stop at the first disagreement.
  - text: "[5, 2]"
    why: Disagreeing still tells you the target's own token for that position, so use it. Every round adds at least one token.
hint: Walk the two lists together and stop at the first place they differ. Which list's token goes in at that place?
```

```card
type: calc
id: tokens_per_round
eyebrow: Do the math
title: Each guess survives with probability 0.8, and a round stops at the first rejection. How many tokens does a round of k = 3 yield on average?
given:
  - { label: Acceptance α, value: "0.8" }
  - { label: Drafted tokens k, value: "3" }
  - { label: Target's own token, value: always 1 }
unit: tokens
answer: { min: 2.94, max: 2.96 }
placeholder: "0.0"
why_ok: "1 + 0.8 + 0.64 + 0.512 = 2.952. Guess j counts only if guesses 1…j all survive (α^j), and the target adds 1. That's (1 − α^(k+1)) / (1 − α)."
why_wrong: "Guess 2 only counts if guess 1 survived too: 0.8 × 0.8. Add the chance that each guess counts, then 1 for the target's own token."
hint: "The chance guess j makes it is α^j. Sum those for j = 1, 2, 3, then add 1."
```

```card
type: fill_blank
id: accept_blanks
eyebrow: Fill the blanks
title: Complete the accept/reject rule for draft token x at position i.
code: |
  u = torch.rand(1, generator=gen)
  if u < [[0]] / [[1]]:
      out.append(x)
  else:
      residual = (p[i] - q[i]).clamp(min=[[2]])
      out.append(int(torch.multinomial(residual, 1, generator=gen)))
tokens: ["p[i, x]", "q[i, x]", "0", "1", "p[i].max()"]
answer: ["p[i, x]", "q[i, x]", "0"]
tips:
  "1": "clamp(min=1) lifts every weight to at least 1, so the replacement would be close to uniform."
  "p[i].max()": The rule compares the probabilities of the one drafted token x, not the row's best.
why_ok: "Keep x with probability min(1, p(x) / q(x)). On rejection, sample from what the target wants more than the draft offered: max(0, p − q)."
why_wrong: "The coin compares the target's probability of x with the draft's. The replacement keeps only the tokens where the target wants more than the draft gave."
hint: Target over draft, for the drafted token. A replacement weight can't be negative.
```

```card
type: quiz
ref: lossless
```

```card
type: quiz
ref: bad_draft
```

```card
type: quiz
ref: why_cheap
```

## Build steps

### Draft and verify, greedy

```challenge
id: spec_greedy
file: speculative.py
minutes: 45
start: { code: "import torch\n\n" }
setup:
  - say: "Two toy models over 8 tokens. Each is a bigram table, so a forward pass over a list of tokens returns next-token logits after every position, just like a transformer. The draft is the target's table, blurred."
    type: "V = 8\ng = torch.Generator().manual_seed(0)\nT = torch.randn(V, V, generator=g) * 2          # target: next-token logits after each token\nD = T + torch.randn(V, V, generator=g)          # draft: the same table, blurred\n\ncalls = {\"target\": 0, \"draft\": 0}\n\ndef target(tokens):\n    # one forward pass: next-token logits after every position, shape [len(tokens), V]\n    calls[\"target\"] += 1\n    return T[tokens]\n\ndef draft(tokens):\n    calls[\"draft\"] += 1\n    return D[tokens]\n\n"
    pause: 3.2
  - say: "Plain greedy calls the target once per new token. That's what you have to match, token for token."
    type: "def greedy(prompt, n_new):\n    tokens = list(prompt)\n    for _ in range(n_new):\n        tokens.append(int(target(tokens)[-1].argmax()))\n    return tokens\n\n"
    pause: 2.4
  - say: "Here's the stub. Each round the draft guesses k tokens, one target pass checks them all, and you keep what the target agrees with. The lines at the bottom compare with greedy and count target passes."
    type_file: spec_greedy/stub.py
    pause: 1.2
tests: spec_greedy/tests.py
solution: spec_greedy/solution.py
alternatives:
  - spec_greedy/alternatives/for_else.py
  - spec_greedy/alternatives/count_accepted.py
  - spec_greedy/alternatives/argmax_rows.py
hints:
  logic: |
    repeat k times:
      draft's best next token after tokens + proposal so far
        → add it to proposal
    one target pass over tokens + proposal → logits
    for i in 0 … k:
      choice = target's best token from row n − 1 + i
        → append choice to tokens
      i is k, or choice differs from proposal[i]?
        yes → end this round
        no  → the draft was right, check the next one
  tools: |
    the draft proposes k tokens, one at a time
      range(n)       counts 0 … n-1
      draft(toks)    next-token logits for every position
      t[-1]          the last row of t
      t.argmax()     index of the biggest value
      int(x)         tensor → plain Python int
      lst.append(x)  adds x to the end of lst
    one target pass
      target(toks)   logits, one row per position
      a + b          joins two lists
    check the rows
      len(lst)       how many items
      t[i]           row i of t
      break          leaves the loop
  nudge: Three moves per round. The draft guesses k tokens, the target scores all of them in one pass, and you keep the guesses it agrees with. Which row of the target's logits checks which guess?
  concept: "Build `proposal` one token at a time: `draft(tokens + proposal)[-1].argmax()` is the draft's next guess after the ones so far. Then call `target(tokens + proposal)` once. With `n = len(tokens)`, row `n - 1 + i` holds what the target itself would put at position `n + i`, where `proposal[i]` sits. Append the target's choice each time. If it matches the guess, you've kept the guess. If it doesn't, you've corrected it, so stop. If all k match, the last row is a free extra token."
  example: |
    Trace one round with `tokens = [3, 1]` (so `n = 2`) and `k = 2`. The draft proposes `proposal = [5, 2]`. The target gets `[3, 1, 5, 2]` and returns 4 rows. Row 0 isn't used.

    Suppose the target's best token in rows 1, 2 and 3 is `5`, `7` and `4`:

    - `i = 0`: row 1 says `5`. The draft guessed `5`. Append `5`, keep going.
    - `i = 1`: row 2 says `7`. The draft guessed `2`. Append `7`, the target's token, and stop.
    - `tokens = [3, 1, 5, 7]`. Two new tokens from one target pass.

    If rows 1, 2 and 3 had said `5`, `2` and `4` instead:

    - `i = 0` and `i = 1` agree, so append `5` and `2`.
    - `i = 2` is the extra row, so append `4` and stop.
    - `tokens = [3, 1, 5, 2, 4]`. Three new tokens from one pass, the most a round can give with `k = 2`.
  skeleton: |
    proposal = []
    for _ in range(_):
        proposal.append(int(draft(tokens + ________)[-1].argmax()))
    logits = ______(tokens + proposal)
    n = len(tokens)
    for i in range(k + _):
        choice = int(logits[n - _ + i].argmax())
        tokens.append(______)
        if i == k or choice != ________[i]:
            break
  solution: |
    proposal = []
    for _ in range(k):
        proposal.append(int(draft(tokens + proposal)[-1].argmax()))
    logits = target(tokens + proposal)
    n = len(tokens)
    for i in range(k + 1):
        choice = int(logits[n - 1 + i].argmax())
        tokens.append(choice)
        if i == k or choice != proposal[i]:
            break
on_solve: "The same 24 tokens as greedy, from 14 target passes instead of 24. Each round costs one target pass and yields between 1 and k + 1 tokens, and the output can't differ from greedy, because every token you keep is one the target chose. The draft only decides how many tokens each pass yields. Try k = 1 and k = 8 at the bottom."
mistakes:
  - { file: spec_greedy/mistakes/placeholders.py, fails: matches_greedy, says: "returns just the prompt" }
  - { file: spec_greedy/mistakes/keep_draft.py, fails: matches_greedy, says: "you kept the draft's token" }
  - { file: spec_greedy/mistakes/no_break.py, fails: matches_greedy, says: "Stop at the first disagreement" }
  - { file: spec_greedy/mistakes/off_by_one.py, fails: matches_greedy, says: "Row j of `logits`" }
  - { file: spec_greedy/mistakes/only_matches.py, fails: matches_greedy, says: "Every round must append at least one token" }
  - { file: spec_greedy/mistakes/per_position.py, fails: one_target_pass, says: "Calling the target per position" }
  - { file: spec_greedy/mistakes/no_bonus.py, fails: one_target_pass, says: "for free" }
  - { file: spec_greedy/mistakes/not_autoregressive.py, fails: one_target_pass, says: "Each draft token must see the ones before it" }
  - { file: spec_greedy/mistakes/target_drafts.py, fails: one_target_pass, says: "The draft is the cheap model" }
  - { file: spec_greedy/mistakes/no_main.py, fails: script_runs, says: "are missing" }
```

### Keep sampled outputs exact

```challenge
id: verify
file: spec_sample.py
minutes: 45
start: { code: "import torch\n\n" }
setup:
  - say: "With sampling, matching the argmax isn't enough: every token must follow the target's distribution. Here's a tiny case over 4 tokens. p is the target's probabilities at 3 positions, q the draft's at the 2 drafted ones."
    type: "# Target probabilities at 3 positions (k + 1 rows) and the draft's at the 2 drafted ones, over 4 tokens\np = torch.tensor([[0.10, 0.50, 0.30, 0.10],\n                  [0.25, 0.25, 0.25, 0.25],\n                  [0.70, 0.10, 0.10, 0.10]])\nq = torch.tensor([[0.40, 0.40, 0.10, 0.10],\n                  [0.10, 0.60, 0.20, 0.10]])\n\n"
    pause: 3.2
  - say: "The rule from Leviathan et al.: keep draft token x with probability min(1, p(x) over q(x)). If you reject it, sample a replacement from max(0, p minus q) and stop. If every draft survives, the target's last row gives a bonus token."
    pause: 4.6
  - say: "Here's the stub. Every random draw goes through gen, so a seed repeats a run exactly. The loop at the bottom runs 2,000 rounds with drafts sampled from q, and compares the first token's frequencies with p."
    type_file: verify/stub.py
    pause: 1.2
tests: verify/tests.py
solution: verify/solution.py
alternatives:
  - verify/alternatives/item_min.py
  - verify/alternatives/where_residual.py
  - verify/alternatives/maximum.py
hints:
  logic: |
    for each drafted token x at position i:
      random number < p ÷ q for x?   (keeps with chance min(1, p/q))
        yes → keep x, go to the next draft
        no  → fix = sample from (p − q), negatives set to 0
                → add fix to out
                → stop and return out
    every draft survived:
      bonus = sample from the target's last row of p
        → add bonus to out
        → return out
  tools: |
    the keep-or-reject coin
      torch.rand(1, generator=gen)  uniform number in [0, 1)
      t[i, x]        the entry at row i, column x
      plain arithmetic: /   and compare: a < b
    what the target wants more of
      t.clamp(min=lo)  raises values under lo up to lo
    sampling a token
      torch.multinomial(w, 1, generator=gen)
                     one index, chance in proportion to w
                     (w needn't sum to 1)
      int(x)         tensor → plain Python int
    the target's last row
      t[-1]          the last row of t
  nudge: "Three draws, all from gen: a coin that says keep with probability p(x) / q(x), a replacement from what the target wants more than the draft offered, and a bonus from the target's last row."
  concept: "`torch.rand(1, generator=gen)` is uniform in [0, 1), so `rand < p[i, x] / q[i, x]` is True with probability min(1, p/q). On a rejection, `(p[i] - q[i]).clamp(min=0)` keeps the tokens the draft under-proposes, and `torch.multinomial(weights, 1, generator=gen)` samples one of them (the weights needn't sum to 1). The bonus is `torch.multinomial(p[-1], 1, generator=gen)`. Wrap the samples in `int(...)`."
  example: |
    Trace one position over 3 tokens. Say `p[i] = [0.2, 0.5, 0.3]` and `q[i] = [0.5, 0.3, 0.2]`.

    - Draft token `x = 0`: `p / q = 0.2 / 0.5 = 0.4`. The draw is `rand < 0.4`: a draw of `0.35` keeps it, a draw of `0.7` rejects it.
    - Draft token `x = 1`: `p / q = 0.5 / 0.3 ≈ 1.67`. Every draw in [0, 1) is below that, so it is always kept.
    - On a rejection, `p[i] - q[i] = [-0.3, 0.2, 0.1]`. Clamped at 0: `[0, 0.2, 0.1]`. The weights add up to 0.3, which is fine.
    - Sampling from those weights gives token 1 about two times in three and token 2 about one time in three. It never gives token 0, the one just rejected.
    - If all drafts survive, the bonus comes from the last row of `p`, whatever the drafts were.
  skeleton: |
    keep = torch.rand(1, generator=gen) < p[i, x] / _______
    fix = int(torch.multinomial((p[i] - ____).clamp(min=_), 1, generator=gen))
    bonus = int(torch.multinomial(_____, 1, generator=gen))
  solution: |
    keep = torch.rand(1, generator=gen) < p[i, x] / q[i, x]
    fix = int(torch.multinomial((p[i] - q[i]).clamp(min=0), 1, generator=gen))
    bonus = int(torch.multinomial(p[-1], 1, generator=gen))
on_solve: "The first tokens come out at about [0.10, 0.49, 0.32, 0.09], the target's p, though every draft came from q = [0.4, 0.4, 0.1, 0.1]. That's the result in Leviathan et al.: keep with min(1, p/q), replace from max(0, p − q), and each token is distributed exactly as if the target had sampled alone. The draft only changes the speed."
mistakes:
  - { file: verify/mistakes/placeholders.py, fails: keeps_agreement, says: "`bonus` is still `...`" }
  - { file: verify/mistakes/no_bonus.py, fails: keeps_agreement, says: "nothing came after them" }
  - { file: verify/mistakes/keep_placeholder.py, fails: accept_rule, says: "If `keep` is still `...`" }
  - { file: verify/mistakes/deterministic.py, fails: accept_rule, says: "never survived" }
  - { file: verify/mistakes/inverted.py, fails: accept_rule, says: "upside down" }
  - { file: verify/mistakes/p_only.py, fails: accept_rule, says: "which is p(x) = 0.3 on its own" }
  - { file: verify/mistakes/unclamped.py, fails: accept_rule, says: "Clamp at 0 first" }
  - { file: verify/mistakes/fix_placeholder.py, fails: replacement, says: "`fix` is still `...`" }
  - { file: verify/mistakes/fix_from_p.py, fails: replacement, says: "a replacement must never pick it" }
  - { file: verify/mistakes/fix_from_q.py, fails: replacement, says: "That's the draft's row q" }
  - { file: verify/mistakes/fix_argmax.py, fails: replacement, says: "always token 1" }
  - { file: verify/mistakes/bonus_argmax.py, fails: bonus_sampled, says: "always token 2" }
  - { file: verify/mistakes/bonus_from_q.py, fails: bonus_sampled, says: "The bonus comes from the target's *last* row" }
  - { file: verify/mistakes/no_main.py, fails: script_runs, says: "is missing" }
```

### When does it pay off?

```challenge
id: speedup
file: spec_math.py
minutes: 20
start: { code: "" }
setup:
  - say: "Last piece: when is it worth it? Say each drafted token survives with probability alpha, independently, and a round stops at the first rejection. A draft pass costs c of a target pass."
    pause: 3.6
  - say: "Here's the stub. Fill in both functions, then read the sweep over k for a 1B draft and an 8B target."
    type_file: speedup/stub.py
    pause: 1.2
tests: speedup/tests.py
solution: speedup/solution.py
alternatives:
  - speedup/alternatives/sum_terms.py
  - speedup/alternatives/loop.py
  - speedup/alternatives/named_cost.py
hints:
  logic: |
    expected_tokens:
      draft j survives with chance alpha^j, for j = 1 … k
        → add them up
        → plus 1 for the token the target always adds
    speedup:
      cost = k draft passes at c each + 1 target pass
        → expected_tokens ÷ cost
        → return that
  tools: |
    chance that draft j survives
      x ** n         x to the power n
    add them up (a geometric series has a closed form)
      plain arithmetic: + - /
    cost of a round
      plain arithmetic: * +
    tokens per round over cost
      expected_tokens(alpha, k)  your function above
      plain arithmetic: /
  nudge: Draft j only counts if drafts 1 through j all survive. And a round's cost has two parts, the draft's k passes and the target's one.
  concept: "Draft j survives with probability α^j, so the drafts that survive average α + α² + … + α^k. The target always adds one token, a replacement or the bonus, giving 1 + α + … + α^k = (1 − α^(k+1)) / (1 − α). A round costs k draft passes at c each, plus one target pass: k·c + 1. Plain decoding makes 1 token per target pass, so the speedup is tokens per round over that cost."
  example: |
    Trace `alpha = 0.5`, `c = 0.2`, with `k = 2`:

    - Draft 1 survives with chance `0.5`, draft 2 with `0.25`. Plus the target's own token: 1 + 0.5 + 0.25 = `1.75`. The closed form gives (1 − 0.125) / 0.5 = `1.75`, the same.
    - Cost: 2 × 0.2 + 1 = `1.4` target passes.
    - `speedup` = 1.75 / 1.4 = `1.25`.

    With `k = 3`: tokens per round `1.875`, cost `1.6`, speedup about `1.17`. A third draft adds only 0.125 tokens but 0.2 of cost, so it makes things worse.
  skeleton: |
    return (1 - alpha ** (k + _)) / (1 - _____)
    return expected_tokens(alpha, k) / (k * _ + _)
  solution: |
    return (1 - alpha ** (k + 1)) / (1 - alpha)
    return expected_tokens(alpha, k) / (k * c + 1)
on_solve: "With α = 0.75 and c = 0.15 the best k is 4, at 1.91×. Past that, each extra draft is less likely to survive than it costs. Set c = 0.3 (a bigger draft) and the best k drops to 2. Set α = 0.5 and the best you can get is about 1.35×. Acceptance is everything, which is why the draft should be trained on the same kind of text as the target."
mistakes:
  - { file: speedup/mistakes/placeholders.py, fails: expected_tokens, says: "still returns `...`" }
  - { file: speedup/mistakes/no_target_token.py, fails: expected_tokens, says: "one token from the target itself" }
  - { file: speedup/mistakes/exponent_k.py, fails: expected_tokens, says: "You're one term short" }
  - { file: speedup/mistakes/linear.py, fails: expected_tokens, says: "a round stops at the first rejection" }
  - { file: speedup/mistakes/no_target_pass.py, fails: speedup, says: "You count only the draft passes" }
  - { file: speedup/mistakes/free_draft.py, fails: speedup, says: "Drafting isn't free" }
  - { file: speedup/mistakes/one_draft_pass.py, fails: speedup, says: "the draft runs once per drafted token" }
  - { file: speedup/mistakes/full_cost_drafts.py, fails: speedup, says: "A draft pass costs c" }
  - { file: speedup/mistakes/hardcoded.py, fails: speedup, says: "Use the arguments" }
  - { file: speedup/mistakes/no_main.py, fails: script_runs, says: "is missing" }
```

### Draft your 8B with a 1B

```replay
id: spec_8b
intro: Sweeping k with a real draft and target takes about 30 minutes on the GPU. Watch a sample run first so you know what to look for.
log: replays/spec-8b-1b.jsonl
speed: 40
recorded:
  hardware: RTX 3090 24 GB
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
summary:
  - { label: Best k on Mongolian, value: "2" }
  - { label: Decode speed, value: 46.8 → 66.0 tok/s (1.41×) }
  - { label: Greedy match, value: 50 / 50 prompts }
run_yourself: |
  # on your GPU machine, from the course repo
  python spec_decode.py --target runs/mn-8b-4a \
    --draft meta-llama/Llama-3.2-1B-Instruct \
    --k 1,2,3,4,6 --prompts mn_eval.jsonl --n 50 --max-tokens 128
```

### Record your metrics

```metrics
id: record
intro: One quality metric (is the output still the target's?) and one systems metric (how much faster one user gets tokens), from your own run.
```

## Hints

- The draft must share the target's tokenizer. Your 4A.5 model keeps Llama 3.1's vocabulary, so Llama 3.2 1B can draft for it. The 4B model's extended vocabulary has no small sibling: use prompt-lookup (n-gram) drafting there, which proposes tokens copied from the prompt and needs no draft model.
- α depends on the text. The 1B draft knows English far better than Mongolian, so it's accepted less often on Mongolian prompts. Fine-tuning the draft on your Mongolian corpus raises α, and with it the best k.
- When sampling, compute p and q with the same temperature and top-p, then use the rule from step 2.
- Speculative decoding helps most when the GPU has spare compute: one user, memory-bound decode. At large batch sizes the target is already compute-bound (5.1's roofline), and the extra positions stop being free.

## Done when

- [ ] Speculative greedy matches plain greedy with fewer target passes {#spec_greedy auto=challenge:spec_greedy}
- [ ] Your accept/reject rule reproduces the target's distribution {#verify auto=challenge:verify}
- [ ] You've swept k for your draft and target on the GPU {#best_k}
- [ ] Output match and speedup recorded {#metrics_recorded auto=metrics}
