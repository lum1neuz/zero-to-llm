---
id: "2.1"
title: Naive generation loop
summary: Write the loop that turns a model into a writer. Run it on the whole sequence, read the last row, pick, append. Start greedy, then add temperature.
phase: "2"
hours: 3
type: build
status: ready
protected: false
cut: null
prerequisites:
  - "1.3"
papers: []
quality_metric:
  id: sample_match
  label: Samples match the reference at temperature 0
  unit: "%"
  inputs:
    - { id: match_pct, label: Greedy tokens matching the reference (%), hint: "Your loop vs the course repo's generate(), 512 tokens, temperature 0" }
  compute: value
  reference: 100
  pass: { min: 99 }
systems_metric:
  id: tokens_per_s
  label: Generation speed, no cache
  unit: tok/s
  note: Your 30M model, 512 new tokens, greedy, no KV cache. Lesson 2.2 makes this faster.
  inputs:
    - { id: tok_s, label: tokens/s without a cache, hint: Printed at the end of the run }
  compute: value
  reference: 216
  pass: { min: 50 }
quiz:
  - id: last_row
    q: "`model(idx)` returns logits of shape [T, V]. Why does the loop read only `logits[-1]`?"
    answer: 1
    options:
      - text: The last row is the most accurate, because it has seen the most training
        why: Every row is trained equally. Row −1 is just the only one that predicts a token you don't have yet.
      - text: Row t predicts the token after position t, so only the last row predicts a new token
        why: Rows 0…T−2 predict tokens that are already in `idx`. Training needs every row, because every row has a target. At generation time they're wasted work, which 2.2 stops computing.
      - text: The other rows are padding
        why: There's no padding here. Every row is a real prediction, just for a position you already know.
  - id: greedy_loops
    q: "The tiny model's greedy output is \"the the the the…\". Why does greedy decoding fall into loops like this?"
    answer: 0
    options:
      - text: It always takes the single most likely token, so once a phrase makes itself likely again, nothing breaks the cycle
        why: Greedy has no randomness. If "the " makes "t" the top guess, it picks "t" every time it gets there. Sampling breaks the cycle, and 2.3's top-k and top-p keep the randomness sensible.
      - text: The model ran out of context window
        why: The tiny model has room for 64 positions and the loop starts repeating within the first ten. Greedy loops start long before any limit.
      - text: argmax has a bug with ties
        why: Exact ties are rare with float logits, and argmax takes the first one when they happen. The loop is the model's honest top choice, repeated.
  - id: low_temperature
    q: Logits are [2, 1, 0]. What does temperature 0.5 do to them before softmax?
    answer: 2
    options:
      - text: Halves them to [1, 0.5, 0], so the distribution gets flatter
        why: That's multiplying by 0.5. Temperature divides, and dividing by a number under 1 makes the logits bigger.
      - text: "Nothing: temperature only applies after softmax"
        why: After softmax, dividing by T only rescales the probabilities, and sampling renormalises them. Temperature has to act on the logits.
      - text: Doubles them to [4, 2, 0], so the top token gets more likely
        why: Dividing by 0.5 doubles the gaps. softmax goes from [0.67, 0.24, 0.09] to [0.87, 0.12, 0.02]. As T → 0 it turns into greedy.
---

## Goal

Write the loop that turns a model into a text generator. Run the model on the whole sequence, read the last position's logits, pick an id, append it, repeat. Start with **greedy** decoding, then add **temperature** sampling with a seeded generator. Then run the same loop on your TinyStories 30M model and measure how fast it goes without a cache.

## Learn

```card
type: concept
id: one_token_at_a_time
eyebrow: How text comes out
title: A model predicts one token. A loop writes the story.
visual:
  kind: bars
  max: 520
  bars:
    - { label: token 1, value: 5, text: "5" }
    - { label: token 128, value: 132, text: "132" }
    - { label: token 256, value: 260, text: "260" }
    - { label: token 512, value: 516, text: "516" }
body: |
  `model(idx)` returns logits of shape `[T, V]`, one row per position. Row t is the model's guess for the token *after* position t. To write text, run the model, read the **last** row, pick an id, put it on the end of `idx`, and go again. Without a cache, every step re-runs the model on the whole sequence. The bars show how many positions each step processes from a 5-token prompt.
```

```card
type: predict
id: predict_last_row
eyebrow: Predict the output
title: Which id does greedy pick next?
code: |
  logits = torch.tensor([[0.1, 2.0, 0.3],
                         [1.5, 0.2, 0.4],
                         [0.0, 0.1, 3.0]])   # [T=3, V=3]
  print(logits[-1].argmax().item())
options:
  - text: "2"
    correct: true
    why: "`logits[-1]` is the last row, [0.0, 0.1, 3.0]. Its largest logit, 3.0, sits at position 2, so the next token id is 2."
  - text: "1"
    why: That's the argmax of row 0, the prediction for the token after position 0. That token is already in the sequence.
  - text: "8"
    why: "That's `logits.argmax()` without `[-1]`: the index of 3.0 in the flattened 3×3 table. 8 isn't a token id here, since V is 3."
  - text: "3.0"
    why: That's the largest value. argmax returns its position, and the position is the token id.
hint: Find the last row first. Then find where its biggest number sits, counting from 0.
```

```card
type: reorder
id: sample_step_order
eyebrow: Put it in order
title: Arrange one step of sampling.
lines:
  - logits = model(idx)
  - last = logits[-1]
  - probs = F.softmax(last / temperature, dim=-1)
  - next_id = torch.multinomial(probs, 1, generator=g)
  - idx = torch.cat([idx, next_id])
shuffle: [3, 0, 4, 1, 2]
why_ok: "Run the model, keep the last row, turn it into probabilities, draw one id, append it. `multinomial` returns a 1-element tensor, so it joins `idx` directly."
why_wrong: "Something runs before its input exists. Follow the names: `last` needs `logits`, `probs` needs `last`, `next_id` needs `probs`."
hint: Follow the variable names. Each line can only use names that an earlier line created.
```

```card
type: fill_blank
id: temperature_blank
eyebrow: Fill the blanks
title: Complete temperature sampling.
code: |
  def sample(logits, temperature, generator):
      probs = F.softmax(logits [[0]] temperature, dim=-1)
      return torch.[[1]](probs, 1, generator=generator)
tokens: ["*", "/", argmax, multinomial, "+", randint]
answer: ["/", multinomial]
tips:
  "*": Multiplying by T < 1 shrinks the gaps between logits and flattens the distribution, the opposite of what low temperature should do.
  argmax: argmax always returns the top id. That's greedy, and the generator would never be used.
  "+": Adding the same number to every logit changes nothing after softmax.
  randint: randint draws every id equally often and ignores the model's probabilities.
why_ok: Dividing by T < 1 sharpens the distribution and T > 1 flattens it. multinomial then draws one id in proportion to `probs`, from the seeded generator.
why_wrong: "Check both blanks: temperature *divides* the logits, and the draw has to follow `probs`."
hint: Low temperature should make the top token *more* likely. Which operation makes the logits bigger when T is 0.5?
```

```card
type: calc
id: positions_no_cache
eyebrow: Do the math
title: Without a cache, how many positions does the model process?
given:
  - { label: Prompt, value: 4 tokens }
  - { label: New tokens, value: "60" }
  - { label: Each step runs on, value: the whole sequence so far }
unit: positions
answer: { min: 2010, max: 2010 }
placeholder: "0"
why_ok: "4 + 5 + … + 63 = 60 × (4 + 63) / 2 = 2010. Your script prints the same number. With a KV cache (2.2) it's 64: the 4 prompt positions once, then each of the 60 new tokens once."
why_wrong: Step 1 runs on 4 positions, step 2 on 5, and step 60 on 63. Add them up as (first + last) × count / 2.
hint: Step 1 sees the 4-token prompt. Each step after that sees one more token than the step before.
```

```card
type: quiz
ref: last_row
```

```card
type: quiz
ref: greedy_loops
```

```card
type: quiz
ref: low_temperature
```

## Build steps

### Write the greedy loop

```challenge
id: greedy_loop
file: generate.py
minutes: 30
start:
  code: |+
    import torch
    import torch.nn.functional as F

    # A tiny character-level model with fixed weights: a bigram table counted from one line of
    # text, plus one attention head, so each next character depends on the whole context.
    # It stands in for your 30M model: ids [T] in, logits [T, V] out.
    text = "the cat sat on the mat. the dog sat on the log. a cat and a dog ran to the mat. "
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    encode = lambda s: torch.tensor([stoi[c] for c in s])
    decode = lambda ids: "".join(chars[i] for i in ids.tolist())


    class TinyLM:
        def __init__(self, seed=0):
            g = torch.Generator().manual_seed(seed)
            V, C, CTX = len(chars), 16, 64
            self.C = C
            counts = torch.ones(V, V) * 0.1
            for a, b in zip(text, text[1:]):
                counts[stoi[a], stoi[b]] += 1
            self.bigram = (counts / counts.sum(dim=1, keepdim=True)).log()
            self.wte = torch.randn(V, C, generator=g)
            self.wpe = torch.randn(CTX, C, generator=g) * 0.5
            self.Wq = torch.randn(C, C, generator=g) / C**0.5
            self.Wk = torch.randn(C, C, generator=g) / C**0.5
            self.Wv = torch.randn(C, C, generator=g) / C**0.5
            self.Wo = torch.randn(C, V, generator=g) * 0.15
            self.positions = 0  # how many positions the model has processed, in total

        def __call__(self, idx):
            T = idx.shape[0]
            self.positions += T
            x = self.wte[idx] + self.wpe[:T]
            q, k, v = x @ self.Wq, x @ self.Wk, x @ self.Wv
            att = (q @ k.T) / self.C**0.5
            att = att.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf"))
            h = F.softmax(att, dim=-1) @ v
            return self.bigram[idx] + h @ self.Wo


    model = TinyLM()

setup:
  - say: At the top is a tiny character model with fixed weights. Like your 30M model, it takes a tensor of ids and returns logits, one row per position. Give it a prompt.
    type: "prompt = encode(\"the \")\n\n"
    pause: 2.6
  - say: "Row t of the logits is the model's guess for the token after position t. To write text: run the model, read the last row, pick an id, append it, and go again."
    pause: 4.0
  - say: "Here's the stub. Your turn: fill in greedy, then the three lines of the loop."
    type_file: greedy_loop/stub.py
    pause: 1.2
tests: greedy_loop/tests.py
solution: greedy_loop/solution.py
alternatives:
  - greedy_loop/alternatives/argmax_dim.py
  - greedy_loop/alternatives/torch_fns.py
  - greedy_loop/alternatives/tolist.py
  - greedy_loop/alternatives/softmax_first.py
hints:
  logic: |
    greedy(logits):
      → return the position of the largest logit
    generate:
      repeat max_new_tokens times:
        run the model on ALL of idx → logits, one row per position
        keep only the last row   (it predicts the next token)
        pick an id from that row → next_id
        make next_id a 1-element tensor
          → add it to the end of idx
      → return idx
  tools: |
    position of the largest logit
      t.argmax()          index of the biggest value in t
    run the model on everything so far
      model(ids)          logits, one row per position
    the last row
      t[-1]               the last item (row) of t
    add one id to the end
      t.view(n)           the same values reshaped to n items
      torch.cat([a, b])   joins tensors a and b end to end
  nudge: "Four blanks, one idea: the newest prediction lives in the last row. What do you feed the model, which row do you read, and how does the new id join the sequence?"
  concept: "`greedy` returns the *position* of the largest logit, which is the token id. In the loop, run the model on all of `idx`, hand `pick` the last row of `logits`, and join the old ids and the new one with `torch.cat`. argmax gives a 0-d tensor and `torch.cat` needs at least one dimension, so make it a 1-element tensor first."
  example: |
    Take a tiny model with 5 token ids that always predicts "(last id + 1) mod 5". The prompt is `idx = [2, 3]`.

    `greedy` on one row: `[0.1, 2.0, -1.0, 0.5, 0.0]`. The biggest value is `2.0`, at position `1`. So it returns `1`, not `2.0`.

    Step 1 of the loop:

    - The model sees `[2, 3]` and returns `logits` with shape `[2, 5]`: row 0 predicts what follows the `2`, row 1 predicts what follows the `3`.
    - Last row: its biggest logit is at position `4`, so `next_id` is `tensor(4)`, a 0-d tensor.
    - As a 1-element tensor it's `[4]`. Joined to the old ids: `idx = [2, 3, 4]`.

    Step 2: the model sees all 3 ids, `[2, 3, 4]`, and `logits` has shape `[3, 5]`. The last row peaks at `0` (4 + 1 wraps around), so `idx` grows to `[2, 3, 4, 0]`.
  skeleton: |
    return logits.______()
    logits = _____(idx)
    next_id = pick(logits[____])
    idx = torch.cat([idx, next_id.____(1)])
  solution: |
    return logits.argmax()
    logits = model(idx)
    next_id = pick(logits[-1])
    idx = torch.cat([idx, next_id.view(1)])
on_solve: "Solved. Your loop writes \"the the the…\", and it processed 2,010 positions to write 60 characters: step 60 re-read all 63 before it. Greedy found a loop it can't leave. Step 2 adds randomness."
mistakes:
  - { file: greedy_loop/mistakes/placeholders.py, fails: greedy_pick, says: "`greedy` still returns `...`" }
  - { file: greedy_loop/mistakes/max_value.py, fails: greedy_pick, says: "the largest *value*" }
  - { file: greedy_loop/mistakes/idx_placeholder.py, fails: returns_tokens, says: "`idx` is still the placeholder" }
  - { file: greedy_loop/mistakes/cat_0d.py, fails: returns_tokens, says: "it's a 0-d tensor" }
  - { file: greedy_loop/mistakes/replaced.py, fails: returns_tokens, says: "replaced by the new id instead of growing" }
  - { file: greedy_loop/mistakes/whole_logits_flat.py, fails: returns_tokens, says: "vocabulary has only 10 ids" }
  - { file: greedy_loop/mistakes/whole_logits_dim.py, fails: returns_tokens, says: "because `pick` got every row of `logits`" }
  - { file: greedy_loop/mistakes/last_token_only.py, fails: whole_context, says: "The model saw 1 token at each step" }
  - { file: greedy_loop/mistakes/first_row.py, fails: last_position, says: "the prediction made at position 0" }
  - { file: greedy_loop/mistakes/no_main.py, fails: script_runs, says: "are missing" }
```

### Add temperature

```challenge
id: temperature
file: generate.py
minutes: 25
start: { from: greedy_loop }
setup:
  - say: Greedy always takes the top id, so it can never leave a loop. Sampling draws the next id at random, in proportion to its probability, and temperature decides how adventurous the draw is.
    type_file: temperature/stub.py
    pause: 2.0
  - say: "The loop at the bottom reuses your generate with a different pick. Write sample, then compare the three temperatures."
    pause: 1.2
tests: temperature/tests.py
solution: temperature/solution.py
alternatives:
  - temperature/alternatives/methods.py
  - temperature/alternatives/num_samples.py
  - temperature/alternatives/exp_normalise.py
hints:
  logic: |
    divide logits by temperature
      (below 1 sharpens, above 1 flattens)
      → softmax into probs
      → draw one id at random, weighted by probs
          using the generator given, so a seed repeats
      → return the id
  tools: |
    divide by the temperature
      plain arithmetic: /   (acts on every logit)
    scores to probabilities
      F.softmax(t, dim=-1)  makes the last dimension add up to 1
    draw one id at random
      torch.multinomial(p, n, generator=g)
                            draws n ids, weighted by p,
                            using the random generator g
  nudge: "Two moves: reshape the distribution with the temperature, then roll the dice. Should temperature act on the logits or on the probabilities?"
  concept: "Divide the logits by the temperature *before* softmax. T < 1 stretches the gaps (sharper, closer to greedy), T > 1 shrinks them (flatter, more random). Then `torch.multinomial` draws one id in proportion to `probs`. Pass it the generator so a seed reproduces the draws."
  example: |
    Take `logits = [2.0, 1.0, 0.0, -1.0]` and see how the temperature reshapes the probabilities:

    - `T = 1.0`: nothing changes, `probs ≈ [0.644, 0.237, 0.087, 0.032]`.
    - `T = 0.5`: the logits become `[4.0, 2.0, 0.0, -2.0]`, and `probs ≈ [0.865, 0.117, 0.016, 0.002]`. The top id wins even more often, so it's close to greedy.
    - `T = 2.0`: the logits become `[1.0, 0.5, 0.0, -0.5]`, and `probs ≈ [0.455, 0.276, 0.167, 0.102]`. The choice is flatter, so unlikely ids show up more.

    Then the draw: with the `T = 1.0` probabilities, 800 draws give id `0` about 515 times, id `1` about 190 times, and so on. Each call returns a 1-element tensor holding one id.

    Check: if you divide *after* the softmax, `T = 0.5` and `T = 2.0` would give the same draws as `T = 1.0`. Printing your `probs` at each T shows you quickly if the temperature has any effect.
  skeleton: |
    probs = F.softmax(logits ___ temperature, dim=-1)
    return torch.___________(probs, 1, generator=_________)
  solution: |
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1, generator=generator)
on_solve: "Compare the three lines. T=0.5 still says \"the the\" a lot, T=1.0 makes word-like chunks, and T=1.5 drifts into noise. Run it again: the seed gives the same text every time. Lesson 2.3 adds top-k and top-p, which cut off the unlikely tail that T=1.5 wanders into."
mistakes:
  - { file: temperature/mistakes/placeholders.py, fails: returns_id, says: "`sample` still returns `...`" }
  - { file: temperature/mistakes/returns_probs.py, fails: returns_id, says: "the probabilities themselves" }
  - { file: temperature/mistakes/log_softmax.py, fails: returns_id, says: "log-probabilities" }
  - { file: temperature/mistakes/argmax.py, fails: uses_generator, says: "That's greedy decoding" }
  - { file: temperature/mistakes/no_generator.py, fails: uses_generator, says: "doesn't use the `generator` you were given" }
  - { file: temperature/mistakes/multiplied.py, fails: temperature_math, says: "You multiplied by the temperature" }
  - { file: temperature/mistakes/after_softmax.py, fails: temperature_math, says: "the temperature has no effect" }
  - { file: temperature/mistakes/no_main.py, fails: script_runs, says: "is missing" }
```

### Generate with your 30M model

```replay
id: generate_30m
intro: On your GPU, the same loop runs your TinyStories 30M model. Watch a recording first. Time per token climbs as the story grows, because every step re-reads everything before it.
log: replays/generate-30m.jsonl
speed: 40
recorded:
  hardware: RTX 3090
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: ms per token, by position
  flag_above: 6.0
  x_labels: ["1–64", "449–512"]
  caption: Each step re-runs the model on every earlier token, so later tokens cost more. With a KV cache (2.2) this climb mostly goes away.
summary:
  - { label: Speed, value: 216 tok/s }
  - { label: Positions processed, value: "133,376 for 512 tokens" }
  - { label: Greedy match, value: 100% }
run_yourself: |
  # on your machine or Colab, from the course repo
  python generate.py --ckpt runs/tinystories-30m/best.pt \
    --prompt "Once upon a time" --new-tokens 512 --temperature 0 --no-cache
```

### Record your metrics

```metrics
id: record
intro: One quality metric and one systems metric, from your own run. They're compared with the reference run and the pass limit.
```

## Hints

- On the GPU, wrap the loop in `with torch.no_grad():` (or `torch.inference_mode()`). Without it, autograd keeps every step's activations for a backward pass that never comes.
- Your 30M model has a fixed context, `block_size`. Once `idx` is longer than that, feed only the last `block_size` ids. The tiny model here has room for 64.
- Your 1.x model takes `[B, T]` and returns `[B, T, V]`, so its last row is `logits[:, -1, :]`. This lesson drops the batch dimension to keep shapes short. Lesson 2.4 brings it back.
- If your tokenizer has an end-of-text token, check `next_id` after each step and stop when it appears.

## Done when

- [ ] Your greedy loop reproduces the reference text {#greedy auto=challenge:greedy_loop}
- [ ] Temperature sampling draws from softmax(logits / T) with a seeded generator {#temperature auto=challenge:temperature}
- [ ] Your 30M model writes a story with your loop, and its match rate and tokens/s are recorded {#metrics_recorded auto=metrics}
