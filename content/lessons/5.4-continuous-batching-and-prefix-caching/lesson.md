---
id: "5.4"
title: Continuous batching and prefix caching
summary: Write the scheduler that lets requests join and leave the batch every step, then hash KV blocks so a shared system prompt is computed once, not once per user.
phase: "5"
hours: 3
type: build
status: ready
protected: false
cut: null
prerequisites:
  - "5.2"
  - "2.4"
fallbacks:
  "2.4": Start from the padded-batching starter in 5.4's folder of the course repo. It covers the attention-mask code 2.4 would have built.
papers: []
quality_metric:
  id: output_match
  label: Outputs unchanged under batching
  unit: "%"
  note: Greedy outputs with 16 requests in flight vs one at a time, 50 prompts. Batched kernels can round differently and flip a rare near-tie, so a point or two under 100% is normal.
  inputs:
    - { id: match_pct, label: Greedy outputs identical at batch 16 vs batch 1 (%), hint: From bench_prefix.py's batching check }
  compute: value
  reference: 96
  pass: { min: 90 }
systems_metric:
  id: ttft_ms
  label: Time to first token with prefix cache
  unit: ms
  note: p50 TTFT at 16 users with a 1,214-token shared system prompt. With prefix caching off it was 471 ms.
  inputs:
    - { id: ttft_ms, label: TTFT p50 with prefix caching (ms), hint: Run 2 of bench_prefix.py }
  compute: value
  reference: 58
  pass: { max: 150 }
quiz:
  - id: long_request
    q: "Static batching, a batch of 8: seven requests want 20 tokens and one wants 500. What happens to the seven short ones?"
    answer: 2
    options:
      - text: They finish at step 20, and new requests take their slots
        why: That's continuous batching. A static batch only changes when it's finished.
      - text: They slow the long request down, so it takes longer than 500 steps
        why: Each step makes one token for every sequence, so the long one still takes 500 steps. The cost is what happens to the slots.
      - text: They're done at step 20 but hold their slots for 480 more steps
        why: Their slots sit idle, and every waiting request waits for the 500-token one too. Continuous batching hands a slot to the next request the step it frees up.
  - id: why_chain
    q: Why does a block's hash include the hash of the block before it?
    answer: 0
    options:
      - text: A token's keys and values depend on every token before it
        why: Attention mixes in the whole past, so the same 16 tokens after a different prefix have different K and V. Chaining makes the hash stand for the entire prefix up to that block.
      - text: To make the hash table smaller
        why: It's the same number of entries either way, one per cached block. Chaining changes what a hash *means*, not how many there are.
      - text: So that blocks are evicted in order, oldest first
        why: Eviction is a separate policy (vLLM evicts least recently used). Chaining is about correctness, never reusing K/V that was computed after a different past.
  - id: what_prefix_speeds
    q: Prefix caching speeds up which number?
    answer: 1
    options:
      - text: Time per output token, because every new token is faster
        why: Decode still reads all the weights and every cached K/V once per token. Reusing blocks changes where they came from, not the per-token work.
      - text: Time to first token, because cached blocks skip prefill
        why: Prefill computes K/V for the whole prompt before the first token. Blocks already in the cache don't have to be computed again, so only the new part of the prompt is.
      - text: Neither. It only saves memory
        why: It does save memory (requests share blocks), but the big win is skipping the prefill compute for the shared prefix.
---

## Goal

Two scheduler tricks that make a serving engine fast under load. **Continuous batching**: instead of running a batch until its longest request is done, decide every step who's in, so finished requests leave and waiting ones take their slots at once. **Prefix caching**: hash each full KV block together with everything before it, so a system prompt shared by every user is prefilled once and its blocks are reused. Simulate both in the browser, then measure them on your vLLM server.

## Learn

```card
type: concept
id: continuous
eyebrow: Continuous batching
title: Decide who's in the batch every step, not every batch.
visual:
  kind: bars
  max: 14
  bars:
    - { label: static, value: 13.3, text: "13.3 steps" }
    - { label: continuous, value: 4.8, text: "4.8 steps" }
body: |
  A static batch runs until its longest request is done. Short requests finish early but keep their slots, and new arrivals wait for the whole batch. **Continuous batching** (iteration-level scheduling, from the Orca paper) checks every decode step: finished requests leave, waiting ones take their slots. PagedAttention (5.2) makes it cheap, because a newcomer only needs free blocks, not one contiguous chunk. The bars show mean latency for 6 requests and 3 slots, from the simulation you'll write.
```

```card
type: predict
id: who_joins
eyebrow: Predict the step
title: Two slots, continuous batching. A wants 3 tokens, B wants 1, C wants 2, and all arrive at step 0. When does C make its last token?
code: |
  # step 0: A and B run. B makes its only token and leaves.
  # step 1: C takes B's slot. A makes its 2nd token, C its 1st.
  # step 2: ?
options:
  - text: Step 2
    correct: true
    why: "C joined at step 1, the step after B left, and makes one token per step: its 2nd at step 2. A finishes at step 2 as well."
  - text: Step 4
    why: That's static batching. A and B hold both slots until A is done at step 2, so C runs at steps 3 and 4.
  - text: Step 1
    why: C makes one token per step. It joined at step 1, so its second token comes at step 2.
  - text: Step 3
    why: C didn't have to wait for A. B's slot freed up after step 0, and C took it at step 1.
hint: B frees its slot after step 0. From then on C makes one token per step.
```

```card
type: calc
id: prefill_saved
eyebrow: Do the math
title: A 1,210-token system prompt is cached in blocks of 16. A new request adds a 90-token question. How many tokens still need prefill?
given:
  - { label: System prompt, value: "1,210 tokens" }
  - { label: User question, value: "90 tokens" }
  - { label: Block size, value: "16" }
unit: tokens
answer: { min: 100, max: 100 }
placeholder: "0"
why_ok: "75 full blocks (1,200 tokens) are cached. Block 76 holds the last 10 system tokens and the first 6 of the question, so it's different for every user and never matches. That leaves 1,300 − 1,200 = 100 tokens to prefill instead of 1,300."
why_wrong: "Only full blocks are cached, and only if everything in them matches. How many full 16-token blocks fit in 1,210 tokens? The rest of the prompt, system and question, needs prefill."
hint: 1,210 // 16 full blocks are reusable. Everything after them gets computed.
```

```card
type: fill_blank
id: hash_chain
eyebrow: Fill the blanks
title: Hash each full block so the hash stands for the whole prefix.
code: |
  parent = None
  for i in range(len(tokens) // block_size):
      block = tuple(tokens[i * block_size:(i + 1) * block_size])
      parent = hash(([[0]], [[1]]))
      hashes.append(parent)
tokens: [parent, block, i, "tuple(tokens)"]
answer: [parent, block]
tips:
  i: The position alone isn't enough. Two prompts can have the same block 3 after different pasts, and their K/V differ.
  "tuple(tokens)": That's the whole prompt, so even block 0's hash would change whenever the user's question does. Hash only up to the end of this block.
why_ok: "Each hash folds in the previous hash and this block's tokens, so two prompts get the same hash for block i exactly when their first i + 1 blocks match."
why_wrong: "A block can be reused only if its own tokens and everything before them match. Which name carries everything before?"
hint: One blank carries the past, the other is this block's own tokens.
```

```card
type: quiz
ref: long_request
```

```card
type: quiz
ref: why_chain
```

```card
type: quiz
ref: what_prefix_speeds
```

## Build steps

### Schedule every step

```challenge
id: scheduler
file: batching.py
minutes: 35
start: { code: "" }
setup:
  - say: "Six requests: when each arrives, in decode steps, and how many tokens it wants. Request a wants 12. Most of the others want 2 to 4. There are 3 slots."
    type: "# Each request: when it arrives (in decode steps) and how many tokens it wants\nrequests = [\n    {\"id\": \"a\", \"arrive\": 0, \"out\": 12},\n    {\"id\": \"b\", \"arrive\": 0, \"out\": 3},\n    {\"id\": \"c\", \"arrive\": 1, \"out\": 2},\n    {\"id\": \"d\", \"arrive\": 2, \"out\": 4},\n    {\"id\": \"e\", \"arrive\": 2, \"out\": 3},\n    {\"id\": \"f\", \"arrive\": 5, \"out\": 2},\n]\nMAX_BATCH = 3\n\n"
    pause: 2.8
  - say: "Static batching, the way a simple generate loop works: fill a batch, run it until its longest request is done, then start the next batch."
    type: "def static_batching(requests, max_batch):\n    done, step = {}, 0\n    queue = sorted(requests, key=lambda r: r[\"arrive\"])\n    while queue:\n        batch, queue = queue[:max_batch], queue[max_batch:]\n        step = max(step, max(r[\"arrive\"] for r in batch))   # wait until the whole batch has arrived\n        step += max(r[\"out\"] for r in batch)                # run until its longest request is done\n        for r in batch:\n            done[r[\"id\"]] = step - 1                        # everyone is returned when the batch ends\n    return done\n\n"
    pause: 2.8
  - say: "Continuous batching decides every step. Here's the stub: admit, make a token, retire. The loop at the bottom compares both schedules."
    type_file: scheduler/stub.py
    pause: 1.2
tests: scheduler/tests.py
solution: scheduler/solution.py
alternatives:
  - scheduler/alternatives/countdown.py
  - scheduler/alternatives/explicit_loop.py
  - scheduler/alternatives/index_queue.py
hints:
  start:
    try: |
      waiting = sorted(requests, key=lambda r: r["arrive"])
      running = []
      print(waiting[0]["id"], waiting[0]["arrive"] <= 0, len(running) < MAX_BATCH)
    see: |
      a True True
    next: "That's the question the admit step asks at step 0: has the oldest waiting request arrived, and is a slot free? Both yes, so `a` moves from `waiting` into `running` with 0 tokens made. Admitting is a `while` that keeps asking this until one answer is no."
  logic: |
    each step:
      admit: while the oldest waiting request has arrived
             and running has a free slot (fewer than max_batch):
        move it from waiting into running, with 0 tokens made
      make: for each slot in running:
        made = made + 1
      retire: for each slot in running:
        made equals the request's out?
          yes → done[request id] = this step
          no  → nothing
      keep only the slots that still need tokens
    when nothing is waiting or running → return done
  tools: |
    admit while the oldest has arrived and a slot is free
      lst[0][key]          a field of the first item of lst
      len(lst)             how many items lst holds
      lst.pop(i)           removes and returns item i
      lst.append(x)        adds x at the end
    make one token for each slot
      for x in lst         visit every item
      slot[i] += 1         adds 1 to item i of a list
    record finished requests
      for a, b in lst      unpacks two items per entry
      d[key] = value       stores value under key in d
    keep only slots that still need tokens
      [x for x in lst if cond]   keeps items where cond holds
  nudge: Each step does three things in order. Let new requests into free slots, make one token for everyone running, and let finished requests go. Which of these does static batching never do mid-batch?
  concept: "`waiting` is sorted by arrival, so `waiting[0]` is the oldest. Admit in a `while` loop as long as it has arrived (`waiting[0][\"arrive\"] <= step`) and a slot is free (`len(running) < max_batch`): `running.append([waiting.pop(0), 0])`. Each slot is `[request, made]`, so add 1 to `slot[1]` for every slot. A request is done when `made == r[\"out\"]`: record `done[r[\"id\"]] = step`, then keep only the slots that still need tokens."
  example: |
    Trace it with 2 slots and three requests: `p` (arrives 0, wants 3), `q` (arrives 0, wants 1), `r` (arrives 1, wants 2).

    - Step 0: `p` and `q` are admitted, both make 1 token. `q` has `made == out`, so `done["q"] = 0`, and it leaves. `running` holds `p` with 1 made.
    - Step 1: `r` has arrived and a slot is free, so it joins. `p` makes its 2nd token, `r` its 1st. Nobody is done.
    - Step 2: `p` makes its 3rd, `r` its 2nd. Both are done: `done["p"] = 2`, `done["r"] = 2`.
    - Final `done`: `{"q": 0, "p": 2, "r": 2}`.

    Check: `q` freed its slot after one step, so `r` didn't have to wait for `p` to finish.
  skeleton: |
    while waiting and waiting[0]["arrive"] <= ____ and len(running) < _________:
        running.append([waiting.___(0), 0])
    for slot in running:
        slot[_] += 1
    for r, made in running:
        if made == r["___"]:
            done[r["id"]] = ____
    running = [slot for slot in running if slot[1] < slot[0]["___"]]
  solution: |
    while waiting and waiting[0]["arrive"] <= step and len(running) < max_batch:
        running.append([waiting.pop(0), 0])
    for slot in running:
        slot[1] += 1
    for r, made in running:
        if made == r["out"]:
            done[r["id"]] = step
    running = [slot for slot in running if slot[1] < slot[0]["out"]]
on_solve: "Same requests, same slots: the last token comes at step 11 instead of 16, and mean latency drops from 13.3 to 4.8 steps. b finishes at step 2 instead of waiting for a's 12 tokens, and its slot goes straight to d. That's iteration-level scheduling from the Orca paper, now the default in vLLM, TGI and SGLang."
mistakes:
  - { file: scheduler/mistakes/placeholders.py, fails: all_finish, says: "finished no requests" }
  - { file: scheduler/mistakes/never_remove.py, fails: all_finish, says: "still hold their slots" }
  - { file: scheduler/mistakes/ignore_arrival.py, fails: arrivals, says: "Admit a request only once" }
  - { file: scheduler/mistakes/no_limit.py, fails: max_batch, says: "but max_batch is" }
  - { file: scheduler/mistakes/drain_first.py, fails: schedule, says: "drained" }
  - { file: scheduler/mistakes/one_per_step.py, fails: schedule, says: "Only one request joins per step" }
  - { file: scheduler/mistakes/retire_first.py, fails: schedule, says: "one step late" }
  - { file: scheduler/mistakes/make_last.py, fails: schedule, says: "one step late" }
  - { file: scheduler/mistakes/no_main.py, fails: script_runs, says: "is missing" }
```

### Reuse a shared prefix

```challenge
id: prefix_cache
file: prefix_cache.py
minutes: 30
start: { code: "" }
setup:
  - say: "Every request starts with the same system prompt. Two users, blocks of 4 tokens so you can follow along."
    type: "BLOCK = 4\nsystem = [101, 7, 7, 42, 9, 13, 250, 3, 3, 88]   # a 10-token system prompt, shared by every request\nfirst = system + [5, 61, 17, 2, 40, 40]          # the first user's request\nsecond = system + [5, 61, 99, 8, 12]             # a second user, same system prompt\n\n"
    pause: 2.6
  - say: "A block's keys and values depend on every token before it, so a block can only be reused if its tokens and everything before them match. One chained hash per block captures exactly that."
    pause: 3.6
  - say: "Here's the stub. block_hashes chains the hashes, and cached_prefix_len counts how many leading tokens are already cached."
    type_file: prefix_cache/stub.py
    pause: 1.2
tests: prefix_cache/tests.py
solution: prefix_cache/solution.py
alternatives:
  - prefix_cache/alternatives/whole_prefix.py
  - prefix_cache/alternatives/while_walk.py
  - prefix_cache/alternatives/range_step.py
hints:
  start:
    try: |
      print(first[:BLOCK] == second[:BLOCK])
      print(first[8:12] == second[8:12])
      print(first[12:16], second[12:16])
    see: |
      True
      True
      [17, 2, 40, 40] [99, 8, 12]
    next: "Blocks 0 and 2 match in both prompts; block 3 doesn't, and the second prompt's last block has only 3 tokens, so it isn't full. `block_hashes` cuts `tokens` into full blocks with slices like these, one per loop pass, and turns each into one number to remember. Write the cutting loop first and print what it cuts."
  logic: |
    block_hashes:
      parent = nothing yet
      for each FULL block of block_size tokens (skip leftovers):
        turn the block into a tuple
        hash (parent, block) together → that is the new parent
        add it to hashes
      return hashes

    cached_prefix_len:
      n = 0
      for each block hash of the prompt, in order:
        is it in cache?
          no  → stop looking
          yes → n = n + block_size
      return n
  tools: |
    count the full blocks
      len(lst) // n        how many whole times n fits
      range(n)             0, 1, ... n-1
    cut out one block
      lst[i:j]             items i up to j of lst
      tuple(lst)           a fixed copy that Python can hash
    hash with the parent
      hash(x)              a number that stands for x
      lst.append(x)        adds x at the end
    look up in cache
      x in s, x not in s   is x one of the set's items
      break                leave the loop at once
      plain arithmetic: +=
  nudge: A block's KV depends on every token before it, so it can be reused only when its tokens and everything before them match. How can one hash stand for all of that?
  concept: "Chain the hashes: `parent = hash((parent, block))`, starting from `None`. Each hash then depends on the block's own tokens and, through `parent`, on every block before it. `block` must be a tuple, because Python won't hash a list, and only `len(tokens) // block_size` blocks are full. To look up, walk the hashes from the start, add `block_size` for each hit, and `break` at the first miss."
  example: |
    Trace it with `block_size = 4`. Prompt `A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]` has 10 tokens.

    - Full blocks: 10 // 4 = 2, so the blocks are `(1, 2, 3, 4)` and `(5, 6, 7, 8)`. The leftover `[9, 9]` is skipped.
    - `hashes` has 2 entries. `h0` covers block 0 alone. `h1` covers `h0` plus block 1, so it stands for all 8 tokens.
    - Put both in `cache`.

    Now look up `B = [1, 2, 3, 4, 5, 6, 0, 0, 7, 7, 7, 7]`:

    - Its blocks are `(1, 2, 3, 4)`, `(5, 6, 0, 0)`, `(7, 7, 7, 7)`: 3 hashes.
    - Block 0 matches `h0`, which is in `cache`, so `n = 4`.
    - Block 1 has different tokens, so its hash isn't in `cache`. Stop there.
    - The answer is `4`. The third block is never checked, even though it would also be a miss.

    Your own `hashes` will be big integers. Check the first one matches across `A` and `B`, and the second one doesn't.
  skeleton: |
    for i in range(len(tokens) __ block_size):
        block = _____(tokens[i * block_size:(i + 1) * block_size])
        parent = hash((______, block))
        hashes.append(parent)
    n = 0
    for h in block_hashes(tokens, block_size):
        if h ___ __ cache:
            _____
        n += __________
    return n
  solution: |
    for i in range(len(tokens) // block_size):
        block = tuple(tokens[i * block_size:(i + 1) * block_size])
        parent = hash((parent, block))
        hashes.append(parent)
    n = 0
    for h in block_hashes(tokens, block_size):
        if h not in cache:
            break
        n += block_size
    return n
on_solve: "The second request reuses 12 of its 15 tokens: the whole system prompt, plus `5, 61`, because both users' questions start the same way and block 2 ended up identical. Only 3 tokens need prefill. Real system prompts often run past 1,000 tokens, so prefix caching takes most of the prefill out of time to first token."
mistakes:
  - { file: prefix_cache/mistakes/placeholders.py, fails: full_blocks, says: "returns no hashes" }
  - { file: prefix_cache/mistakes/partial_block.py, fails: full_blocks, says: "still being filled" }
  - { file: prefix_cache/mistakes/list_block.py, fails: full_blocks, says: "Turn the block into a tuple" }
  - { file: prefix_cache/mistakes/whole_prompt.py, fails: shared_prefix, says: "nothing after it" }
  - { file: prefix_cache/mistakes/unchained.py, fails: chained, says: "include its parent's hash" }
  - { file: prefix_cache/mistakes/position_only.py, fails: chained, says: "include its parent's hash" }
  - { file: prefix_cache/mistakes/prefix_placeholder.py, fails: prefix_len, says: "returns nothing" }
  - { file: prefix_cache/mistakes/counts_blocks.py, fails: prefix_len, says: "that counts blocks" }
  - { file: prefix_cache/mistakes/no_stop.py, fails: prefix_len, says: "Stop at the first miss" }
  - { file: prefix_cache/mistakes/no_main.py, fails: script_runs, says: "are missing" }
```

### Measure it on your server

```replay
id: prefix_16_users
intro: Two benchmark runs and a batching check take about 30 minutes on the GPU. Watch a sample run first so you know what to look for.
log: replays/prefix-cache-16-users.jsonl
speed: 40
recorded:
  hardware: RTX 3090 24 GB
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
summary:
  - { label: TTFT p50 at 16 users, value: 471 → 58 ms }
  - { label: Prefill tokens, value: "271,400 → 32,900" }
  - { label: Batch 16 vs batch 1, value: 48 / 50 identical }
run_yourself: |
  # on your GPU machine, from the course repo
  python bench_prefix.py --model runs/mn-8b --system-prompt prompts/mn_system.txt \
    --users 16 --requests 200
```

### Record your metrics

```metrics
id: record
intro: One quality metric (does batching change the outputs?) and one systems metric (how fast the first token comes with a cached prefix), from your own run.
```

## Hints

- The prefix-caching flag has changed between vLLM versions, and recent versions turn it on by default. Check `vllm serve --help` for yours, and turn it off explicitly for the baseline run.
- Put shared content (the system prompt, few-shot examples) at the very start of the prompt and keep it byte-identical. One changed token early invalidates every block after it, because each hash chains through it.
- Continuous batching changes which requests share a forward pass, and batched kernels can round differently. A few greedy outputs may split from the batch-1 run at near-ties. A large mismatch means a bug, often in padding or attention masks.
- Real schedulers also split a long prompt's prefill into chunks (chunked prefill), so one newcomer's 2,000-token prompt doesn't stall everyone else's decode for a whole step.

## Done when

- [ ] Your scheduler beats static batching on mean latency {#scheduler auto=challenge:scheduler}
- [ ] Prefix hashes reuse the shared system prompt and nothing they shouldn't {#prefix_cache auto=challenge:prefix_cache}
- [ ] TTFT measured with prefix caching on and off on your server {#ttft_measured}
- [ ] Output match and TTFT recorded {#metrics_recorded auto=metrics}
