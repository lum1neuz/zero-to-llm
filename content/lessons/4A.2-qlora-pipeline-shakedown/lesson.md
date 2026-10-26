---
id: "4A.2"
title: QLoRA pipeline shakedown
summary: Work out where 12 GB goes when you fine-tune Llama 3.1 8B in 4 bits, then run a short QLoRA job end to end before the real one.
phase: "4A"
hours: 5
type: run
status: ready
protected: false
cut: null
prerequisites:
  - 4A.1
  - "3.5"
papers:
  - title: QLoRA
    authors: Dettmers et al.
    year: 2023
    url: https://arxiv.org/abs/2305.14314
quality_metric:
  id: val_loss
  label: Shakedown validation loss
  unit: nats
  note: 200 steps over 6,000 Alpaca-cleaned examples packed into 2048-token sequences (one epoch), eval on 200 held out. You're testing the pipeline, not the model. Anything that falls steadily and ends near the reference is healthy.
  inputs:
    - { id: val_loss, label: Eval loss after 200 steps (nats), hint: The eval line at the end of your run }
  compute: value
  reference: 1.12
  pass: { max: 1.3 }
systems_metric:
  id: peak_vram_gb
  label: Peak VRAM
  unit: GB
  note: torch.cuda.max_memory_reserved() / 1e9 at the end of training. It leaves out the ~0.5 GB CUDA context, so nvidia-smi shows a bit more.
  inputs:
    - { id: peak_vram_gb, label: Peak reserved memory (GB), hint: The run prints it last }
  compute: value
  reference: 9.9
  pass: { max: 12.0 }
quiz:
  - id: paged_optimizer
    q: A paged optimizer is training your adapters when one step needs more VRAM than is free. What does it do?
    answer: 1
    options:
      - text: Compresses Adam's moments to 8 bits until the step fits
        why: That's 8-bit Adam, a different trick that's always on. Paging doesn't change any numbers.
      - text: Moves pages of optimizer state to CPU RAM and brings them back when the update runs
        why: The state lives in CUDA unified memory, so under pressure the driver pages it out to CPU RAM instead of raising out-of-memory. Slower for that step, but the run survives the spike.
      - text: Skips that step and moves on to the next batch
        why: Nothing gets skipped. Every step still updates the adapters. Paging only changes where the optimizer state sits for a moment.
  - id: compute_dtype
    q: The base weights sit in VRAM as 4-bit NF4. What precision do the matmuls run in?
    answer: 0
    options:
      - text: bf16. Each block of weights is dequantized just before its matmul
        why: That's `bnb_4bit_compute_dtype=torch.bfloat16`. Weights are stored in 4 bits to save memory and expanded to bf16 for the arithmetic, then thrown away.
      - text: 4-bit integer matmuls, which is why it's fast
        why: There are no 4-bit matmuls here. QLoRA saves memory, not compute. It's a bit slower than bf16 LoRA because of the dequantizing.
      - text: fp32, to keep the gradients accurate
        why: The adapters' optimizer state is fp32, but the forward and backward matmuls run in bf16. fp32 would double the activation memory.
  - id: frozen_backprop
    q: The 4-bit base is frozen. Does the backward pass still go through it?
    answer: 2
    options:
      - text: No. Gradients stop at frozen layers, which is where the memory saving comes from
        why: Then only the last layer's adapters would get a gradient. The adapters in layer 1 need the gradient to travel back through layers 32 to 2.
      - text: Yes, and it updates the NF4 weights a little too
        why: The base never changes. There's no optimizer state for it and no fp32 copy to update.
      - text: Yes. Backprop passes through every frozen weight to reach the adapters, but never updates them
        why: Gradients flow through the dequantized base weights to the activations below. Nothing stores a gradient for the base weights themselves, which is the saving.
---

## Goal

Fine-tune Llama 3.1 8B on a 12 GB GPU. The base model goes to 4-bit **NF4** with double quantization and stays frozen, **LoRA** adapters train on top in higher precision, and a **paged optimizer** rides out memory spikes. First do the VRAM arithmetic yourself: what the 4-bit weights cost, what the adapters and their optimizer state cost, and the part that surprises people, the logits. Then shake the pipeline down with a 200-step run on a small English dataset, so the long Mongolian run in 4A.4 doesn't die at hour three.

## Learn

```card
type: concept
id: qlora_pieces
eyebrow: The idea
title: Three tricks fit an 8B fine-tune in 12 GB.
visual:
  kind: bars
  max: 21
  limit: { value: 12, label: 12 GB VRAM }
  bars:
    - { label: LoRA, value: 20.4, text: 20.4 GB }
    - { label: QLoRA, value: 10.1, text: 10.1 GB }
body: |
  A full fine-tune of 8B parameters needs about 16 bytes each for weights, gradients and Adam, well over 100 GB. LoRA drops the base model's gradients and optimizer state, but a bf16 base is still 16 GB. **QLoRA** stores the frozen base in **NF4** (4 bits, plus a small scale per block of 64), trains LoRA adapters on top, and uses a **paged optimizer** so a memory spike pages optimizer state to CPU RAM instead of crashing.
```

```card
type: calc
id: dq_bits
eyebrow: Do the math
title: With double quantization, how many bits does each NF4 weight really cost?
given:
  - { label: NF4 weight, value: 4 bits }
  - { label: One absmax per, value: "64 weights, stored in 8 bits" }
  - { label: One fp32 constant per, value: 256 absmaxes }
unit: bits
answer: { min: 4.12, max: 4.13 }
placeholder: "0.00"
why_ok: 4 + 8 / 64 + 32 / (64 × 256) = 4.127 bits. Without double quantization each absmax is fp32, so 4 + 32 / 64 = 4.5 bits. On the 7B weights in Llama's Linear layers that's 0.33 GB saved.
why_wrong: Spread each piece over the weights that share it. An 8-bit absmax is shared by 64 weights, and one fp32 constant by 64 × 256 of them.
hint: 8 / 64 for the absmax, 32 / (64 × 256) for the constant, plus the weight's own 4 bits.
```

```card
type: predict
id: logits_memory
eyebrow: Predict the output
title: Llama 3.1's vocabulary has 128,256 tokens. What prints?
code: |
  seq, vocab = 2048, 128256
  logits = torch.empty(seq, vocab, dtype=torch.float32)
  print(f"{logits.nbytes / 1e9:.2f} GB")
options:
  - text: "0.02 GB"
    why: That would be 2048 × 4096 hidden numbers in fp16. The output head widens each token from 4096 to 128,256 numbers.
  - text: "0.53 GB"
    why: That's the bf16 size. The loss upcasts the logits to fp32 first, so it's 4 bytes each.
  - text: "1.05 GB"
    correct: true
    why: 2048 × 128,256 × 4 bytes = 1.05 GB, for one sequence of 2048 tokens. That's a sixth of the 4-bit model, just to score the next token. Batch 4 would need 4.2 GB.
  - text: "4.20 GB"
    why: That's a batch of 4. Here there's one sequence.
hint: Multiply the three numbers, seq × vocab × 4 bytes.
```

```card
type: fill_blank
id: bnb_config
eyebrow: Fill the blanks
title: Load the base model the QLoRA way.
code: |
  bnb = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type=[[0]],
      bnb_4bit_use_double_quant=[[1]],
      bnb_4bit_compute_dtype=[[2]],
  )
tokens: ['"nf4"', '"fp4"', "True", "False", torch.bfloat16, torch.float32]
answer: ['"nf4"', "True", torch.bfloat16]
tips:
  '"fp4"': "FP4 spreads its 16 levels like a tiny float. NF4 puts them at the quantiles of a normal distribution, which suits roughly normal weights better (3.5)."
  "False": Double quantization shrinks the absmaxes from 32 bits to 8, about 0.37 bits per weight for free.
  torch.float32: The 4-bit weights get expanded for each matmul. fp32 would double the activation memory and run slower than bf16.
why_ok: "NF4 for the 16 levels, double quantization for the scales, and bf16 for the arithmetic. This is the configuration from the QLoRA paper."
why_wrong: "Check each line against the paper's recipe: which 4-bit grid suits normally distributed weights, do the scales get quantized too, and what precision do the matmuls run in?"
hint: The paper's name for its 4-bit grid is NormalFloat.
```

```card
type: reorder
id: qlora_setup
eyebrow: Put it in order
title: Arrange the QLoRA setup.
lines:
  - model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb)
  - model = prepare_model_for_kbit_training(model)
  - model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear"))
  - trainer = Trainer(model=model, args=args, train_dataset=train_ds)
  - trainer.train()
shuffle: [2, 4, 0, 3, 1]
why_ok: "Load in 4 bits, prepare (freeze the base, turn on gradient checkpointing, keep the norms in fp32), add the adapters, then train. Preparing after `get_peft_model` would freeze the adapters too."
why_wrong: "Two lines look swappable but aren't. `prepare_model_for_kbit_training` freezes every parameter it finds, so it has to run before the adapters exist."
hint: The adapters have to be the last thing added before training, or something freezes them.
```

```card
type: quiz
ref: paged_optimizer
```

```card
type: quiz
ref: compute_dtype
```

```card
type: quiz
ref: frozen_backprop
```

## Build steps

### What do 8B weights cost in NF4?

```challenge
id: nf4_memory
file: vram.py
minutes: 20
start: { code: "# VRAM budget for a QLoRA fine-tune of Llama 3.1 8B on a 12 GB GPU\n\n" }
setup:
  - say: "Here's Llama 3.1 8B in numbers: 32 blocks, each with seven Linear layers. The key and value projections are narrow, 1024 wide, because 32 query heads share 8 key-value heads."
    type: "# Llama 3.1 8B\nLAYERS = 32\nHIDDEN, INTER, KV = 4096, 14336, 1024\nVOCAB = 128256\n\n# (in, out) of the seven Linear layers in every block\nSHAPES = [(HIDDEN, HIDDEN), (HIDDEN, KV), (HIDDEN, KV), (HIDDEN, HIDDEN),   # q, k, v, o\n          (HIDDEN, INTER), (HIDDEN, INTER), (INTER, HIDDEN)]                # gate, up, down\nn_linear = LAYERS * sum(i * o for i, o in SHAPES)   # quantized to NF4\nn_embed = 2 * VOCAB * HIDDEN                        # embed_tokens and lm_head stay bf16\nprint(f\"{(n_linear + n_embed) / 1e9:.2f}B parameters, {n_linear / 1e9:.2f}B of them in Linear layers\")\n\n"
    pause: 3.4
  - say: "bitsandbytes quantizes only the Linear layers. The embedding and the output head stay bf16, a billion weights between them."
    pause: 3.0
  - say: "Here's the stub. Your turn: work out the absmax overhead per weight, with and without double quantization, then the total."
    type_file: nf4_memory/stub.py
    pause: 1.2
tests: nf4_memory/tests.py
solution: nf4_memory/solution.py
alternatives:
  - nf4_memory/alternatives/if_else.py
  - nf4_memory/alternatives/bits_first.py
hints:
  start:
    try: |
      print(n_linear, n_embed)
      print(32 / BLOCK)
      print(n_linear * 4 / 8 / 1e9)
    see: |
      6979321856 1050673152
      0.5
      3.489660928
    next: "That's the two weight counts, the absmax cost per weight when every 64 weights share one fp32 absmax (32 / 64 bits), and the 4-bit weights alone in GB. `scale_bits` is that 0.5 in the plain case, and double quantization makes it smaller by storing each absmax in 8 bits. Get the plain total working first, then the double-quant version."
  logic: |
    scale_bits = absmax overhead per weight, in bits
    double_quant on?
      no  → one fp32 absmax (32 bits) per BLOCK weights
              → scale_bits = 32 ÷ BLOCK
      yes → an 8-bit absmax per BLOCK weights
            + one fp32 constant per BLOCK × DQ_BLOCK weights
              → scale_bits = the two shares added
    quantized weight = 4 bits + scale_bits   → ÷ 8 for bytes
    bf16 weight = 2 bytes
    total_bytes = n_quant × its bytes + n_bf16 × its bytes
      → return total_bytes ÷ 1e9
  tools: |
    absmax overhead per weight
      plain arithmetic: / +    spread a cost over n weights
      a if cond else b         picks one of two values
    bytes for the quantized weights
      plain arithmetic: + * /  bits to bytes, count × size
    bf16 weights and the total
      plain arithmetic: * +    count × 2 bytes, then add
  nudge: Every NF4 weight pays 4 bits for itself plus its share of the block's scale. Spread each scale over the weights that share it.
  concept: "Without double quantization each block of 64 weights has one fp32 absmax, 32 / 64 = 0.5 bits per weight. With it, the absmax is stored in 8 bits (8 / 64) and every 256 absmaxes share one fp32 constant (32 / (64 × 256)). A quantized weight costs (4 + scale_bits) / 8 bytes, a bf16 weight 2 bytes."
  example: |
    Trace it on one full double-quant group: `BLOCK * DQ_BLOCK = 16384` weights, 256 blocks of 64.

    - No double quant: 16384 weights × (4 + 0.5) bits = 73728 bits = `9216` bytes.
    - Double quant: the 256 absmaxes cost 256 × 8 = 2048 bits, plus one fp32 constant, 32 bits. Weights themselves: 16384 × 4 = 65536 bits. Total 67616 bits = `8452` bytes.
    - Per weight that's 4.5 bits without, and ≈ `4.127` bits with.
    - 16384 bf16 weights: 16384 × 2 = `32768` bytes.

    Check: for 1 billion weights the tests expect `0.5625` GB without double quant, ≈ `0.5159` GB with it, and `2.0` GB for bf16.
  skeleton: |
    scale_bits = ___ / BLOCK + ___ / (BLOCK * DQ_BLOCK) if double_quant else ___ / BLOCK
    total_bytes = n_quant * (___ + scale_bits) / ___ + n_bf16 * ___
  solution: |
    scale_bits = 8 / BLOCK + 32 / (BLOCK * DQ_BLOCK) if double_quant else 32 / BLOCK
    total_bytes = n_quant * (4 + scale_bits) / 8 + n_bf16 * 2
on_solve: "5.70 GB instead of 16.06. Double quantization saved 0.33 GB, and over a third of what's left is the 1.05B bf16 embedding weights. Step 2 is unlocked. Add everything that training puts on top."
mistakes:
  - { file: nf4_memory/mistakes/placeholders.py, fails: absmax_overhead, says: "`total_bytes` is still `...`" }
  - { file: nf4_memory/mistakes/scale_placeholder.py, fails: absmax_overhead, says: "`scale_bits` is still `...`" }
  - { file: nf4_memory/mistakes/no_overhead.py, fails: absmax_overhead, says: "so the absmaxes cost nothing" }
  - { file: nf4_memory/mistakes/fp16_absmax.py, fails: absmax_overhead, says: "Your absmax costs 16 bits per block" }
  - { file: nf4_memory/mistakes/bits_not_bytes.py, fails: absmax_overhead, says: "How many bits in a byte?" }
  - { file: nf4_memory/mistakes/dq_ignored.py, fails: double_quant, says: "Does `double_quant` change anything" }
  - { file: nf4_memory/mistakes/dq_no_constant.py, fails: double_quant, says: "You left out the fp32 constants" }
  - { file: nf4_memory/mistakes/dq_constant_only.py, fails: double_quant, says: "You counted only the fp32 constants" }
  - { file: nf4_memory/mistakes/bf16_fp32.py, fails: bf16_part, says: "That's fp32" }
  - { file: nf4_memory/mistakes/bf16_dropped.py, fails: bf16_part, says: "Add them in" }
  - { file: nf4_memory/mistakes/bf16_quantized.py, fails: bf16_part, says: "You quantized the bf16 weights too" }
```

### Budget the whole fine-tune

```challenge
id: lora_budget
file: vram.py
minutes: 25
start: { from: nf4_memory }
setup:
  - say: "The weights fit with room to spare. Training adds three things: the adapters with their gradients and optimizer state, the activations kept for backward, and the logits."
    pause: 3.6
  - say: "Here's the stub and the budget table. Gradient checkpointing keeps one hidden state per layer and recomputes the rest, so those lines are filled in. Write the three functions."
    type_file: lora_budget/stub.py
    pause: 1.2
tests: lora_budget/tests.py
solution: lora_budget/solution.py
alternatives:
  - lora_budget/alternatives/loop.py
  - lora_budget/alternatives/spelled_out.py
hints:
  start:
    try: |
      i, o = SHAPES[1]
      print(i, o)
      print(16 * i, o * 16)
    see: |
      4096 1024
      65536 16384
    next: "That's the key projection, 4096 in and 1024 out. With `r = 16`, `A` holds 16 × 4096 numbers and `B` holds 1024 × 16. `lora_params` adds those two for each of the seven Linear layers of a block, then multiplies by the number of layers. Do it for this one layer, then ask how to cover all seven."
  logic: |
    lora_params:
      for each (in, out) Linear layer in shapes:
        A holds r × in numbers, B holds out × r numbers
          → add them
      sum over the layers in a block → times layers
    adapter_gb:
      each trainable number keeps weight, gradient, Adam m, Adam v
        → 4 fp32 values = 16 bytes
      → count × bytes each ÷ 1e9 → return GB
    logits_gb:
      one fp32 (4 bytes) per vocab entry, for every token
      tokens = seq × batch → return GB
  tools: |
    A and B sizes for every layer
      sum(expr for i, o in seq)   adds one value per pair
      plain arithmetic: * +       sizes of A and B, times layers
    optimizer state in GB
      plain arithmetic: * /       bytes per number, bytes to GB
    logits in GB
      plain arithmetic: * /       numbers × 4 bytes, bytes to GB
  nudge: "Three counts: the numbers in A and B across every adapted layer, the bytes training keeps per trainable number, and the size of the logits."
  concept: "Per adapted Linear, A has r × in numbers and B has out × r, so r × (in + out), times every block. Each trainable number keeps four fp32 values while training: the weight, its gradient, and Adam's m and v, 16 bytes. The logits are one fp32 number per vocabulary entry for every token in the batch."
  example: |
    Trace the small case the tests use: `shapes = [(4, 6), (6, 3)]`, `r = 2`, `layers = 3`.

    - Layer `(4, 6)`: A is 2 × 4 = 8, B is 6 × 2 = 12. Together `20`.
    - Layer `(6, 3)`: A is 2 × 6 = 12, B is 3 × 2 = 6. Together `18`.
    - One block: 20 + 18 = 38. Three blocks: `114`.
    - `adapter_gb(10**9)`: a billion numbers × 16 bytes = `16.0` GB.
    - `logits_gb(1000, 50_000, batch=3)`: 1000 × 50,000 × 3 = 150 million numbers × 4 bytes = 600 million bytes = `0.6` GB.

    Mind the traps: in × out (126) is the full weight matrix, not A plus B, and 38 is a single block.
  skeleton: |
    return ______ * sum(r * (___ + ___) for i, o in shapes)
    return n_trainable * ___ / 1e9
    return seq * _____ * _____ * ___ / 1e9
  solution: |
    return layers * sum(r * (i + o) for i, o in shapes)
    return n_trainable * 16 / 1e9
    return seq * vocab * batch * 4 / 1e9
on_solve: "About 10.1 GB, so it fits, with under 2 GB to spare. The adapters, all 41.9M of them with their optimizer state, cost 0.67 GB. The logits for one 2048-token sequence cost more than that. Now set `SEQ = 4096`: the logits and checkpoints double, and you're at 11.6 GB before the recomputed layer grows too. That's why the shakedown runs at 2048, one sequence at a time."
mistakes:
  - { file: lora_budget/mistakes/placeholders.py, fails: lora_counts, says: "`lora_params` still returns `...`" }
  - { file: lora_budget/mistakes/full_matrix.py, fails: lora_counts, says: "which is in × out, the full weights" }
  - { file: lora_budget/mistakes/r_in_out.py, fails: lora_counts, says: "their sizes add" }
  - { file: lora_budget/mistakes/no_layers.py, fails: lora_counts, says: "the count for one block" }
  - { file: lora_budget/mistakes/only_a.py, fails: lora_counts, says: "You count only A" }
  - { file: lora_budget/mistakes/globals.py, fails: lora_counts, says: "Does it use its arguments" }
  - { file: lora_budget/mistakes/bf16_weights.py, fails: adapter_state, says: "only a bf16 copy of the weights" }
  - { file: lora_budget/mistakes/fp32_weights.py, fails: adapter_state, says: "You count only the fp32 weights" }
  - { file: lora_budget/mistakes/no_adam.py, fails: adapter_state, says: "but not Adam's two moments" }
  - { file: lora_budget/mistakes/one_moment.py, fails: adapter_state, says: "You count one of Adam's moments" }
  - { file: lora_budget/mistakes/logits_bf16.py, fails: logits_size, says: "upcasts the logits to fp32" }
  - { file: lora_budget/mistakes/logits_no_batch.py, fails: logits_size, says: "Every sequence in the batch has its own" }
  - { file: lora_budget/mistakes/logits_hidden.py, fails: logits_size, says: "You used the hidden size" }
```

### Shake the pipeline down

```replay
id: shakedown
intro: "The shakedown fine-tunes Llama 3.1 8B with QLoRA for 200 steps on 6,000 short English instruction examples, packed into 2048-token sequences, about 35 minutes. Its job is to break early: out-of-memory, a loss that doesn't fall, an adapter that won't save. The recording starts with the classic first failure."
log: replays/shakedown.jsonl
speed: 40
recorded:
  hardware: RTX 3060 12 GB
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: Peak VRAM per logging window (GB)
  flag_above: 11.0
  x_labels: [step 20, step 200]
  caption: "One window spiked to 11.8 GB, a fragmented allocator asking for one more block than was free. The paged optimizer moved some Adam state to CPU RAM and the run carried on. Everything else sits between 9.4 and 9.9 GB, close to your budget minus the CUDA context."
summary:
  - { label: Trainable, value: "41.9M of 8.07B (0.52%)" }
  - { label: Eval loss, value: "1.12" }
  - { label: Peak VRAM, value: 9.9 GB reserved }
  - { label: Throughput, value: 800 tokens/s }
run_yourself: |
  # on your machine or Colab, from the course repo
  python qlora_train.py \
    --model meta-llama/Llama-3.1-8B --data alpaca-cleaned --n-examples 6000 --pack \
    --steps 200 --seq 2048 --rank 16 --alpha 32 \
    --grad-checkpointing --optim paged_adamw_32bit
```

### Record your metrics

```metrics
id: record
intro: Your shakedown's eval loss and peak VRAM. They're compared with the reference run and the pass limit. If peak VRAM is near 12 GB, cut the sequence length before 4A.4.
```

## Hints

- Turn on gradient checkpointing before anything else. Without it every layer keeps its activations for backward, and an 8B model at 2048 tokens runs out of memory on step one.
- Keep the batch at 1 sequence and use gradient accumulation (4 or 8) for a bigger effective batch. Every extra sequence in the batch adds another 1.05 GB of logits.
- `torch.cuda.max_memory_reserved()` is what the allocator held at its peak; `max_memory_allocated()` is what tensors actually used. Fragmentation lives in the gap between them.
- The adapter saves on its own, about 168 MB in fp32. You don't merge it into the 4-bit base. To ship a merged model, load the base in bf16, merge, then quantize again.
- If the loss sits flat from step 1, check `model.print_trainable_parameters()`. Calling `prepare_model_for_kbit_training` after `get_peft_model` freezes the adapters, and training quietly does nothing.

## Done when

- [ ] Your NF4 weight sizes match bitsandbytes, with and without double quantization {#nf4_memory auto=challenge:nf4_memory}
- [ ] Your budget for the whole fine-tune fits in 12 GB {#lora_budget auto=challenge:lora_budget}
- [ ] A 200-step QLoRA run finishes, saves its adapter, and both metrics are recorded {#metrics_recorded auto=metrics}
