---
id: "3.6"
title: llama.cpp and k-quants
summary: Build a Q4_K super-block by hand, a scale and a min per 32 weights with the scales themselves quantized to 6 bits, then quantize Llama 3.1 8B to GGUF and measure perplexity and decode speed.
phase: "3"
hours: 6
type: run
status: ready
protected: true
cut: null
prerequisites:
  - "3.5"
papers: []
quality_metric:
  id: ppl_delta
  label: Q4_K_M perplexity change vs fp16
  unit: "%"
  note: llama-perplexity on wikitext-2 at context 512, against the F16 GGUF of the same model.
  inputs:
    - { id: ppl_f16, label: F16 GGUF perplexity, hint: "llama-perplexity, partial GPU offload" }
    - { id: ppl_q4km, label: Q4_K_M perplexity, hint: "Same command, Q4_K_M file" }
  compute: pct_change
  reference: 2.8
  pass: { max: 4.0 }
systems_metric:
  id: tokens_per_s
  label: Decode speed on your GPU
  unit: tok/s
  note: llama-bench tg128 with every layer on the GPU (-ngl 99), Q4_K_M.
  inputs:
    - { id: tokens_per_s, label: tg128 tokens per second, hint: From llama-bench }
  compute: value
  reference: 57
  pass: { min: 30 }
quiz:
  - id: q4km_mix
    q: Q4_K is 4.5 bits per weight, yet llama-quantize reports 4.89 for Llama 3.1 8B in Q4_K_M. Where do the extra bits come from?
    answer: 2
    options:
      - text: GGUF's header and metadata
        why: The header and tokenizer take a few MB, nowhere near 0.39 bits × 8 billion weights ≈ 390 MB.
      - text: The super-block scales are stored in fp32
        why: d and dmin are fp16, and they're already counted in Q4_K's 4.5 bits.
      - text: "The _M mix keeps the most sensitive tensors in Q6_K"
        why: Q4_K_M quantizes half of the attn_v and ffn_down tensors, and the output layer, to Q6_K (6.56 bits), where errors hurt most. Everything else is Q4_K. The mix averages 4.89 bits.
  - id: why_min
    q: Q4_K stores a min for every 32 weights as well as a scale. What does the min buy you over 3.2's symmetric absmax?
    answer: 0
    options:
      - text: The 16 levels span lo…hi instead of −amax…amax
        why: A sub-block whose weights run from −0.12 to 0.07 wastes most of a symmetric grid on values that never occur. Scale plus min fits the grid to the actual range and uses all 16 codes, not 15.
      - text: It removes the need for a scale
        why: You still need the step size. The min only says where the grid starts.
      - text: It lets the matmul skip zero weights
        why: Nothing is skipped. Every weight is decoded as d · sc · q − dmin · m and multiplied.
  - id: why_faster
    q: Decoding one token reads every weight once. Llama 3.1 8B in Q4_K_M is 4.9 GB, and an RTX 3060 moves 360 GB/s. What's the best decode speed you could hope for?
    answer: 1
    options:
      - text: About 7 tokens/s
        why: That's 360 / 49, as if the file were ten times bigger. Divide 360 GB/s by the 4.9 GB read per token.
      - text: About 73 tokens/s
        why: 360 / 4.9 ≈ 73. Decode is memory-bound, so fewer bytes per weight means more tokens per second. Real runs reach about 80% of the bound, so ~57 tok/s.
      - text: About 1,800 tokens/s
        why: That's prompt processing, where one weight read serves hundreds of tokens at once. Decode makes one token per full read of the weights.
---

## Goal

llama.cpp runs quantized models on almost anything, from the single GGUF file you'll serve in phase 5. Its **k-quants** stack two ideas from this phase. Each block of 32 weights gets a **scale and a min**, so the 16 levels cover the block's real range rather than a symmetric one. Then 8 such blocks form a **super-block** whose scales and mins are themselves quantized to 6 bits, double quantization again. Build a Q4_K super-block by hand and work out its 4.5 bits per weight. Then convert Llama 3.1 8B to GGUF, quantize it to Q4_K_M, and measure perplexity and decode speed with llama.cpp's own tools.

## Learn

```card
type: concept
id: k_family
eyebrow: The k-quant family
title: Bits per weight, from Q8_0 down to Q2_K.
visual:
  kind: bars
  max: 9
  bars:
    - { label: Q8_0, value: 8.5, text: "8.5" }
    - { label: Q6_K, value: 6.5625, text: "6.56" }
    - { label: Q5_K, value: 5.5, text: "5.5" }
    - { label: Q4_K, value: 4.5, text: "4.5" }
    - { label: Q3_K, value: 3.4375, text: "3.44" }
    - { label: Q2_K, value: 2.625, text: "2.63" }
body: |
  Q8_0 is the older plain format: blocks of 32 int8 weights with one fp16 scale. Every k-quant, Q2_K to Q6_K, uses super-blocks of 256 weights split into sub-blocks of 16 or 32. Each sub-block has a scale (and, for Q4_K, Q5_K and Q2_K, a min), quantized to 4, 6 or 8 bits against one or two fp16 numbers per super-block. The fractional bits are those block constants spread over the weights. The "_S/_M/_L" variants mix types per tensor.
```

```card
type: predict
id: predict_asym
eyebrow: Predict the output
title: A sub-block runs from lo = −0.10 to hi = 0.50. Which code does w = 0.21 get?
code: |
  scale = (hi - lo) / 15          # 0.04
  q = round((w - lo) / scale)     # codes 0…15, counted up from lo
options:
  - text: "8"
    correct: true
    why: (0.21 + 0.10) / 0.04 = 7.75, which rounds to 8. Decoded, 8 × 0.04 − 0.10 = 0.22, one hundredth off.
  - text: "7"
    why: That truncates 7.75 instead of rounding it. Round to the nearest code.
  - text: "3"
    why: That's 3.2's symmetric absmax, 0.21 / (0.50 / 7) ≈ 2.94. Q4_K counts up from lo, so the grid covers −0.10…0.50, not −0.50…0.50.
  - text: "5"
    why: That's 0.21 / 0.04 without subtracting lo first. Code 0 sits at lo, −0.10, not at zero.
hint: Shift w so that lo becomes 0, then count steps of 0.04.
```

```card
type: calc
id: q4k_bits
eyebrow: Do the math
title: Bits per weight in one Q4_K super-block.
given:
  - { label: Weights, value: 256 at 4 bits }
  - { label: Sub-blocks, value: "8 of 32, each with a 6-bit scale and a 6-bit min" }
  - { label: Super-block, value: "d and dmin, fp16 each" }
unit: bits
answer: { min: 4.49, max: 4.51 }
placeholder: "0.00"
why_ok: (256 × 4 + 8 × 12 + 2 × 16) / 256 = (1024 + 96 + 32) / 256 = 4.5. With fp16 scale and min per 32 weights instead, it would be 4 + 32 / 32 = 5.0.
why_wrong: "Add up every bit in the super-block: the codes, eight 6-bit scales, eight 6-bit mins and two 16-bit numbers. Then divide by the 256 weights."
hint: Count the whole super-block's bits first, then divide by 256.
```

```card
type: calc
id: gguf_size
eyebrow: Do the math
title: How big is Llama 3.1 8B as a Q4_K_M GGUF?
given:
  - { label: Parameters, value: 8.03 × 10⁹ }
  - { label: llama-quantize reports, value: 4.89 bits per weight }
unit: GB
answer: { min: 4.85, max: 4.97 }
placeholder: "0.00"
why_ok: 8.03 × 10⁹ × 4.89 / 8 = 4.91 GB, and the file on disk is 4.92 GB with its metadata. Down from 16.1 GB in F16, and it fits in 12 GB with room for the KV cache.
why_wrong: Multiply the parameter count by the bits per weight, then divide by 8 to get bytes.
hint: Bits ÷ 8 = bytes, and 10⁹ bytes = 1 GB.
```

```card
type: reorder
id: llama_cpp_pipeline
eyebrow: Put it in order
title: From a Hugging Face checkpoint to a measured GGUF.
lines:
  - python convert_hf_to_gguf.py → model-f16.gguf
  - llama-imatrix on calibration text → imatrix.dat
  - llama-quantize --imatrix imatrix.dat model-f16.gguf Q4_K_M
  - llama-perplexity -f wiki.test.raw on F16 and Q4_K_M
  - llama-bench -ngl 99 for prompt and decode speed
shuffle: [3, 0, 4, 1, 2]
why_ok: "Convert, measure importance, quantize, then evaluate. The importance matrix plays AWQ's role: it tells llama-quantize which weights the activations lean on, so their rounding counts more."
why_wrong: "Everything starts from the F16 GGUF, and llama-quantize needs the importance matrix before it runs. Evaluation comes last."
hint: Which files does each command read, and which does it write?
```

```card
type: quiz
ref: q4km_mix
```

```card
type: quiz
ref: why_min
```

```card
type: quiz
ref: why_faster
```

## Build steps

### A scale and a min per 32 weights

```challenge
id: subblocks
file: kquants.py
minutes: 30
start: { code: "import torch\n\n" }
setup:
  - say: One super-block, 256 weights from a normal distribution. llama.cpp quantizes every tensor in chunks exactly this size.
    type: "w = 0.05 * torch.randn(256, generator=torch.Generator().manual_seed(0))\n\n"
    pause: 2.2
  - say: "The baseline: 3.2's symmetric absmax on blocks of 32, 15 levels centred on zero."
    type: "def int4_blocks(w, block):\n    wb = w.reshape(-1, block)\n    scale = wb.abs().amax(dim=1, keepdim=True) / 7\n    return (torch.round(wb / scale).clamp(-7, 7) * scale).reshape(-1)\n"
    pause: 2.0
  - say: "Q4_K does it differently. Each sub-block of 32 keeps its lowest value, lo, and spreads all 16 codes from lo up to its highest, hi. The way back is typed for you: code times scale, plus lo. Your turn: fill in scale and q."
    type_file: subblocks/stub.py
    pause: 1.2
tests: subblocks/tests.py
solution: subblocks/solution.py
alternatives:
  - subblocks/alternatives/no_clamp.py
  - subblocks/alternatives/sub_method.py
hints:
  logic: |
    for each sub-block (lo = lowest, hi = highest):
      scale = the gap from lo to hi, cut into 15 equal steps
      for each weight:
        count how many steps it sits above lo → round to a whole number
        outside 0…15?
          yes → pull it back to the edge
          no  → keep it
      store the codes in one unsigned byte each
    return codes, scale, lo
  tools: |
    step size from the range
      plain arithmetic: - /
    count the steps above lo
      plain arithmetic: - /
      torch.round(t)   nearest whole number
    keep the codes in range
      t.clamp(lo, hi)  squeezes values into [lo, hi]
    store in one unsigned byte
      t.to(dtype)      converts; torch.uint8 is 1 unsigned byte
  nudge: "Code 0 should decode to lo and code 15 to hi. How big does each step have to be, and how do you count steps up from lo?"
  concept: "16 codes means 15 steps between lo and hi, so `scale = (hi - lo) / 15`. A weight's code is how many steps it sits above lo: subtract lo, divide by scale, round to the nearest whole number, keep it in 0…15, and store it in one unsigned byte, `torch.uint8`."
  example: |
    Trace it on two sub-blocks of 4 weights (the real ones have 32), `wb = [[-0.10, 0.045, 0.20, 0.00], [0.30, 0.50, 0.38, 0.35]]`:

    - `lo = [[-0.10], [0.30]]`, `hi = [[0.20], [0.50]]`.
    - `scale = [[0.02], [0.01333]]`, since 0.30 / 15 and 0.20 / 15.
    - Steps above `lo`: `[[0, 7.25, 15, 5], [0, 15, 6, 3.75]]`.
    - Rounded: `q = [[0, 7, 15, 5], [0, 15, 6, 4]]`. The lowest weight of each block is code 0 and the highest is code 15.
    - The way back, `q * scale + lo`: `[[-0.10, 0.04, 0.20, 0.00], [0.30, 0.50, 0.38, 0.3533]]`. Each is within half a step of `wb`.
  skeleton: |
    scale = (__ - __) / __
    q = torch.round((wb - __) / _____).clamp(_, __).to(torch._____)
  solution: |
    scale = (hi - lo) / 15
    q = torch.round((wb - lo) / scale).clamp(0, 15).to(torch.uint8)
on_solve: "Scale plus min beats symmetric absmax by about 20% on the same 32-weight blocks, because the grid follows each block's real range and uses all 16 codes. The price is storage: an fp16 scale and an fp16 min per 32 weights is a whole extra bit per weight. Next, shrink that."
mistakes:
  - { file: subblocks/mistakes/placeholders.py, fails: shapes, says: "are still `...`" }
  - { file: subblocks/mistakes/q_placeholder.py, fails: shapes, says: "`q` is still `...`" }
  - { file: subblocks/mistakes/float_q.py, fails: shapes, says: "Which dtype stores one byte" }
  - { file: subblocks/mistakes/symmetric.py, fails: codes_fit, says: "wrapped around" }
  - { file: subblocks/mistakes/no_clamp_16.py, fails: codes_fit, says: "doesn't fit in 4 bits" }
  - { file: subblocks/mistakes/div16.py, fails: round_trip, says: "16 codes have 15 steps" }
  - { file: subblocks/mistakes/no_lo.py, fails: round_trip, says: "counted from zero" }
  - { file: subblocks/mistakes/truncated.py, fails: round_trip, says: "chops off the decimals" }
  - { file: subblocks/mistakes/hi_only.py, fails: round_trip, says: "only covers 0…hi" }
```

### Quantize the scales: the super-block

```challenge
id: superblock
file: kquants.py
minutes: 30
start: { from: subblocks }
setup:
  - say: "Eight scales and eight mins at fp16 cost 256 bits for 256 weights. Q4_K stores them in 6 bits each, against one fp16 super-scale per super-block, like NF4's double quantization."
    type_file: superblock/stub.py
    pause: 2.6
  - say: "Your turn: write six_bit. The rest decodes the super-block the way llama.cpp does and compares bits and error."
    pause: 1.0
tests: superblock/tests.py
solution: superblock/solution.py
alternatives:
  - superblock/alternatives/amax.py
  - superblock/alternatives/no_clamp.py
hints:
  logic: |
    find the largest value in v   (all positive)
      → d = that value ÷ 63   (one for all the values)
    for each value:
      value ÷ d → round to a whole number
      outside 0…63?
        yes → pull it back to the edge
        no  → keep it
    store the whole numbers in one byte each
      → return them and d
  tools: |
    find the largest value
      t.max()          the biggest value in t
    one super-scale
      plain arithmetic: /
    whole numbers of d
      t / x            divides every value by x
      torch.round(t)   nearest whole number
    keep them in range
      t.clamp(lo, hi)  squeezes values into [lo, hi]
  nudge: "It's 3.1's absmax again, on 8 positive numbers instead of weights. Which number lands on the top of a 6-bit range, and what is that top?"
  concept: "6 bits hold 0…63. One super-scale d covers all the values, chosen so the largest lands on exactly 63: `d = v.max() / 63`. Each value becomes the nearest whole number of d's, kept inside 0…63. The values here are scales and mins, all positive, so no negative side is needed."
  example: |
    Trace it on four scales, `v = [0.020, 0.015, 0.010, 0.025]` (the real one has 8):

    - The largest is `0.025`, so `d = 0.025 / 63 ≈ 0.000397`.
    - `v / d ≈ [50.4, 37.8, 25.2, 63.0]`. The largest lands exactly on 63.
    - Rounded: `q = [50, 38, 25, 63]`.
    - Decoded, `q * d ≈ [0.01984, 0.01508, 0.00992, 0.02500]`. Each is within half a step (`d / 2 ≈ 0.0002`) of `v`.
  skeleton: |
    d = v.___() / __
    q = torch.round(v / _).clamp(_, __)
  solution: |
    d = v.max() / 63
    q = torch.round(v / d).clamp(0, 63)
on_solve: "That's a Q4_K super-block: 4.5 bits per weight instead of 5.0, for about 3% more error. Q4_K_M uses this format for most tensors and Q6_K for the sensitive ones, which is how it lands at 4.89 bits. llama.cpp also searches each sub-block's scale and min to minimise error instead of taking the plain lo and hi, and with an importance matrix it weights that search by the activations, AWQ's idea again."
mistakes:
  - { file: superblock/mistakes/placeholders.py, fails: largest_63, says: "still `...`" }
  - { file: superblock/mistakes/per_value.py, fails: largest_63, says: "one super-scale for all" }
  - { file: superblock/mistakes/div64.py, fails: largest_63, says: "6 bits hold 0…63" }
  - { file: superblock/mistakes/div31.py, fails: largest_63, says: "5 bits" }
  - { file: superblock/mistakes/inverted.py, fails: largest_63, says: "Flip it" }
  - { file: superblock/mistakes/truncated.py, fails: within_half_step, says: "chops off the decimals" }
```

### Quantize Llama 3.1 8B with llama.cpp

```replay
id: llama_cpp_8b
intro: "Convert to F16 GGUF, build an importance matrix, quantize to six types, then measure each one. The F16 baseline doesn't fit in 12 GB, so llama-perplexity offloads 24 of 32 layers for it. About 3 hours in all, most of it perplexity runs. Watch a recording first."
log: replays/llama-cpp-8b.jsonl
speed: 40
recorded:
  hardware: RTX 3060 12 GB
  illustrative: true
  note: Sample data written for the course, not a real run. Replace it with a recording of your own run.
chart:
  title: Perplexity by quant type
  flag_above: 6.6
  x_labels: [Q8_0, Q2_K]
  caption: "Down to Q4_K_M the cost is under 3%. Below 4 bits per weight it climbs fast: Q3_K_M loses 10%, and Q2_K is barely usable. For a 12 GB card, Q4_K_M or Q5_K_M is the usual choice."
summary:
  - { label: Q4_K_M file, value: "16.1 → 4.92 GB (4.89 bits/weight)" }
  - { label: Perplexity, value: "6.233 → 6.407 (+2.8%)" }
  - { label: Decode, value: "57 tok/s (F16, partly offloaded: 8)" }
  - { label: Prompt, value: "1,950 tok/s" }
run_yourself: |
  # in a llama.cpp checkout, built with CUDA
  python convert_hf_to_gguf.py ~/models/Llama-3.1-8B --outtype f16 --outfile llama-8b-f16.gguf
  ./llama-imatrix -m llama-8b-f16.gguf -f calibration.txt -o imatrix.dat -ngl 24
  ./llama-quantize --imatrix imatrix.dat llama-8b-f16.gguf llama-8b-q4_k_m.gguf Q4_K_M
  ./llama-perplexity -m llama-8b-q4_k_m.gguf -f wiki.test.raw -ngl 99
  ./llama-bench -m llama-8b-q4_k_m.gguf -ngl 99
```

### Record your metrics

```metrics
id: record
intro: Q4_K_M's perplexity change against F16, and its decode speed on your card.
```

## Hints

- Compare perplexities only from the same tool, file and context length. llama-perplexity at context 512 gives different absolute numbers from the Python eval in 3.4 and 3.5, so compare F16 and Q4_K_M with each other, not with those.
- `-ngl` is how many layers go on the GPU. `-ngl 99` means all of them. If the F16 baseline runs out of memory, lower it until it fits; perplexity stays the same, only speed changes.
- llama-bench's `tg128` is decode (one token at a time, memory-bound) and `pp512` is prompt processing (batched, compute-bound). Your metric is tg128.
- Keep the Q4_K_M file. Lesson 5.1 serves it with llama.cpp's server and races it against vLLM.

## Done when

- [ ] Scale and min per sub-block beats symmetric absmax {#asym auto=challenge:subblocks}
- [ ] Your Q4_K super-block decodes at 4.5 bits per weight {#superblock auto=challenge:superblock}
- [ ] Perplexity change and decode speed recorded {#metrics_recorded auto=metrics}
