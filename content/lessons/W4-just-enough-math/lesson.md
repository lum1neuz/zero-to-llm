---
id: "W4"
title: Just enough math
summary: The four bits of math the course leans on, with everyday numbers. Chances that multiply, powers, logarithms, and "surprise", which is how a model's mistakes get measured.
phase: "W"
hours: 1.5
type: intro
status: ready
protected: false
cut: null
prerequisites: ["W3"]
papers: []
quality_metric:
  id: challenges_solved
  label: Challenges solved
  unit: count
systems_metric:
  id: minutes_spent
  label: Time spent on this lesson
  unit: min
quiz:
  - id: sure_and_right
    q: A model said the next letter would be "a" with 100% certainty, and it was "a". How surprised should we be?
    answer: 0
    options:
      - text: Not at all. The surprise is 0
        why: "Right. −log(1) = 0. A certain guess that turns out right costs nothing."
      - text: Very surprised. The surprise is large
        why: "Surprise is large when the model gave the right answer a *small* chance. Here it gave it 100%."
      - text: It depends on the length of the word
        why: "Surprise only depends on the chance the model gave to what actually happened. 100% means a surprise of 0."
  - id: lower_loss
    q: The loss is the average surprise over many guesses. What does a lower loss mean?
    answer: 1
    options:
      - text: The model is guessing less often
        why: "The model makes the same number of guesses. The loss measures how good they are."
      - text: The model gave the right answers higher chances, so its guesses are better
        why: "Yes. Higher chances for what really happened means less surprise, which means a lower loss. Training is all about pushing the loss down."
      - text: The model is more random
        why: "A more random model spreads its chances thin, so the right answer gets less, and the loss goes *up*."
---

## Goal

Get comfortable with four ideas, using small everyday numbers: chances that multiply, powers, logarithms, and **surprise**, which is minus the log of a chance. The average surprise is the **loss**, the single number every model in this course tries to make smaller.

## Learn

```card
type: concept
id: chances_multiply
eyebrow: Chances in a row
title: To get the chance of two things in a row, multiply.
body: |
  Flip a coin: heads has a chance of **0.5** (a half).

  Flip it twice. The chance of **heads, then heads again** is

  **0.5 × 0.5 = 0.25**, a quarter.

  A model writing a word works the same way. The chance it writes "hi" is (chance of "h") × (chance of "i" after "h"). Long texts multiply many chances together, and the result gets tiny very fast. Logarithms fix that, a few cards from now.
```

```card
type: calc
id: two_in_a_row
eyebrow: Do the math
title: "A model gives \"h\" a chance of 0.5, then \"i\" a chance of 0.2. What's the chance of \"hi\"?"
given:
  - { label: Chance of "h", value: "0.5" }
  - { label: Chance of "i" after "h", value: "0.2" }
unit: ""
answer: { min: 0.1, max: 0.1 }
placeholder: "0.0"
why_ok: "0.5 × 0.2 = 0.1, which is 10%. Two fairly likely steps already make a fairly unlikely word."
why_wrong: "For things in a row, multiply the chances: 0.5 × 0.2."
hint: Multiply, don't add.
```

```card
type: concept
id: powers_and_logs
eyebrow: Powers and logs
title: A logarithm asks "how many times did I multiply?"
body: |
  A **power** is repeated multiplying: **2 × 2 × 2 = 8**, written **2³**.

  A **logarithm** (log) goes the other way. It asks *how many* times you multiplied:

  | Question | Answer |
  |---|---|
  | How many 2s multiply to 8? | log₂(8) = **3** |
  | How many 10s multiply to 1000? | log₁₀(1000) = **3** |
  | How many 10s multiply to 1? | log₁₀(1) = **0** |

  So a log turns huge or tiny numbers into small, friendly ones.
```

```card
type: predict
id: log_of_100
eyebrow: Predict the answer
title: What is log₁₀(100)?
code: |
  How many 10s multiply to make 100?
options:
  - text: "2"
    correct: true
    why: "10 × 10 = 100: two 10s. So log₁₀(100) = 2."
  - text: "10"
    why: "10 is what you multiply by. The log counts how many times: 10 × 10 is two times."
  - text: "100"
    why: "100 is the result. The log asks how many 10s it took to get there."
hint: Write out 10 × 10 × … until you reach 100, then count the 10s.
```

```card
type: concept
id: surprise
eyebrow: Measuring mistakes
title: "Surprise = −log(chance). Unlikely things are more surprising."
body: |
  When a model guesses, we look at the chance it gave to what *actually* happened. Then we take **minus its log**. That's the **surprise**:

  ```chart
  kind: line
  title: Surprise against chance
  x: { label: chance given to the right answer, values: [0.1, 0.25, 0.5, 1.0] }
  y: { label: surprise, min: 0, max: 2.5 }
  series:
    - { name: surprise, values: [2.30, 1.39, 0.69, 0.00] }
  highlight: { x: 0.1, label: "unlikely, 2.30" }
  caption: The lower the chance the model gave the right answer, the bigger the surprise.
  ```

  A confident right guess costs nothing. A right answer the model thought unlikely costs a lot.

  The **minus** is there because logs of chances (numbers below 1) are negative, and a surprise should be positive.
```

```card
type: reorder
id: loss_steps
eyebrow: Put it in order
title: Put the steps for working out a model's loss in order.
lines:
  - For each guess, find the chance it gave the right answer
  - Turn each chance into a surprise with −log
  - Add all the surprises up
  - Divide by how many guesses there were
shuffle: [3, 1, 0, 2]
why_ok: "That's the loss: the average surprise. In lesson 0.3 it's called the negative log likelihood, and it's exactly this."
why_wrong: "You need the chances before you can turn them into surprises, and you can only average after adding up."
hint: It's an average, like W2's. What are you averaging?
```

```card
type: quiz
ref: sure_and_right
```

```card
type: quiz
ref: lower_loss
```

## Build steps

### Write surprise

```challenge
id: surprise
file: loss.py
minutes: 10
start: { code: "import math\n\n" }
setup:
  - say: "Python's math module has logs built in. math.log(0.5) is about −0.69: negative, because 0.5 is below 1."
    pause: 3
  - say: "Your turn: return the surprise for a chance p. That's minus the log of p."
    type_file: surprise/stub.py
    pause: 1
tests: surprise/tests.py
solution: surprise/solution.py
alternatives:
  - surprise/alternatives/times_minus_one.py
  - surprise/alternatives/log_of_inverse.py
hints:
  start:
    try: |
      print(round(math.log(0.5), 4))
      print(round(math.log(1.0), 4))
    see: |
      -0.6931
      0.0
    next: "That's the log of a half, which is below zero, and the log of a sure thing, which is `0`. Surprise should be big and positive for a small chance. What one small change turns the first number into a positive one?"
  logic: |
    take the natural log of p   (zero or negative when p is 1 or less)
      → flip the sign           (a small chance becomes a big surprise)
      → return it
  tools: |
    take the natural log
      math.log(x)      the natural log of x (base e)
    flip the sign
      plain arithmetic: -x   (a minus sign in front)
  nudge: The log of a chance below 1 is negative. How do you make it positive?
  concept: "`math.log(p)` is the natural log of p. Put a minus sign in front of it so that unlikely chances give big positive numbers."
  example: |
    Trace it on a few chances:

    - `p = 1.0`: the log is `0.0`, so the surprise is `0.0`. A sure thing surprises nobody.
    - `p = 0.5`: the log is about `-0.6931`, so the surprise is about `0.6931`.
    - `p = 0.25`: the log is about `-1.3863`, so the surprise is about `1.3863`.
    - `p = 0.1`: the log is about `-2.3026`, so the surprise is about `2.3026`.

    The smaller the chance, the bigger the surprise, and every answer is positive (or zero).
  skeleton: |
    return ___.___(p)
  solution: |
    return -math.log(p)
on_solve: "Solved. surprise(0.5) is 0.69 and surprise(1) is 0. Next: average many surprises to get a loss."
mistakes:
  - { file: surprise/mistakes/placeholder.py, fails: sure_is_zero, says: "placeholder" }
  - { file: surprise/mistakes/no_minus.py, fails: matches_minus_log, says: "put a minus in front" }
  - { file: surprise/mistakes/log10.py, fails: matches_minus_log, says: "That's log10" }
  - { file: surprise/mistakes/log2.py, fails: matches_minus_log, says: "That's log2" }
  - { file: surprise/mistakes/one_minus.py, fails: matches_minus_log, says: "which is 1 − p" }
```

### Average the surprise: your first loss

```challenge
id: average_surprise
file: loss.py
minutes: 10
start: { from: surprise }
setup:
  - say: "A model makes many guesses. Its loss is the average surprise over all of them. This is the number training tries to push down."
    type_file: average_surprise/stub.py
    pause: 2.5
  - say: "Your turn: inside the loop, add each guess's surprise to the total."
    pause: 1
tests: average_surprise/tests.py
solution: average_surprise/solution.py
alternatives:
  - average_surprise/alternatives/plus_equals.py
  - average_surprise/alternatives/minus_log_inline.py
hints:
  start:
    try: |
      print(round(surprise(0.5), 4))
      print(round(surprise(0.25), 4))
      print(round(surprise(0.5) + surprise(0.25), 4))
    see: |
      0.6931
      1.3863
      2.0794
    next: "That's two guesses and their surprises added together. Your loop does this one guess at a time, keeping the running total in `total`. Inside the loop, make `total` bigger by the surprise of `p`."
  logic: |
    total starts at 0
    for each guess p in probs:
      work out the surprise of p
        → add it onto total   (don't replace total)
    after the loop
      → average = total ÷ number of guesses   (the stub does this)
      → return the average
  tools: |
    work out the surprise of p
      surprise(x)      your function from step 1
    add it onto total
      plain arithmetic: a + b, then store the result back
  nudge: "It's an average, like in W2. What should go into `total` for each guess?"
  concept: "Use the `surprise` function you just wrote. Each time round the loop, make `total` bigger by `surprise(p)`."
  example: |
    Trace it on `probs = [0.5, 0.25, 1.0]`:

    - Start: `total = 0`.
    - `p = 0.5`: the surprise is about `0.6931`, so `total` becomes about `0.6931`.
    - `p = 0.25`: the surprise is about `1.3863`, so `total` becomes about `2.0794`.
    - `p = 1.0`: the surprise is `0.0`, so `total` stays about `2.0794`.
    - There are `3` guesses, so the result is `2.0794 / 3`, about `0.6931`.

    Print `total` inside the loop and compare. If it jumps back down instead of growing, you replaced it instead of adding to it.
  skeleton: |
    total = _____ + ________(p)
  solution: |
    total = total + surprise(p)
on_solve: "Solved. That number is a loss. For [0.5, 0.25, 1.0] it's 0.69. In lesson 0.3 you'll compute exactly this for a real model of names."
mistakes:
  - { file: average_surprise/mistakes/placeholder.py, fails: loss, says: "nothing was added" }
  - { file: average_surprise/mistakes/overwrites.py, fails: loss, says: "Only the last guess counted" }
  - { file: average_surprise/mistakes/adds_probs.py, fails: loss, says: "average probability" }
```

## Hints

- You never need to work out a log by hand. Python's `math.log` does it.
- If your answers come out negative, you're missing the minus in front of the log.

## Done when

- [ ] Finish the Learn cards {#cards auto=learn}
- [ ] Write surprise {#surprise auto=challenge:surprise}
- [ ] Average the surprise {#loss auto=challenge:average_surprise}
