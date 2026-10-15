---
id: "W1"
title: What is a language model?
summary: No code yet. See what a language model does, in the same way your phone's keyboard guesses your next word.
phase: "W"
hours: 1
type: intro
status: ready
protected: false
cut: null
prerequisites: []
papers: []
quality_metric:
  id: quiz_right
  label: Quiz questions right on the first try
  unit: count
systems_metric:
  id: minutes_spent
  label: Time spent on this lesson
  unit: min
quiz:
  - id: what_it_does
    q: What does a language model do at every single step?
    answer: 1
    options:
      - text: It looks up the answer in a big list of saved sentences
        why: "It doesn't store sentences to copy. It learned patterns, and it uses them to guess."
      - text: It guesses the next small piece of text
        why: "Yes. One small piece at a time: guess it, add it, then guess again. Long answers are many guesses in a row."
      - text: It understands the question the way a person does
        why: "It doesn't think like a person. It's very good at guessing what text usually comes next."
  - id: why_numbers
    q: Why do letters and words get turned into numbers?
    answer: 0
    options:
      - text: Because computers can only do math with numbers
        why: "Right. To a computer, everything (pictures, music, text) ends up as numbers, because numbers are what it can calculate with."
      - text: To keep the text secret
        why: "It isn't a code to hide things. Every letter always gets the same number, so anyone can turn it back."
      - text: Because numbers are shorter to type
        why: "Length isn't the reason. The computer needs numbers so it can do arithmetic on them."
---

## Goal

Understand, in plain words, what a language model does: it guesses the next piece of text, again and again. It learns those guesses by counting what usually comes next in lots of examples. No code and no math in this lesson.

## Learn

```card
type: concept
id: phone_keyboard
eyebrow: You already use one
title: Your phone's keyboard is a tiny language model.
body: |
  When you type **"good"**, your keyboard suggests **"morning"**, **"night"** or **"luck"**.

  It isn't reading your mind. It has seen lots of messages, and it noticed which words usually come after "good".

  ChatGPT, Claude and the model you'll build in this course work on the same idea. They're just much, much bigger.
```

```card
type: concept
id: one_piece_at_a_time
eyebrow: The big idea
title: It writes one small piece at a time.
body: |
  A language model never writes a whole sentence at once. It does one tiny step, over and over:

  1. Look at the text so far.
  2. Guess the next small piece: a letter, or part of a word.
  3. Add it to the end.
  4. Go back to step 1.

  A long answer is just thousands of these small guesses in a row.
```

```card
type: predict
id: guess_next_word
eyebrow: Your turn to guess
title: Which word would the keyboard suggest after "good"?
code: |
  Messages it has seen:
  "good morning"   5 times
  "good night"     3 times
  "good luck"      2 times
options:
  - text: morning
    correct: true
    why: "It came after \"good\" most often: 5 times out of 10. A model picks the most likely next word, or picks at random with those odds."
  - text: night
    why: "\"night\" appeared 3 times, which is fewer than \"morning\" with 5."
  - text: luck
    why: "\"luck\" appeared only 2 times, the fewest of the three."
hint: Count which word followed "good" most often.
```

```card
type: calc
id: chance_of_morning
eyebrow: Do the math
title: Out of all 10 messages, what percent were "good morning"?
given:
  - { label: "\"good morning\"", value: "5 times" }
  - { label: All messages starting with "good", value: "10" }
unit: "%"
answer: { min: 50, max: 50 }
placeholder: "0"
why_ok: "5 out of 10 is one half, which is 50%. This is a probability: how likely \"morning\" is to come next. Models think in probabilities like this."
why_wrong: "Divide the part by the whole: 5 ÷ 10 = 0.5. Then multiply by 100 to get a percent."
hint: "A percent is (part ÷ whole) × 100."
```

```card
type: reorder
id: writing_loop
eyebrow: Put it in order
title: Put the model's writing steps in order.
lines:
  - Look at the text so far
  - Guess the next piece
  - Add the piece to the end
  - Repeat from the top
shuffle: [2, 0, 3, 1]
why_ok: "That's the whole loop. It reads, guesses, adds, and repeats. Lesson 2.1 has you write this loop in code."
why_wrong: "Think about writing a word yourself. You read what's there, decide what comes next, write it, and carry on."
hint: You can't guess the next piece before you've looked at the text.
```

```card
type: concept
id: letters_are_numbers
eyebrow: One more idea
title: Computers only work with numbers, so letters become numbers.
body: |
  A computer can add and multiply, but it can't "read" the letter **e**. So we give every letter a number:

  | letter | a | b | c | … | z |
  |---|---|---|---|---|---|
  | number | 1 | 2 | 3 | … | 26 |

  Then the word **"cab"** becomes **3, 1, 2**. The model only ever sees numbers like these. At the end we turn its numbers back into letters.

  Learning means slowly adjusting millions of numbers inside the model until its guesses get good. You'll see how in the next lessons.
```

```card
type: quiz
ref: what_it_does
```

```card
type: quiz
ref: why_numbers
```

## Hints

- There's nothing to install for this lesson. Just read and tap.
- If a card feels hard, read the primer first. It explains everything here slowly, with examples.

## Done when

- [ ] Finish the Learn cards {#cards auto=learn}
