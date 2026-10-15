---
gist: A language model is a program that guesses the next bit of text, over and over. It learns good guesses by looking at lots of examples.
minutes: 8
---

## Start here {#start}
> This course teaches you how AI chat programs work by building one, step by step. This first lesson has no code and no math.

You've probably used ChatGPT, Claude or a similar chat program. You type a question and it writes an answer. It can feel like magic.

It isn't magic. Underneath is a program called a **language model**. By the end of this course you'll have built small ones yourself, and you'll understand the big ones.

You don't need to be good at math or programming to start. Each lesson explains the new ideas when they come up. Go slowly. Reading something twice is normal.

## The keyboard on your phone {#keyboard}
> Your phone's keyboard already guesses your next word. That's a tiny language model.

Open a messaging app and type **"good"**. Above the keyboard, your phone suggests a few words, maybe **"morning"**, **"night"** or **"luck"**.

How does it know? It doesn't understand you. It has simply seen many messages, and it noticed that after "good", people often write "morning".

That's the core idea of every language model:

> Look at the text so far, and guess what usually comes next.

The big chat programs do exactly this. They've just seen far more text, and they're far better at guessing.

## One small piece at a time {#loop}
> A language model writes by guessing one small piece, adding it, and guessing again.

When a chat program answers you, it doesn't write the whole answer at once. It repeats four steps:

1. **Look** at all the text so far.
2. **Guess** the next small piece. That might be a letter or a part of a word.
3. **Add** that piece to the end of the text.
4. **Repeat** from step 1.

Here's a model writing "the cat sat", one word at a time:

| Text so far | Guess |
|---|---|
| the | cat |
| the cat | sat |
| the cat sat | on |

Each row is one guess. A long answer is just this table with thousands of rows. That's why answers appear word by word on screen.

## Guessing with counts {#counting}
> The simplest way to guess is to count what came next in examples, then pick the most common.

Imagine the keyboard has seen these messages:

| Message | How many times |
|---|---|
| good morning | 5 |
| good night | 3 |
| good luck | 2 |

After "good", the word "morning" came 5 times out of 10. So the best guess is "morning".

That's a real, working language model. It's tiny, but it works. In lesson 0.3 you'll build exactly this kind of model, but for letters in names instead of words in messages.

## Chances, not certainties {#probability}
> A model doesn't know the next word. It knows how likely each word is.

Out of 10 messages starting with "good":

- "morning" came 5 times: **5 out of 10**, or **50%**
- "night" came 3 times: **30%**
- "luck" came 2 times: **20%**

These percentages are called **probabilities**. A probability is just "how likely": 0% means never, 100% means always.

Notice that they add up to 100%. Something has to come next, so all the chances together make one whole.

A model can always pick the most likely word. But then it would always write the same thing. So often it picks at random, using these chances: "morning" half the time, "night" a bit less, "luck" sometimes. That's why a chat program can answer the same question differently twice.

## Letters become numbers {#numbers}
> Computers can only calculate with numbers, so every letter or word gets a number.

A computer is a very fast calculator. It can't read the letter "e" the way you do. So before a model sees any text, we swap each letter for a number:

| letter | a | b | c | d | e | … | z |
|---|---|---|---|---|---|---|---|
| number | 1 | 2 | 3 | 4 | 5 | … | 26 |

The word **"cab"** becomes **3, 1, 2**. The model works only with these numbers. When it's done, we swap its numbers back into letters so you can read them.

This swap table is called a **vocabulary**. You'll build one in lesson 0.0.

## How does a model learn? {#learning}
> Learning means slowly adjusting lots of numbers inside the model until its guesses match the examples better.

Counting works for tiny examples. Real models do something more flexible. Inside a model there are many numbers, millions or billions of them. They work a bit like the knobs on a mixing desk.

Training goes like this:

1. Show the model some real text, with the last piece hidden.
2. Let it guess the hidden piece.
3. Check how wrong the guess was.
4. Turn the knobs a tiny bit, so next time the guess is a little less wrong.

Do this millions of times and the guesses become very good. You'll see exactly how the knobs get turned in lesson 0.2, with only a few lines of code.

## What you'll build in this course {#path}
> The course goes from tiny models you build by hand to a real model that's smaller, adapted to a new language, and served to users.

Here's the path, in plain words:

| Part | What you'll do |
|---|---|
| Start here | Learn the basic ideas, a little Python, and the bits of math you need |
| Phase 0 | Build tiny models by hand, including the name-guesser from this lesson |
| Phase 1 | Build the transformer, the design inside every modern chat model |
| Phase 2 | Make a model write text quickly |
| Phase 3 | Shrink a model so it fits on smaller computers |
| Phase 4 | Teach a model a new language |
| Phase 5 | Serve a model to many people at once |

You don't need to remember this table. Each lesson tells you where you are.

## Words you'll see {#glossary}
> A short list of the new words from this lesson.

- **Language model**: a program that guesses the next piece of text.
- **Probability**: how likely something is, from 0% (never) to 100% (always).
- **Vocabulary**: the table that turns letters or words into numbers and back.
- **Training**: showing a model many examples and adjusting it so its guesses improve.
- **Parameters**: the many adjustable numbers ("knobs") inside a model.
