---
paper: "Training language models to follow instructions with human feedback"
gist: Fine-tune GPT-3 on human-written demonstrations, then on human preference rankings with reinforcement learning, and much smaller models end up preferred over the original 175B one.
---

## The problem {#problem}
> A model pretrained to predict web text isn't trying to follow your instructions, so making it bigger doesn't make it more helpful.

GPT-3 continues text plausibly, which isn't the same as doing what a user asks. It can ignore the instruction, make things up or produce toxic output. The authors call this a misalignment between the training objective and what users want, and they set out to fix it with human feedback.

## Step 1: supervised fine-tuning {#sft}
> Labelers wrote good responses to prompts, and the model was fine-tuned on those demonstrations.

Prompts came from early API users and from the labelers themselves. A team of contractors wrote the responses they wanted the model to give. Fine-tuning on these pairs is ordinary supervised training on what the assistant should say. It's the stage 4A.5 reproduces, with the loss masked to the assistant's turn.

## Step 2: a reward model {#reward}
> Labelers ranked several model outputs for the same prompt, and a model learned to predict which output people prefer.

Ranking is easier and more consistent than writing, so it yields more data per hour of labeler time. The reward model takes a prompt and a response and outputs a score, trained so that preferred responses score higher.

## Step 3: reinforcement learning {#rl}
> The fine-tuned model was then optimised with PPO to maximise the reward model's score, with a penalty for drifting too far from the supervised model.

The penalty keeps the policy from exploiting quirks of the reward model. A variant that also mixes in pretraining updates reduced the drop in performance on standard NLP benchmarks, which the authors call an "alignment tax".

## What it achieves {#results}
> Labelers preferred outputs of the 1.3B InstructGPT over the 175B GPT-3, and the models were more truthful and less toxic.

The gap shows that the training signal mattered more than the parameter count for this kind of helpfulness. The authors are explicit about the limits: the model reflects the preferences of a particular group of labelers and researchers, and it still makes mistakes and can be pushed into harmful output.

## Evaluate what you trained for {#evaluation}
> The main metric was human preference on held-out prompts from real users, not benchmark accuracy.

Since the goal was following instructions, they judged that directly and reported standard benchmarks separately to watch for regressions. 4A.5's harness does the same on a small scale: questions held out from training, scored the same way for every model, with contamination checked before any score is trusted.

## How the course uses it {#course}
> 4A.5 does step 1 only, supervised fine-tuning on Mongolian instructions, and builds the eval that 4B reuses.

There's no reward model or reinforcement learning in this course. The masked loss, the chat template and the end-of-turn token are the supervised stage's mechanics, and they're what turns the control model into something that answers.
