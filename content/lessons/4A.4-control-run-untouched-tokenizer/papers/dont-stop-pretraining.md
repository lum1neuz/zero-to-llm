---
paper: "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks"
gist: A second phase of pretraining on text from the target domain keeps improving a strong pretrained model, and a short extra phase on the task's own unlabeled text helps on top of that.
---

## The question {#question}
> Does a model pretrained on huge, broad data still gain from more pretraining on the narrower text you actually care about?

By 2020, models like RoBERTa had been pretrained on a large, diverse corpus, and it seemed possible that they already covered every domain well enough. The authors tested whether continued pretraining on in-domain text still pays off.

## Domain-adaptive pretraining {#dapt}
> Continue the language-modelling objective on a large unlabeled corpus from the target domain before fine-tuning on the task.

They took RoBERTa and continued pretraining it separately on four domains: biomedical papers, computer science papers, news and product reviews. Then they fine-tuned each version on classification tasks from the matching domain. It's the same objective as pretraining, only on different text. That's what 4A.4's control run does with Mongolian.

## Task-adaptive pretraining {#tapt}
> A much smaller extra phase on the unlabeled text of the task itself helps too, and the two phases add up.

Task-adaptive pretraining uses only the task's own inputs, without labels, so it's cheap. It helped on its own, and domain then task adaptation together gave the best results across their tasks.

## Relevance matters {#relevance}
> Continued pretraining on an unrelated domain didn't help, and sometimes hurt.

As a control, they also adapted to a domain that didn't match the task. The gains vanished. So what helps is relevant text, not extra training steps as such. It's the same reasoning behind 4A.4: to credit something, compare against a run that changes only that thing.

## Choosing the data {#selection}
> When in-domain text is scarce, picking the unlabeled examples most similar to the task's data recovers much of the benefit.

They also tried retrieving the domain sentences nearest to the task's data and pretraining on those. It's a reminder that what's in the corpus counts as much as how big it is, which is 4A.3's whole point.

## How the course uses it {#course}
> 4A.4 is domain-adaptive pretraining of Llama 3.1 8B on Mongolian, kept deliberately plain so 4B can use it as a control.

You continue next-token training on 20M tokens of clean Mongolian with QLoRA and leave everything else alone, the tokenizer included. Held-out perplexity falls from 5.84 to 3.12 per token. 4B asks how much a Mongolian tokenizer adds on top.
