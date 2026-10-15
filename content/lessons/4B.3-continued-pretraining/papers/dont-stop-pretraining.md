---
paper: Don't Stop Pretraining
gist: A second phase of pretraining on unlabeled text from the target domain improves a pretrained model on that domain's tasks, even for a large, broadly trained model.
---

## The question {#question}
> Does a model pretrained on huge, diverse text still gain from more pretraining on text from the domain you care about?

RoBERTa was trained on a large mix of web, news and book text. The paper asks whether that breadth is enough, or whether continuing the same unsupervised objective on in-domain text helps before fine-tuning on a task.

## Domain-adaptive pretraining {#dapt}
> Continue pretraining on a large unlabeled corpus from the domain, then fine-tune on the task as usual.

The authors pick four domains, biomedical papers, computer science papers, news and reviews, with two classification tasks in each. Domain-adaptive pretraining (DAPT) on each domain's text improves the tasks in that domain, in both high- and low-resource settings. The further the domain is from the original pretraining data, the bigger the gain.

## Task-adaptive pretraining {#tapt}
> Pretraining on the task's own unlabeled text is far cheaper and also helps, and doing it after DAPT helps most.

Task-adaptive pretraining (TAPT) uses just the unlabeled inputs of the task's training set, a much smaller corpus. It still improves results, and DAPT followed by TAPT gives the best results overall. Adding more task-like unlabeled text, selected automatically from the domain, helps further.

## The right domain matters {#irrelevant}
> Continued pretraining on an unrelated domain doesn't help and can hurt.

As a control, the authors pretrain on a domain that doesn't match the task. That is usually worse than not continuing at all. Continued pretraining helps because of what the text is, not because the model trained for longer.

## How the course uses it {#course}
> 4A.4 and 4B.3 are domain-adaptive pretraining with Mongolian as the domain, and 4A.4 is the control the paper's logic calls for.

Mongolian is far from Llama 3.1's pretraining mix, the case where the paper finds the largest gains. 4A.4 measures what continued pretraining alone does. 4B.3 adds one change, the extended vocabulary, and keeps the text and settings the same, so the difference between them is the tokenizer.
