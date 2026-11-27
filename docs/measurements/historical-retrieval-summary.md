# Historical retrieval summary

Preserved from `docs/ROADMAP.md` at commit `fa999b65fd60039fa1400eb198d2360d72737f0a`
on 2026-11-27. The text and measurements below retain that summary's original claims.
The Qwen3-Embedding-0.6B experiment's exact date and raw run artifacts were not located in
this repository; its values are a historical summary, not a newly reproduced result.
The linked September report supports the 8b run. For the later synthetic run, see the
[November reliability recheck](2026-11-reliability-recheck.md#end-to-end-evidence).

## Original bundled-probe summary

`tests/memory_quality/` holds 60 labelled probes over an 86-memory corpus, half of it
Russian, with every query written to share few or no words with its target so the keyword
signal cannot carry it. `pytest --live` runs them against a real embedding endpoint. With
`qwen3-embedding:8b` at 4,096 dimensions and no floor
([`2026-09-phase0-baseline.md`](2026-09-phase0-baseline.md)):

| | recall@8 | MRR | stale-first | abstain |
| --- | --- | --- | --- | --- |
| overall | 0.89 | 0.51 | 0.70 | 0.00 |
| single-hop | 0.90 | 0.58 | — | — |
| temporal | 1.00 | 0.66 | — | — |
| knowledge-update | 1.00 | 0.50 | 0.70 | — |
| multi-hop | 0.62 | 0.16 | — | — |
| unanswerable | — | — | — | 0.00 |

Stale-first measures order, not presence: the same memory is forbidden by "where do I live
now" and expected by "where did I live before", so a metric counting presence could only be
satisfied by suppressing it, which breaks the question that asks for it.

Semantic retrieval works: a query and its answer sharing no token find each other, in both
languages. Two categories do not, and both are open work rather than regressions.

The bundled probes are built so the keyword signal cannot carry them, so they cannot show
what keyword search or stored names add. A real archive does.

## On a real archive

~2,000 imported conversation turns and three sets of questions: 90 a chat model wrote from
sampled turns, each labelled with its source; 60 lookups, each naming a term the keyword index
finds in at most three memories; and 38 authored questions the keyword index confirms the
archive never mentions. Qwen3-Embedding-0.6B.

| recall signals | paraphrase recall@8 | MRR | lookup recall@8 | MRR |
| --- | --- | --- | --- | --- |
| vector only | 0.92 | 0.77 | 0.88 | 0.65 |
| **vector + keyword (recall as it is)** | **0.90** | **0.72** | **1.00** | **0.88** |
| + the entity ranking fused | 0.83 | 0.59 | 1.00 | 0.94 |
| + the entity ranking, narrowed by a semantic index | 0.76 | 0.53 | 0.97 | 0.91 |

Lookups were chosen by what the keyword index finds, so keyword search finding all of them is
partly by construction; what they show is that an embedding alone misses one name in eight.
A stored name is in the memory's text, so fusing the entity ranking counts the keyword match
twice: it ranks lookups a little higher and paraphrases much lower. Narrowing the search to
the memories that share a name with the question (VoiceMem's semantic index) cut the answer
out of questions and put none in.

The relevance floor separates the two kinds of question: a random answerable question has
the wider margin 97% of the time. At 0.11, on the half of the questions held out of the fit,
it kept 24 of 25 answers and silenced 12 of 13 unanswerable questions; over all of them, 141
of 150 answers and 34 of 38 silences.

With `qwen3-embedding:8b`, over 3,010 imported memories and 128 labelled questions, the floor
is 0.08. On the 38 questions sealed from the fit, recall@8 is 0.84 against 0.88 with no floor,
and 85% of the unanswerable questions among them are silenced; on the 90 fitted ones, 0.88
against 0.89, and 84%. A higher margin silences no more of the sealed questions and costs
recall ([`2026-09-phase0-baseline.md`](2026-09-phase0-baseline.md)).
