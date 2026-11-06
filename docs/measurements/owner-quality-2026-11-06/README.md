# Ownership and renewed-request selection: rejected prompt change

**Decision: retain the production prompt and mandatory owner review.** The candidate
reduced missed task lines but introduced false positives, without improving exact-note
accuracy. No task matching, automatic confirmation or service deployment is added.

## Frozen comparison

On 2026-11-06, 32 new synthetic notes were frozen before execution: eight development
notes and 24 held-out notes, evenly divided between Russian and English. The held-out
set contains eight ownership, eight renewed-request and eight control cases. Wording
is new relative to the previous experiments; labels and the candidate were written by
the same investigator. This is not an externally blind or representative user benchmark.

One candidate only, with no prompt changes after observing results. It explicitly
recognizes reminders about the author's unfinished task and distinguishes the author's
duty to remind another person from that person's duty. The frozen acceptance condition
requires better exact accuracy and fewer missed lines, no added false positives and
no worse language/category results; existing regression probes would run only after
passing that first gate. The candidate failed, so that additional model run was unnecessary.

Artifacts: [cases](cases.json), [protocol and hashes](protocol.json),
[baseline prompt](baseline.txt), [candidate prompt](candidate.txt),
[all outputs](results.json), [grouped metrics](summary.json).
The corpus SHA-256 is `2dc49c10e371aa858897d62aa75c5c768273a0cb5ce69917fc649e73eed2a502`.

Runtime was a normal wheel from `9628401befa379bf6a2c5e223e61102b991b4a09`
(the same application code merged in PR #105), on macOS arm64. The existing local
Granite 4.1 8B GGUF was selected by its recorded model blob identifier
`ed902ac9eb6adce5a90c6a08c8ea201b50e23fdc5976d1cd0362006afac5309e`.
Native llama-server ran through the existing private-socket guardian: four CPU threads,
no GPU, 1024-token context, temperature zero, at most 20 seconds per request. Eight
model cycles were limited to 120 seconds each. Arm order alternated within each batch.
No model downloads, personal messages, network inference or task/memory writes occurred.
All 64 responses were structurally valid. That does not establish semantic correctness.

## Results

Exact means the entire selected line set matches the label. TP/FP/FN count task-bearing
source lines; they are not counts of executed tasks or delivered reminders.

| Held-out slice | Current exact | Candidate exact | Current FP/FN | Candidate FP/FN |
| --- | --- | --- | --- | --- |
| All 24 notes | 16/24 | 16/24 | 5/4 | 6/2 |
| English, 12 | 9/12 | 9/12 | 2/1 | 2/1 |
| Russian, 12 | 7/12 | 7/12 | 3/3 | 4/1 |
| Ownership, 8 | 3/8 | 3/8 | 4/2 | 4/1 |
| Renewed requests, 8 | 6/8 | 6/8 | 1/1 | 2/0 |
| Controls, 8 | 7/8 | 7/8 | 0/1 | 0/1 |

Held-out recall improved from 12/16 (75%) to 14/16 (87.5%), while precision fell from
12/17 (70.6%) to 14/20 (70%). Development exact improved from 4/8 to 5/8; those notes
are excluded from the held-out results. These are small-sample single-run observations,
not evidence of a statistically established improvement. They cannot be compared directly
with the older 16/20 score because the notes differ.

Three held-out cases improved: `en_owner_delegate`, `ru_owner_delegate`, `ru_renew_two`.
Three regressed: `en_quoted`, `ru_quoted`, `ru_renew_then_cancel`. The first two regressions
selected requests quoted from another person, and the third retained a cancelled reminder.
The baseline also failed an English cancelled-renewal case. Both prompts missed a real
English task beside somebody else's duty. An exact source quote is not proof of ownership.

## Reproduce and interpret

Use the recorded pinned runtime and trusted existing server/model files. The script only
reads these synthetic fixtures and writes into a **new** private output directory:

```sh
"$python" -I docs/measurements/owner-quality-2026-11-06/reproduce.py \
  --server "$server" --model "$model" --output "$new_output_directory"
```

The script checks the frozen corpus/candidate hashes and verifies that the installed
baseline prompt still matches. It uses the real production selection and validation path,
intercepting only the candidate system-prompt field within this isolated experiment.
It never patches installed files, opens owner memory or installs an agent. A startup
failure can abort a batch; inspect coverage and validity before comparing results.
The runnable version adds explicit paths, formatting and baseline checking to the executed
local experiment. Raw stored metrics were independently recomputed for all 64 outputs.

The next useful quality investigation should separately test ownership evidence and
latest-status resolution, including quoted requests and cancellation after renewal.
That is a hypothesis, not an implemented or proven solution. Explicit task selection,
source review and confirmation remain necessary. No seven-day pilot was started.
