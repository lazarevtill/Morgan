# Local alternatives and a second-pass validator: neither adopted

On 2026-11-06 the existing local models and one frozen validation method were tested.
**Keep Granite and mandatory owner review.** The alternative did not meet the current
request deadline; validation removed false tasks but also removed a valid replacement.
No production prompt, model default or automatic task authority changes follow.

## Protocol and scope

[Cases](cases.json) are 24 newly frozen synthetic notes: 12 parallel English/Russian pairs,
eight notes each for ownership, cancellation/replacement and renewed requests. They were
not used to design another prompt after execution. Labels are by the same investigator,
not an external blind benchmark; translated pairs are not independent statistical samples.
The [protocol](protocol.json) and [execution plan](execution-plan.json) precede corpus runs.

The validator gets the entire note plus the baseline's proposed indices. It can remove
proposed lines but never add one; added indices invalidate its response. It checks owner,
quotation and latest status, using the fixed [prompt](validator.txt). Empty results remain
fallible, not proof of no obligations. The acceptance gate requires a strict improvement
without more FP or FN overall, per language or per category, complete valid coverage,
request bounds and a child RSS below 12 GiB. No candidate tuning occurred.

All inference used existing weights and the native llama-server, four CPU threads,
GPU disabled, private Unix sockets, 1024-token context and temperature zero. Each cycle
was bounded to 120 seconds; each selection/validation request to 20 seconds. No owner
messages, secrets, downloads or paid APIs were used. The already running owner pilot's
model lock serialized the experiment with owner work; the same STOP applied and remaining
batches would defer if any owner message arrived. The pilot configuration was unchanged.

## Existing alternatives

The local inventory contained Granite, nimble, ornith, a qwen3 embedding model, and
Gemma/Phonon artifacts for a different runner. Only the installed GGUF chat alternatives
were tried through the existing bounded native adapter; no new runner was installed.
The local Ollama loaded-model list was empty before the probes.

| Single synthetic canary | Total wall | Loading | Child CPU | Peak child RSS |
| --- | --- | --- | --- | --- |
| nimble 9b-q4_K_M | 34.968 s | 18.782 s | 72.537 s | 8.70 GB |
| ornith-1.5 9b | 31.419 s | 15.328 s | 72.962 s | 8.95 GB |

Both selected the single requested line correctly. The protocol chose the faster valid
canary, ornith, before seeing corpus results. This is a compatibility/latency choice,
not evidence that one model has better task understanding.

Ornith produced no valid result within the 20-second request bound on its first two
corpus cases. Per the frozen stop rule, its run ended after 55.562 seconds: two failed
attempts, 22 unattempted, **zero quality-scored responses**. Child CPU was 164.069 seconds,
peak child RSS 8.99 GB. This does not establish poor semantic quality: it establishes
failure to qualify under this exact CPU/runtime/deadline configuration. Nimble's quality
on the corpus was not measured. See [canaries](ornith-canary.json),
[nimble canary](nimble-canary.json), [attempts](ornith-results.json) and [run](ornith-run.json).

## Granite versus validated Granite

All 48 outputs were structurally valid. Exact scores the complete selected source-line
set; FP/FN count lines, not executed tasks. Latencies exclude model loading; the validated
latency includes the baseline and the additional call, when the baseline selected anything.

| Slice | Baseline exact | Validated exact | Baseline FP/FN | Validated FP/FN |
| --- | --- | --- | --- | --- |
| All 24 | 19/24 | 20/24 | 4/1 | 2/2 |
| English 12 | 10/12 | 11/12 | 2/0 | 1/0 |
| Russian 12 | 9/12 | 9/12 | 2/1 | 1/2 |
| Ownership 8 | 4/8 | 6/8 | 4/0 | 2/0 |
| Cancellation/replacement 8 | 8/8 | 7/8 | 0/0 | 0/1 |
| Renewals 8 | 7/8 | 7/8 | 0/1 | 0/1 |

Median time rose from **4.210 s to 7.179 s**; observed maximum from 7.068 s to 15.622 s.
The combined six-cycle experiment took 266.226 s and 793.329 child CPU-seconds, peaking
at 9.11 GB child RSS. CPU includes loading, baseline and validation; it cannot be assigned
to the validator alone. RSS is macOS RUSAGE_CHILDREN's reaped child-tree peak in bytes,
not a hard memory cap or whole-machine usage. See [raw outputs](granite-results.json),
[run measurements](granite-run.json) and [group metrics](summary.json).

Validation correctly removed two duties belonging to somebody else. But it also dropped
this valid Russian replacement: “Большую отменяю; вместо неё мне нужно купить маленькую.”
The earlier line concerned a large shelf; the new obligation is a small shelf. Thus recall
fell from 11/12 to 10/12 despite higher exact accuracy. The non-regression gate failed.
Do not hide this tradeoff by reporting only accuracy, precision or fewer selected lines.
These observations use a different corpus from PR #106 and are not a direct trend measure.

## Reproduction and next practical limit

Use the recorded baseline runtime `9628401befa379bf6a2c5e223e61102b991b4a09`, existing
trusted model files, and an explicitly selected empty owner workspace for its model lock.
Do not contend with another model: use the cooperating pilot lane or stop independent
model work. The script reads only the frozen synthetic corpus for inference and stops
remaining batches if that workspace has owner messages. Output must be a new directory.

```sh
"$python" -I docs/measurements/task-validation-2026-11-06/reproduce.py \
  --name granite --server "$server" --model "$granite_weights" \
  --workspace "$empty_owner_workspace" --output "$new_output_directory"
```

Use `--name ornith` with its recorded weights for the alternative arm. Hashes, raw outputs
and measurements are retained. The runnable copy generalizes the executed experiment's
local paths into explicit arguments; it does not modify installed files or a service.
A failure before model readiness may leave partial coverage; never count it as success.

The next concrete limitation addressed was deployment lifetime: the owner loop stopped
at expiry but its login entry could start another expired check at each future login.
The separate finite-login-expiry change disables that exact current-account job after
lifetime exhaustion, preserving data. It is independent of these rejected quality changes.
Owner review, explicit target selection and cancellation checks remain necessary.
