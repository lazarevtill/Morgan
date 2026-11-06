# A finite, scheduled local model cycle

The personal-note workflow can now own a local model for one explicitly requested operation.
It does not install a daemon, recurring timer or background queue. One bounded delay allows
scheduling within the invocation; stopping or crashing the caller never creates a later retry.

```bash
python -m morgan_brain.model_cycle /path/to/private-cycle \
  --server /path/to/trusted/llama-server --model /path/to/existing-model.gguf \
  --seconds 90 --delay 5 --threads 4 < /path/to/note.txt
```

After the five-second delay, the model loads, proposes task-bearing lines and exits. The private
workspace receives immutable `proposal.json`, compatible with the reviewed intake from
[personal note intake](personal-note-intake.md). The result remains `agent_inferred`, unapproved
and review-required. No memory database, task permission or external-action adapter is opened.

## Ownership, scheduling and resources

The caller takes the existing workspace supervisor lock. A short-lived guardian takes its worker
lock and owns exactly one native child. Thus a caller that dies cannot let another cycle in the
same workspace overlap the model still being shut down. The guardian starts its own monotonic
expiry before the scheduling delay and enforces it independently of the caller.

The model listens on a Unix socket inside a fresh private temporary directory. No TCP port is
opened, discovered or reused; there are no credentials or proxy settings. Model HTTP operations
stay in the provider module. The fixed command loads only an explicitly selected local file,
with zero GPU layers, context 1,024, batch 64 and one to four CPU threads. The model file must
be at most 8 GB. No downloads, new quantization, remote routing or server reconfiguration occur.
The selected executable and weights must be trusted; this is process ownership, not a sandbox
for untrusted binaries or same-account adversaries.

A cycle is 5–120 seconds including the delay, which is 0–60 seconds and shorter than the cycle.
Each selection is at most 20 seconds or the remaining cycle budget. Native shutdown gets one
second for TERM, then one second for KILL/reap. The caller waits up to five seconds for guardian
cleanup. Process startup and cleanup add overhead around these deadlines. Thread/context/weight
limits constrain requested resources; they do not enforce a hard RSS or CPU-percentage quota.
The native process may allocate more RAM than its model-file size.

The SDK context manager can keep the model loaded for several bounded selections within one
cycle, as used by the quality experiment. It does not extend the guardian expiry. Arbitrary
caller code can outlive the context budget, but the guardian still stops the native model.
The CLI performs only one selection. Stdin is bounded after reading; an unclosed input pipe can
wait for its writer before the cycle begins.

## Stop and recovery

Create the existing workspace STOP marker with:

```bash
python -m morgan_brain.local_notes_cli /path/to/private-cycle stop
```

The guardian checks it every 50 ms, including while waiting for the scheduled start and while
inference is in flight. The caller also checks STOP before and after inference. Native TERM is
followed by KILL if needed, and the owned process is reaped. SIGTERM/SIGINT to the guardian use
that same cleanup. Caller cancellation requests guardian shutdown. Hard-killing only the caller
leaves the guardian alive until its finite expiry; the worker lock prevents overlap meanwhile.

A crash of the model ends the operation. There is no automatic restart or replay of inference.
After fixing the problem, a new explicit invocation can run again. If `proposal.json` already
contains a complete matching grounded proposal for the same source hash, the caller returns
`already_written` without inference. A conflicting or malformed artifact is rejected. Publication
is atomic and never overwrites a prior result. This does not promise exactly-once computation:
concurrent attempts can race before publication, but cannot overwrite or multiply the artifact.

STOP is persistent. Use the existing `resume` command only after checking why the cycle stopped;
it requires that both locks are free. No saved PID is ever used for control. If both caller and
guardian are forcibly killed, the native process can survive: external OS-level supervision is
still needed for that stronger failure model. A killed caller may also leave an empty private
temporary directory. The code never searches and kills arbitrary processes or adopts old PIDs.

## Quality evidence on 2026-11-06

The real cached Granite 4.1 8B model ran through this new Unix-socket lifecycle on the Mac,
using bundled llama-server 0.8.0-dev (`39febd2ad`), four CPU threads and the limits above.
The [baseline](../measurements/2026-11-model-cycle-baseline.json) covered 16 public synthetic
notes: English/Russian, cancellation, late completion, changed plans, third-party duties,
hypothetical wishes and quoted instructions. It produced 13 valid responses and three refusals;
10/16 cases were exact (10/13 among valid responses). False positives included another person's
duty, a completed historical task and a hypothetical statement. Total lifecycle: 68.99 seconds,
including 13.89 seconds until ready.

The prompt now explicitly considers the note author's own duties, later cancellations/completions
and a successful empty result when there are no tasks. The
[revised measurement](../measurements/2026-11-model-cycle-revised.json) includes the same 16 cases
plus four newly added holdout examples. All 20 responses were valid; 16/20 were exact: 13/16 on
the reused cases and 3/4 on holdout. Aggregate task-line counts: 12 true positives, one false
positive and three false negatives. Total lifecycle: 89.25 seconds, including 12.97 seconds until
ready. The guardian stopped the model after each run.

The improvement has a tradeoff: fewer false selections but more missed tasks. The model still
selected a third party's duty and missed some changed or quoted-context obligations. Reused
examples are not independent validation, and four holdout examples cannot establish reliability.
There is no automatic promotion to user-stated commitments or execution authorization. Broader
user-selected evaluation and a reviewed intake interface remain necessary before useful 24/7
operation. Persistent scheduling and OS-level service management remain separate, unimplemented
steps; this change establishes a finite model lifecycle that they can later invoke.

A later [frozen ownership/renewal comparison](../measurements/owner-quality-2026-11-06/README.md)
found 16/24 exact held-out notes for both prompts. The candidate reduced missed lines
from four to two but increased false-positive lines from five to six, including quoted
foreign requests and a cancelled reminder. It was rejected; production selection and
mandatory review remain unchanged. This corpus differs from the earlier 20-note test.
