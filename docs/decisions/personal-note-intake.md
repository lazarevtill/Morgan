# Personal notes, reviewed commitments and a daily plan

This workflow makes the existing planner usable with explicit personal input. A local model
proposes task-bearing lines from one note. The owner reviews those lines, supplies planning
fields and stores a commitment through the normal memory gate. No inferred text grants authority.

## 1. Propose from one note

Use an already running trusted local llama-server, configured with an explicit model alias.
For a deliberately started foreground instance, the measured CPU options were:

```bash
llama-server --model /path/to/existing-model.gguf --alias morgan-granite \
  --host 127.0.0.1 --port 11436 --gpu-layers 0 --ctx-size 1024 \
  --threads 4 --batch-size 64 --no-warmup
```

Stop that instance after use. No command below starts or installs it. Then prepare a proposal:

```bash
python -m morgan_brain.note_proposal_cli \
  --endpoint http://127.0.0.1:11436 --model morgan-granite \
  --seconds 20 --output /path/to/private-drafts/proposal.json < /path/to/note.txt
```

This sends the piped note to the chosen server. Nothing scans personal directories or reads
credentials. Input is bounded to 4,096 UTF-8 bytes and 32 lines. The output directory must be
private; the file is created mode 0600 and never overwritten. The JSON draft records the source
hash, verbatim selected lines, `agent_inferred`, `review_required: true` and `approved: false`.
Stdout contains only status and a count. No memory, permission file or task journal is opened.

The client requires an exact `/v1/models` identity and matching response model, schema-constrained
chat output and normal completion. Invalid references, tool calls, refusal and truncation fail
closed. Its whole HTTP operation has a maximum 30-second deadline and no retry. The shared local
endpoint guard rejects DNS, public addresses, credentials, redirects and environment proxies.
The operator still must trust the server: a private IP does not prove no remote forwarding.
Do not pipe secrets into this workflow. Cancellation cannot guarantee remote compute has stopped.

The CLI bounds inference after stdin is read; an open pipe can still wait for its writer.
The note must fit the server's configured context. A context failure produces no draft.

## 2. Store a reviewed commitment

Review the proposal and explicitly choose the fields. Example values below are illustrative:

```bash
morgan commitment 'Water the plants' --project personal \
  --due 2026-11-06 --minutes 10 --priority normal --status open \
  --source user_stated --author-id OWNER --event-id plants-20261106 \
  --effective-at 2026-11-06T08:00:00Z --json
```

The normal Morgan settings choose the owner and database; `--author-id` is provenance, not an
identity switch. `--source` defaults to `unknown`, exactly as `remember` does. Only use
`user_stated` for an actual user statement or reviewed confirmation; model proposals remain
`agent_inferred`. The planner rejects inferred commitments. Every field is validated before
opening the memory context, and capture reuses `cmd_remember`, its replay and revision rules,
repository classification and sole gate write path. No second commitment database is created.

The memory backend may need its configured embedding server. The existing explicit `hash`
backend works without inference for exact-ID workflows in a new isolated database, but does not
provide semantic search. Do not switch a populated embedding database to hash as a fallback.

## 3. Plan and record completion

Feed the returned current evidence ID to `daily_brief_cli` as documented in
[memory-backed daily brief](memory-backed-daily-brief.md). Delivery of the plan remains a
separate explicit local-note approval. Storing a commitment never authorizes doing its contents.

To mark it complete, use `morgan commitment` with the reviewed fields, `--status done`, a new
stable `--event-id`, `--revises-event-id plants-20261106`, and the actual later `--effective-at`.
This preserves the original event. Use the new ID in subsequent plans; old/superseded evidence
is rejected. The completed commitment is listed as excluded, not silently forgotten.

## What ran on the Mac

On 2026-11-06 no inference server was already listening on the inspected local sockets.
The earlier Ollama log showed a GPU-discovery watchdog timeout. The existing Ollama bundle
also contains llama-server 0.8.0-dev (`39febd2ad`). A finite direct CPU invocation used the
already cached Granite 4.1 8B GGUF (5.35 GB), four threads, zero GPU layers, context 1,024,
batch 64 and no warmup. No model was downloaded, quantized or changed.

The first raw-completion trial generated eight tokens at about 9.7 tokens/second but failed the
exact answer format. Switching to the model's chat template and JSON schema gave three exact
results: English two tasks, Russian two tasks excluding a completed item, and zero tasks for
completed/cancelled lines. Server readiness took 14.3 seconds; requests took 4.62, 4.22 and
2.63 seconds. The full run ended in 26.1 seconds and its owned server was stopped.

The actual proposal CLI was then tested with a separate synthetic Russian note. It selected
exactly two unfinished tasks and excluded the completed purchase. Readiness took 12.64 seconds;
the full invocation and cleanup took 18.30 seconds. After explicit fixture review, both selected
lines were captured using `morgan commitment` in an isolated hash-backend memory database and
produced a two-item, 20-minute focus plan through the real planner. The server was stopped and
the loopback port was verified free.

This is feasibility evidence, not a reliability qualification. Three small examples cannot
justify autonomous intake. The 0.11 GB cached Phonon model is speech-only; cached Gemma is
9.46 GB and other completion models are at least 5.35 GB. No smaller suitable cached text model
was established. The earlier LAN endpoint remains unverified for inference.

For sustained usefulness Morgan still needs broader real-world evaluation with user-selected
data, a deliberately configured inference lifecycle, reviewed intake UX and a real scheduling
policy. These commands are explicit and finite; they install no service or background agent.

The structured chat API follows the upstream
[llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
checked 2026-11-06. The existing Ollama synthetic benchmark remains available independently.
