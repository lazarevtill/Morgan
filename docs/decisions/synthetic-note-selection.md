# Synthetic note selection before autonomous task intake

Morgan needs to turn notes into useful task proposals, but model accuracy has not yet been
established on the available hardware. The first experiment selects source lines containing
explicit unfinished obligations. It deliberately does not infer deadlines, durations, priorities,
ownership, permissions or executable actions.

## Reproduce

Against an already running, trusted Ollama server and an exact model name from its inventory:

```bash
python -m morgan_brain.eval.note_selection \
  --endpoint http://127.0.0.1:11434 --model INSTALLED_MODEL_NAME --seconds 45
```

The CLI sends only six bundled synthetic English/Russian notes, including completed/cancelled
items, a hypothetical wish and a quoted instruction. It reads no environment configuration,
credentials, private notes or memory database. Output is JSON on stdout; nothing is persisted.
The provider module contains all HTTP traffic. The existing daily-brief workflow is unchanged.

The model returns `status` and integer line numbers. Strict validation rejects coercion,
out-of-range or duplicate references, extra fields, mismatched model identity, incomplete output
and refusal with selections. Accepted text is copied verbatim from the source. This prevents
invented task wording; it does **not** prove the line describes a real unfinished obligation.
The negative fixtures measure that distinction. Human review is still necessary.

## Bounds and failure handling

- Explicit literal loopback, RFC1918 or IPv6 ULA HTTP address and port only; no DNS, embedded
  credentials, redirects or environment proxies. Private addressing is not server authentication.
- Exact model must already appear in `/api/tags`. Advertised remote models and cloud names are
  rejected. A server can hide forwarding: the operator must independently trust its configuration.
  This is not proof that inference is local or free.
- One attempt per case, at most six; at most 15 seconds per case and 120 seconds per run
  (45 default). Each deadline covers inventory, generation and response consumption together.
- At most 256 generated tokens, 2,048 context tokens and 256 KiB decoded response per request.
  No pulls, retries, keep-alive changes, server starts, unload requests or service reconfiguration.
- Timeout, refusal, malformed output, HTTP failure and unavailable host are separate outcomes.
  Unavailable, timeout or unavailable model stops the remaining cases. Remote error bodies are
  never emitted. Cancellation closes the client; it cannot guarantee remote compute stops.

The scorecard reports valid-case exact accuracy and true/false positive/negative counts alongside
coverage, failures and per-case wall time. A failed connection has null quality metrics, never
zero accuracy. Latency includes inventory, model loading and transport; it is not tokens/second.
Six public fixtures cannot establish production quality; `production_qualified` is always false.
Even a perfect result cannot grant permission, write a commitment or enqueue a task.

## Measurement on 2026-11-06

The user-designated LAN Ollama endpoint refused TCP connections (errno 61). Inventory endpoints
were unavailable. The finished benchmark repeated one bounded attempt with a cached-model name
as the requested candidate: `ibm/granite4.1:8b`. It stopped after 0.0376 seconds, with zero valid
cases and five unattempted cases. No generation request was reached, so the candidate's presence
on that server, quality and inference speed remain unknown. The local Mac cache is not evidence
of the LAN inventory.

A preceding isolated Mac experiment found cached model manifests, but one synthetic Granite
request timed out after 90 seconds. Its temporary process was stopped. That experiment does not
establish model quality either. This stage did not start or modify any model service.

Next: restore access to the existing server, inspect its installed model names, and run this
small benchmark with a finite budget. Only after useful measured output should Morgan gain a
reviewable note-to-commitment conversion. Preserve `agent_inferred` provenance and explicit
source references; obtain reviewed structured commitments before the daily planner consumes them.

## API references

Checked 2026-11-06: Ollama documents JSON-schema `format`, `stream: false`, generation options,
and completion fields in [Generate](https://docs.ollama.com/api/generate), and installed model
metadata in [List models](https://docs.ollama.com/api/tags). No generic OpenAI capability fallback
is assumed for this experimental native Ollama probe.
