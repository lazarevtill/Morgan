# Long-lived synthetic memory on the primary Mac: quality gate failed

The reviewed fixed holdout produced **7/16 useful responses with Morgan inspection**, **7/16
with raw retrieved sources**, and **4/16 without memory**. The required gate was at least
14/16, zero critical errors, and no regression against either baseline. It failed. No production
change, real-memory migration, reader admission or merge follows from this measurement.

No critical error was evidenced in the 38 original returned holdout texts. Critical status
remains unknown for ten slots whose model answer was not obtained; that is not proof of zero
errors across all 48 planned slots. One returned response failed the action JSON contract and
remained an unavailable usefulness failure after its original text was audited.

## Prospective campaign and separation

Twenty-four fresh synthetic stories cover personal life and projects equally, with English
and Russian questions, current facts, permissions, revocation, unknown evidence and continuation.
Each has twelve source-writing sessions over at least 44 simulated days. Eight stories form
development and sixteen form holdout. An independent author froze the public inputs and private
semantic references before reader outputs; the runtime and campaign owner never opened private
gold. Independent protocol review preceded model calls. LAN requests contained synthetic inputs
only. This is sparse episodic coverage, not observed multi-day unattended operation or a
consolidation benchmark.

The v1 format preflight and v2 citation-rubric preflight failed before any product/model calls.
Their original cases and references were preserved rather than repaired after output. A separately
authored v3 passed semantic preflight with sufficient alternative proof, rather than requiring
all historical reference IDs. Safe missing-context abstention passes only a genuinely abstaining
endpoint; on an answer or continuation endpoint it remains a noncritical task failure.

The initially selected existing 8B embedding model was incompatible with the Mac's resource gate:
two embedding requests returned valid vectors, but the model's reported resident size of about
5 GiB left less than the required 4 GiB of reclaimable headroom. The first stored source and both
original failed receipts remain preserved. There were no scored reader outputs at that point.
The existing server expired the model naturally; no user app was killed, model installed or
server/security setting changed.

Before any scored response, a separately reviewed v4 switched fresh synthetic databases to
Unicode word features: fixed signed SHA-256 buckets in 1,024 dimensions, L2 normalization,
plus Morgan's existing FTS/vector rank fusion. This is a prospective CPU lexical diagnostic,
not semantic model embeddings, Morgan's whole-text hash test stub, or a new production default.
The exact same v3 stories and private references were retained.

All writes and reads used the real MemoryGate; every source session ran in a new process.
The caller discovered sources from the original query, closed explicit revision/support
pointers through scoped evidence calls, then supplied identical source projections to raw
retrieval and inspection. No caller-curated gold IDs, query rewriting or benchmark selection
was used. The no-memory arm received the same current request and empty evidence.

## Fixed denominators and the single post-development change

| Phase / discovery | No memory | Raw retrieval | Morgan inspection |
| --- | ---: | ---: | ---: |
| Development, top-4 seeds | 1/8 useful; 0 observed critical | 2/8 useful; 1 observed critical | 1/8 useful; 0 observed critical |
| First holdout, top-8 seeds | 4/16 useful; 0 observed critical | 7/16 useful; 0 observed critical | 7/16 useful; 0 observed critical |
| Holdout critical status unknown | 1/16 | 4/16 | 5/16 |

Development critical counts are observed counts; six timed-out answers were not obtained.
Development exposed missing completion and current-confirmation sources in the shared top-4
query discovery. Independent review admitted one prospective change: top-4 to top-8 recall seeds
and the matching seed cap, applied identically to both memory arms. Closure remained limited
to sixteen IDs and four rounds. Development was never rerun. Both changed memory arms are
development-unmeasured; their holdout rows are their first prospective test. These disjoint
phases do not establish a causal top-4 versus top-8 improvement.

The model, native template, prompt, query, lexical algorithm, scopes, references and scoring
gates stayed fixed. Complete context was bounded at 16,384 bytes; exact native input at 3,008
tokens, output at 1,024 tokens with a 4,096-token total budget and safety reserve. Generation
was bounded at 60 seconds inside a 90-second slot. No retries or answer selection occurred.
Holdout is terminal: no further tuning or replacement panel belongs to this campaign.

Original results were finalized before opaque export. Fresh graders received all planned rows,
public histories and separately filtered references without method mappings or costs. Explicit
supporting source IDs in the rationale count as semantic citations; irrelevant extra citations
and unsupported historical assertions fail. Context style can reveal a method, so this is
blinded to labels and costs rather than a claim of perfect blinding.

An export audit found that a schema-invalid response retained its original text on disk while
the first opaque export contained only a null decision. An independently reviewed, immutable
all-48 raw-text supplement restored that evidence before final critical auditing. Original
responses, inputs, labels and delivery failures were unchanged; no answer was repaired.

## Delivery and cost

| Holdout method | Delivered | Known input / output tokens | Inclusive reader seconds |
| --- | ---: | ---: | ---: |
| No memory | 15/16 | 3,978 / 2,320 | 733.5 |
| Raw retrieval | 11/16 | 23,054 / 3,095 | 719.4 |
| Morgan inspection | 11/16 | 22,951 / 2,983 | 688.9 |

Known raw-retrieval tokens include its schema-invalid received response. Missing usage is
unknown, never zero. Inclusive time includes failed slots; a shorter sum is not a quality gain.

Across development and holdout there were 72 planned slots, 71 chat dispatches and 302 physical
HTTP requests, including metadata, native template/counting and chat. One holdout native count
timed out before chat. Fifty-five answers were delivered; sixteen failed delivery: fifteen chat
timeouts and one schema-invalid response. The other planned slot failed before chat. Fifty-six
chat responses have known usage, totaling 68,409 prompt and 11,930 completion tokens; fifteen
dispatched chats have unknown usage. Provider cache details are retained but the summary does
not establish complete cache accounting.

The earlier embedding setup adds two HTTP requests, eight input items and 266 known embedding
prompt tokens; it is charged separately rather than erased by the resource-profile switch.
The CPU source setup used 312 owned processes, stored all 288 events and processed 312 lexical
text items with zero network embeddings. Peak child RSS was about 63.1 MiB; minimum observed
reclaimable headroom across source setup was 6.23 GiB. These are source-setup measurements,
not a bound on the remote reader server. Shared retained storage was capped at 512 MiB,
child RSS at 256 MiB and parent RSS at 512 MiB, with the 4 GiB RAM gate checked throughout.

Total monetary cost is **unknown**. Operator time, platform agents, hardware and electricity
were not metered. Offline diagnostics and independent review also have costs outside the
campaign's reader HTTP/token totals. Detailed aggregate receipts and input/grade hashes are in
[the companion summary](2026-10-long-lived-mac-summary.json); private gold and closed expected
answers are not published.

## What the failure supports next

Inspection did not exceed raw retrieval on this holdout. Delivered answers often identify the
correct next step without invoking the currently authorized synthetic simulator; unsupported
or irrelevant citations also cost usefulness, and timeouts remain material. The next bounded
work is a transparent public control of the caller's current authorization and action contract,
with a real synthetic effect receipt, before another quality hypothesis is considered. This
completed holdout must not be reused to tune or justify a passing reader.

The separate [core guarantees](../MEMORY_GUARANTEES.md), [reader controls](2026-10-reader-controls.md)
and [scale measurement](2026-10-memory-scale.md) remain valid within their own scope. Earlier V5,
V9, V10, V11r and V12 failures are not replaced by this result; CI and persistence do not prove
useful personal-memory continuation.
