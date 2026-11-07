# Conservative credential admission

## Decision

Reject a small set of credential-shaped strings instead of silently redacting source evidence.
The provider-rule work preserved at
[`archive/phase1a-capture-and-digest`](https://github.com/lazarevtill/Morgan/tree/archive/phase1a-capture-and-digest)
(tip `7172a524654191c11da0f85fdef4e7a6dbf8a56c`) motivated this subset. Its archive tables,
31-rule scanner, generic entropy/assignment/PII rules, placeholder exemptions and redaction
metadata are not imported. No migration or schema change is needed.

Replacing different source values with the same placeholder would change exact quotations
and could collapse distinct assertions during immutable replay. Refusal keeps accepted text,
IDs, attribution, event times and revision parents unchanged. No new rule can grant authority.

## Supported shapes

`memory/credentials.py` scans the entire supplied string using four fixed rule families:

- GitHub: `gh[pousr]_` followed by at least 36 alphanumeric characters, or `github_pat_`
  followed by at least 22 alphanumeric/underscore characters.
- OpenAI: explicit `sk-proj-` or `sk-svcacct-` prefix followed by at least 20 URL-safe token
  characters. Ambiguous legacy `sk-` strings are not covered.
- Anthropic: explicit `sk-ant-api03-` prefix followed by at least 20 URL-safe token characters.
- Armored private keys: a supported PEM or PGP private-key header, a newline, then at least
  32 base64-body characters, allowing whitespace between them and blank lines or recognized
  `Version`, `Comment`, `Proc-Type` and `DEK-Info` armor headers before the body.
  A header quoted alone is not
  refused; a truncated block with enough body is refused even without an end marker.

These are conservative **shape heuristics, not validation of a live credential**. Full-shaped
synthetic examples, revoked tokens and matching substrings inside larger identifiers also
refuse. Ordinary prefix documentation, hashes, public keys and contact details are not
redacted or blocked by these rules. There are no placeholder exemptions or scanning windows.
The error contains only the fixed rule name and a retry instruction, never the matching value,
source text, surrounding context or offset. Nothing calls a provider to validate a credential.

## Integration and persistence

- A new memory event is checked after canonical identity/revision validation and before its
  first embedding. Existing exact store replay still returns without embedding, rescanning
  or writing. A changed assertion under an existing ID still raises the identity conflict.
- An atomic chat turn checks both deep-copied events before either is embedded. A rejected
  assistant answer leaves no new user event, answer event or history row; the model call
  itself has already happened and cannot be undone.
- `ask` checks the question before strict token counting, recall embedding or model calls.
- Fact subject, predicate and object are checked before their write transaction. Checkpoint
  facts reuse that path. Accepted evidence and canonical fact identities are not rewritten.
- ChatGPT import checks each eligible visible source turn in full before its first non-replay
  piece can be embedded. Checking individual pieces could miss a token across a boundary.
  A refusal stops the import; prior accepted turns stay committed, but no new piece of the
  refused turn is stored. Exact historical resume retains canonical replay, attribution and
  erasure checks without embedding or rescanning. If an old turn is only partly imported,
  the entire source must pass before any new piece proceeds. Hidden/tool messages remain
  excluded under the existing importer contract.

Erasure generations, preparation outside write locks, revision validation and atomic index
writes are unchanged. A refusal cannot create a project or partially commit a turn/fact.

## Boundaries

This is **write admission and ask-question screening, not complete egress protection**.
It does not scrub or delete historical memory, snapshots, backups or legacy exact replays.
Old recalled memories and history can still enter model prompts. Standalone recall queries,
metadata fields (including repository remotes, paths and supplied entities), separate owner
and task journals, model-only drafting paths and direct low-level store use are not screened.

Unknown providers, generic passwords, JWTs, personal identifiers, encoded/escaped values,
secrets split across separate source strings, atypically wrapped private keys, unrecognized
armor metadata and future provider formats can pass. The guard must
never be described as permission to store a secret or a guarantee of no disclosure. A broader
redaction/egress policy needs a separate provenance and migration design.

## Verification

Synthetic tests cover positives and negatives, punctuation/JSON/Unicode adjacency, long
strings, placeholder-like text, partial key blocks, payload-free exceptions, exact replay,
identity conflict, rejected revisions, fact atomicity, whole-turn preflight, question admission,
answer refusal, whole-source import screening and historical import resume. The existing
immutable-event, revision, provenance,
import and forget suites remain the regression contract. No live model or real credentials
are needed.
