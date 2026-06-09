# Ecosystem Research — Ground Truth as of June 2026

**Status:** reference (research snapshot, 2026-06-09; round 2 included below)
**Parent:** [Personal Agent OS vision](2026-06-09-personal-agent-os-vision.md)
**Method:** three parallel web-research agents (interop standards, on-device AI, platform
landscape), June 9 2026. Condensed; source URLs inline. Facts here decay — re-verify before
building on anything load-bearing more than a quarter out.

---

## 1. Interop standards

**MCP — won the tool layer.** Current spec 2025-11-25; the **2026-07-28 revision** (RC frozen
May 2026) is a big break: stateless core (session handshake removed), Tasks moved to an
extensions framework, six OAuth-hardening SEPs, MCP Apps (sandboxed UI) as extension, W3C Trace
Context, and **deprecation of Roots/Sampling/Logging** (12-month removal).
Official registry ~9,650 servers (pre-GA); 97M+ monthly SDK downloads; all major clients.
([changelog](https://modelcontextprotocol.io/specification/2025-11-25/changelog),
[RC](https://blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate/),
[adoption](https://www.digitalapplied.com/blog/mcp-adoption-statistics-2026-model-context-protocol))
→ *Bet hard; build stateless-native; avoid deprecated features.*

**A2A v1.0** (Linux Foundation, early 2026; ACP merged into it Aug 2025): 150+ orgs, production
at Microsoft/AWS/SAP/ServiceNow — **enterprise cross-vendor traction, not personal-agent**.
([a2a-protocol.org](https://a2a-protocol.org/latest/),
[LF](https://www.linuxfoundation.org/press/a2a-protocol-surpasses-150-organizations-lands-in-major-cloud-platforms-and-sees-enterprise-production-use-in-first-year))
→ *Agent Card only; endpoint deferred.*

**Agent Skills (SKILL.md)** open-sourced Dec 2025 ([agentskills.io](https://agentskills.io/home));
~32 tools and ~490K skills by March 2026. **AGENTS.md**: 60K+ repos; both governed by the
**Agentic AI Foundation** (Linux Foundation; Anthropic/OpenAI/Block; 170+ members).
→ *De facto standards; adopt.*

**Agent↔UI:** fragmented — AG-UI (CopilotKit, $27M raise), Google A2UI (v0.8 preview), MCP Apps
(rides MCP itself — strongest gravity), OpenAI Apps SDK (ChatGPT-only).
→ *MCP Apps first if ever; ignore the rest.*

**Identity/payments:** Web Bot Auth (RFC 9421-based) got an IETF WG early 2026
([draft](https://datatracker.ietf.org/doc/draft-meunier-web-bot-auth-architecture/)); payments
(x402 v2, Google AP2, OpenAI ACP) unconsolidated. → *Watch Web Bot Auth; ignore payments.*

**Memory interop: no standard exists, none coming.** Vendor silos (Anthropic memory tool,
OpenAI file_search, Google Memory Bank); Mem0 the de facto OSS layer.
([state of memory](https://mem0.ai/blog/state-of-ai-agent-memory-2026))
→ *The vacuum the Memory Passport fills.*

**OTel GenAI semconv** still experimental, incl. agent/MCP spans; MCP 2026-07-28 bakes in trace
context. ([spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/)) → *Adopt names, expect churn.*

## 2. On-device / local AI

**Models:** a Qwen 3.5 (Feb 2026, Apache-2.0, 0.8B–27B dense + MoE; 9B ≈ 6–8 GB @ Q4, 262K ctx,
tuned for tool calling) / Gemma 4 (Mar 2026; E4B natively multimodal; 26B-A4B MoE) duopoly.
Mistral Small 4, Ministral 3, Phi-4 current; **Llama is out of the local race** (no small Llama 4;
Meta pivoted closed). Tool-calling small-model leaders: GLM-4.5/Qwen3-32B on BFCL v3.
([comparison](https://codersera.com/blog/gemma-4-vs-qwen-3-5-comparison-2026/),
[BFCL](https://pricepertoken.com/leaderboards/benchmark/bfcl-v3))

**Runtimes:** Ollama 0.24 (MLX runner, >2x spec-decode on Apple Silicon), llama.cpp MTP merged
(~2x dense), vLLM 0.21 for concurrency, LM Studio 0.4.14. **Multi-token-prediction speculative
decoding is now mainstream and free.** NPUs fragmented (ExecuTorch/QNN for mobile is the real
path; ipex-llm/Lemonade vendor servers; ANE still not the LLM path).
([roundup](https://codersera.com/blog/local-ai-runtimes-may-2026-update/))

**Phones:** Apple Foundation Models framework (iOS 27, WWDC Jun 2026): ~3B on-device model for
any app with structured output, tool calling, image input, Spotlight-RAG, and a `LanguageModel`
protocol for remote swap-in. Android AICore/ML Kit Prompt API with structured output (Gemini
Nano 4 preview). BYO 3–4B @ Q4 runs 20–50 tok/s on flagships.
([Apple](https://developer.apple.com/documentation/FoundationModels),
[Android](https://android-developers.googleblog.com/2026/05/android-ai-intelligence-system.html))
→ *Thin client + platform-API fallback; never bundle a runtime.*

**Embeddings/voice:** Qwen3-Embedding/Reranker 0.6B–8B the local default; STT: Parakeet TDT /
Whisper-v3 baseline; TTS: Kokoro (82M) cheap, Voxtral TTS (Mar 2026); **full-duplex: NVIDIA
PersonaPlex-7B (Jan 2026, Moshi-based) is open SOTA** — Morgan's seam validated.
([NVIDIA ADLR](https://research.nvidia.com/labs/adlr/personaplex/))

**Hardware:** unified-memory mini-PCs are the home-brain category — Mac Studio M4 Max
($1,999, 410–546 GB/s, best value), Ryzen AI Max/Framework Desktop ($2,348, 128 GB), DGX Spark
($4,699, bandwidth-limited). ([review](https://www.tomshardware.com/pc-components/gpus/nvidia-dgx-spark-review/3))

## 3. Platform & memory landscape

**Memory systems:** consolidation around Mem0 (fact extraction; graph paywalled), **Zep/Graphiti
(bi-temporal KG — the only engine with t_valid/t_invalid invalidation; the consensus-SOTA
pattern = temporal KG + supersession + attribution, i.e. Morgan's pattern)**, Letta (memory-OS;
pivoted Mar 2026 to **skills as the unit of agent learning**). Benchmarks vendor-gamed; neutral:
Zep 63.8% vs Mem0 49% on LongMemEval; BEAM deliberately unsaturated.
([Zep paper](https://arxiv.org/abs/2501.13956), [Letta](https://www.letta.com/blog/our-next-phase))

**Big labs deepen lock-in:** OpenAI "Dreaming V3" memory (Jun 4 2026: background synthesis,
self-rewriting memories, Gmail context, **reduced audit trail**, non-exportable); Claude memory
for all users (Mar 2026, transparency-forward); Google Personal Intelligence + 24/7 "Gemini
Spark". ([Dreaming](https://dataconomy.com/2026/06/05/chatgpt-memory-dreaming-architecture-upgrade/),
[Google](https://blog.google/innovation-and-ai/products/gemini-app/personal-intelligence/))

**Agent-OS attempts:** **OpenClaw** (~347K stars by Apr 2026; self-hosted, channel-native,
proactive; NVIDIA productized it; creator hired by OpenAI) proved the demand at mainstream scale
— **but has memory files, not a learning loop**. Microsoft made Windows an agent platform
(Agent Store + always-on "Scout", Build 2026). Dead: Humane, Rabbit R1, Sora app.
([timeline](https://inbounter.com/blog/openclaw-2026-timeline),
[Microsoft](https://www.microsoft.com/en-us/microsoft-365/blog/2026/06/02/introducing-microsoft-scout-your-always-on-personal-agent/))

**Self-improvement:** GEPA (ICLR 2026) is the production-standard reflective optimizer (+20% over
GRPO, 35x fewer rollouts); ACE formalizes context-as-learning playbooks; MIT SDFT for
forgetting-free weight updates. **No consumer product ships measurable, eval-gated per-user
learning — the gap.** ([GEPA](https://github.com/gepa-ai/gepa), [ACE](https://arxiv.org/abs/2510.04618))

**Proactivity UX that stuck:** morning briefs (ChatGPT Pulse), named scheduled routines
(Claude Code Routines, Apr 2026), event-triggered background agents, sleep-time consolidation.

## 4. The opening (gaps nobody fills)

1. **Measurable per-user learning** — eval-gated GEPA/ACE-style optimization in a self-hosted
   assistant: unclaimed.
2. **Temporal-KG personal memory end-to-end** in an assistant (not infra): unclaimed.
3. **Auditability** — inspectable, exportable learned profile while labs retreat from it.
4. **Provider-agnostic portability** — every lab deepens lock-in; OpenClaw is flexible but
   learning-free.

Morgan sits on all four. The vision doc's bets follow directly.

---

## 5. Round 2 — deep dives (same day; four agents: MCP implementation, 12–24mo foresight, adversarial spec review, strategy red-team)

### MCP implementation ground truth
2026-07-28 RC: stateless core (SEP-2567 removes `Mcp-Session-Id` + handshake; `server/discover`
replaces it), `Mcp-Method`/`Mcp-Name` routing headers, extensions framework, Tasks → extension.
Python SDK: v1.27.2 stable supports `FastMCP(stateless_http=True)` today; **SDK V2 has no alpha
— expect H2 2026**; 12-month deprecation policy makes shipping on v1.27 safe.
Mounting in FastAPI works but the MCP lifespan must merge into the outer app's lifespan
([python-sdk #1367](https://github.com/modelcontextprotocol/python-sdk/issues/1367)).
Auth reality: Claude Code does static headers; claude.ai's connector has a known bug dropping
Bearer tokens to self-hosted servers — static bearer first, CIMD/OAuth (SEP-2468/2352) later.
Prior-art failure catalog: mem0/OpenMemory output-shape mismatches break clients (declare
`outputSchema`), `delete_all_memories` footgun, **Zep CVE-2026-32247: Cypher injection via
prompt injection** — write-path memory servers are actively exploited. #1 attack class for us:
**memory poisoning** (stored text replayed as trusted context) — sanitize at store *and* recall.
MCP Apps production-ready since Jan 2026 (Claude, VS Code, Goose render it).
([RC](https://blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate/),
[OWASP MCP cheat sheet](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html))

### 12–24 month foresight
Context-as-learning-substrate won: ACE playbooks + GEPA (both ICLR 2026) + Letta Skill Learning;
2027 shape = playbook deltas + learned skills + nightly consolidation with principled forgetting
(FOREVER/FSFM Ebbinghaus-style replay) + *optional* TTT/SDFT weight tier (consumer-practical
late 2027 at earliest). OpenAI's "Dreaming" consolidates offline too → differentiation narrows
to **ownership + auditability + portability + the eval gate**, not the loop's existence.
Proactivity research (CHI 2025): persistent suggestions get disabled; digest-first +
high-precision interrupts + a *learned* per-user interrupt policy is the 2027 differentiator.
Regulation: EU AI Act GPAI enforcement Aug 2026 (high-risk pushed to Dec 2027); EDPB Art-22
guidance on AI memory expected 2026; portability academics target "cognitive lock-in" —
**regulation favors self-hosted/auditable**. Long context does NOT kill memory (economics +
distractor interference, H); watch TTT-E2E context-into-weights (L-M, 2027–28).
Wildcards: Siri AI (Gemini-powered) / Spark / Scout normalize OS-account assistants (M);
MCP risk low; skills-as-learning-unit strongly validated.
([ACE](https://arxiv.org/abs/2510.04618), [SEAL](https://arxiv.org/abs/2506.10943),
[FOREVER](https://arxiv.org/abs/2601.03938), [EU timeline](https://www.legiscope.com/blog/eu-ai-act-timeline-deadlines.html))

### Strategy red-team (applied to the spec set)
No moat — every component copyable; defensible: data gravity, category authorship, rigor.
Passport deliberately trades gravity for sovereignty (now stated in vision §2). "Network
effect" → **single-user flywheel**; the indispensable loop is cross-surface recall.
**Importers > export evangelism** (ChatGPT/Claude export ingestion = cold-start fix + exodus
wedge). H1 re-sequenced: morning brief in, desktop profile + SKILL.md → H2. Kill criteria added
to the roadmap. Biggest risk: optimizing for an ecosystem audience that doesn't exist for a
single-maintainer project while deferring the daily value that feeds the signal pipeline.

### Adversarial spec-vs-code review (10 findings, all folded into the specs)
Verdict: the layering maps honestly onto the code and the kernel invariants survive — but the
draft shipped two false feasibility assumptions, both fixed: (1) /v1 client tool passthrough is
unimplementable on current seams (descoped to H2 with a named mechanism); (2) the single
`system_override` slot is occupied by the champion, and **the streaming path already drops the
champion silently** (now an H1 prerequisite fix, with the unwritten-`SessionHistoryStore`-on-
redis bug). Other corrections: `user_stated` rule needed an identity-class model
(external-agent vs owner-device); `memory_store` must ride `MEMORY_STORED`, not
`RESPONSE_GENERATED` (signal pollution); port turns are signal-excluded; consolidation is the
wrong (LLM-cost, non-deterministic) path for passport *restore* → direct bi-temporal merge;
no `CacheProtocol` exists (profile text was fiction); hybrid-burst sensitive-fact guarantee
needs recall-time filtering + role resolution before context build; SKILL.md "conformance" is
honestly greenfield; structured provenance + schema migrations are a cross-cutting prerequisite
named in no draft — now H1 item 0.
