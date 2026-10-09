# Opt-in adapter request controls

`OpenAICompatAdapter(..., request_options=ChatRequestOptions(max_output_tokens=512,
enable_thinking=False))` declares a positive output-token ceiling and a compatible
server template's thinking flag. Import `ChatRequestOptions` from
`morgan_brain.providers.request_budget`. All fields default to `None`; omitting
options or constructing empty options preserves the old request. No factory,
CLI, MCP or environment default changes. Options are frozen and accept no arbitrary
payload, model, message or authentication override.

The output ceiling is sent as `max_tokens`, including reasoning tokens, and is
not a guarantee of a usable final answer. The thinking flag becomes the request's
`chat_template_kwargs.enable_thinking` through the OpenAI SDK's `extra_body`.
It requires a compatible endpoint and template; no model name is detected and no
server configuration is changed. Explicit `True` is also supported.
The installed llama.cpp b11330 [server API](https://github.com/ggml-org/llama.cpp/blob/b11330/tools/server/README.md)
documents this extension; the actual loaded Ornith template has an explicit
boolean-false branch. This is request compatibility, not a memory quality result.

Both non-streaming and streaming requests use these options. Structured calls
read the adapter's typed options for each initial attempt and re-ask. The shared
canonical payload includes model, messages, schema, output limit and template
flag before byte accounting. SDK `extra_body` is an argument envelope, not an
extra JSON field on the wire. Custom clients without this optional typed property
retain existing accounting; arbitrary custom backend wire additions are not
measured by this contract.

Local compatibility controls restored final content but did not pass the strict
public development output check. Its prompt said reports were not verified and
also said to use null without evidence. The expected `verified_inventory=false`
can be read as verification status, while the returned null can be read as unknown
inventory truth. That ambiguity does not erase the recorded strict failure, and
Markdown fences independently violated JSON-only instructions. Any later semantic
evaluation must preregister an explicit verification-status or truth-status meaning
on fresh development cases. No prior scores or heldout data change.


Optional `temperature` accepts a finite number from 0 to 2 and rejects booleans.
It defaults to `None`, preserving the provider's sampler settings. It uses the
standard top-level API field and participates in the same complete canonical byte
measurement on every initial and re-ask request. Explicit zero is retained.

The earlier successful development compatibility controls explicitly sent
`temperature=0`; the adapter integration and fixed rollout monitor preserve the
loaded provider default of 0.6. This difference is a calibration caveat, not proof
that temperature caused the subsequent timeouts. The optional field lets callers
reproduce the known request settings without patching SDK methods. Adding this
field does not modify the active monitor or retry a failed fixture. No new model
calls, global defaults, server settings or old scores change in this local variant.
