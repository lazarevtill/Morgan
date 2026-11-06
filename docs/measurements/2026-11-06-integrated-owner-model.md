# Integrated owner/model pilot — 2026-11-06

Tested revision: `a07e744dd4518cdfd8ceb4c083b8c80b9cf59979` (PR #104).
A fresh normal wheel runtime verified 101 package files, not an editable checkout.
macOS, existing Granite 4.1 8B GGUF (5,347,914,400 bytes), bundled llama-server,
four CPU threads, no GPU layers, private Unix socket, 90-second model budget.
Host RAM/CPU were independently checked as 16 GiB / eight CPUs.

## One vertical scenario

The actual `OwnerUI` controller received scripted responses while its `OwnerModel`
ran the real local model. Source: “Нужно вернуть книгу в библиотеку.” The model
selected that exact line. The operator fixture then supplied title, ten minutes,
normal priority and an explicit date, and separately confirmed the task.
Before that final confirmation, a real owner-input status read found zero tasks.
Afterwards there was one task. These are fixture decisions, not user approval.

- 21.04 seconds total controller flow; 13 scripted dialog responses.
- One model draft; no external API calls, service or credentials.
- Recreated controller displayed the cached result with no inference thread or retry.
- A second real local operation was stopped during loading through ordinary owner-input
  STOP. It finished and released both guardian locks in 0.992 seconds; STOP persisted.
- Completed child process tree consumed 38.575 CPU-seconds and reported peak RSS
  8,800,813,056 bytes. This is `RUSAGE_CHILDREN` on macOS, not an isolated instantaneous
  model RSS sample or a memory quota. Concurrent applications still need headroom.

This removes separate model CLI/file handoff, not all manual planning. Thirteen responses
are not evidence of improved human completion time; native button interaction is still
unverified because the computer-use tool could not target the previous osascript window.
The real model and real memory/receipt lifecycle were exercised, while dialog I/O was scripted.
A single simple note is a smoke test, not a new quality benchmark or evidence that renewal,
negation and ownership inference are reliable.

## Regression scope and remaining limits

48 focused tests passed, covering previous UI/model behavior and eleven new integrated cases:
shared STOP, resource lane contention, multiple source caches, interrupted recovery, malformed
cache refusal, expiry, process failure, cancellation, separate task review, idle close and
retaining the UI-session lock until model cleanup finishes.
Full-suite and CI outcomes are recorded in the PR.

The operation remains finite and requires a logged-in foreground session. Sleep/reboot,
phone delivery, a continuous receiver and headless confirmations are not implemented or
tested here. One shared workspace prevents overlapping model jobs there; independent
workspaces are not globally resource-limited. No LaunchAgent or other persistent service
was installed, and no private owner data was sent to a model or external service.
