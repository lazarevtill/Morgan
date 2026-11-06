# Native owner review validation — 2026-11-06

Environment: macOS 15.8.2 arm64, isolated synthetic workspace, Europe/Belgrade.

## Usability measurement

The scripted Russian and English scenario creates a book-return task, renews it with an
explicit target, then STOPs and exits. The controller uses real owner-input persistence and
MemoryGate; only dialog responses are fixtures. Both runs retain one current task family,
move its due date from November 6 to November 7, preserve both source statements, and STOP.

- One launch command; zero copied message IDs, evidence IDs or digest strings afterward.
- New task: nine dialog responses including menu, text, action, four fields, confirmation
  and acknowledgement. Renewal: seven. STOP and exit: two. Total: eighteen responses.
- This reduces command syntax and opaque-ID handling, **not demonstrated click count or
  completion time**. It is not a human usability study and not automatic language matching.

## Failure coverage

Controller and native-adapter tests cover cancellation, wrong confirmation, malformed numeric
fields, empty renewal targets, ambiguity, stale reviews, STOP before commit and while waiting,
explicit resume, lost acknowledgement with exact replay, session expiry, invalid native
responses and child termination. Real owner-input CLI regression tests remain applicable.

An actual native confirmation dialog returned cancellation on its bounded timeout in 4.21s.
A full temporary native session was started with synthetic data. The computer-use tool could
not select the osascript dialog as an application; Finder inspection did not expose that
window. Consequently **actual native button clicks and long-text layout are not verified**.
Controller fixtures do not substitute for that manual GUI acceptance test.

## Remaining acceptance work

Manually open a pinned build, create/renew a synthetic task, cancel and reopen a saved review,
then STOP and resume. Inspect the full preview at the maximum supported source length.
A persistent owner channel, host restart policy and real notification delivery are still
separate deployment decisions; this bounded foreground UI does not resolve them.

## Runtime and regression evidence

A normal wheel of PR revision `1175e043072a5b4e5309a79bc8e6ed14d30a2045` was
built and installed offline into a fresh workspace runtime. Verification checked 100
package files with `editable: false`; `python -I -m morgan_brain.owner_ui --help`
resolved the installed module independently of the checkout. This does not activate it.

The initial Mac suite produced 1841 passes, four skips and one failure because the test
invocation used a Python path containing `..`. The existing launch-plan validator correctly
rejected it; rerunning that test with a canonical absolute interpreter path passed.
The Linux type check additionally exposed platform-dependent inference after the Darwin
constructor guard; explicit field annotations fix it without changing runtime behavior.

Review follow-up added nine regression cases: embedded parser failures never exit callers,
dash-prefixed titles remain literal values, initialization failures are reported, and a SQLite
failure preserves `applying` state, exits without a second storage read, and recovers after
explicit reconfirmation. All 31 UI/native tests pass after those fixes.
