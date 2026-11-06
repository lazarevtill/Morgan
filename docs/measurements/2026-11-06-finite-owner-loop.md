# Finite owner-loop probe, 2026-11-06

Tested a normal hash-verified wheel from `2a4d618a28aa42893518eb82cd52a9f6105194c3`
on macOS arm64 with the existing local GGUF and llama-server. No service was installed,
no real owner text used, and no network inference or credentials were involved.

A separate headless process received “I need to return the library book.” through the
owner-input API and produced a source-grounded, unapproved draft in **24.330 seconds**.
The saved task list remained empty. SIGTERM ended the idle process in **0.044 seconds**.
A second process retained the original expiry and source cache, entered waiting state
without another inference, and exited **0.118 seconds** after shared owner STOP.
These are one-run observations, not latency percentiles or a seven-day endurance test.

Fixture tests additionally exercise two sequential EN/RU messages, in-flight STOP,
in-flight expiry and shutdown, failed-attempt replay prevention, concurrent-loop refusal,
launch-count/corrupt-state rejection, real process SIGTERM/restart and disabled plist
rendering. Closing an unrelated review window leaves background-owned work alone.

The model's earlier 8.8 GB peak RSS measurement remains the resource planning input;
this probe did not remeasure peak memory. Draft output still needs owner review, and
results must be opened locally. No automatic notification or phone channel was tested.
Actual login/logout/reboot/sleep transitions were not performed. A LaunchAgent would
only run after login and while the Mac is awake; the pilot authorization is pending.
