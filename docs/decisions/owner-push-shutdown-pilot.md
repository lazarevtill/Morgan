# Owner push shutdown and restart pilot

## Problem and change

The one-shot push CLI previously relied on the default SIGTERM behavior. An interrupted task
stayed fenced as running, but the process did not persist the owner's STOP control. A later
invocation with a different task ID could therefore proceed without an explicit resume.

The CLI now runs its send through `send_with_stop_signal`, which maps SIGTERM to the existing
private workspace STOP marker. The adapter's current 0.1-second readiness polling cancels
its awaited request and the execution journal records `outcome_unknown`. The signal handler
is removed when this foreground invocation finishes. This wrapper owns its event loop's
SIGTERM handling; it is not intended to replace a host application's signal policy.

The handler applies while the send is executing, not while reading stdin or prompting for a
token. Those earlier phases have no journal claim or network effect. SIGKILL cannot run cleanup:
a running task remains unresolved and non-replayable. Neither outcome proves that a provider
did not accept the message. STOP is not message retraction or a phone-delivery receipt.

## Reproduce without credentials or a service

From an installed development checkout:

```sh
python -m pytest -q tests/integration/test_push_shutdown.py
```

This test starts only its own child processes and uses a temporary private workspace. The
child fixture `tests/fixtures/owner_push_shutdown.py` injects httpx MockTransport and a synthetic
credential, prints a deterministic request-start boundary and waits. No DNS lookup, TLS
connection, model call or external publication occurs. Do not replace that transport with a
real provider while running termination tests.

The two test cases perform the following sequence:

| Step | Graceful SIGTERM | Abrupt SIGKILL |
| --- | --- | --- |
| Reach synthetic effect boundary | One request started | One request started |
| Terminate the owned child | Normal cleanup, exit 2 | Signal termination |
| Inspect durable state | STOP + outcome_unknown | No guaranteed STOP; running |
| Start another process | STOP refuses before request | Unresolved task returned, no request |
| Explicit resume, same ID | outcome_unknown, no request | Already fenced; no replay |

The fresh subprocess uses the same production wrapper and journal path. It cannot silently
recreate an attempt or turn a missing response into a success. Combined with the prior real
loopback HTTP test, this covers protocol handling and process lifecycle separately. It does
not measure a real server accepting a request at the exact crash instant.

## Reboot boundaries and deployment plan

No OS reboot, user logout, service installation or launchctl mutation is part of this pilot.
Fresh-process tests exercise durable recovery, not FileVault unlock, GUI availability, network
recovery or operating-system scheduling. Those remain separate deployment acceptance tests.

The prepared `launch_agent_plan` still targets the finite local-note worker, not owner push.
It is disabled, has no token, does not renew an approval and does not install itself. The
[hosting decision](owner-channel-hosting.md) documents calendar behavior and the exact paths
and controls to review. As [Apple documents](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html),
a user agent runs for a logged-in user; it is not an always-available service before login.

Minimal next deployment sequence:

1. Confirm an actual private notification destination and owner subscription. An installed
   messaging app is not proof of a configured bot, private topic, signed-in owner or phone.
2. Approve one exact synthetic external message using a transient token. Record provider
   acceptance, device arrival and owner acknowledgement separately, then test explicit STOP.
3. For unattended notification, review a concrete token lifecycle and exact recurring send
   policy. The current interactive CLI intentionally has no saved token source or auto-grant.
4. Separately review stable interpreter/workspace paths and installation of the disabled
   local-note LaunchAgent. Never imply that installing it also enables phone publication.
5. After explicit installation approval, test logout/login, reboot/unlock and sleep/wake with
   synthetic grants. Check STOP survives and expired/unknown work does not replay. Do not
   disable FileVault, enable automatic login or change power settings merely to pass the test.

A personal channel inventory belongs in the owner's local report, not in the public repo.
Checking standard app/config/service names cannot rule out remote hosts, containers or jobs
with custom names. No token search or account/database inspection is needed for this pilot.
