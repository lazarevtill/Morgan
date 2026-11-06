# Owner communication and macOS hosting preparation

Checked 2026-11-06 against main after PR #93. The working scenario is one explicitly approved
owner notification, followed by process restart without duplicate publication. No personal
information was sent outside the Mac, and no service or standing credential was installed.

## Available surfaces versus proposed integrations

| Surface | Current project/environment evidence | Meaning of a successful result |
| --- | --- | --- |
| CLI / MCP stdio | Implemented in current package | A caller gets a tool response; no proactive phone delivery |
| MCP HTTP over a private network | Implemented; non-loopback requires a bearer key | Remote tool access, including writes; not a push channel |
| Native macOS dialog | PR #93, actual timed-out synthetic probe | OS button/timeout response; needs a desktop session |
| Telegram bot | Referenced in archived designs; absent from current package | `sendMessage` returns a Message, not proof of device delivery or reading |
| ntfy | No configured owner subscription verified | Publish response identifies a server-accepted message; phone receipt remains unknown |
| Email / SMS / commercial push | No active Morgan adapter verified | Extra account/destination/billing setup; provider acceptance is not reading |

No secrets, phone accounts or private service configurations were searched. "Not verified"
is not a claim that the owner has no such account. The current MCP server is not an appropriate
unauthenticated read-only phone endpoint: its tool surface includes mutations.

For this pilot choose ntfy's small HTTP publish API with existing httpx, rather than restoring
an archived chat framework. It can address a phone subscription after configuration, with no
new Python dependency. Telegram remains a reasonable future choice if the owner identifies
an existing bot/chat. This choice does not establish which app is installed on their phone.

Primary references:

- [Telegram sendMessage](https://core.telegram.org/bots/api#sendmessage)
- [ntfy publish and caching](https://docs.ntfy.sh/publish/)
- [ntfy access control and iOS forwarding](https://docs.ntfy.sh/config/)
- [Apple launchd agents and daemons](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html)

ntfy normally caches published content; a private topic requires actual server ACLs. Supplying
a bearer token does not by itself prove anonymous readers are denied. Public anonymous topics
are unsuitable for personal content. Phone push may involve FCM/APNs or an upstream service;
self-hosting alone does not guarantee wholly local notification delivery. Review that route
before publishing personal data. No public ntfy topic was created or used in this work.

## Explicit one-shot adapter

```sh
printf '%s' 'Synthetic owner notification' |
  python -m morgan_brain.owner_push_cli /private/path/outbox \
    --server https://YOUR-APPROVED-NTFY-HOST --topic YOUR-PRIVATE-TOPIC \
    --user demo --project personal --task-id owner-notice-1
```

This is a template, not a configured destination. The CLI prompts for a transient token via
the terminal and refuses the echo fallback. It reads no dotenv file, environment token or
keychain and writes no credential. Exact content, owner/project, server and topic are bound
to the journal task; invoking `send` explicitly authorizes that one request. Never let a model
invoke this surface without a separate trusted authorization decision.

ntfy trims surrounding whitespace. This adapter rejects blank text or surrounding whitespace
(including a trailing newline) before requesting a token or sending. Use `printf`, not plain
`echo`; content is never silently normalized after approval. This follows the upstream
[text-body handler](https://github.com/binwiederhier/ntfy/blob/main/server/server.go).

The SDK `owner_push.send` reuses the private workspace lock and task journal. `OwnerPush`
requires an authorizer through TaskControl. Only HTTPS origins are accepted, no embedded
credentials, redirects, inherited proxies, attachments, callbacks, remote actions or retries.
The response is bounded and checked against the exact topic and text. Token and raw provider
error text are excluded from saved receipts. As with other SDKs, a custom transport and the
local account are trusted; diagnostics controlled by that account are outside this boundary.

Receipt meanings:

- `server_accepted: true`: a matching ntfy response with a message ID arrived.
- `device_delivered: null`: no device receipt was observed.
- `read: null`: no human acknowledgement was observed.
- An error, timeout, STOP during the request or lost receipt leaves `outcome_unknown`.
  A prior `running` task after a hard crash also blocks replay. A new task ID is not a safe
  automatic workaround: the original message may already have been published.

The request has a 12-second overall adapter deadline (10-second HTTP operation timeouts).
STOP is polled every 0.1 seconds while awaiting the response. It cannot retract a message
already accepted by a server. STOP before send prevents journal creation and the request.
The existing `local_alert_cli OUTBOX stop` and `resume` manage the same workspace marker;
resume sends nothing. Journals retain text separately from Morgan's memory/forget lifecycle.

## Hosting after reboot: prepare, do not install

Use launchd rather than another permanent Python supervisor. A user LaunchAgent starts after
login; it does not provide execution before login. FileVault unlock, sleep, logout and network
availability still affect service availability. A LaunchDaemon would require a separate system
installation and cannot supply a normal user GUI session. A dedicated always-on host could be
more suitable, but no host availability or deployment authorization was established here.

The installed macOS 15.8.2 SDK `launchd.plist(5)` distinguishes missed StartInterval firings
from StartCalendarInterval events coalesced on wake. The older public launchd man page differs
on StartInterval, so this plan uses calendar minutes, not assumptions about interval catch-up.
Calendar minutes use the Mac's local clock; this is not an IANA-zone task scheduler.
Neither calendar launch nor RunAtLoad proves continuous availability or task timeliness.

Generate a disabled review artifact:

```sh
python -m morgan_brain.launch_agent_plan \
  --python /ABSOLUTE/STABLE/VENV/bin/python \
  --workspace /ABSOLUTE/PRIVATE/APPROVED-NOTES > proposed-agent.plist
plutil -lint proposed-agent.plist
```

It contains a fixed invocation of the existing `local_notes_cli run`, one action/cycle, ten
seconds per invocation, calendar minutes 0/15/30/45, RunAtLoad, no KeepAlive, no shell or secrets.
Logs go to `/dev/null`; existing private worker observations remain available via `diagnose`.
The worker's existing explicit finite grants, expiry and STOP still apply. The recurring OS
launch does not renew them. This plan runs **local note delivery only**, not ntfy, a model or
an unlimited agent. Phone publication and recurring execution are deliberately not connected
without an exact destination/content policy and approved credential lifecycle.

The generator prints bytes only: no launchctl, file installation, chmod, sleep changes or
permission grants. A generated Disabled key is not a security boundary against an administrator
or an existing launchctl override. Nothing was bootstrapped or enabled in this work.

For a concrete installation review, the remaining configuration/actions are:

1. Choose an existing, verified HTTPS ntfy origin and private topic with owner-only read ACL;
   identify the owner phone subscription, OS notification permissions and forwarding route.
2. Approve one exact synthetic external pilot to that destination. Verify server acceptance,
   phone arrival and a separate explicit owner acknowledgement independently. No such send
   has occurred; no provider credential was inspected.
3. For unattended sends, specify the exact token scope, storage/retrieval method, expiration,
   revocation, allowed content, rate/quiet-hours policy and recurring authority. This is not
   covered by the transient interactive token prompt and is not configured here.
4. Choose a stable installed Python/package and private note workspace. Review installation
   at `~/Library/LaunchAgents/org.morgan.approved-notes.plist`, removal of Disabled/explicit
   enabling and bootstrap in that owner's GUI domain. Those persistent actions remain unperformed.
5. Test actual logout/login, reboot/unlock, sleep/wake and network loss with synthetic data.
   Check STOP before and after each transition and that no expired grant or unknown send retries.
   Do not weaken FileVault or enable automatic login to make this test appear successful.

## Evidence

Twenty tests cover matched/mismatched receipts, redirects, oversized/lost replies, STOP before
and during requests, exact replay, changed destinations, credential exclusion,
refusal of echoed token input, whitespace preflight and plist bounds.
A real loopback HTTP server exercised request bytes and response parsing with a test-only
transport mapping; no external TLS/service or phone was exercised. A separate subprocess test
ran the generated worker argv twice with a temporary finite synthetic note grant, then with
STOP. The receipt attempt was preserved. This tests fresh-process recovery, not an actual reboot.
The generated plist also passed the native `plutil -lint` validator on this Mac.
