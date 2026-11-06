# Seven-day personal-agent pilot: readiness and acceptance plan

This is a concrete proposed pilot, not a statement that Morgan is running 24/7. The currently
verified useful path is local note → model suggestion → explicit owner review → durable
commitment → daily preview → owner-reported completion → updated preview. Continuous hosting
and an owner phone destination are not configured by this document.

## Verified local path

A fresh synthetic Russian note contained two open tasks and one completed purchase. The cached
Granite 4.1 8B GGUF (5,347,914,400 bytes), bundled llama-server and four CPU threads completed
the guided flow in 23.035 seconds on 2026-11-06. Six scripted responses confirmed the two open
tasks and completed one: planned work changed from 20 to 10 minutes. This excludes human
thinking/typing time and is one smoke test, not a new accuracy benchmark. Earlier extraction
errors remain relevant: review is mandatory and model suggestions have no execution authority.

The new `--status --preview-day YYYY-MM-DD` query reused the same current evidence for the next
day and returned the remaining ten-minute task. It did not rerun inference or mutate the saved
day. Without an explicitly selected timezone it returned `unknown_timezone`, not a guessed
schedule. Tests also cover a constraint blocking one day but not the next, future-due tasks,
completion, pending review and unchanged durable files. See [status](guided-status.md).

The tested model is an already present local file; no model was downloaded. The known LAN
model endpoint could not be reached in a read-only probe, so it is not a dependency of this
pilot. Endpoint addresses and owner application inventory remain in the private local report.

## Proposed limits and success criteria

| Item | Concrete proposed limit / criterion | Enforcement today |
| --- | --- | --- |
| Trial | Seven days after an explicitly agreed start | Operator-managed trial; not installed |
| New input | At most two owner-selected notes daily, fourteen total | Operator policy; each run bounded |
| Model | Existing Granite; four CPU threads; 90-second guided cycle | Existing guardian per invocation |
| Memory | Private session workspace, explicit owner and project | Gate and private workspace checks |
| Task admission | At most eight reviewed lines per session | Session schema and confirmation |
| Morning use | Preview the explicit local date; show sources and uncertainties | Read-only query implemented |
| Evening use | Owner reports completion/corrections in the guided dialogue | Immutable revision path implemented |
| External effects | Initially zero; later one approved synthetic phone message | No destination/grant configured |
| Notifications | Only after destination approval; proposed maximum two daily | No recurring external-send grant exists |
| Stop/recovery | STOP persists; unknown/running sends are never retried | Process/journal tests pass |
| Exit criterion | All admitted tasks reviewed, no duplicate sends, useful owner feedback | Needs real owner participation |

Measure task correctness after review, manual interactions, stale/unknown answers, duplicate
notifications (target zero), actual phone arrival and owner acknowledgement separately. Do not
count an HTTP receipt as phone delivery, scripted answers as user satisfaction, or a generated
plist as uptime. A single fresh-process test does not establish seven-day availability.

## Resource and expense envelope

The completed smoke test made zero paid API calls and bought/downloaded nothing. The local
path has no provider request charge. Electricity, hardware wear and any existing subscription
cost are not measured. Four threads and a model file below 8 GB do not impose a hard RSS cap.
Fourteen 90-second model windows total 21 minutes of admitted wall-time windows (up to 84
thread-minutes); startup/cleanup and other work add overhead. These are proposed invocation
counts, not an implemented seven-day global compute limiter.

A phone service's actual plan, quota, forwarding route and cost cannot be determined without
a verified deployment/account. No paid tier, SMS, cloud model, hosting purchase or automatic
renewal is authorized or configured by this plan. Establish an exact zero-new-spend or explicit
currency cap for that chosen service before unattended use; do not infer it from a model key.

## Configuration matrix: known versus required

| Setting | Current readiness | Exact decision required before live deployment |
| --- | --- | --- |
| Local inference | Existing executable and cached model passed | Pin trusted installed paths/version |
| Memory scope | Synthetic `pilot` / `personal` tested | Owner's actual ID, project, input sources |
| Time | Explicit ISO dates work | Owner IANA zone and first/last trial timestamps |
| Telegram | App presence alone is insufficient; no current adapter | Verified bot/chat and consent if selected |
| ntfy | Adapter tested locally, no owner destination established | HTTPS origin, private topic ACL, phone subscription |
| NetBird | Existing client is not proof of a service route | Verified intended peer/service, without broad scans |
| Credentials | Interactive transient token only | Named secret store, scope, expiry and revocation for unattended use |
| Hosting | Disabled launchd artifact only | Stable Python/workspace, account, installation and removal actions |
| Costs | Zero paid calls in this validation | Exact existing service quota and explicit spending ceiling |

Do not guess missing values or inspect messenger account databases/token stores to fill them.
The smallest remaining owner decision is the actual existing private notification destination
(or confirmation that none exists), plus timezone and intended host. Prefer proving that
channel with one synthetic message before provisioning another service.

## Deployment sequence and stop controls

1. Complete the local reviewed workflow first. The model can be unavailable while status and
   previews still work over confirmed evidence. Fresh notes require review, not auto-acceptance.
2. Identify one real recipient/subscription and approve a single synthetic external test with
   the exact destination and transient credential. Observe server receipt, device arrival and
   explicit acknowledgement independently. Telegram app installation does not verify a bot.
3. Decide whether phone push needs unattended access. Approve the exact credential lifecycle
   and bounded content/destination policy before connecting it to any recurring runner. The
   current ntfy CLI prompts; it cannot be silently used as a credential-backed daemon.
4. Review the existing disabled local-note LaunchAgent separately. It invokes the finite
   local-note worker at quarter-hour calendar minutes; it does not run inference, refresh a
   personal plan or send ntfy messages. Fourteen exact local-note occurrences can be compiled
   by `repeat` with interval 43,200 seconds and a 900-second window, provided the chosen first
   start and final window fit within the existing seven-day grant bound. Missed windows skip.
5. Installation/enable/bootstrap and end-of-trial bootout/removal require concrete paths and
   confirmation. Expired note grants prevent new effects but do not unregister launchd itself.
   A user agent starts after login, not before FileVault unlock; sleep/network loss remain gaps.
6. After installation is authorized, run real login, reboot/unlock and sleep/wake acceptance
   tests with synthetic data. Never bypass these limits through automatic login, disabled
   FileVault, indefinite model loops or renewed grants. STOP every active workspace before
   teardown; SIGTERM during push persists STOP, and unresolved effects remain fenced.

The OS scheduling and phone-push pieces are intentionally not represented as a connected
24/7 agent. Their missing authorization/configuration is the concrete live-deployment blocker;
the local reviewed-plan workflow and day preview are available now.
