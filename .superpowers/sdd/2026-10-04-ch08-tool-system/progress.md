docs/superpowers/plans/2026-10-04-ch08-tool-system.md
Spec: docs/superpowers/specs/2026-10-04-ch08-tool-system-design.md
Method: Native, approved 2026-10-04. Tasks1-9 pending.
Ruling: continue in existing checkout per persisted user preference; preserve Ch07 uncommitted verified fixes as a separate baseline commit before Ch08 implementation.
Preflight: catalog CallContext must not enter checkpoint; engine preflight cannot write final audit when waiting; ticket replay cannot repeat previous batch calls; all caches pass through engine.
Preflight: Task3 fixture confirmation uses a trusted fake implementing Task5 claim contract; actual capability is built only by backend.
Preflight: SDK v1 range is a compatible implementation choice within the fixed official stack; verify installed API before coding.
Task1: in progress.
Task1 complete: 5 RED→GREEN, baseline686; BASE419e6ca. Ruling: factory may return list for existing build_tools; each definition still independently validated. Task2 in progress.
Task2 complete: 3 RED→GREEN; real MySQL apply + check twice current. Task3 in progress.
Task3 complete:8 new tests,41 related pass. Default executor exact policy; legacy tests explicitly register fixture tools. Result JSON now normalized object. Task4 in progress.
Task4 complete:3 MCP tests + tools 28 pass; real HTTP/server-only reload/offline isolation. Ruling: historical build_tools fixture retained, production bind_builtins removes logistics and policy forbids its execution. Task5 in progress.
Task5 complete: 7 ticket tests RED→GREEN; 27 related tests pass; full pytest 712 passed/1 existing warning. Persistent claim + ticket/receipt atomic commit; legacy button engine audited. BASE2ff1f85.
Task5 Ruling: legacy graph tests explicitly use fixture logistics grant, FAQ assertions use normalized object — production policy deliberately removes builtin logistics; no runtime permission broadening.
Task5 Ruling: confirmation lookup will be added only at its Task6 consumer with ownership tests; no unauthenticated lookup API. Existing confirmation methods currently return None and replay safely accepts completed/executing.
Task6 in progress: Context7 rechecked interrupt node restart, checkpoint superstep boundaries, nested pending state.
Task6 complete: full pytest 732 passed/1 existing warning; nested interrupt resumes without repeated read, SQLite reopen pending, owner/stale checks, supersede audit and checkpoint closure.
Task6 Ruling: conservative user-origin imperative matching plus verbatim grounded description — fail closed on ambiguous request, real-model annotated evaluation remains Task9; cost: some paraphrases require clearer request.
Task7 in progress: dynamic registry/service and prefix budget tests RED (3 expected failures).
Task7 verified: initial full-suite 743 passed/1 warning; after an additional budget-skipped-call audit fix, focused 15 tests passed; rerun full suite before final.
Task8 complete: isolated headless Edge, all requests intercepted; 27 requests. Preview/XSS-safe rendering, confirm/double-click exactly once, cancel, refresh restore, stale card invalidation, complaint handoff/ticket independence, and order selection passed; screenshots/reports saved under data/ch08/reports.
Task9 verified: full pytest 744 passed/1 pre-existing Starlette warning; local eval 57 passed including real MCP subprocesses and current MySQL migration; browser 27 requests passed. Fresh backend reviewer found no high-priority findings; frontend excluded. User-authorized cloud data scope does not yet include Chapter 8 evaluation samples, so live eval remains gated.
Task9 environment ruling: the restricted tool identity cannot write `.git/index` (ACL denies; no permission changes made). Local source/tests/reports are ready but cannot be staged/committed in this environment; prior user authorization for local commits noted, no shared/published action attempted.
