# Ch02 Function Calling Implementation Plan

> **For agentic workers:** Implement task by task under the approved spec. Use the Superpowers execution workflow for backend tasks. The chat page follows the user's Vibe Coding exception.

**Goal:** Let the existing SSE customer service chat choose and run one business tool, save the full exchange in MySQL, and stream the final answer.

**Architecture:** FastAPI route delegates to a chat service. SQLAlchemy repositories persist conversations, messages, FAQ and tickets. LangChain `@tool` declarations are bound to the selection model; a guarded executor runs at most one call; the final model response streams without tools.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2.x, MySQL 8 in Docker Compose, LangChain, uv, pytest.

**Spec:** `docs/superpowers/specs/2026-09-19-ch02-function-calling-design.md`

## Global Constraints

- Keep the user supplied DDL exact in column type, nullability, defaults, indexes, foreign keys, table engine and charset.
- No order, product or logistics tables or external API integration. No Agent Loop, RAG or vector search.
- Only one tool is actually executed per user turn. Final output remains SSE token streaming.
- For non-code prompt and seed data, verify against the three labeled acceptance samples. Chat page implementation uses Vibe Coding.
- Append `dev-notes/ch02.md` after each completed stage with user words, output, corrections and rework.
- Before implementation of version-sensitive APIs, refresh Context7 references. If the fixed stack cannot work, stop and ask the user.

## Files and responsibilities

- `db/ch02.sql`, `db/seed_ch02.py`, `compose.yaml`: exact DDL, repeatable FAQ data, MySQL service.
- `app/db/models.py`, `app/db/session.py`: SQLAlchemy 2 mappings and connection factory.
- `app/repositories/conversations.py`, `messages.py`, `faq.py`, `tickets.py`: focused persistence operations.
- `app/tools/business.py`, `app/services/tool_executor.py`: five LangChain tools and guarded execution.
- `app/services/chat.py`: turn orchestration and SSE event source.
- `app/api/chat.py`, `app/schemas/chat.py`, `app/core/prompts.py`, `app/core/llm.py`, `app/config.py`: existing entry point and configuration adaptation.
- `app/static/index.html`: chat bubble status and tool badge.
- `tests/`: focused tests by component; `scripts/`: demo and labeled acceptance runner.

## Task 1: Database foundation and exact mapping

**Files:** Create `compose.yaml`, `db/ch02.sql`, `db/seed_ch02.py`, `app/db/models.py`, `app/db/session.py`; modify `pyproject.toml`, `.env.example`, `app/config.py`; test `tests/test_db_models.py`.

- [ ] Copy the supplied SQL verbatim except unescaping the message formatting. Start MySQL 8 with a health check and persistent volume; keep credentials in environment variables.
- [ ] Add SQLAlchemy and a MySQL driver with uv; add `DATABASE_URL` configuration. Use `mysql.BIGINT(unsigned=True)`, dialect ENUM, JSON, DATETIME, and explicit server-side `ON UPDATE CURRENT_TIMESTAMP` mapping where present.
- [ ] Write a failing schema comparison test that checks all four tables, PKs, indexes, foreign keys, enum values and defaults against an initialized MySQL test database; run it red.
- [ ] Implement mappings and short-lived Session factory; execute the DDL and inspect `SHOW CREATE TABLE` plus ORM reflection; run the test green.
- [ ] Add idempotent seed containing “退货政策是什么”, with no “邮费” phrase. Run seed twice and assert unchanged row count; record the stage in dev-notes and commit.

## Task 2: Persistence and conversation ownership

**Files:** Create `app/repositories/conversations.py`, `messages.py`, `faq.py`, `tickets.py`; modify `app/schemas/chat.py`; test `tests/test_repositories.py`.

- [ ] Write failing tests for new numeric conversation ID, owner check, ordered message loading, assistant `tool_calls` JSON, matched tool `tool_call_id`, FAQ escaped literal LIKE, and ticket insert plus status update in one transaction.
- [ ] Implement repository methods: `get_or_create_conversation`, `append_message`, `load_messages`, `search_faq`, `create_ticket`. The route must reject a conversation owned by another `user_id`.
- [ ] Verify rollback on invalid ticket type or foreign key, and idempotent seed lookup. Replace `SessionStore` in chat flow only after the new history path passes tests; keep any independent memory utility tests intact.
- [ ] Run focused tests, record the stage, commit.

## Task 3: Five tools and executor

**Files:** Create `app/tools/business.py`, `app/services/tool_executor.py`; test `tests/test_tools.py`, `tests/test_tool_executor.py`.

- [ ] Write failing tests for the three mock tool schemas and results, FAQ exact keyword hit/miss, ticket write, unknown name, invalid arguments, timeout, retry limits, and no retry after uncertain ticket write.
- [ ] Define five `@tool` functions with Pydantic argument schemas and useful descriptions. Inject the authenticated conversation context into ticket creation from the service rather than accepting a model supplied ID as authority.
- [ ] Register tools by name. Validate inputs before execution; enforce 5 second timeout; retry transient read-only failures at most twice with short backoff. Produce serializable success/error results and a `ToolMessage` with matching `tool_call_id`.
- [ ] Run focused tests, record the stage, commit.

## Task 4: Chat orchestration and SSE

**Files:** Create `app/services/chat.py`; modify `app/api/chat.py`, `app/core/prompts.py`, `app/core/llm.py`; test `tests/test_chat_api.py`, `tests/test_chat_service.py`.

- [ ] Write failing tests for no-tool answer, one tool then streamed answer, multiple requested tools with only first executed and all IDs answered, selection/model error, tool error, message order, owner mismatch, and client disconnect after ticket write.
- [ ] Update prompt: tools provide demonstration data; use results only; FAQ misses must be admitted; do not promise real order status. Preserve extract prompt behavior.
- [ ] Build selection call with `bind_tools(..., tool_choice="auto")`; persist user and requested calls; emit `conversation` and `tool_status` frames. Run executor once and persist results. Call model again without tools and stream `delta` frames, then persist final assistant and emit `[DONE]`.
- [ ] Keep database sessions short and closed before waiting on upstream output. Preserve the existing `event: error` behavior and make error frames safe for the UI.
- [ ] Run focused and full tests, record the stage, commit.

## Task 5: Chat page Vibe Coding

**Files:** Modify `app/static/index.html`.

- [ ] Change browser request to send stored anonymous `user_id` and server issued numeric `conversation_id`; save the `conversation` frame ID.
- [ ] Show `tool_status` inside the current assistant bubble and a compact tool name badge after success or failure. Keep current streaming text, typing behavior, keyboard send and scrolling.
- [ ] Open the page and manually check one normal answer and one tool call; adjust layout based on observed behavior. Record the stage and commit. This task follows the user's Vibe Coding exception, without brainstorm, TDD or code review gates.

## Task 6: Acceptance and delivery

**Files:** Create `scripts/demo_ch02.py` or equivalent; modify `README.md`, `dev-notes/ch02.md`.

- [ ] Refresh Context7 guidance for any changed library APIs, run the current model's real tool-call smoke test, and stop for user input if the fixed stack is incompatible.
- [ ] With MySQL running and FAQ seeded, run the three labeled samples: “订单 1001 的物流到哪了” calls `query_logistics` and answers from its result; “退货政策是什么” calls `query_faq` and answers from the row; “邮费是多少” calls `query_faq` and returns no match. Record the expected miss for Ch03.
- [ ] Inspect the database: correct conversation owner; user, assistant tool request, tool result and final assistant in order; ticket write when triggered. Verify the browser badge and gradual final response.
- [ ] Run the full pytest suite and share exact demo commands, counts/results and `dev-notes/ch02.md` path. Do final review and commit documentation.
