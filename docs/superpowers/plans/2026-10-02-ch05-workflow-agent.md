# Ch05 Workflow + Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended here) or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 在原SSE聊天入口交付固定Workflow分流、受预算控制的多步Agent、SQLite检查点与独立用户动作。

**Architecture:** 父StateGraph负责透传、规则闲聊、意图与四出口；Agent子图继承父图checkpointer。MySQL保留业务账本，AsyncSqliteSaver保存图State；create_ticket仅在确认动作API调用。

**Tech Stack:** FastAPI、SQLAlchemy/MySQL、LangChain @tool、LangGraph、langgraph-checkpoint-sqlite、Milvus第四章检索器、原生HTML/CSS/JS。

**Spec:** `docs/superpowers/specs/2026-10-02-ch05-workflow-agent-design.md`，用户2026-10-02已确认。

## Global Constraints

- SQLite仅承诺单实例本机持久化；不增加其他数据库或更换技术栈。
- 每轮最多5次工具执行、6次LLM调用、12,000累计输入输出token、120秒，全部可配置。
- 预算包含意图、检索Query理解、FAQ证据检查、Agent决策和最终生成；嵌入/rerank单独记录。
- 商品咨询/退款退货先检索后闸；物流/订单/售后直达Agent；投诉不进Agent；明确闲聊零LLM。
- Agent不能自主调用create_ticket；转人工纯前端模拟，两个按钮互不绑定。
- 先Context7核对实际依赖版本/API再写对应实现；固定选型走不通停下询问。
- 代码TDD；Prompt/阈值用标注评估；每任务当场更新dev-notes/ch05.md并限定提交。
- 不重跑第四章300题；不升级正式指代消解/上下文策略，不新增业务工具。

## Review Focus

1. 重复确认工单或提交后断连：同一offer只生成一张ticket，重试返回原结果（Task6）。
2. 同一会话并发/越权请求：拒绝冲突和越权，不能污染其他线程（Task7）。
3. 服务重启或断连时工具申请未闭合：留下对应中断结果，不重复展示部分答案（Task2/5/7）。
4. 知识轮后紧接闲聊/业务轮：清除本轮证据、动作、预算，避免历史状态污染（Task7）。
5. 模型返回非法意图、工具或非有限分数：Schema/阈值拒绝，不能绕过固定路由（Task3/4/5）。

## 文件与接口约定

新建 `app/workflow/`：contracts.py为State和决策Schema，budget.py为每轮预算，model.py为计费模型适配器，intent.py为分类，retrieval.py为检索闸，agent.py为子图，graph.py为父图，runtime.py为生命周期，service.py为SSE适配。裸循环为`app/workflow/bare_loop.py`和`scripts/demo_agent_loop.py`。动作业务为`app/services/chat_actions.py`、`app/repositories/chat_actions.py`、`app/api/chat_actions.py`。

共享接口：`TurnBudget(limits, clock)`；`BudgetedModel(model, budget).ainvoke(messages, **kwargs)`/`.astream(messages, **kwargs)`；`AgentDependencies(model, tools, executor)`；`build_agent_graph(deps)`；`build_workflow(deps, checkpointer)`；`WorkflowChatService.events(req)`产生现有二元SSE事件；`make_turn_input(req, conversation_id, turn_id)`每轮清空临时State。

State为TypedDict，messages使用add_messages，budget_usage为可序列化字典、运行时model/session/锁不放State。模型、executor和依赖通过运行时上下文传递。决策Schema使用`kind=tools|answer|clarify|stop`、标准tool_calls列表、actions类型数组与简短reason，不请求隐藏推理。

## Task 1：裸循环与预算

**Files:** 新建workflow/__init__.py、contracts.py、budget.py、model.py、bare_loop.py、scripts/demo_agent_loop.py、tests/test_ch05_budget.py、tests/test_ch05_bare_loop.py；修改app/config.py、.env.example。

**Interfaces:** 定义上文TurnBudget、BudgetedModel、AgentDependencies；`run_bare_loop(messages, deps, budget)`返回完整消息和停止原因。

- [ ] 查Context7的LangChain bind_tools、usage_metadata、max_tokens和流式usage接口，记录锁定版本；不立即安装LangGraph。
- [ ] 写预算边界和两步工具的失败测试；脚本化模型/工具定义在tests/ch05_fakes.py，可注入clock、usage和调用序列。

```python
async def test_bare_loop_uses_second_observation():
    from tests.ch05_fakes import bare_scenario
    result, calls = await bare_scenario(["query_order", "query_logistics"])
    assert calls == ["query_order", "query_logistics"]
    assert result.stop_reason == "answered"
    assert result.messages[-1].content == "演示物流结果"
```

- [ ] `uv run pytest tests/test_ch05_budget.py tests/test_ch05_bare_loop.py -q`观察缺接口失败。
- [ ] 实现循环：bind_tools→AIMessage→执行/匹配ToolMessage→回灌→无tool_calls返回；过滤create_ticket；申请越界工具返回明确error。Budget每次请求先reserve输入/输出，completion结算，重试单独计数，计时用monotonic；无usage按保守估计标estimated。

```python
budget.reserve_llm(input_tokens=input_tokens, output_cap=output_cap)
result = await model.ainvoke(messages, max_tokens=output_cap)
budget.settle_llm(result.usage_metadata)
```

- [ ] 验证恰好上限、再调用拒绝、缺usage、token预留超限、超时与工具多申请；运行示例`uv run python -m scripts.demo_agent_loop --offline`。
- [ ] 写dev-note并提交`feat(ch05): add bounded bare agent loop`。

## Task 2：可重试消息落账与工单解耦

**Files:** 修改app/db/models.py、app/repositories/messages.py、app/repositories/tickets.py；新建scripts/migrate_ch05.py、tests/test_ch05_persistence.py；保留旧ChatService接口供旧测试。

**Interfaces:** `append_once(session, conversation_id, message_key, message)`；message_key按turn_id+角色+调用ID稳定生成；`load_history`沿用完整轮次裁剪。

- [ ] 查Context7 SQLAlchemy MySQL事务、唯一索引及upsert；读取真实messages模型后制定兼容迁移。
- [ ] 写同一message_key重复落账只一条、不同会话不可复用key、ToolMessage对应工具申请、工单创建不改会话status测试。

```python
def test_ticket_does_not_transfer_conversation(session, conversation):
    from app.repositories.tickets import create_ticket
    ticket = create_ticket(session, conversation.id, "投诉演示", "投诉")
    assert ticket.conversation_id == conversation.id
    assert conversation.status == "进行中"
```

- [ ] `uv run pytest tests/test_ch05_persistence.py -q`确认失败；实现nullable message_key VARCHAR(128)+唯一索引，历史NULL不回写；append_once冲突读取既有记录且校验参数一致。
- [ ] migrate_ch05有`--check`只读模式与`--apply`幂等模式，按实际DDL核对字段/索引后才操作，支持既有数据库与新建库；不重建/清空数据。处理mysql utf8mb4和SQLite测试差异。
- [ ] 验证调用申请已落账、执行被取消时补interrupted ToolMessage；历史数据仍能载入。
- [ ] 聚焦`uv run pytest tests/test_ch05_persistence.py tests/test_repositories.py tests/test_tools.py -q`，当场dev-note并提交。

## Task 3：七类意图与固定分流

**Files:** 新建app/workflow/intent.py、app/core/workflow_prompts.py、data/ch05/intent_cases.jsonl、scripts/eval_ch05_intent.py、tests/test_ch05_intent.py。

**Interfaces:** `detect_smalltalk(raw_question)->bool`；`classify_intent(model, raw_question)->IntentDecision`；`route_intent(intent)->knowledge|business|complaint|smalltalk`。

- [ ] 测固定七类映射、非法标签/额外字段拒绝、纯问候命中、混合业务不命中；构造至少28条标注样例，七类各4条，包含注入与业务混合口语。

```python
def test_smalltalk_does_not_hide_business():
    from app.workflow.intent import detect_smalltalk
    assert detect_smalltalk("你好")
    assert not detect_smalltalk("你好，订单1001到哪了")
```

- [ ] 运行失败测试；实现完整短句白名单+简单JSON分类prompt，模型标签经Pydantic校验，投诉优先规则写入prompt，多诉求单标签确定优先级为投诉→退款退货→售后→物流→订单→商品咨询→闲聊并标为简化实现。
- [ ] Prompt验证使用`uv run python -m scripts.eval_ch05_intent --offline`校验数据，再live标注评估；输出accuracy与每类错误，不能用prompt文本包含词语冒充评估。
- [ ] 聚焦测试、dev-note、限定提交。

## Task 4：强制检索与前置置信度闸

**Files:** 新建app/workflow/retrieval.py、data/ch05/gate_cases.jsonl、scripts/eval_ch05_gate.py、tests/test_ch05_gate.py；修改app/knowledge/retrieval.py、app/tools/business.py以注入budgeted模型而保留现有调用默认值。

**Interfaces:** `retrieve_for_turn(query, budget)->RetrievalResult`；`gate_result(result, threshold)->GateDecision`；`CitationRegistry.merge(hits)`为同轮唯一编号；query_faq的检索及证据检查注入同一预算。

- [ ] 用现有第四章命中/库外样例构造小型阈值标注集，记录分数，不重跑全量；找到有限阈值使样例符合标签后填入config并保存评估依据。
- [ ] 写空命中、NaN/Inf、阈值等号、模型预算耗尽、外部依赖关闭、失败入池不宣称成功、raw_question保留和引用重编号测试。

```python
def test_nonfinite_score_never_passes_gate():
    from app.workflow.retrieval import gate_result
    from tests.ch05_fakes import retrieval_result
    assert not gate_result(retrieval_result(score=float("nan")), threshold=0.5).allowed
```

- [ ] 运行失败测试后实现复用检索器、budgeted Query理解、显式阈值与reason；失败source/reason沿用low_confidence repository，所有网络等待结束后开启数据库事务。
- [ ] 知识检索通过闸直接供Agent，不再重复自动调用query_faq；Agent主动FAQ仍复用现有工具、计入预算、保持引用编号一致。
- [ ] 聚焦`uv run pytest tests/test_ch05_gate.py tests/test_hybrid_retrieval.py tests/test_tools.py -q`，dev-note和提交。

## Task 5：LangGraph Agent子图

**Files:** 新建app/workflow/agent.py、runtime.py、tests/test_ch05_agent_graph.py、tests/test_ch05_checkpoint.py；修改pyproject.toml、uv.lock、.gitignore。

**Interfaces:** `build_agent_graph(deps)`继承父checkpointer；runtime管理AsyncSqliteSaver的async context；state仅可序列化数据。

- [ ] 查Context7和安装包版本元数据，核对AsyncSqliteSaver.from_conn_string/setup、StateGraph、add_messages、astream/custom writer；锁定兼容版本再`uv add langgraph langgraph-checkpoint-sqlite`，不使用记忆中未核对API。
- [ ] 写两次工具循环、追问、未知工具、恶意create_ticket、预算耗尽不额外LLM、完整工具对应、最终节点才有delta测试。

```python
async def test_final_stream_hides_decisions():
    from tests.ch05_fakes import graph_scenario
    events, state = await graph_scenario(["query_order", "query_logistics"])
    assert state["tool_steps"] == 2
    assert all(data["node"] == "final_answer" for kind, data in events if kind == "delta")
```

- [ ] 运行失败测试；实现decision→tools→decision及answer/clarify/stop出口，tool_calls写标准消息；仅final_answer调用无工具astream，决策中不生成最终正文。
- [ ] 将budget state写入检查点，不依赖重新创建budget后归零；tools节点提前验证并预留数量，取消补悬空结果；最终文本带建议事件，不通过文本解析猜按钮。
- [ ] 用临时SQLite关闭/重开runtime验证已完成状态可读；真实节点失败保存检查点，恢复不重复写ticket/已发delta。运行裸循环与子图同样例对比。
- [ ] 聚焦`uv run pytest tests/test_ch05_agent_graph.py tests/test_ch05_checkpoint.py -q`、dev-note和提交。

## Task 6：独立确认动作与幂等工单

**Files:** 新建app/services/chat_actions.py、app/repositories/chat_actions.py、app/api/chat_actions.py、tests/test_ch05_actions.py；扩展models.py与migrate_ch05.py。

**Interfaces:** `issue_offer(session, conversation_id, turn_id, action, defaults)->ActionOffer`；`confirm_ticket(user_id, conversation_id, offer_id, description, ticket_type)->TicketResult`；POST `/api/chat/actions/ticket`。

- [ ] 测忽略动作0写入、转人工0ticket、工单提交一次写一条、重复返回原工单、越权/参数改动/非ticket offer拒绝、并发确认不重复。

```python
def test_double_confirmation_returns_same_ticket(action_client, ticket_offer):
    first = action_client.confirm(ticket_offer)
    second = action_client.confirm(ticket_offer)
    assert first["ticket_no"] == second["ticket_no"]
    assert action_client.ticket_count() == 1
```

- [ ] 失败测试后实现chat_action_offers表：offer_id VARCHAR(64)主键、conversation_id外键、turn_id VARCHAR(64)、action ENUM(handoff,ticket)、defaults JSON、confirmed_params JSON NULL、ticket_no VARCHAR(32) NULL外键、created_at DATETIME；(conversation_id,turn_id,action)唯一约束。
- [ ] 唯一行FOR UPDATE事务内执行create_ticket工具；build_tools增加可选session注入，默认行为保持，ticket工具参与当前事务、不内部另开事务。commit后才回成功；重试读同offer参数/结果。不得改工具入参/返回契约或新建业务工具。新增最小迁移覆盖全新/已有表检查。
- [ ] API严格Schema与长度限制，服务端先验证归属；不提供handoff服务端执行接口。SQLite本地测试用事务逻辑，MySQL双连接真实竞态验收。
- [ ] 聚焦`uv run pytest tests/test_ch05_actions.py tests/test_tools.py -q`、dev-note和提交。

## Task 7：父图、聊天入口和前端接入

**Files:** 新建app/workflow/graph.py、service.py、tests/test_ch05_workflow.py、tests/test_ch05_chat_api.py；修改app/api/chat.py、app/main.py、app/static/index.html。

**Interfaces:** 父图依赖Agent子图、intent、retrieval、actions；WorkflowChatService保持events(req)接口，新SSE workflow_status/actions。

- [ ] 写路由节点轨迹、纯闲聊0模型、投诉0Agent、业务无预检索、弱证据0Agent、下一轮清空证据/动作、并发409、越权403、cancel闭合工具、启动关闭SQLite测试。

```python
async def test_policy_cannot_skip_gate():
    from tests.ch05_fakes import workflow_scenario
    state = await workflow_scenario("退货政策是什么", intent="退款退货", sufficient=False)
    assert state["node_trace"] == ["resolve", "intent", "retrieval", "confidence_gate", "fallback", "log"]
    assert state["agent_calls"] == 0
```

- [ ] 运行失败测试；父图配置固定路由，在既有入口替换生产dependency，保留旧ChatService作为第2章回归参考。application lifespan统一管理checkpointer和job shutdown；DB同步IO放线程，进程内会话锁在StreamingResponse前获得，流finally释放。
- [ ] stable message_key实现MySQL落账、日志节点及中断日志，取消后终结旧轮并开启新轮，启动检查pending申请；不盲目自动重放图任务。
- [ ] 前端为actions各渲染独立按钮：handoff显示固定两句话，ticket编辑/确认后POST；只锁定所点击按钮；拒绝重复提交、失败恢复按钮、取消确认0请求。继续发消息不受旧offer阻塞。保持引用弹窗/feedback。
- [ ] 原生DOM smoke或真实浏览器验证SSE分片、事件顺序、独立按钮、重复点击、确认取消与引用。抽取内联脚本作`node --check`，不写镜像实现的前端单测。
- [ ] 聚焦`uv run pytest tests/test_ch05_workflow.py tests/test_ch05_chat_api.py tests/test_ch05_actions.py -q`，dev-note和提交。

## Task 8：验收、独立审查和交付

**Files:** 新建scripts/eval_ch05.py、data/ch05/reports/acceptance.json；修改README.md、dev-notes/ch05.md。

**Interfaces:** `uv run python -m scripts.eval_ch05 --offline`脚本化端到端；`--live`只跑章样例并存独立报告，不改ch04报告。

- [ ] 验证本章迁移`uv run python -m scripts.migrate_ch05 --check`，缺表时apply；真实库应用前核实现有数据和预期DDL，不清库。运行MySQL动作并发探针，记录新增演示ticket范围。
- [ ] 离线验证所有路径/预算/恢复；live依次验证政策、订单1001物流、投诉、你好、先查订单再查物流，明确mock结果；保存node_trace、tool_calls、usage、stop_reason。检查点关闭/重启可恢复完成轮次。
- [ ] 跑意图与阈值标注集，记录失败题，不用mock输出声称线上通过；云数据目的地沿用既有明确授权范围，超出范围先澄清。
- [ ] `uv run pytest -q`、`uv run python -m scripts.validate_eval_ch04`、内联JS语法、diff空白检查；按剩余风险扩大验证，不重跑无关云评估。
- [ ] 请求独立整章review，检查Spec、预算/异常/动作事务/恢复，逐项修复后复审；当场追记结论。此前任务完成时记录局部自审结论，执行方式选native则仅此处独立审查。
- [ ] 用户浏览器确认投诉两个按钮互不绑定、忽略继续聊天、引用/feedback；记录真实结果，不用静态smoke冒充交互验收。
- [ ] README写裸循环、迁移、启动与验收命令、SQLite本机范围及多进程不支持；finish记录四项留痕，限定提交并交付报告路径。

## 计划自审结论

八任务覆盖spec全部章节；Review Focus五类各有明确归属测试；协议接口统一为BudgetedModel/TurnBudget/WorkflowChatService；采用正式图State与custom SSE，不在前端猜工具或动作。跨库不声称事务一致，消息/offer分别唯一键幂等；动作工具复用且在确认事务内执行。缺订单号追问、输入恶意标签、零模型闲聊均有行为样例。

## 执行建议

推荐Native：由主代理按任务TDD实施并阶段提交，最后独立审查。八个任务接口依赖紧密，统一实施可以减少反复移交上下文；用户仍可选择每任务独立实现/审查的Subagent-driven方式。当前状态为计划自审完成、待用户审阅并选择执行方式，尚未实施。
