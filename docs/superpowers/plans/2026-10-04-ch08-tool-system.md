# Ch08 Unified Tools and MCP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the preserved Native execution method. Steps use checkbox (`- [ ]`) syntax for tracking. No per-task subagent reviews; one fresh backend reviewer at the end, excluding the frontend per the user's instruction.

**Goal:** 让内置/MCP 工具动态登记并通过统一校验、权限、执行和审计，建单经用户预览确认后安全落库。

**Architecture:** 单例目录保存定义/工厂，每轮生成已授权且绑定会话的工具快照；执行引擎持有可信调用上下文。两个独立官方 MCP Server 提供 mock 查询；LangGraph 独立确认节点暂停，既有 ChatActionOffer 保存幂等状态，审计单独最佳努力写入。

**Tech Stack:** FastAPI、SQLAlchemy/MySQL、LangGraph + AsyncSqliteSaver、LangChain @tool、官方 MCP Python SDK（Streamable HTTP）、langchain-mcp-adapters MultiServerMCPClient、jsonschema。

**Spec:** `docs/superpowers/specs/2026-10-04-ch08-tool-system-design.md`（用户2026-10-04已确认）。

## Global Constraints

- 本章新增一个 MySQL 表 tool_audit_logs，严格沿用用户 DDL：InnoDB、utf8mb4、SET NAMES utf8mb4，不挂外键；不新增业务表。
- 新 MCP 工具采用逐工具授权，客服侧权限配置热更新；本章唯一可执行写工具是 builtin/create_ticket。
- 物流 MCP 接管 query_logistics，旧内置同名工具从运行清单下线；不得静默回退旧物流实现。
- 只读且 retry_safe 的临时故障最多重试2次，写工具重试0次；query_faq 有入池副作用，retry_safe=false。
- 复用 ChatActionOffer 防重复；审计失败不能阻止业务或作为权限/幂等的唯一依据。
- 当前会话的完整历史、摘要、引用、退款和订单选择器继续工作；动态工具 Schema 纳入真实上下文预算。
- 工单卡片按 Vibe Coding 实施，不做前端 brainstorm/TDD/code review，仍进行浏览器功能验收。
- 用户自行在 VS Code 启动聊天服务；测试只能使用自己启动的独立临时进程，不打开用户网页或停止用户8000/8001服务。
- 技术选型走不通就说明并停下询问，不替换 MCP SDK、传输或 Client。
- 每任务完成立即更新 dev-notes/ch08.md 四项记录并提交任务文件；第七章未提交的已验收改动不可丢弃或混作第八章改动。

## Review Focus

1. 确认已到但进程在数据库写入前后崩溃：恢复只能核查结果，不能再建一单（Task5/6测试）。
2. 工具发现后撤权、改 Schema 或发生同名冲突：本轮不得利用旧快照绕过执行检查（Task1/3/4测试）。
3. 父请求取消/超时先于同步写线程完成：审计标未知结果且重试0，不能误告知“未创建”（Task3/5测试）。
4. 暂停期间切会话、刷新、新消息、重复或错会话确认：回载正确卡片，旧凭据不触发新操作（Task6 API测试、Task8浏览器验收）。
5. 工具新增使 Schema 前缀膨胀，同时有第七章摘要：在调用模型前更新预算并安全拒绝装不下的请求（Task7测试）。

## 文件与接口分工

| 文件 | 职责 |
|---|---|
| app/tooling/contracts.py | ToolDefinition、ToolPolicy、CatalogSnapshot、CallContext、ExecutionOutcome、AuditRecord |
| app/tooling/policy.py / registry.py | 本地热授权、可信插件登记、全局名称检查、每轮工具快照 |
| app/tooling/validation.py / results.py | JSON Schema 校验、结果分诊与有界中文 JSON |
| app/tooling/mcp_client.py | MultiServerMCPClient 发现及异常转换，不决定权限 |
| app/services/tool_executor.py | 唯一执行入口、超时重试、许可检查、审计 |
| app/repositories/tool_audits.py | 审计独立事务、失败日志 |
| app/services/ticket_confirmation.py | 预览、确认/取消、持久化写许可及幂等查询 |
| app/workflow/ticket_flow.py | 信息收集状态与独立 interrupt 节点 |
| app/mock/business.py / mcp_servers/*.py | 一致 mock 规则与两个独立 Server |
| config/tool_policy.json / tool_plugins.json | 精确授权和可信内置插件登记，热更新 |
| sql/ch08-ddl.sql / scripts/migrate_ch08.py | 原始DDL及显式重复可运行迁移 |
| scripts/eval_ch08.py / data/ch08/* | 标注集、独立故障注入与可选真实验收 |
| docs/ch08-runbook.md / dev-notes/ch08.md | 操作指南与阶段记录 |

所有 app/tooling 和 mcp_servers 新包创建空 __init__.py。业务上下文留在 CallContext，活工具句柄不序列化进 WorkflowState。

## Task 1：统一目录、参数定义与精确热授权

**Files:** 创建 app/tooling/{__init__,contracts,policy,registry,validation}.py、config/tool_policy.json、config/tool_plugins.json、tests/test_ch08_registry.py；修改 pyproject.toml、uv.lock。

**Interfaces:**

- ToolDefinition(name, description, parameters, source, mcp_server, tool)：tool 是已绑定 LangChain BaseTool；key 为 `(source,mcp_server,name)`，definition_hash 覆盖名称/描述/Schema。
- ToolPolicy(enabled, access, timeout_seconds, max_retries, retry_safe)：access 仅 read/write；write 的重试强制为0。
- CatalogSnapshot(version, definitions)：immutable tuple；`model_tools()` 返回 LangChain 工具列表，`get(name)` 返回定义或 None。
- PolicyStore(path).lookup(key) -> ToolPolicy | None：每次检查读取已验证配置；格式错误不开放外部能力。
- ToolRegistry(policy).register_builtin(factory)：factory 以 CallContext 为参数返回 BaseTool；`snapshot(context, discovered=()) -> CatalogSnapshot`。
- `validate_arguments(schema:dict,args:object) -> list[dict]`：每项包含 path/message，不强制转换；`check_schema(schema)` 注册时运行。
- CallContext 字段：conversation_id、user_id、turn_id、session_factory、raw_question、budget、model、ticket_request（来自服务端）、confirmation（来自Task5）、trusted_results（服务端缓存）。

- [ ] 执行前按 using-git-worktrees 核对当前 main 与未提交第七章成果；沿用已授权的本地工作方式，禁止丢弃/回滚。保存 git 状态和初始测试基线。
- [ ] Context7 核对官方 SDK 的1.x FastMCP、adapters 对应安装版本和 jsonschema；依赖命令 `uv add 'mcp>=1.12.4,<2' langchain-mcp-adapters 'jsonschema>=4.23,<5'`，解析失败则停止说明。锁文件记录精确版本，不能拿v2 MCPServer示例混进1.x代码。
- [ ] 写 RED：下列类型校验，以及未授权工具不进入快照、相同名称不覆盖、同名跨Server冲突、配置更新下一次 lookup 生效、外部 annotations 无法改变权限。

```python
def test_json_types_are_not_coerced():
    from app.tooling.validation import validate_arguments
    schema={'type':'object','properties':{'count':{'type':'integer','minimum':1}},'required':['count']}
    assert validate_arguments(schema, {'count':'2'})
    assert validate_arguments(schema, {'count':True})
    assert validate_arguments(schema, {})
    assert not validate_arguments(schema, {'count':2})
```

- [ ] `uv run pytest tests/test_ch08_registry.py -q`，确认新行为失败。
- [ ] 实现数据类型及校验；使用 Draft202012Validator，拒绝远程 $ref 网络读取，仅允许当前 Schema 本地引用，避免校验触发外部请求。

```python
validator = Draft202012Validator(schema)
errors = sorted(validator.iter_errors(args), key=lambda e: '/'.join(map(str,e.absolute_path)))
return [{'path':'/'.join(map(str,e.absolute_path)), 'message':e.message} for e in errors]
```

- [ ] 配置使用精确key（如 `mcp:logistics:query_logistics`），未知默认拒绝；内置工厂登记与会话绑定分离。插件仅显式导入 app.tools.plugins 包内模块；快照整体构建后原子替换，不修改运行中快照。
- [ ] 测试 GREEN，更新dev-notes，提交 `feat: add unified tool catalog and hot local policy`。

## Task 2：审计模型、仓储与显式迁移

**Files:** 修改 app/db/models.py；创建 app/repositories/tool_audits.py、sql/ch08-ddl.sql、scripts/migrate_ch08.py、tests/test_ch08_audit.py、tests/test_ch08_migration.py。

**Interfaces:** ToolAuditLog 模型严格对应spec第8节；AuditRecord 包含DDL除id/created_at外字段；`ToolAuditRepository(session_factory).write(record) -> bool` 独立提交，异常记录日志并返回False。

- [ ] 写 RED：模型 unsigned、nullable、5种中文ENUM、3个普通索引、无外键、默认值/注释一致；迁移已有匹配表无操作、错误类型/外键停止；仓储故障不向调用方抛出。

```python
def test_audit_does_not_require_conversation_foreign_key():
    from app.db.models import ToolAuditLog
    assert not ToolAuditLog.__table__.foreign_keys
    assert set(ToolAuditLog.__table__.c.status.type.enums)=={'成功','失败','超时','校验拦下','权限拒绝'}
```

- [ ] `uv run pytest tests/test_ch08_audit.py tests/test_ch08_migration.py -q`，确认RED。
- [ ] 复制用户DDL全文到sql文件（SET NAMES和COMMENT保留），实现模型；迁移复用ch07检查风格，单独检查主键自增、列类型/长度/默认/注释、索引及无外键，不自动修坏表。
- [ ] 独立仓储使用短超时数据库连接，失败只写日志；调用方也限制等待时长。result_summary限制UTF-8字节，error_message最多512字符；duration_ms与retry_count有界。

```python
try:
    with session_factory() as session, session.begin():
        session.add(ToolAuditLog(**record))
    return True
except Exception:
    logger.exception('tool audit write failed')
    return False
```

- [ ] `uv run python -m scripts.migrate_ch08 --check`；缺表后显式 `--apply`，再连续两次 `--check`，确认幂等；不启动时迁移。
- [ ] 测试GREEN，记录真实MySQL结构检查结果，提交 `feat: persist best effort tool audits`。

## Task 3：统一执行、分诊、重试与结果整理

**Files:** 修改 app/services/tool_executor.py；创建 app/tooling/results.py、tests/test_ch08_executor.py；更新 tests/test_tool_executor.py。

**Interfaces:**

- `ToolExecutor(snapshot,policy,audits,context).preflight(call) -> ExecutionOutcome`：纯权限/Schema检查；允许的未确认建单返回needs_confirmation。
- `async execute(call, *, before_attempt=None) -> ToolMessage`：未持有可信确认直接返回permission_denied；不在引擎内部调用interrupt。
- `async reject(call,reason) -> ToolMessage`：不派发、记录权限拒绝，用于取消/作废。
- ExecutionOutcome(kind,payload)，kind取 ready/invalid/denied/needs_confirmation。
- `normalize_result(definition,raw) -> dict`：成功result或error；`serialize_result(payload) -> str` 中文JSON、标识符不截断。

- [ ] 写RED：参数错误、未授权、unknown、业务空结果各不执行/不重试；连接失败后成功重试1次；持续超时重试2次；write超时0；audit异常不影响返回；撤权即时拒绝。

```python
async def test_write_timeout_has_zero_retries(executor_fixture):
    engine,calls,audits=executor_fixture(access='write', confirmed=True, failure=TimeoutError())
    result=await engine.execute({'name':'create_ticket','args':{'description':'包裹未收到','ticket_type':'售后'},'id':'w1'})
    assert calls.count==1
    assert audits[-1]['status']=='超时' and audits[-1]['retry_count']==0
    assert 'timeout' in result.content
```

fixture在本测试文件构造Task1目录、内存审计列表、计数调用器、Task5确认接口的受控假对象；只替换外部I/O，不替换引擎。

- [ ] `uv run pytest tests/test_ch08_executor.py -q` 观察RED。
- [ ] 实现状态机；attempt前校验并调用before_attempt预算计费，`asyncio.wait_for`包调用；只对白名单临时网络异常重试，退避0.1/0.2秒，重试上限受本地策略约束。

```python
attempts = 1 + (policy.max_retries if policy.access=='read' and policy.retry_safe else 0)
for attempt in range(attempts):
    if before_attempt: before_attempt()
    try:
        raw = await asyncio.wait_for(tool.ainvoke(args), policy.timeout_seconds)
        break
    except (TimeoutError, ConnectionError) as exc:
        if attempt+1==attempts: raise
        await asyncio.sleep(0.1*(attempt+1))
```

外层将最终异常分类并在finally审计；预算异常不伪装成功。父取消使用有界shield审计然后重新抛取消，不能吞掉LangGraph暂停异常。

- [ ] MCP传输异常集中适配成上述暂时错误；isError业务响应不重试。规范化嵌套JSON时兼容旧消费者，更新消费者统一接收result对象；为FAQ保留citations/sufficient。
- [ ] 真正副作用超时案例用线程事件控制晚完成，不靠长sleep；确认重试0、审计超时、最终结果未误报未执行。
- [ ] GREEN并更新记录，提交 `feat: centralize tool execution permissions retries and outcomes`。

## Task 4：两个官方 MCP Server 与动态发现

**Files:** 创建 mcp_servers/{__init__,logistics,after_sales}.py、app/mock/{__init__,business}.py、app/tooling/mcp_client.py、tests/test_ch08_mcp.py；修改 app/tools/business.py、app/config.py、config/tool_policy.json。

**Interfaces:** `MCPDiscovery(connections).discover() -> tuple[ToolDefinition,...]`，内部按Server调用get_tools；`mock_order_snapshot(order_id)->dict`沿用既有确定性规则；Server工具名query_logistics/query_warranty/query_return_progress。

- [ ] Context7按Task1锁定版本核查 FastMCP、MultiServerMCPClient、返回isError/structuredContent接口，记录版本及引用。
- [ ] RED测试：两个真实HTTP子进程端到端调用；单Server宕机仍发现另一台；仅重启Server增加工具客户端可发现；同订单订单/物流状态一致；所有模型工具无重复query_logistics。

```python
async def test_http_discovery_keeps_server_provenance(mcp_pair):
    definitions=await mcp_pair.discovery.discover()
    assert {d.mcp_server for d in definitions}=={'logistics','after_sales'}
    assert sum(d.name=='query_logistics' for d in definitions)==1
```

mcp_pair在tests/test_ch08_mcp.py提供：用sys.executable启动两个测试端口进程，轮询HTTP就绪，yield后只终止自身PIDs；不碰用户服务。

- [ ] 实现Server（端口默认8101/8102，可用环境变量配置）；Task1锁定1.x时使用：

```python
from mcp.server.fastmcp import FastMCP
mcp=FastMCP('logistics',host='127.0.0.1',port=8101,stateless_http=True,json_response=True)
@mcp.tool()
def query_logistics(order_id:str)->dict:
    return logistics_result(order_id)
if __name__=='__main__': mcp.run(transport='streamable-http')
```

`logistics_result(order_id)->dict`定义在app/mock/business.py，使用共享mock_order_snapshot构造与生命周期一致的轨迹；其它Server函数同样只使用纯mock模块，不能初始化数据库。

- [ ] 客户端使用指定 MultiServerMCPClient，按Server分别超时发现并处理异常；连接transport采用安装版本验证的Streamable HTTP枚举（文档 http），不用旧SSE。显式关闭localhost代理继承的测试HTTP连接，用户Clash配置不动。
- [ ] 移除运行内置清单query_logistics，保留旧历史名称回载；fake工具测试可以自行定义物流替身，不在生产路径回退内置物流。
- [ ] `uv run pytest tests/test_ch08_mcp.py tests/test_tools.py -q` GREEN，记录实际HTTP及热发现证据，提交 `feat: serve logistics and after sales through MCP`。

## Task 5：工单预览账本、可信确认与防重复

**Files:** 创建 app/services/ticket_confirmation.py、tests/test_ch08_ticket_confirmation.py；修改 app/services/chat_actions.py、app/repositories/chat_actions.py。

**Interfaces:** TicketConfirmationService(session_factory)：`prepare(context,call,definition_hash)->dict`，`confirm(user_id,cid,confirmation_id)->dict`，`cancel(...)->dict`，`claim(context,call)->dict`，`lookup(confirmation_id)->dict`；`execute_ticket_once(context,args)->dict`是注册的写调用器，保证票据与offer结果同事务。

可信confirmation只由后端在校验归属/固定参数后写入CallContext；客户端和模型参数无法构造它。预览返回confirmation_id/turn_id/tool_call_id/ticket_type/description，内部元数据存ChatActionOffer.defaults的_ch08键。

- [ ] RED：两次确认仅一行ticket；更改参数拒绝；wrong user/cid/turn拒绝；cancel不写；claim后崩溃再恢复不派发；写提交后checkpoint未保存恢复返回同ticket_no；旧投诉按钮一次确认仍有效。

```python
async def test_duplicate_confirmation_returns_existing_ticket(ticket_flow):
    first=await ticket_flow.confirm_and_execute()
    second=await ticket_flow.confirm_and_execute()
    assert first['ticket_no']==second['ticket_no']
    assert ticket_flow.ticket_count()==1
```

ticket_flow使用独立合成会话+真实MySQL事务，测试结束只删本用例数据；审计无外键可保留或按本用例call_id清理。

- [ ] `uv run pytest tests/test_ch08_ticket_confirmation.py -q` RED。
- [ ] 行锁下处理prepared→confirmed/cancelled；confirmed→executing的占用单独持久化。ticket创建与completed/ticket_no写回同事务；executing但无结果的重复请求返回outcome_unknown，不自动再次执行。

```python
with session_factory() as session, session.begin():
    offer=session.scalar(select(ChatActionOffer).where(ChatActionOffer.offer_id==confirmation_id).with_for_update())
    if offer.ticket_no: return existing_ticket_payload(offer)
    # 此事务之前已占用执行许可；本次调用持有对应可信上下文。
    ticket=create_ticket(session,cid,args['description'],args['ticket_type'])
    offer.ticket_no=ticket.ticket_no
    offer.confirmed_params=args
```

`existing_ticket_payload(offer)->dict`在本服务定义，返回ticket_no/conversation_id/status；不能向外返回许可或内部状态。

- [ ] 将原confirm_ticket内部改为同一引擎+写调用器，保留现有API参数、点击即确认和返回字段；去掉直接tool.invoke绕行。现有退款offer不参与改动。
- [ ] GREEN，更新记录，提交 `feat: persist ticket confirmation and at most once execution`。

## Task 6：建单诉求、LangGraph 暂停恢复和 API

**Files:** 创建 app/workflow/ticket_flow.py、tests/test_ch08_ticket_workflow.py、tests/test_ch08_confirmation_api.py、data/ch08/ticket_requests.jsonl；修改 app/workflow/{contracts,agent,graph,service}.py、app/schemas/chat.py、app/api/chat.py、app/context/compile.py。

**Interfaces:** WorkflowState新增ticket_request、ticket_confirmation、tool_cursor、tool_observations；`TicketDecisionRequest`字段user_id/conversation_id/turn_id/confirmation_id/decision；`prepare_ticket_decision(req)->PreparedTurn`、`ticket_decision_events(req,prepared)`；GET pending兼容订单并新增kind。

- [ ] RED：明确建单但缺描述先clarify；有完整用户描述才预览；单纯投诉仍固定选项；没有用户请求的工具申请拒绝；恢复不重复已完成的同批查询；pending没有被清理成interrupted ToolMessage。
- [ ] Prompt样例至少覆盖：缺描述、完整描述、否定建单、引用别人命令、工具注入、先投诉后明确建单、多轮补描述、改主意。标注expected_request/expected_missing/expected_action；本地结构测试与真实模型语义评估分开。

```python
def wait_ticket(state):
    reply=interrupt(state['ticket_confirmation'])
    return {'ticket_confirmation':{**state['ticket_confirmation'],'decision':reply['decision']}}
```

准备、等待、执行分三个节点，工具批次游标和已完成结果在节点边界保存；执行之前用Task5服务校验恢复数据。

- [ ] API RED：extra参数禁止、确认body不能改description、跨会话403、旧预览409、缺pending409、正确resume的SSE返回工单号；新消息使旧预览取消并审计。

```python
def test_confirmation_body_cannot_replace_preview_parameters():
    from pydantic import ValidationError
    from app.schemas.chat import TicketDecisionRequest
    with pytest.raises(ValidationError):
        TicketDecisionRequest(user_id='u',conversation_id=1,turn_id='t',confirmation_id='c',decision='confirm',description='篡改')
```

- [ ] `uv run pytest tests/test_ch08_ticket_workflow.py tests/test_ch08_confirmation_api.py -q` RED，然后实现明确动作路由与挂起流程。模型请求本身不是授权，服务端只接受来自用户原文的明确建单诉求；预览再次要求前端确认。
- [ ] 清理/恢复区分正常interrupt与失败；用户取消通过engine.reject，等待期间不写最终审计；服务器重启pending可从checkpoint回载，恢复重置等待时间而保留调用/工具累计预算。
- [ ] GREEN，连同 `tests/test_ch06_resume.py tests/test_ch05_actions.py` 回归，提交 `feat: interrupt ticket requests for customer confirmation`。

## Task 7：全入口接入及动态 Schema 预算

**Files:** 修改 app/main.py、app/workflow/{service,agent,after_sales,bare_loop}.py、app/services/chat.py、app/context/budget.py、app/core/prompts.py；创建 tests/test_ch08_integration.py、tests/test_ch08_context_budget.py。

**Interfaces:** `with_tool_prefix(config,system_messages,schemas)->ContextConfig`，prefix=max原预留与实际估算，后续calculate_budget只使用新cfg；目录及发现服务由lifespan管理，按请求绑定CallContext。

- [ ] RED：全工具包括缓存结果、投诉按钮、售后预查订单均经过engine/audit；裸循环/旧ChatService也不能绕过写权限；query_faq引用仍可映射；动态注册工具下一轮可被模型绑定。
- [ ] 预算RED：动态前缀增长改变历史额度；有摘要时仍扣实际摘要；超过窗口在LLM之前报ContextCapacityError；相同目录版本system/Schema前缀固定。

```python
def test_dynamic_prefix_is_charged():
    from dataclasses import replace
    from app.context.budget import calculate_budget
    from app.context.contracts import ContextConfig
    cfg=ContextConfig(window=20000,output=2000,user_input=2000,steps=3,tool_result=1200,rerank_k=5)
    assert calculate_budget(cfg).history-calculate_budget(replace(cfg,prefix=cfg.prefix+500)).history==500
```

另测18000窗口增加前缀后不足以容纳一轮时抛出ContextCapacityError，不能带着负余量继续请求模型。

- [ ] 先 `uv run pytest tests/test_ch08_integration.py tests/test_ch08_context_budget.py -q` RED，再实现目录刷新先于历史预算/投影，Agent每次bind使用本轮快照；恢复重新核对权限/定义，不能带旧schema确认新参数。
- [ ] 把agent节点的query_order/query_faq复用分支迁入引擎trusted_results通道；模型无权声明缓存或审计来源。旧consumer兼容result对象，完整ToolMessage只留checkpoint。
- [ ] GREEN及相关ch05/ch06/ch07回归，记录，提交 `feat: integrate dynamic tools with workflow context budgets`。

## Task 8：工单预览卡片（Vibe Coding）

**Files:** 修改 app/static/index.html；创建 scripts/verify_ch08_browser.py。

**Interfaces:** ticket_confirmation SSE及Task6 POST；pending kind为ticket_confirmation时渲染，旧order selection路径保持可用。

- [x] 直接沿用订单卡片样式：展示类型/描述、确认提交/取消；发送中禁用，成功显示ticket_no，取消锁定；unknown显示“结果待核实”且不显示自动重试按钮。
- [x] POST仅发送身份/confirmation_id/turn_id/decision；不将DOM内容当成新业务参数。使用textContent呈现工具描述及工单内容，避免外部文本变HTML。
- [x] 直接浏览器验收：确认/取消、连点、刷新回载、旧卡失效、新消息、原投诉按钮及订单卡。截图/结果记录到data/ch08/reports；不先写前端单测，不发前端review。
- [x] 页面请求测试使用拦截路由；不打开用户网页、不重启用户服务。修改效果经真实 Edge 截图与页面断言验收。
- [ ] 记录交互结果，提交 `feat: show ticket confirmation cards in chat`。（当前 Git index ACL 拒绝暂存/提交）

## Task 9：完整验收、后端审查与交付

**Files:** 创建 scripts/eval_ch08.py、data/ch08/tool_cases.jsonl、docs/ch08-runbook.md；更新dev-notes、计划勾选和必要回归测试。

**Interfaces:** `eval_ch08 --local`执行真实MCP进程/数据库和可控模型替身；`--live`使用合成样例和已授权云模型；报告标明model_mode、版本、工具来源、call_id、审计查询、票据增量、分项结论。未获外发授权时先完成所有本地验收。

- [x] 标注Prompt样例与执行案例：新增内置工具、MCP热发现、未知工具、Schema错误、空结果、网络重试、写超时、审计失败、缺描述追问、确认、取消、错会话、重复确认、撤权、崩溃恢复。失败即明确记录，不静默改变预期。
- [x] 使用真实两个MCP进程和同一客服运行时发现工具；Server 单独重启热发现及故障隔离通过测试；审计验证物流source=mcp、运行目录不含内置物流。
- [x] 运行完整 `pytest`（744 passed）、迁移检查、离线验收（57 passed）及浏览器脚本（27请求）；全部本地门槛通过。
- [ ] 获得第八章合成工单/问题发送授权后执行 `uv run python -m scripts.eval_ch08 --live`；逐条人工检查追问/预览/工单号及工具来源，不能只看JSON可解析。超时重试用本地故障注入证明，不通过反复云请求制造超时。
- [x] 按 Native 做一次新上下文后端独立审查，明确排除前端文件；review未发现高优先级问题。
- [x] runbook给出：

```powershell
uv run python -m scripts.migrate_ch08 --apply
uv run python -m mcp_servers.logistics
uv run python -m mcp_servers.after_sales
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
uv run python -m scripts.eval_ch08 --local
uv run python -m scripts.eval_ch08 --live
```

两个Server与客服分别占用终端，命令不会自动打开浏览器；环境变量端口/Server地址、权限登记样例、审计查询、工单确认/取消演示语句必须写入指南。

- [x] 按要求逐阶段追记用户原话、关键产出/评审、纠偏、失败返工；真实模型与 Git 提交门槛仍阻塞，finish 未记录。

## 计划自审与交接

- Spec1–4注册/权限：Task1/4/7；Spec5执行：Task3；Spec6 MCP：Task4；Spec7确认：Task5/6/8；Spec8 SQL审计：Task2/3；Spec9预算前端：Task7/8；Spec10验收：Task9。
- Review Focus五项均有所属任务测试；真实外发、SDK兼容和确认恢复风险单列，不靠模型自行判断权限。
- 类型接口保持 ToolDefinition→CatalogSnapshot→ToolExecutor，CallContext不进checkpoint；持久化只存JSON状态与稳定ID。
- 先前已选择Native，本章沿用；书面计划待用户审阅，确认后开始实施。前端例外贯穿Task8及最终审查范围。
