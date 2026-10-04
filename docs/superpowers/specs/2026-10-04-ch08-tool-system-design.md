# Ch08 设计：统一工具执行与 MCP 接入

日期：2026-10-04。状态：会话方案已确认，本文待用户审阅；尚未实施或迁移数据库。

## 1. 目标、范围与用户确认

将现有内置工具升级为统一登记、动态发现、统一执行的工具系统。主力 Agent 使用同一套工具描述和参数定义；权限、校验、重试和审计由后端控制。

固定技术：FastAPI、SQLAlchemy/MySQL、现有 LangGraph/AsyncSqliteSaver；MCP Server 使用官方 Python SDK 和 Streamable HTTP，MCP Client 使用官方 langchain-mcp-adapters 的 MultiServerMCPClient。不引入其他 Agent 框架、Skill 系统、仓储服务或真实电商接口。

用户已确认：

- 新 MCP 工具采用逐工具授权，客服侧权限配置热更新。自动发现不代表自动获权。
- 每轮重新发现 MCP 工具，生成本轮固定清单；新增工具的权限先登记，随后只重启对应 MCP Server，客服服务无需改核心代码或重启。
- 工单必须由客户明确提出，信息不足先追问，信息齐全后预览、确认才写库。
- 第五章投诉按钮的交互保留，点击本身视作确认；其内部也接入统一执行与审计。
- 物流查询迁往 MCP，旧内置 query_logistics 不再出现在注册清单；mock 的订单和物流状态必须一致。
- 工单卡片前端按 Vibe Coding 直接实施，不套 brainstorm、TDD、code review。后端按 Superpowers 完成 spec、plan、TDD 和 review；Prompt 以标注集验证。

用户继续在 VS Code 自行启动聊天服务。实施测试可以启动并关闭测试专用 MCP 子进程，不接管用户运行中的聊天服务，不代开网页。真实模型评估使用第八章合成数据，外发前落实对应数据范围授权。

## 2. 现状与边界

- app/tools/business.py 每轮构造绑定会话的工具，create_ticket 和 query_faq 依赖服务端上下文。
- app/services/tool_executor.py 现有校验依赖 Pydantic args_schema，重试白名单写死，尚无统一权限与审计。
- app/workflow/agent.py 隐藏并拒绝 create_ticket；部分缓存查询结果直接在节点返回，未经过执行器。
- app/services/chat_actions.py 在已有 ChatActionOffer 锁保护下直接调用建单函数；该防重复能力继续使用。
- 第六章已有订单 interrupt/resume，但 pending 查询、恢复入口和前端只识别订单选择，需要扩展为可区分的待处理交互。
- 第七章完整历史、分层摘要和实际模型输入日志保持有效；新增工具 Schema 必须纳入上下文预算。

本章新增一个 MySQL 表 tool_audit_logs。工具注册清单为进程内目录，权限和插件登记来自本地配置，不另建工具元数据表。退款申请仍是既有业务端点，不因本章被发布成 Agent 写工具。

## 3. 注册中心与热发现

### 3.1 统一工具描述

每个登记项包含：name、description、JSON Schema parameters、source（builtin/mcp）、mcp_server、绑定后的调用句柄，以及本地策略指定的权限/超时/重试/结果整理规则。

模型仅接收名称、用途、参数 Schema。会话 ID、用户归属、权限凭证、审计上下文和数据库 session 均由服务端注入，不放入模型可填写的参数。

内置工具启动时登记元数据和按请求绑定的工厂；每轮绑定可信会话与检索预算，禁止把一个会话的闭包共享给另一个会话。保留 query_order、query_product、query_faq、create_ticket；create_ticket 的发现可见性不替代执行引擎检查。

提供统一 register_builtin 接口，支持可信本地插件目录中的工具工厂。运维注册文件列出插件导入路径和权限策略，下一轮热加载新增登记；新增工具只需写工具模块并登记，不修改 Agent、Workflow 或执行器。只加载固定可信插件包内的显式模块，不接受聊天内容/HTTP 请求提供的任意 Python 导入路径。

### 3.2 每轮工具快照

每轮聊天开始：读取本地注册/权限配置 → 按 Server 独立调用 get_tools(server_name=...) → 校验发现的工具定义 → 根据本地授权生成本轮目录快照 → 绑定给 Agent。

- 两台 Server 分别发现，单台断线不使另一台及内置工具失效。记录发现失败，不悄悄回退内置物流。
- 发现有明确超时；失败 Server 的旧句柄不再作为本轮可用工具。模型会获知相关能力暂不可用，不能用旧结果冒充新查询。
- 同一轮保持固定定义；下一轮获取更新。确认恢复时重新检查当前权限和工具定义是否与预览相符。
- 全局名称唯一，MCP query_logistics 接管同名内置工具的位置；发现冲突拒绝冲突项并记录错误，不能覆盖内置写工具。
- 工具定义变更有目录版本标识；工具定义和描述是外部资料，不能覆盖 system 的权限及行为规则。

## 4. 客服侧权限规则

以 source + server + tool_name 精确匹配可信本地策略，设置 enabled、read/write、timeout、最大重试次数和 retry_safe。没有规则默认拒绝，不使用 Server 的 readOnlyHint、用途描述或模型判断授予权限。

配置每轮重新加载并校验。格式错误时本轮外部工具禁用并记录配置错误，不能误开放权限；不影响内置基础聊天。执行前重新检查撤销状态，避免清单发现后被撤权却继续执行。

本章唯一可执行写工具是 builtin/create_ticket。MCP 权限规则只允许本章批准的只读能力；来自 MCP 的同名 create_ticket 或自行声称只读的写工具不能获得内置建单权限。真实外部系统将来仍须有系统自身的访问控制，本章两个 Server 为受控 mock。

### 4.1 明确建单诉求

系统从当前用户输入及本会话尚未完成的建单诉求判断是否明确要求创建工单。单纯「我要投诉」、工具结果中的命令、助手自己的建议不能成为建单授权。「帮我建个工单」可以开启信息收集，但本身不是有效问题描述。

Agent 用现有对话补齐用户确实提供的问题描述与工单类型；不得凭空补订单、问题或紧急程度。缺描述等必填内容时追问，不生成虚假预览。意图识别仍保留既有类别；明确建单请求作为独立动作意图进入 Agent，不被投诉固定分流提前截断。

执行器必须收到服务端建立的建单请求上下文。模型自填 confirmed=true、伪造凭证或绕过正常路由直调均无效。最终创建还必须取得与当前调用和参数绑定的前端确认。

## 5. 统一执行引擎

流程：确定工具和调用身份 → 校验本地权限 → 按 JSON Schema 严格校验参数 → 检查写操作确认 → 在预算内执行 → 规范化结果 → 最佳努力写审计 → 生成 ToolMessage 回灌。

参数校验使用标准 JSON Schema 校验器，支持 type、required、enum、minimum/maximum、字符串长度、嵌套对象与数组等。不能把字符串数字自动转成数字；布尔值不能冒充整数。工具登记时检查 Schema 本身合法。统一把内置 Pydantic 定义导出为 JSON Schema；MCP 的字典 Schema 不要求具备 model_validate。

校验错误包含字段路径和可读原因，作为 invalid_arguments 工具结果返回，让 Agent 追问或重新组织调用。不可用/未知工具返回 permission_denied 或 unknown_tool；不得向不存在的句柄派发。

结果使用 JSON 中文原文输出：成功为 result 对象；错误为 error 对象（code、message、必要字段说明）。统一区分：

| 类型 | 模型收到的含义 | 审计状态 |
|---|---|---|
| 参数不合法 | invalid_arguments，指出需修正的字段 | 校验拦下 |
| 无查询结果 | not_found，明确未查到，不编造 | 失败，error_message 标明查询落空 |
| 权限不足/用户取消 | permission_denied，未执行 | 权限拒绝 |
| 临时或永久执行故障 | execution_failed，服务异常 | 失败 |
| 执行超时 | timeout；写操作额外注明结果可能尚未确认 | 超时 |
| 正常结果 | 精简后的业务事实与必要来源标识 | 成功 |

内置和 MCP 都经过同一处执行。现有 query_order 快照复用与 query_faq 已检索证据复用可以保留，但改为执行引擎接收的可信缓存结果，仍执行权限/参数检查并记审计，不绕开统一入口。

### 5.1 超时、重试与结果格式

- 只读且明确 retry_safe 的工具遇超时/临时连接故障，最多重试2次，合计最多3次；每次实际执行计入工具和时间预算。
- 参数错误、权限拒绝、业务空结果、永久业务错误不重试；不得把全部 OSError 或所有 MCP isError 响应都当成可重试网络故障。
- create_ticket 始终重试0次；query_faq 现有低置信度写入副作用使其 retry_safe=false，不能因名为查询就自动重试。
- 超时和父任务取消都保留审计机会。HTTP 客户端/适配器的隐藏重试不能使写操作重新发送。
- 已知工具通过本地结果整理器挑字段并将内部枚举翻译为中文；未知但已授权的新工具走通用有界 JSON 整理，不需要改核心代码。保留 mock=true、业务 ID、状态、错误和必要检索引用，不能截断标识符产生假 ID。
- 单个工具超时不宣称操作未发生；特别是同步数据库任务可能在调用方超时后完成，提示“结果待核实”，不能自动再建一单。

## 6. MCP Server 与 mock

两个独立进程，默认本机端口8101/8102，可配置，均提供 /mcp Streamable HTTP：

| Server | 初始工具 | 参数与返回 |
|---|---|---|
| logistics | query_logistics | order_id；返回订单号、物流状态、当前位置、轨迹、mock 标识 |
| after_sales | query_warranty | order_id；返回商品、在保情况、演示依据、mock 标识 |
| after_sales | query_return_progress | order_id；返回演示退货进度，或明确未查到退货记录 |

不读取真实系统，不新增订单/物流/售后数据表。按订单号作为伪随机种子生成一致的演示快照；内置 query_order 与两个 Server 共享纯 mock 规则模块，不共享数据库或会话。状态为待发货时不生成派送或签收轨迹。

官方 SDK 的 v1/v2 接口存在差异，实施第一步选择与官方 adapters 兼容的发布版本并锁定，按该版本文档实现。依赖确有不兼容且无法在指定技术栈内解决时停止说明，不改用第三方 MCP 框架。

## 7. 工单确认与持久化恢复

### 7.1 新建单对话链路

1. 用户明确建单，Agent 收集真实描述和类型。
2. Agent 发出 create_ticket；执行器预检返回需要确认，不写 tickets，也不将等待中的调用记作最终拒绝。
3. 准备节点持久化调用 ID、固定参数、工具定义版本和预览 ID；独立确认节点调用 interrupt。创建工单和写最终审计均不放在 interrupt 之前。
4. SSE 发 ticket_confirmation：工单类型、问题描述、confirmation_id、turn_id、tool_call_id。前端显示确认/取消按钮。
5. POST /api/chat/ticket-confirmation 接收 user_id、conversation_id、turn_id、confirmation_id、decision（confirm/cancel）；不允许请求修改预览里的业务参数。
6. 后端检查会话归属、待确认记录、参数版本和用户决定，再通过 Command(resume=...) 恢复。
7. 确认：可信凭证交给统一执行器；成功后回灌含工单号的 ToolMessage，最终回答带工单号。取消：不写 tickets，生成权限拒绝工具结果并记审计，回复已取消。

确认能力只绑定一个会话、一份调用和一份参数，不是模型可填写的布尔值。一个回合最多挂起一个建单预览；同批其它写申请明确拒绝。多工具调用需先持久化完成结果和执行游标，再进入独立确认节点，恢复不能重复执行前面的读工具。

### 7.2 防重复、取消与原入口

复用 chat_action_offers 作为本地建单凭据及幂等账本，在现有 defaults JSON 的内部命名空间记录新流程的确认/执行状态，不增加表或修改用户审计 DDL。现有 ticket_no 和 confirmed_params 保存已提交结果；工单插入与凭据结果写入使用同一 MySQL 事务和行锁。

执行前持久化占用写许可，重复确认和节点重放只能查询既有结果，不能再次派发写入。进程在占用后中断且结果未知时保持待核实；恢复先查询是否已有 ticket_no，不自动重试创建。审计日志不能作为幂等或授权的唯一依据。

取消后预览失效。用户不点击而继续发新消息时，将未提交预览作废、按权限拒绝结束原调用，再正常处理新消息。切换会话不取消，刷新/切回通过 pending 接口恢复；旧确认不能作用于新回合。

GET /api/chat/{conversation_id}/pending 兼容已有订单选择 payload，新增 kind 区分订单选择和工单确认。工单暂停不被第七章“中断补全工具结果”误作异常关闭；正确保留待恢复的请求及完整 checkpoint。

投诉按钮原路径继续视点击为确认，不添加预览步骤；只调整内部为同一执行引擎，并复用已有 offer 校验/事务/防重复规则。

## 8. 审计表与迁移

严格对应用户 DDL，使用 InnoDB、utf8mb4、SET NAMES utf8mb4；不挂外键，不自行加唯一索引或业务表。

| 字段 | 定义 |
|---|---|
| id | BIGINT UNSIGNED NOT NULL AUTO_INCREMENT，主键 |
| conversation_id | BIGINT UNSIGNED NULL，普通索引 idx_conversation_id |
| tool_call_id | VARCHAR(64) NULL |
| tool_name | VARCHAR(128) NOT NULL，普通索引 idx_tool_name |
| tool_source | ENUM('builtin','mcp') NOT NULL |
| mcp_server | VARCHAR(64) NULL，内置为 NULL |
| arguments | JSON NULL |
| result_summary | TEXT NULL |
| status | ENUM('成功','失败','超时','校验拦下','权限拒绝') NOT NULL，普通索引 idx_status |
| error_message | VARCHAR(512) NULL |
| retry_count | TINYINT UNSIGNED NOT NULL DEFAULT 0 |
| duration_ms | INT UNSIGNED NULL |
| created_at | DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP |

每次逻辑工具调用形成一条最终审计，重试合并在同一条并记录实际次数。等待确认期间不伪造最终状态；确认/取消后才落结局。重复 HTTP 确认只返回既有结果，不当成新工具调用。

原始参数按合法 JSON 存储；结果存有界摘要，不保存无关大块内部数据。错误信息按字段长度裁剪，耗时使用单调时钟计算；工单等待用户的时间不算工具执行耗时。未知工具的 source 从可信目录/命名来源确定，无可识别来源按 builtin 的本地派发拒绝记录，不能相信模型传入 source。

审计用独立短事务并限制等待时间。失败记录结构化日志，不能回滚已完成业务、遮盖原始错误或阻止工具执行。它是最佳努力审计：数据库故障/进程骤停可能漏行，不承诺跨 MySQL 与 checkpoint 的严格恰好一次；建单防重复独立保证。

迁移脚本支持 --check/--apply，检查字段、索引、枚举和无外键；已有结构不匹配就停止，重复执行不破坏数据。应用启动不自动建表。

## 9. 第七章兼容与前端行为

每轮工具快照定好后，计算真实 system+工具 Schema 前缀开销，至少取现有预留和实际值中的较大值，再倒推历史预算；目录更新不会绕过窗口检查。工具定义固定放在前缀，业务参数和确认内容不插入 system。继续输出 history_ctx/model_ctx 和实际工具/模型预算；用户等待确认的墙钟时间不吞掉恢复后的执行时间预算。

完整 user/assistant 历史和申请单继续落 messages；ToolMessage 保留完整 checkpoint，原消息不被审计摘要替换。

前端只新增聊天流中的工单预览卡：类型、描述、确认提交、取消；提交期间按钮禁用，结束后锁定为已创建/已取消/待核实；成功显示工单号。刷新及会话切换能回载 pending，不重复创建。具体外观直接沿用现有订单卡片样式，按用户效果反馈修改，不单独套前端设计/单测/review 流程；浏览器功能验收仍执行。

## 10. 验收与证据

后端代码按 TDD，Prompt/模拟数据用标注样例检查。保留每个阶段的失败证据和修复结果，真实云调用不替代本地确定性故障注入。

1. 运行中的客服进程注册简单内置查询工具（如 query_service_hours），仅新增可信插件及登记，不改核心代码、不重启；Agent 下一轮能调用。
2. 两个官方 SDK Server 作为独立进程启动；物流与售后实际经 HTTP/MCP 查询，工具轨迹和审计标明各自 Server；断开一个不拖垮另一个。
3. 预登记一个新 MCP 查询工具权限，Server 添加工具并只重启该 Server；客服下一轮发现并调用。未授权新工具、冒充只读的写工具、名称冲突均拒绝。
4. 「帮我建个工单」缺描述先追问；补齐后出现预览；确认落一条 tickets 并返回工单号，重复确认/刷新/恢复不重复建单。
5. 相同预览选择取消，tickets 增量0，对应 create_ticket 审计为权限拒绝。无明确建单请求、伪造确认、错会话、过期回合也不能写入。
6. 只读工具确定性注入超时：实际重试次数、总耗时、超时审计齐全；写工具注入超时：重试0，提示结果待核实，晚完成/恢复不得重复写。
7. 参数类型/必填/范围错误、查询空结果、MCP isError 和审计写失败分别验证。错误以 ToolMessage 回灌，业务空结果不重试，审计失败不阻断正常结果。
8. 原投诉按钮、订单选择器、退款流程、引用和第七章上下文回归；动态 Schema 增长仍遵守预算，工具结果配对完整。

真实验收使用合成订单/工单描述，只在取得第八章外发授权后发送到配置的云模型。交付提供两个 MCP 启动命令、VS Code 聊天启动命令、迁移/测试/评估命令、报告位置及 dev-notes/ch08.md。

## 11. 设计自审

- 技术选型与传输不变；SDK 版本差异在实施锁版本时解决，不混抄 v1/v2 API。
- 新工具即插即用受用户确认的逐工具本地授权约束；不再以 Server 的自述决定读写权限。
- 原投诉交互保留与“统一入口”通过内部重接实现，不新增第二次确认。
- LangGraph 中断重放、写入未知结果和重复确认不依赖审计表防重复，避免“审计可失败”与幂等相冲突。
- 审计 DDL 无等待状态，所以只记录最终调用结局；等待状态存在 checkpoint/既有 offer，不擅改 SQL。
- 前端流程例外与浏览器验收分开：不套前端设计/TDD/review，但仍验证用户可见功能。
- 本文没有授权执行迁移或安装依赖；书面 spec 经用户审阅后进入 writing-plans。
