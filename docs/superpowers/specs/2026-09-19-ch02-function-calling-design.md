# Ch02 设计：客服聊天内的单次 Function Calling

## 目标与边界

用户在现有 `/` 聊天页提问，`POST /api/chat` 在同一 SSE 流内完成模型选工具、执行一次、回灌结果、逐 token 输出最终回答。保留 `/api/extract`。本章不做 Agent Loop、向量检索或 RAG。订单、商品、物流只由工具内部随机生成演示数据，不接外部系统，也不建对应表；界面与回答须标明这些是演示数据。

## 已确认决策

- 数据库以用户提供的四张 MySQL DDL 为唯一结构依据；实施时把原文保存为 `db/ch02.sql` 并执行，SQLAlchemy 映射与其列、类型、默认值、索引、外键一致。MySQL 由 Docker Compose 启动，InnoDB、utf8mb4。
- 服务端生成 `conversations.id` 数字 ID，首次 SSE 发 `conversation` 帧；浏览器在 localStorage 保存该 ID 与匿名 `user_id`，后续请求提交两者。旧 UUID `session_id` 不再作为数据库主键。服务端核对会话所有者，不允许凭别人的 ID 续聊。
- `user_id` 为演示用匿名标识，不等于生产鉴权。浏览器清理 localStorage 后开启新会话。
- 一轮最多实际执行一个工具。模型未选工具时直接回答；模型同时申请多个时只执行第一个，对其余申请写入“本轮未执行” ToolMessage，以满足协议，再收敛。第二次模型调用禁用工具。

## 架构与数据流

`api/chat.py` 只管 HTTP、SSE 与输入校验；`services/chat.py` 编排两次模型调用；`services/tool_executor.py` 管注册、校验、超时、重试、错误结果；`tools/business.py` 放五个 `@tool`；`db/models.py` 对应四表；`db/session.py` 管连接；`repositories/` 管会话、消息、FAQ、工单读写。现有 `core/llm.py` 工厂与配置继续使用。历史从数据库读取，仍按 token 预算裁剪；裁剪要保留 assistant tool_calls 与对应 tool 消息成组，避免发出孤立工具消息。

1. 收到 `{user_id, conversation_id?, message}`；首次创建会话，持久化 user 消息，再发 `conversation` 帧。
2. 历史加新消息送绑定五工具的模型，`tool_choice=auto`；选工具阶段非流式。未选工具则跳到最终流式回答。
3. 若有申请，先持久化带原始 `tool_calls` JSON 的 assistant 消息，发 `tool_status` 帧；执行第一个合法申请，保存带 `tool_call_id` 的 tool 消息。其余申请逐一写“本轮未执行” tool 消息。工具结果作为 `ToolMessage` 按 ID 回灌。
4. 最终模型调用不绑定工具，`astream` 的文本片段逐个推送现有 `delta` 帧；结束后持久化 assistant 回答并发 `[DONE]`。即使没有工具调用，也记录 user 和最终 assistant。
5. 上游或数据库出错后推 `event: error` 并关流；不会把异常堆栈发给用户。若断流发生在工具已落库之后，保留已写消息，不盲目重新执行副作用工具。

## 数据模型与种子

- `conversations`: `BIGINT UNSIGNED` 自增主键；`user_id VARCHAR(64)`；中文 MySQL ENUM 状态；`created_at` 与 `updated_at` 由数据库默认值维护；`idx_user_id`。
- `messages`: `BIGINT UNSIGNED` 自增主键与外键 `conversation_id`；role 为 user/assistant/tool；`content TEXT NULL`、`tool_calls JSON NULL`、`tool_call_id VARCHAR(64) NULL`；创建时间、索引与外键按 DDL。
- `faq`: 问题 `VARCHAR(512)`、答案 `TEXT`、分类 `VARCHAR(64)`，时间戳与分类索引按 DDL。
- `tickets`: `ticket_no VARCHAR(32)` 业务主键；会话外键；描述 `TEXT`；类型 售后/投诉/咨询，状态 待处理/已处理；创建时间和索引按 DDL。
- 初始化 FAQ 至少有“退货政策是什么”及答案，不含“邮费”及其同义描述。种子脚本重复执行不产生重复行。会话与工单不预填，以免污染演示。

## 五个工具

`query_order(order_id: str)`、`query_product(product_name: str)`、`query_logistics(order_id: str)` 均返回结构化、可序列化的随机演示数据及 `mock: true`。同一轮的随机值生成一次，工具重试不改变已成功结果。订单 1001 的物流问题优先使用 `query_logistics`，返回的订单号与入参一致。

`query_faq(keyword: str)` 只做 `faq.question LIKE %keyword%` 的字面关键词匹配；通配符须转义，参数绑定，不做分词、同义词扩展或答案列搜索。命中可返回有限条问题/答案/分类；未命中明确返回空结果，模型不得补造政策。“退货政策”应命中；“邮费”应不命中，此漏召回记录为 Ch03 升级依据。

提示词要求：凡询问平台规则、政策、费用，先调用 `query_faq`，再按查询结果回答；找不到时明确说 FAQ 未查到。这使“邮费是多少”样例可检验关键词漏召回，而不是模型凭常识直接回答。

`create_ticket(conversation_id: int, description: str, ticket_type: Literal['售后','投诉','咨询'])` 写 `tickets`，生成不超过 32 字符的唯一工单号；会话 ID 从服务端上下文注入并验证，不信任模型自报。成功后把会话状态改为“已转人工”；工单插入与状态更新同一事务。重试仅用于确认未发生写入的失败；不对提交结果不确定的写入自动重试，避免重复工单。

五工具用 LangChain `@tool` 和明确的 Pydantic 参数 Schema；执行器只接受注册表中的名称，先校验参数，再以每次 5 秒超时执行。只对无副作用的三个 mock 查询与 FAQ 的暂时性失败最多重试 2 次，短退避；校验错误、未知工具、FAQ 空结果不重试。所有失败转为带错误码的 ToolMessage 让模型解释，日志保留诊断细节但不泄露密钥。

## SSE 与页面

沿用 `data: {"delta":"..."}`、`data: [DONE]` 和 `event: error`。新增 `event: conversation`，数据含数字 `conversation_id`；`event: tool_status`，数据含工具名、阶段（执行中/成功/失败）和简短展示文案。聊天页在当前助手气泡显示本轮工具名徽章，状态阶段展示等待提示；工具结果不直接当作客服回答显示。网页改造实施时按用户指定的 Vibe Coding 方式直接进行。

## 验证与风险

- 代码走 TDD：DDL 与 ORM 列对齐、会话归属、消息顺序与工具 ID、参数校验、超时重试、工单事务、SSE 帧及流式完成。
- 种子数据与提示词走标注样例评估：订单 1001 物流、退货政策命中、邮费未命中；真实模型与浏览器手工验收工具徽章和逐 token 输出。
- DeepSeek 当前文档支持 `tool_choice=auto`，但模型能力随具体 `CHAT_MODEL` 变化；实施前用现有上游做真实工具调用冒烟。若固定技术栈下无法调用，停下询问，不换模型或框架。
- SSE 中数据库会话不跨上游流式等待长时间持有连接；每次读写用短生命周期 Session。超时处理与断开后的写入边界在测试中覆盖。

## 文档依据（Context7，2026-09-19）

- [LangChain 工具绑定与 ToolMessage](https://docs.langchain.com/oss/python/langchain/models)
- [FastAPI 流式响应的依赖生命周期](https://fastapi.tiangolo.com/advanced/advanced-dependencies)
- [SQLAlchemy MySQL 类型及 ON UPDATE 注意事项](https://docs.sqlalchemy.org/en/20/dialects/mysql.html)
- [DeepSeek tool_choice 约束](https://api-docs.deepseek.com/api/create-chat-completion)
