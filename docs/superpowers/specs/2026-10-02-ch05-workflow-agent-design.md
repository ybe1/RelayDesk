# Ch05：确定性 Workflow 与主力 Agent

日期：2026-10-02。状态：设计自审完成，待用户审阅；尚未实施。

## 目标与已确认决策

在现有 SSE 聊天入口内，用 LangGraph 的确定性 Workflow 组织业务路径，核心业务出口进入可多步调用工具的 Agent。用户在聊天页自然提问即可触发流程。先交付不依赖 Agent 框架的裸循环演示，再以同组样例验证 LangGraph 重构。

用户已确认：官方 AsyncSqliteSaver 保存独立 SQLite 文件；常见问候、感谢、告别先用规则返回固定话术、零模型调用；外层 Workflow 与内层 Agent 子图；每轮最多5次工具执行、6次模型调用、12,000累计token、120秒；建工单与转人工独立；修改已有 create_ticket 的副作用，使其不再将 conversations.status 改为已转人工。

本章定位是生产架构的编排骨架与单实例本机持久化。SQLite checkpointer 适用于本地部署，不承诺多实例共享或生产集群能力。不新增 PostgreSQL，不实现正式指代消解、上下文策略升级、MCP、飞轮入库。

## 工作流和固定路由

入口先验证 user_id 与 conversation_id 归属，再进入指代消解透传节点。明确闲聊只对完整短句进行白名单匹配；包含订单号、业务词或混合诉求的消息不按局部关键词截断为闲聊。未命中规则时，意图节点通过简单 prompt 输出经 Schema 校验的 JSON，intent 仅允许七类。

| 意图 | 出口 | 必须经过的节点 |
| --- | --- | --- |
| 商品咨询、退款退货 | 知识类 | retrieval → confidence_gate → Agent 或 fallback |
| 物流、订单、售后 | 业务数据类 | Agent，无预检索闸 |
| 投诉 | 投诉出口 | 固定安抚 → 两个独立建议选项 |
| 闲聊 | 闲聊出口 | 固定话术 |

七类归属写死在代码里，模型只能输出标签，不能输出目标节点或覆盖分流规则。分类失败返回可解释兜底并记录错误，不随机分流。所有完成出口汇入日志节点。投诉话术明确表示可以选择转人工或创建工单，用户继续发消息时按新一轮正常处理。

## 检索与置信度闸

知识类复用 search_knowledge_ch04 的 dense/BM25/RRF/rerank 及首尾证据顺序，先检索政策再进入 Agent。闸使用非空命中、有限的最高 rerank_score 和显式可配置阈值；阈值初值通过小型标注集校准并记录，不沿用默认0作为有效置信度判断。保留第四章相关规则和数据源，正式语义置信度检查留给后续章节。

证据弱时不调用 Agent、不流出模型答案：返回固定证据不足话术，在 low_confidence_questions 保存用户原话、会话、source=retrieval_low_conf 与原因。检索服务故障和证据不足分别记录；故障不能伪称无相关知识。入池失败可见于日志，不能报告已入池。

通过闸后将有序且有大小上限的证据放入 Agent 上下文，先推送 citations 再推答案 delta。退款退货 Agent 可继续查询订单。业务数据类不走检索闸；必要时 Agent 仍可调用现有 query_faq。多次 FAQ 调用的引用重新分配本轮唯一编号，避免不同来源共用[n]。

## 裸循环与 Agent 子图

裸循环演示直接调用 LLM：有 tool_calls 则经现有 ToolExecutor 执行并回灌 ToolMessage，无工具调用则收敛。保留调用申请、结果与停止原因；只使用现有工具。演示通过独立命令运行，不成为第二个生产聊天入口。

Agent 子图包含 decision、tools、final_answer 和 stop 节点。decision 使用受 Schema 约束的结构化决策，选择继续调用工具、追问或进入回答，并可给出 handoff/create_ticket 建议选项；工具申请必须保持标准 AIMessage.tool_calls / ToolMessage.tool_call_id 对应关系。执行工具来自第2章 query_order/query_product/query_logistics/query_faq；create_ticket 不提供给自主执行节点。

工具串行执行、结果回灌后重新决策。final_answer 禁止工具调用，按当前证据及工具结果真正逐token生成，不能把完整生成文本伪拆成token；决策和中间内容不流给用户。缺必要订单号等信息直接追问并结束本轮，用户下一轮补充后沿用既有历史，不升级上下文策略。mock结果必须明确标示演示性质。最终生成遵守第四章引用、拒答和禁止承诺规则。

停止条件：无待执行工具、追问、明确停止、达到任一预算、取消或错误。多工具申请按实际执行次数计数，未执行申请必须有明确错误ToolMessage，不能留下悬空调用。模型不得通过建议动作绕过工具上限。

## 预算与超时

配置项：工具执行5、LLM调用6、累计输入输出12,000 token、整轮120秒。模型调用包含意图识别、检索内Query理解、Agent决策、最终回答；可见重试也计入调用数。嵌入与rerank单独记录次数和耗时，不计作聊天模型调用。预算对象传入检索相关模型调用边界，不能只统计Agent。

每次请求前预留输入与输出上限，设置实际模型max_tokens，确认足够预算才发起。完成后按usage_metadata结算；上游缺usage时按保守估算记账并标记estimated。达到上限禁止新调用，并使用固定停止话术，不额外花一次模型调用总结。当前一轮不自动重试LLM；工具重试沿用ToolExecutor的读工具策略，受总deadline控制并记录attempt。保留既有按完整用户轮次裁剪的上下文方式。

## State、检查点与业务账本

父图 State 贯穿：可信用户/会话ID、turn_id、用户原话、透传问题、intent、route、当前证据/引用、消息、预算用量、工具步数、action suggestions、最终文本、停止原因和节点轨迹。每轮重置证据、引用、预算、建议与停止原因，防止上一轮状态污染新问题。消息采用LangGraph消息reducer和稳定消息ID，重复更新不叠加同一条。

thread_id 由已核验的会话ID生成，不接受客户端自行指定检查点线程。应用生命周期建立/关闭 AsyncSqliteSaver，一次编译父图，Agent子图继承父图checkpointer。文件位于可配置本机持久目录，运行数据库及WAL/SHM文件不提交Git。

LangGraph保存执行State，MySQL conversations/messages 保存用户可追溯聊天及全部工具申请与结果。无分布式事务：跨库不声称原子提交或恰好一次。消息以稳定ID及检查点进度设计可重试落账，必要的最小迁移在plan明确。对同一会话实行互斥，冲突请求明确拒绝；本章单实例，不将进程锁宣称为分布式锁。

断连时不继续后台自由运行Agent；终止未完成工具申请，保存取消原因。服务重启后已完成会话状态可恢复，新用户消息开始新轮。旧轮中断的只读工具可按显式恢复路径重试，但不自动重放写操作或已经展示的部分回答；记录不完整轮次，避免悬空ToolMessage。默认不承诺断线SSE逐token续传。

## 独立用户动作

后端通过 actions SSE 事件返回可选项，每项携带服务端生成的offer_id、类型、建议描述/工单类型。选项在完成回答后展示；固定投诉同时给两个，Agent可给一个或两个。

转人工仅前端模拟：点后该按钮锁定，展示“已转接人工客服”，追加“您好，我是客服小猫，请问有什么可以帮您的”。不创建tickets、不修改服务端会话处理状态，也不触发模型。其他按钮独立可用。

建工单打开确认框，显示可编辑description与售后/投诉/咨询类型；取消不发请求。确认后调用独立动作API，核验用户/会话、建议选项和参数，再复用现有create_ticket工具写tickets。仅在事务提交后返回工单号及成功状态，UI只锁定该按钮。已有工具名称、入参与返回结构保留，删除“创建工单顺带转人工”的副作用。

动作API使用offer_id保证幂等，同一确认重复提交返回相同工单，不创建多张。最小新增动作记录表存offer、会话、类型、参数与ticket_no；唯一约束和事务保证刷新/重试后仍有效，不仅依赖前端禁用。offer与会话绑定，同一offer不能更改已执行参数。actions不是graph interrupt，用户忽略按钮不会阻塞正常聊天。

## SSE、日志与错误

兼容既有conversation、tool_status、citations、delta、error、done；新增workflow_status及actions，旧前端可以忽略。仅final_answer产生模型delta，固定出口直接发送固定话术。Node轨迹明确区分workflow.retrieval、confidence_gate与Agent内部tool。

日志按conversation_id/turn_id标记节点、意图、route、工具名/调用ID、耗时、模型次数、token/estimated、停止原因。记录tool结果在MySQL，运行日志不输出密钥或隐藏推理。异常返回用户可读消息并留下诊断原因；流已输出不能再改变HTTP状态，使用error帧结束。

## 验证与交付

代码任务走TDD；纯Prompt/阈值用标注样例评估，不用断言提示词字符串冒充行为验证。先小型集覆盖七类、混合闲聊、缺订单号、证据弱、两步工具、提示注入；裸循环和图使用同一脚本化模型/工具验证一致性。云端只跑章验收必要样例，不重跑第四章300题。

必须验证：政策路径强制经过retrieval/gate；订单1001物流由Agent调工具；投诉两按钮独立，点转人工无ticket，确认工单才写一行且重试不重复；忽略选项正常聊天；明确闲聊零LLM；先查订单再查物流至少两次工具节点；四预算停止；超时/取消无悬空调用；重启可读检查点；越权线程拒绝；引用可点击且沿用一次性反馈。

完成后交付裸循环与图演示命令、服务启动命令、测试和标注样例结果、迁移说明、dev-notes/ch05.md路径。每个阶段当场追记用户原话、关键产出、拒绝/纠偏和翻车/返工，独立code review后再finish。

## 文档核对与自审

已通过Context7核对LangGraph官方持久化、子图继承checkpointer、thread_id和custom stream文档。来源：https://docs.langchain.com/oss/python/langgraph/checkpointers 、https://docs.langchain.com/oss/python/langgraph/use-subgraphs 、https://docs.langchain.com/oss/python/langgraph/streaming 。当前项目尚未显式依赖langgraph/checkpoint-sqlite；实施前需核对锁定版本和AsyncSqliteSaver接口、LangChain/FastAPI/SQLAlchemy相关用法，再安装依赖。

自审：无待定实现占位；路由与七类意图一一对应；SQLite本机范围与“生产骨架”区分；两动作与现有工具副作用已按用户确认解耦；预算涵盖外层调用；前置闸位于流式回答之前；幂等迁移限定在必要记录，不新增业务工具。
