# Ch09 对话方案复核（2026-10-05）

状态：设计审查记录，不是已批准的 spec 或实施计划。尚未修改业务代码、安装依赖、建表或调用付费模型。

## 结论和已确认约束

整体方向符合需求，但不能直接照之前的口头方案实施。需补齐下列正确性与恢复性约束后形成正式 spec。

用户已选择方案 A（复用 FastAPI/MySQL/Milvus/JobRunner），审核知识固定分类“用户问题补充”，每周一 02:00 全量评估，并认可反馈/审核/评估后台设计。本机独立 Compose 部署 Langfuse 是此前根据“继续”采用的工作假设。F1 最优只是此前建议，用户尚未选择该校准目标。

## 1. 审核通过必须写入当前聊天使用的集合

证据：scripts/kb.py 的 vectorize 使用 app/knowledge/vector_store.py 的 MilvusStore，集合常量为 knowledge；当前配置默认 MILVUS_COLLECTION=knowledge_ch04，在线使用 HybridMilvusStore。app/knowledge/ch04_indexer.py 提供全量混合索引重建，并明确不修改 Ch03 向量化确认状态。

修正：复用 Ch03 的“SQL 原文先提交、向量按 SQL 主键 upsert、成功再确认、失败可补齐”机制；索引写入必须适配当前混合集合的 dense/text/BM25 字段。不要直接调用旧版 kb-vectorize 就宣称上线成功。以稳定 review_queue 来源标识确保重复批准只产生一份知识；人工批准、SQL 入库和可检索状态要区分。Milvus 故障时显示待索引并支持恢复，不能把“审核通过”等同于“已可检索”。保留已确认的固定分类，但验证其与已有 category 过滤兼容性；不能默认新分类一定会被过滤检索命中。

## 2. 自评入口不能只依赖旧 query_faq

证据：app/tools/business.py 的 query_faq 执行 evidence_checker 并写 source=self_check；app/workflow/graph.py 的强制检索只调用 gate_result；app/workflow/agent.py 可直接生成回答，且已有证据时可复用 FAQ 结果。

修正：明确覆盖“强制检索直接回答”“售后扩写检索”“Agent 主动调用 FAQ”三条路径。知识回答流式输出前检查最终送入模型的证据，useful=false 才以 self_check 入池并拒答，避免重复检查/重复落池。数值证据闸仍在检索之后、Agent 之前。查询超时、模型/API 故障不能被当成知识缺口自动进入待审队列。

## 3. 快照必须按轮次保存，并在拦截前捕获

证据：graph.py 与 after_sales.py 在 gate 不通过时清空 evidence；CitationRegistry 只保存答案摘录且不带精排分数；agent.py 最终消息目前只挂 citations。直接用引用列表无法满足原文+分数快照要求。

修正：检索完成立即捕获按相关性排序的 Top-K 原文、chunk_id、章节路径、原始分数/精排分数和 rank，之后才筛选生成证据。把快照与对应 turn_id/最终消息绑定，保存在持久化 checkpoint 的当轮记录中；👎 从用户拥有的助手消息 ID 反查该轮原话与快照。不能取会话最新 evidence，也不能现场重检索冒充历史快照。明确区分没有检索（NULL）、检索过但零命中、历史快照不可恢复；后者要明确标示缺失，不能伪装成没有检索。覆盖刷新、重启和旧消息反馈验收。

## 4. 去重、反馈和审核要有可重跑语义

现有 DDL 没有独立任务状态或反馈唯一键，不能默认一次点击/一次进程就保证只写一次。

修正：标准化/语义查重调用在数据库事务外进行，提交阶段锁定并复查原话仍未归并；增加 occurrence_count 与回填 matched_review_id 在同一事务内完成。候选待审问题须分批完整遍历，不能任意截取前几条。并发归并需串行化提交并复查候选变化；反馈需定义基于助手消息 ID 的持久幂等处理；批准和知识创建也要防并发重复。严格沿用用户 DDL；若现有字段与锁无法满足所选幂等方案，再明确提出最小表结构调整，不能暗加字段。

标准化还需利用本轮已消解问题解决“这个怎么办”等指代，再脱敏传给模型；只发送脱离上下文的原话可能生成错误缺口。示例答案只能是待核准草稿，不得自动进入知识库；脱敏不能误删商品型号等检索关键内容。

## 5. 校准方法需避免数据泄漏和错误解释

Top1 相关分、有效证据数量与分差都可用，但大分差不必然代表可靠，小分差也可能是多条互补证据。相同 FAQ 的多个片段不能机械算成多份独立支持。

修正：先定义证据足够的标签与有效证据规则，再搜索参数；仅命中章节或字符串不能被直接当成语义充分的金标准。使用 Ch04 标注的答案要点/相关章节，必要时人工核对边界样例。同一知识点的同义问法按组划分校准/验证数据，留出不参与调参的测试样例，单列库外误放行与多事实问题表现。覆盖当前单查询与售后多查询检索路径。记录数据集、语料、模型、Top-K、规则版本；证据分数不能冒充已校准概率。

F1 不是需求强制目标，也尚未得到用户确认。客服场景应同时展示错误放行与误拒答的取舍，不能只凭 F1 最高宣布上线。

## 6. JobRunner 不是调度器或持久队列

证据：app/core/jobs.py 仅维护本进程的子进程表和 threading.Lock，没有周期调度或跨重启任务记录。

修正：在方案 A 内补最小调度与恢复机制。周一 02:00 使用 Asia/Shanghai；多进程、手动启动、自动启动之间防重复运行；每轮独立报告与稳定运行标识，结果写 eval_runs 保证幂等。补跑策略需限量且可见，不能重启一次就重复付费跑全量或补跑无限积压。首次上线不能把历史所有周次当欠账。应用关闭时无法执行任务，应在运行说明中写清。

只把完整成功的评估轮次纳入正常趋势；失败/中断有独立状态与日志，不能将排除失败样本后的高分冒充全量结果。记录各指标真实分母、四策略结果、数据集/知识库/模型版本；两轮必须实际独立执行，不能重复导入同一旧报告。飞轮补知识后，原本库外问题若变成可回答，应版本化更新标注并提示趋势不可直接比较。

## 7. Langfuse 回调、版本与 token 统计需落实到可验证契约

Context7 核对：LangGraph 用 RunnableConfig.callbacks 或编译后的配置绑定回调，不能假设 compile(callbacks=...) 存在。原始编译图的 checkpoint/resume/state 能力必须保留。官方 Langfuse 文档提示自托管 v3 使用 Metrics API v1，v2 指标 API 的兼容性取决于服务端版本。

修正：固定兼容的 Langfuse 服务端/SDK 版本；在应用建图阶段统一安装回调，分配每轮独立 trace/conversation session 并处理 interrupt/resume 的关联。HTTP embedding/rerank、自写执行引擎的校验拒绝/权限拒绝与重试需补观察记录。意图在识别后写入对应 trace，成本聚合需要实际验证受支持的元数据查询方式。只累加模型叶子调用 usage，避免父节点累计用量重复计费；意图判定前的调用也归入最终意图，定时评估与线上客服用量分开。没有价格配置时显示 token 数和未知金额，不编造费用。

Langfuse 不可用不能拖垮聊天；异步上报有限超时，关闭时 flush。自托管仅保证 trace 存本机，业务模型/embedding 调用仍按配置访问云端。此轮审查没有发送数据给这些模型服务。

## 8. 流程修正

此前过多零散确认，没有及时形成可审阅文档；还反复尝试全文输出单行的大体积 Ch04 JSON 报告，造成无必要输出。后续只做结构化摘要读取。此记录作为阶段性审查落地，正式 spec 将集中写清上述契约，再进行一次文档评审；不继续把每个实现细节都变成用户选择题。

## 资料与验证范围

- 本地只读核对：workflow/graph.py、after_sales.py、agent.py、retrieval.py；tools/business.py；core/jobs.py；knowledge/indexer.py、ch04_indexer.py、vector_store.py；scripts/kb.py；反馈页面和消息/问题池模型。
- Context7：/langchain-ai/langgraph；/langfuse/langfuse-docs，重点核对回调配置、checkpoint、Metrics API 的版本兼容性。
- 官方参考：https://github.com/langchain-ai/langgraph/blob/main/pregel/main.py
- 官方参考：https://langfuse.com/docs/metrics/features/metrics-api
- 本次为设计审查，没有运行业务测试，不声称上述修正已经实现。
