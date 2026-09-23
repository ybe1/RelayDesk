# Ch04 设计：混合检索、重排与评估体系

状态：已定稿（2026-09-23）。

前置：Ch03 已运行于 Docker Milvus Standalone 3.0.2，MySQL 是知识原文权威源，线上 `query_faq` 使用 BGE-M3 dense 检索。本章在该基础上升级，不重复迁移 Lite。

## 1. 目标与边界

本章实现 Milvus 原生 BM25、dense + BM25 RRF 混合召回、硅基流动 `BAAI/bge-reranker-v2-m3` 重排、元数据过滤、单轮 Query 理解、引用与双闸拒答、低置信度问题池、五桶评估集、四策略报告、RAG 评估页、编造个案台账与裁判回归。聊天页增加可点引用和一次性满意度反馈。

不做多轮指代消解、多轮改写、关键词数据库 fallback、混合检索之外的检索通道，也不引入 LangGraph 或 Langfuse。

固定选型：Milvus Standalone 原生 BM25 + `hybrid_search` + RRF；BGE-M3 dense；SiliconFlow `BAAI/bge-reranker-v2-m3`。任何红线冒烟失败都停止并向用户报告，不自行更换模型或检索技术。

## 2. 集合升级与安全切换

不直接删除当前 `knowledge`。新建 `knowledge_ch04`，从 MySQL `knowledge_chunks` 全量重放并验证后，通过配置 `MILVUS_COLLECTION=knowledge_ch04` 切换在线流量。旧集合保留用于回退，删除需另行授权。

`knowledge_ch04` 字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | INT64 主键，auto_id=false | 等于 `knowledge_chunks.id` |
| `dense` | FLOAT_VECTOR(1024) | BGE-M3，COSINE |
| `text` | VARCHAR | `category + questions + answer`，中文 analyzer |
| `sparse` | SPARSE_FLOAT_VECTOR | BM25 Function 输出 |
| `question` / `answer` | VARCHAR | 检索和引用原文 |
| `category` | VARCHAR | 过滤字段 |
| `section_path` | VARCHAR | 引用溯源 |
| `content_type` | VARCHAR | faq/policy/manual/spec 等 |
| `is_key_clause` | BOOL | 关键条款标量 |

`text` 配置 `enable_analyzer=true` 和内置 chinese analyzer。`Function(FunctionType.BM25)` 从 `text` 生成 `sparse`。dense 使用 AUTOINDEX/COSINE，sparse 使用 SPARSE_INVERTED_INDEX/BM25。

切换前必须验证 MySQL done 数、Milvus count、ID 集合、dense、BM25、hybrid、filter、引用字段和 rerank 冒烟。失败时继续使用旧集合。

## 3. 在线检索管线

线上继续使用现有 `/api/chat`、一次模型选工具、最多执行一次工具、再流式收敛的结构。订单、商品、物流和工单工具不变，只升级 `query_faq`。

1. Query 理解把当前用户原话转换为 `normalized_query`、BM25 `search_query` 和可选 `category`。不读取历史对话做指代消解。
2. dense 使用 `normalized_query` 的 BGE-M3 向量召回 Top-50。
3. BM25 使用只在检索侧扩展同义词的 `search_query` 召回 Top-50。
4. 元数据过滤以参数化的受控字段生成 Milvus filter，不接受任意表达式。
5. `hybrid_search([dense_request, bm25_request], RRFRanker())` 融合两路。
6. SiliconFlow `/v1/rerank` 使用 `BAAI/bge-reranker-v2-m3` 精排并截取 Top-10。
7. 检索机械闸判断无命中或最高 rerank 分低于阈值。
8. 生成前结构化自评返回扁平的 `useful: bool` 与 `reason: str`。
9. 证据充分时按“第 1 名在首、第 2 名在尾、其余居中”排列，生成带编号证据。

`search_knowledge(query, strategy, top_k, category, ...)` 支持 `vector`、`bm25`、`hybrid`、`hybrid_rerank`。线上默认最后一种，评估调用同一入口。

## 4. 重排客户端

重排手写 HTTP 客户端，不使用 OpenAI SDK。配置为 `RERANK_BASE_URL`、`RERANK_API_KEY`、`RERANK_MODEL`，默认 base URL 为 `https://api.siliconflow.cn/v1`，模型为 `BAAI/bge-reranker-v2-m3`。请求为 query + documents + top_n，响应按 `results[].index` 和 `relevance_score` 恢复原候选。限制批量、超时，对 429/503/504 有限退避；响应索引、数量、分数和越界情况严格校验。

## 5. 拒答、低置信度池和引用

机械闸失败时明确拒答，并写 `low_confidence_questions(source='retrieval_low_conf')`。自评 `useful=false` 时写 `source='self_check'`。真实聊天写 conversation_id；评估或无会话入口可写 NULL。同一问题每次发生保留一行，本章不归并。

`query_faq` 保持模型可见输入 `keyword`，返回升级为 `keyword / sufficient / reason / evidence / citations`。citation 含 `n / chunk_id / section_path / question / answer / content_type`。工具结果本身已经作为 tool message 持久化，因此引用快照随该消息保存，不修改 `messages` DDL。

System Prompt 要求所有业务事实引用 `[n]`，不得使用不存在的编号，不得把常识当规则；证据冲突要说明；禁止承诺退款到账、物流到达、赔偿和人工处理时限；不得把演示规则描述为法律保证。证据不足必须明确拒答。

后端在工具成功后从 tool result 提取 citations，发送 SSE `citations` 事件，再流式输出最终答案。

## 6. 数据库

严格采用用户提供的 `low_confidence_questions` 和 `faith_cases` DDL。ORM 保持 unsigned BIGINT、ENUM、索引、唯一键、默认值、外键和 nullability 一致。

`faith_cases` 按 eval_id 一题一行。再次判为编造时更新答案、理由、全部 Top-K citation 快照，`seen_count + 1`；已处置问题复发时退回未解决并清空 resolution，但保留 resolved_at 作为复发依据。标记已解决或无需解决必须填写 resolution；退回未解决清空 resolution。

## 7. 评估体系

最终评估集为五桶 300 题，每桶 60：

- A_policy：政策和流程；
- B_model：真实存在的具体型号和易混型号族，验证 BM25；
- C_colloquial：口语改写；
- D_out_of_scope：库外问题，应拒答；
- E_multi：需要多个章节共同回答。

可回答桶评 Recall@5、MRR、证据要点覆盖度；E 桶按证据组分别计分并平均，不用最后一块到位的名次。四种策略都生成答案并评答案要点覆盖度；线上策略另评 Faithfulness；D 桶评拒答率。

数据自检必须保证 ID/query 唯一、五桶数量一致、期望章节真实存在、expect_points 出现在目标原文、D 桶无 ground truth。纯 Prompt 和数据任务用标注集验证，不写镜像实现的单测。

`scripts/eval_ch04.py` 输出 `data/ch04/reports/rag_eval.json` 和 `rag_eval.txt`。上游生成或裁判失败时，检索指标仍落盘，generation 标为未完成。页面只读这份 JSON，不重新计算指标。

## 8. Faithfulness 与裁判回归

裁判 temperature=0。提示词列明允许的跨证据合并、不确定提醒、安全提示和常识注解，也明确把凭空数值、条件张冠李戴判为编造，并加入真实边界 few-shot。

判定编造时 upsert `faith_cases`，数据库失败只记录错误，不破坏本轮报告。本轮幻觉率只按本轮报告题号计算，台账累计另列。

`judge-check` 使用已人工处置且带 citation 快照的案例原样重放，不重新检索或生成。已解决映射为“应判编造”，无需解决映射为“应判忠实”，未解决跳过。裁判空响应重试一次，仍失败不计入一致率并返回非零。

## 9. 管理 API 和页面

新增 `/api/rag-eval/overview`，只读 `rag_eval.json`，返回 present、meta、retrieval、evidence_coverage、generation、best 和作业状态。产物缺失或损坏返回 present=false；generation 缺失不影响检索结果展示。

新增 faith_cases 列表、状态筛选、分页和处置 API。状态更新校验 resolution。`eval-rag` 通过现有白名单 JobRunner 执行固定 Python argv，不通过 shell，不接受命令片段。

`/rag-eval` 展示报告头、KPI、四策略分桶图、生成质量、完整数据、编造台账和重跑日志。页面数字全部来自同一报告产物。

## 10. 前端例外

聊天页和 `/rag-eval` 页面按用户要求使用 Vibe Coding，不套 brainstorming、TDD 或 code review。聊天回答中的 `[n]` 变为可点角标，弹层展示 section_path 与 chunk 原文。每条有效 assistant 回答显示一次性 👍/👎，点击后点亮、显示“已反馈”并锁定；本章不落库。

## 11. 测试与红线

确定性代码使用 TDD：schema、BM25/hybrid/filter、rerank 客户端、检索编排、首尾排序、引用传播、低置信度入池、faith_cases、API、报告读取。Prompt 和数据使用标注样例或评估集。

第一阶段先执行两个 fail-fast 冒烟：

1. 当前 Milvus 3.0.2 临时集合跑通 chinese analyzer、BM25、dense + BM25 RRF 和 filter；
2. SiliconFlow `/v1/rerank` 跑通固定模型并验证响应形状。

任一失败即在 `dev-notes/ch04.md` 记录红线并停下询问，不进入替代方案。

最终验收：四策略报告产生数字；具体型号由 BM25 命中；引用可回到原文；库外问题明确拒答并落低置信度池；点赞反馈交互锁定；完整回归通过。

## 12. 已确认决策

1. 实现课程文档完整范围，包括 300 题、评估页、faith_cases 和裁判回归。
2. Milvus 使用现有 Docker Standalone 3.0.2，不降级到文档旧示例。
3. 新建 `knowledge_ch04` 并行验证后切换，旧 `knowledge` 暂留回退。
4. 重排使用 SiliconFlow `BAAI/bge-reranker-v2-m3` `/v1/rerank`。
5. 保持单轮工具调用和现有聊天入口。
6. 前端走 Vibe Coding；其他确定性代码走 TDD，Prompt/数据走标注评估。
