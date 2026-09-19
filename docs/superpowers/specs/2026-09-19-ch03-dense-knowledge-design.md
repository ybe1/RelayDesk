# Ch03 设计：MySQL 权威原文 + Milvus dense 知识检索

## 目标与范围

把现有 `query_faq(keyword: str)` 的内部实现改为 BGE-M3 dense 向量检索。工具的名称、入参 Schema 和返回 JSON 外形 `{"keyword": string, "matches": [{"question": string, "answer": string, "category": string}]}`保持不变；聊天页和 Ch02 单次工具调用链继续使用。新增离线 Markdown 建库、历史对话问答抽取、全局同义去重和可恢复双写。在线不做关键词召回、混合检索、重排或 Agent Loop。

## 已确认决策

- 嵌入模型固定为 `BAAI/bge-m3`，调用硅基流动云端嵌入 API；不在本机加载模型。聊天模型仍用现有 `CHAT_*` 配置。嵌入服务使用独立 `EMBED_BASE_URL`、`EMBED_MODEL`、`EMBED_API_KEY`，密钥只放 `.env`。
- 首批知识由仓库新增的演示 Markdown 文档构成，包含退货政策、运费说明、商品 FAQ 和售后手册；同时将 Ch02 `faq` 现有记录迁入知识源，以保持已有问答可检索。演示内容在页面和文档中标明为演示规则，不能冒称企业正式政策。
- 对话抽取按批次写 `qa_extraction_staging`，所有批次完成后跨批去重；同义且答案一致的问法合并为 `questions` 的多行文本，答案冲突者不合并。
- Milvus 使用 Docker 的 standalone 模式，集合名 `knowledge`，只建 `id INT64` 主键和 `vector FLOAT_VECTOR(1024)`；余下字段仅存 MySQL。索引使用 COSINE，在线 Top-K 默认 5，不加在线 reranker。
- 离线任务有一次性入口和独立的每日 02:00（Asia/Shanghai）调度入口。调度不绑在 FastAPI 进程，避免多 worker 重复运行。一次性入口用于建库与故障恢复演示。

## 数据库结构

实施时把用户给出的 **两张表** DDL 原样保存为 `db/ch03.sql`（包含开头的 `SET NAMES utf8mb4`），在已有 MySQL 数据库上按顺序执行。`knowledge_chunks` 是原文权威源；`qa_extraction_staging` 是对话抽取的中转暂存表。SQLAlchemy 映射必须与 DDL 的 `BIGINT UNSIGNED`、NULL、中文/英文 ENUM、时间默认值、索引和两个自引用外键及 `ON DELETE SET NULL` 完全一致。原 Ch02 四表与 FAQ 数据保留。

`knowledge_chunks.id` 与 Milvus `knowledge.id` 数值相同；`vector_id` 存该 ID 的十进制字符串。`vectorize_status` 仅有 `pending` / `done`。`prev_chunk_id`、`next_chunk_id` 只连接同一来源文档内相邻 chunk，插入行后在同一 MySQL 事务内回填指针。`qa_extraction_staging.status` 按 `extracted` → `kept` 或 `discarded` 转换。

## 离线：文档解析与切分

读取 `knowledge/` 下的演示 Markdown。文件路径和标题栈决定来源及章节路径；首层标题是文档名。先按 Markdown `#`～`######` 结构切分，再将超过约 800 字符的段落递归按段落、换行、句末标点切分，目标重叠约 100 字符，重叠边界回退到最近的 `。！？.!?`。不得留下半句；没有可用句界的单句或单行表格允许超过目标长度并记日志。大表格按完整数据行分块，每块复制表头和对齐行，不拆单行。

商品 FAQ 以文档中的真实问题填 `questions`；政策/手册没有天然问题时，以当前节标题填 `questions`，上级标题路径填 `category`。`answer` 为该块正文。`section_path` 为文件路径加完整标题路径，供同一来源的重跑检查和溯源；`content_type` 取 `faq` / `policy` / `manual` / `extracted_qa`；Markdown `> [!IMPORTANT]` 标记的关键条款映射到 `is_key_clause`。除 `category`、`questions`、`answer` 外的元数据都不进入嵌入文本。嵌入文本固定为 `category + "\n" + questions + "\n" + answer`，保留换行。

文档导入在单个受控任务中按来源比较完整、有序的 chunk 序列；相同则跳过，首次导入则整份文档在一个 MySQL 事务内写入，重复运行不会插入相同 chunk。来源文档内容变更时明确报“需重建该来源”，本章不做增量修订。任务串行运行，避免没有源唯一索引时的并发重复插入。Ch02 FAQ 迁移采用同样的来源检查。若导入 MySQL 的事务中断，该事务回滚；后续向量化只读取已提交行。

## 离线：历史对话挖知识

调度任务从 `conversations` / `messages` 读取完整会话，按几十个会话为一批，保留用户与最终 assistant 文本；剔除 tool 消息、工具调用申请、订单号/物流等一次性事实和明显个人信息。对每批用现有聊天上游 LLM 的结构化输出抽取可复用 QA，先写暂存表，记录 `batch_no` 和会话 `source_ref`。每批可单独重试；已暂存的来源不重复插入相同 QA。

全部待处理批次暂存后执行全局去重：先做规范化文本完全一致去重，再以 BGE-M3 问题向量筛出同义候选，由 LLM 确认问法意图和答案事实是否一致；只有两者均一致才合并。代表行标 `kept`，其余标 `discarded`，把保留组的所有真实问法以换行加入同一 `knowledge_chunks.questions`。答案冲突或判断失败的候选不自动合并；判断失败保留 `extracted` 供重跑。入库前再次与已有知识对照，避免重复。抽取和去重 Prompt 用标注对话样例评估，不用镜像实现的单测。

## 离线：MySQL / Milvus 双写与恢复

1. 先提交 `knowledge_chunks` 行，`vectorize_status='pending'`，`vector_id=NULL`。
2. 批量读取 `pending`，向硅基流动请求 BGE-M3 dense 向量，校验每条向量维度为 1024、数量与输入对应。
3. 用 MySQL chunk ID 作为 Milvus 主键执行 `upsert`。成功后在 MySQL 回填 `vector_id=str(id)`、状态 `done`。不把答案或元数据复制到 Milvus。
4. 若在 Milvus upsert 后、MySQL 更新前中断，重跑会对同一 ID upsert，再完成回填，不产生第二个向量实体。若嵌入或 Milvus 失败，行维持 `pending`，下一次任务拾取。验收会故意制造该中断点并确认恢复。

向量化任务只处理 `pending`；建库命令可以重入，先导入再排空 `pending`。单任务锁防止调度和手工建库同时操作。同一 embedding 模型和维度必须用于离线与在线；若模型配置变化，需全量重建，不能混用不同向量空间。

## 在线 `query_faq`

使用现有工具参数 `keyword` 作为用户查询文本，经同一 BGE-M3 API 向量化；Milvus `knowledge` COSINE Top-K 检索只返回 ID 与分数，再按排名批量从 MySQL 读取状态为 `done` 的权威原文。返回的每项只含 `question=questions`、`answer`、`category`，外层仍为 `keyword` 与 `matches`。无命中返回空列表。`RAG_TOP_K=5`、`RAG_MIN_SCORE=0.55` 为初始配置，在标注集上检验后可调整；不把相似度作为第二路排序或重排。嵌入/Milvus 故障按工具错误处理，不退回 SQL LIKE。

更新工具描述与客服 Prompt 中“字面关键词”措辞，但保留模型调用工具的方式。Ch02 的 `query_faq` 入参/返回契约测试改为验证新实现，聊天 SSE 和工具徽章无需改版。

## 验证、资源与风险

- TDD：标题路径、句界重叠、表格表头、FAQ 映射、ORM/DDL 对齐、暂存状态、同义去重保护、双写中断恢复、Milvus ID 回表、工具契约与错误路径。
- 标注评估：至少包含“邮费是多少”→“运费说明”、退货政策、商品 FAQ、无关问题；对话抽取样例覆盖同义问法、答案冲突、订单私有事实。真实模型与 Milvus 联调确认结果可用。
- 目前 `.env` 没有硅基流动密钥；实施可先完成离线代码和测试，真实云端验收须由用户在 `.env` 填 `EMBED_API_KEY`。不打印或提交密钥。
- 本机约 16 GB 总内存、检查时约 2.8 GB 空闲；Milvus 官方对 standalone 的最低要求为 8 GB RAM。实施先做 Docker 与模型 API 冒烟，若固定选型在本机无法运行，按用户要求停下询问，不自行改用 Milvus Lite 或其他模型。

## Context7 与官方依据（2026-09-19）

- [SiliconFlow BGE-M3 嵌入接口](https://docs.siliconflow.cn/docs/api/embeddings-post)
- [BGE-M3 维度与序列长度](https://github.com/FlagOpen/FlagEmbedding/blob/master/research/BGE_M3/README.md)
- [PyMilvus create / upsert / search](https://github.com/milvus-io/pymilvus/blob/master/_autodocs/api-reference/milvus-client.md)
- [LangChain Markdown 标题切分](https://reference.langchain.com/python/langchain-text-splitters/markdown/MarkdownHeaderTextSplitter)
- [LangChain 递归切分](https://docs.langchain.com/oss/python/langchain/knowledge-base)
- [Milvus standalone 资源要求](https://milvus.io/docs/prerequisite-docker.md)
