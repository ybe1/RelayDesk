# Ch03 设计：MySQL 权威原文 + Milvus dense 知识检索

> 2026-09-22 对照课程《3.2 实战演练： 动手跑通 RAG 链路》修订；用户已要求修改后执行，并纠正为继续使用 Docker Milvus standalone。

## 目标与已确认边界

升级 `query_faq(keyword: str)` 为 BGE-M3 dense 单路检索，保持实际契约 `{"keyword": string, "matches": [{"question": string, "answer": string, "category": string}]}`。课程旧示例的 hits/async/Lite 不覆盖当前代码和用户选择。

- 硅基流动 `BAAI/bge-m3`，1024 维，独立 EMBED_BASE_URL/EMBED_MODEL/EMBED_API_KEY；不输出密钥。
- Milvus Docker standalone，集合 knowledge，COSINE；不自行改用 Lite、其他向量库或模型。
- MySQL 是原文权威源；Milvus 存 id/vector/question/answer/category 在线读模型，在线检索不回 MySQL。
- 历史对话挖知识使用可重跑一次性 CLI；不安装 APScheduler、不启动 cron。
- 只做规范化文本 QA 去重，不做语义合并，不调用模型判重。
- 沿用同步 SQLAlchemy SessionLocal；阻塞工作用同步 def 路由或显式线程卸载，不改数据库架构，不阻塞聊天 SSE。
- Ch02 会话、消息、FAQ、工单数据不改不清；FAQ 仅只读复制为新知识。无关键词 fallback、混合检索、重排或 Agent Loop。
- 后端 TDD；纯 Prompt/数据用标注集评估；前端 Vibe Coding，浏览器验证。

## 两表、切块与幂等导入

计划附录用户 DDL 原样保存为 db/ch03.sql，字段、类型、默认值、索引、自引用外键和中文注释完整保留。只新增 knowledge_chunks 与 qa_extraction_staging，不增加第三张表或改动原表。

四份演示 Markdown 放 data/kb/：退货政策、运费说明、商品 FAQ、售后手册。正文明确标注演示规则。统一 parser 供 CLI、页面预览、录入调用：按标题层级切分，超长段递归切到约 800 字符，约 100 字符重叠按完整句界裁齐；超长单句/表格行保留并提示。大表格按完整行分块，每块复制表头及分隔行。同一章节多块正文全部保留。

政策/手册 questions 为本节标题、category 为上级路径；FAQ/挖取 QA 填真实问法。关键条款用 `> [!IMPORTANT]` 标注。section_path 包含来源及标题路径，超 DDL 限制报错。前后指针仅连接同一来源本次新建块，不改其他来源已存在块。每条新知识只嵌入 `category + "\n" + questions + "\n" + answer`，其他字段只存元数据。在线只嵌入用户查询。

规范化为 Unicode NFKC、小写、去 Unicode 标点和空白；指纹是规范化 `(questions, answer)` 二元组无歧义 JSON 编码的 SHA-256。文档、手工录入、staging 跨批及已有知识共享规则。相同标题不同正文、相同问题不同答案均保留；不同文字的同义问法不合并。预览只读查重，入库在单写者锁内再次与现有库及本次输入判重；来源新增块与指针一次事务提交。重录零新增，文档新内容仅追加未知指纹；本章不做编辑、删除或源文件增量同步。

## 对话挖知识

每批最多 30 个会话，只给聊天模型用户文本及最终 assistant 回答，排除 tool 消息、工具申请、模拟订单和物流、明显个人信息。结构化结果包含原输入会话 ID，拒绝未知 ID；候选先提交 staging(status=extracted)，全部批次结束后对所有 extracted 整体文本去重。第一条新 QA 标 kept 并写 pending chunk，重复标 discarded；知识插入和状态变更同一事务。重跑不会重复或丢失知识。无可复用 QA 是合法空结果。挖取 Prompt 用标注样例检验，不用固定字符串单测替代质量评估。

## 双写、恢复与在线检索

先提交 MySQL pending/vector_id=NULL；批量读 pending 后结束行事务，嵌入验证数量、索引顺序、1024 维及有限数值；Milvus 按 MySQL 主键 upsert `{id,vector,question:questions,answer,category}`；成功后新事务回填 vector_id=str(id)、done。云端或向量库失败保持 pending，upsert 后回填前中断重跑覆盖同一 ID。

Milvus 显式 INT64 主键、auto_id=False、FLOAT_VECTOR(1024)、COSINE；question/answer/category 为显式 VARCHAR，最大长度依锁定版本核实并提前按 UTF-8 字节校验，不静默截断。已有集合核验字段、维度和索引，不兼容则报错，不隐式 drop。模型变化需显式重建派生索引。

所有写任务共享连接级 MySQL advisory lock；锁连接持有至任务结束，finally 释放，进程退出自动断开；远程调用期间不持有行事务。API 和 CLI 同锁，内部步骤用已持锁执行上下文，避免重复抢锁。

搜索返回 id/score/question/answer/category 供自测；query_faq 去掉 id/score，外形保持 keyword/matches。top_k=5、min_score=0.55 是待标注集检验的初值。无命中 matches=[]；失败报工具错误，不降级 SQL LIKE。独立检索器在 MySQL 不可用时仍可读 Milvus。

## API、六区知识库与后台入口

/kb 六区：①手工 Markdown 录入；②建库材料；③对话挖知识；④向量化与双写；⑤检索自测；⑥最近入库。顶部库存展示 total/pending/done/key_clause_count、staging 三态、Milvus count。读数失败为未知，consistent=null；数量一致仅代表计数一致，不等于完整性证明。

| 接口 | 契约 |
|---|---|
| GET /api/kb/overview | 库存、暂存三态、依赖状态和 true/false/null 计数一致性 |
| GET /api/kb/documents | 固定白名单材料、类型、块数、触发特性 |
| POST /api/kb/preview | markdown/content_type/source_name 或白名单 document 二选一；只读 |
| POST /api/kb/ingest | 手工 Markdown，可选 vectorize；返回 inserted/skipped/indexed/pending |
| GET /api/kb/staging | status/limit/offset 分页，limit 上限 100 |
| POST /api/kb/vectorize | 启动同一 kb-vectorize 后台作业，立即返回 job ID |
| POST /api/kb/search | query/top_k，top_k 1..20，仅 dense；返回 hits 带分数 |
| GET /api/kb/recent | 分页最近知识及来源、状态 |
| GET /api/jobs；POST /api/jobs/{name} | 固定白名单与启动；未知 404，正在运行 409 |
| GET /api/jobs/{name}；POST /api/jobs/{name}/stop | 状态、有长度上限日志尾、停止进程树 |
| GET /api/admin/overview | 聚合摘要，单依赖失败不导致整页失败 |

预览显示节路径、问法、正文、字数、关键条款、表格块、重复标志，顶部显示标题切/重叠/表格复制是否触发。空正文、非法类型、路径穿越、两种输入同时提供为 400。录入后向量化失败返回 502，仍含已提交 inserted/skipped/pending 及补跑提示，不回滚原文；前端必须展示这份错误正文。

作业白名单 kb-preview/kb-build/kb-mine/kb-vectorize/seed-conv/kb-reset/eval-retrieval/eval-mining 对应固定 Python 参数数组，shell=False，无任意命令/路径输入。日志位于 log/acceptance/（忽略提交），输出计数/ID和脱敏错误。Windows 隐藏窗口并按受控 PID 停整个进程树，POSIX 用进程组；不能照搬仅 POSIX 的 killpg。终态刷新保留日志尾。关闭应用清理其启动的任务。kb-mine 显示耗时提示；提供 seed-conv（幂等追加带专用标识的合成会话，不改已有会话）与 kb-reset（heavy，页面二次确认且服务端显式 confirmation 校验）。reset 只清 Ch03 两表和 knowledge 集合，不触碰 Ch02 四表或其他集合；执行真实清库前必须得到用户针对本次清库的确认。恢复演示用新增块故障注入，不自行清库。管理面是本机单用户入口，本章不加登录/审计。

/admin 聚合卡和共享导航连接现有聊天 / 与知识库 /kb；聊天页加后台管理入口，不链接不存在模块。沿用 app/static，不新建前端框架。

## 实施门槛和验收

按已安装版本先查 Context7 的 SQLAlchemy/LangChain/PyMilvus/FastAPI 接口并记录依据；工具缺失明确报告，不假装已查。真实阶段检查 Docker daemon、standalone、MySQL、聊天/嵌入凭据存在性。固定选型不可用时继续独立代码测试，单列阻塞项，不能改 Lite 或宣称真实联调成功。

后端先红后绿覆盖切块、指纹、防同节多块误删、事务回滚、staging 原子晋升、双写恢复、在线零 SQL、工具契约、预览只读、部分失败原文保留、白名单与停止。标注检索集覆盖运费换说法、退货、商品、售后、无关问题；抽取集含可复用、重复、冲突、私有事实、无 QA。

浏览器实走材料预览→建库→补向量→自测“邮费是多少”→聊天工具徽章及正确回答；手工贴两节含关键条款和表格的 Markdown→预览→录入→重录零新增→检索。真实故障为 upsert 后回填前进程退出，再重跑确认相同 ID 唯一、pending 清零；页面展示日志/账本，后台按 ID 检查，不只比数量。截图保存在 dev-notes/generated-images/。每阶段即时写 dev-notes/ch03.md 四项。代码测试与真实验收分开报告。
