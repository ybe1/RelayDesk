# Task 13 — 聊天引用、反馈与 RAG 评估页面

## 状态

前端实现和轻量 smoke 已完成。真实浏览器验收未完成：当前环境没有浏览器自动化工具或可执行浏览器，因此没有截图，也不把静态检查当作交互验收。

## 交付

- 聊天流按当前 assistant bubble 保存 citation map。仅该 map 内的 `[n]` 渲染为按钮；弹窗先显示长度受限的 SSE 标签，再请求 `/api/kb/chunks/{chunk_id}` 并显示完整章节路径、问题与回答。弹窗处理加载、404、网络失败、关闭按钮、Escape、背景关闭；所有内容经 `textContent` 创建。
- 仅非空且收到流结束标记的最终回答出现一次性 👍/👎。点击后选中态点亮、两按钮锁定、显示“已反馈”，不发后端请求。
- `/rag-eval` 显示报告状态和元信息、最佳 MRR / 对应 Recall@5 / 对应证据覆盖率 / 生成忠实率、策略及分桶图、检索与生成表、faith 台账筛选分页与处置、`expected_seen_count` 乐观锁和 409 刷新提示。固定评估作业有启动确认、服务端 token、停止、状态轮询和日志尾。缺失或 partial 报告时台账与作业区仍可用。
- 聊天、知识库、后台与评估页共享导航；后台卡片显示报告状态和最佳 MRR。

## 验证

- `node --check`：`admin.js`、`admin-page.js`、`rageval.js` 与聊天页内嵌脚本均通过。
- `.venv/Scripts/python.exe -m pytest tests/test_static.py -q`：1 passed，3 条既有弃用警告；覆盖 `/rag-eval` 和 `/static/rageval.js`。
- `git diff --check`：通过。未启动 `eval-rag`、未发云调用、未修改 live JSON/TXT 报告。
- 浏览器交互场景（引用、反馈、missing/complete 报告、台账处置）及截图：未执行。环境无浏览器工具，Node 中也无 Playwright/Puppeteer/jsdom 包，浏览器可执行文件不在 PATH。

## 范围与风险

本任务只改静态页面、共享导航、静态路由 smoke 和本文档。已存在的 live 报告 JSON/TXT 工作区改动未纳入提交。浏览器交互仍是 Task 14 的验收门槛；台账依赖真实数据库可用，完整报告依赖真实生成/裁判完成。
