# 第七章：当前会话上下文管理

## 在 VS Code 启动

打开仓库目录的终端，先打开 Docker Desktop。检查并应用两步结构扩展（重复执行不会改已有段）：

```powershell
uv run python -m scripts.migrate_ch07 --check
uv run python -m scripts.migrate_ch07 --apply
uv run python -m scripts.migrate_ch07 --check
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

自己在浏览器打开 `http://127.0.0.1:8000/`。终端保持运行，Ctrl+C 关闭。若端口占用，先关闭以前启动的服务。开发助手不代开服务。

侧栏显示该浏览器匿名用户的会话：新在前，首问预览，已摘要标记。新对话保留旧会话，点旧会话回载原文并继续。流式回复时禁用切换；列表失败不会阻止发送。反馈只在该浏览器保存，已提交工单和演示退款以服务端结果为准。

## 默认与演示配置

默认窗口1,000,000，历史额度由窗口扣固定开销/输出预留/当轮峰值计算，最多按40轮稳态预留；能放就不压。以下演示变量必须一起设置，旧 `TOKEN_BUDGET` / `AGENT_MAX_TOKENS` 不控制新历史窗口：

```powershell
$env:MODEL_CONTEXT_WINDOW='18000'
$env:MAX_OUTPUT_TOKENS='2000'
$env:MAX_USER_INPUT_TOKENS='2000'
$env:MAX_AGENT_STEPS='3'
$env:TOOL_RESULT_MAX_TOKENS='1200'
$env:RERANK_TOP_K='5'
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

初始历史5650、层13954、层21695。实际摘要总量增加后历史额度相应变小。恢复默认：关服务，删除上述六个临时环境变量或重新打开终端（检查 `.env` 中是否也设置了同名配置），再启动。启动测实际system+工具前缀，超1600或装不下一轮报明确容量错误。

## 看日志

UTF-8文件：`log/app.log`。完整上下文含该会话业务原话，文件不进Git。密钥不作为日志参数。

```powershell
Get-Content -Encoding utf8 log/app.log -Wait
Select-String -Path log/app.log -Pattern 'history_ctx|model_ctx|layer1 demote|summary trigger|summary start|summary done|summary failed|summary skip'
```

有 grep 的终端可用 `grep -E 'model_ctx|history_ctx|summary|layer1 demote' log/app.log`。

`history_ctx`每轮，包括闲聊；`model_ctx`主力Agent每次实际调用，包含最终messages和固定tools。`context_usage`将估计与上游实际usage分开。摘要在后台，只追加段：trigger→start→done/skip/failed，带边界、段号和耗时。单轮示例太短可能尚未压缩，应连续聊到层2实际用量超额。

## 验证命令

```powershell
uv run pytest -q
uv run python -m scripts.eval_ch07
uv run python -m scripts.eval_ch07 --local-workflow --profile default
uv run python -m scripts.eval_ch07 --local-workflow --profile demo
uv run --with playwright python -m scripts.verify_ch07_browser
```

本地工作流用模型替身，仍走实际LangGraph、MySQL与独立SQLite checkpoint；不证明云端模型质量。浏览器验收使用后台Edge和HTTP替身，无监听端口，不开用户网页。临时Playwright不加入项目运行依赖。

完成数据发送授权后，真实验收：

```powershell
uv run python -m scripts.eval_ch07 --live --profile default
uv run python -m scripts.eval_ch07 --live --profile demo
uv run python -m scripts.eval_ch07 --live --section summary
```

只发送 `data/ch07` 合成会话/订单/虚拟手机号样例到配置的DeepSeek，知识检索沿用SiliconFlow。每套22轮。独立评估checkpoint，不占8000/8001。报告逐轮落 `data/ch07/reports/`，附输入/实现/库版本签名，完成且签名不变不会重跑。失败会停，不在网络错误后继续烧额度；查看错误再处理。摘要评估保存10条真实输出和required/forbidden标注检查，语义仍需逐条人工复核，不能把JSON可解析当成事实正确。

默认验收：22轮收敛、无降级/摘要；演示验收：初始额度准确、完整级联、摘要追加不重写、回忆最早订单/核实诉求和只在早期出现的虚拟手机号。非阻塞由受控摘要模型的SSE时间线测试验证。测试/报告不得把本地替身结果标成真实模型通过。

## 原文与模型输入

MySQL保存用户/助手原文、工具申请、锚点和摘要段。新tool结果仅存完整checkpoint；旧tool行保留。层1原文不改，层2助手80字、用户原话不动、工具一行标识，最远历史按原批次追加摘要。没有跨会话画像或记忆。

只读接口：`GET /api/conversations?user_id=...&cursor=...&limit=100`；`GET /api/conversations/{id}/messages?user_id=...`。匿名user_id延用演示身份，不能当生产登录鉴权。

开发记录：`dev-notes/ch07.md`；设计/计划：`docs/superpowers/specs/2026-10-02-ch07-context-design.md`、`docs/superpowers/plans/2026-10-02-ch07-context.md`。

## 2026-10-04 真实验收结果

- 默认窗口22轮：`data/ch07/reports/eval-live-default-60c8abf19f7b.json`，无降级或摘要，早期事实回忆正确。该报告在摘要联系方式校验修复前生成，默认窗口没有调用摘要。
- 演示窗口22轮：`data/ch07/reports/eval-live-demo-c5301ced81ae.json`，新增5段不可变摘要，最后正确回忆订单、虚拟手机号和诉求。
- 10个真实摘要样例：`data/ch07/reports/eval-live-summary-30db9fd58931.json`，标注和语义复核通过；本地回归686 passed。
- 每套包含1轮证据不足的安全拒答，单列记录，不视为业务题答对。旧失败报告保留。
- 评估模型连接最多重试2次；`--direct`只给评估进程使用独立直连客户端，不关闭Clash，不改变生产服务。
- 摘要专项复测：`uv run python -m scripts.eval_ch07_summary_pipeline`（植入合成历史，真实后台摘要及真实回忆，不能当作22轮真实对话）。
