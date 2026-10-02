# Ch06：正式分流与订单退款演示

## 启动

在项目根目录的VS Code终端运行：

```powershell
docker compose up -d
uv sync
uv run python -m scripts.migrate_ch06 --check
# 仅首次待迁移时运行，已有结构不覆盖：
uv run python -m scripts.migrate_ch06 --apply
uv run uvicorn app.main:app --host 127.0.0.1 --port 8001
```

打开 http://127.0.0.1:8001/ 。使用一个应用进程；不要同时运行8000和8001并共享同一SQLite检查点。保持Clash可用，云端API沿用现有配置。

`CHAT_MODEL`默认主模型；`INTENT_SMALL_MODEL`留空不降级。配置小模型时使用同一聊天上游，低于`INTENT_CONFIDENCE_THRESHOLD=0.75`或非法JSON升级主模型一次。MySQL仍存业务数据；官方AsyncSqliteSaver只存会话图状态。

## 聊天验收

1. “退货政策是什么”：日志应有resolve、intent、retrieval、confidence_gate、decision、final_answer、log；不查订单、不扩写、不出现选单。
2. “订单1001到哪了”→“它能退吗”→“这单物流呢”：第二轮补订单指代，经过slot、fetch_order、expand、policy_retrieval；第三轮切回物流。多个历史订单不明确时选单，不猜最近数字。
3. 新会话说“我要退款”：聊天流显示三张演示订单卡片，点选后续跑；刷新页面可恢复尚未选择的卡片。继续发新问题会废止旧卡片。
4. 未知商品完好等条件时，应追问或needs_review，不凭收货天数直接批准。退款原因不在聊天里追问，只在退款申请表单的固定类目里选择。
5. 只有退款意图、eligible判断、匹配订单与有效政策证据，才显示“提交退款申请”；确认后保存MySQL演示申请，重复同原因返回同号，不执行真实退款。
6. “我要投诉”仍给独立“转人工”“建工单”；“你好”固定回复，零模型调用。怪问题归其他，固定兜底。

订单、商品、物流均为一致性演示数据，无真实电商API。refund_requests只保存演示申请，status=submitted、is_mock=1。

## 验证命令与证据

```powershell
uv run pytest -q
uv run python -m scripts.eval_ch06
# 获得本章演示评估数据发送授权后才运行：
uv run python -m scripts.eval_ch06 --live --output data/ch06/reports/eval-live-final.json
# 只重跑失败题，已成功题跳过：
uv run python -m scripts.eval_ch06 --live --retry-failed --output data/ch06/reports/eval-live-final.json
# 扩写内容核对完成后，导入人工复核；复用已保存输出，不重新发模型请求：
uv run python -m scripts.eval_ch06 --live --output data/ch06/reports/eval-live-final.json --expansion-review data/ch06/reports/expansion-review-final.json
```

本地eval命令只验证40条意图标注、7组×4轮对话、8条扩写与3条资格判断样例的完整性；`quality_status=not_run`不代表模型准确率通过。

真实评估逐题写data/ch06/reports/eval-live.json，网络/额度失败立即停止后续请求；同输入同模型重跑跳过已有结果。数据、被测生产实现、评估器、库版本或非敏感判断配置变更必须用新的`--output`保留旧证据。可用`--section intent/dialogues/expansion/assessment`分段执行；最终all检查整套门槛。意图JSON合规率100%、准确率至少95%，指定多轮指代/切换全部正确才达标。退款样例运行实际Agent图及refund offer资格校验，引用必须来自提供的政策证据。

扩写数量/格式通过不等于内容通过。核心扩写初次输出标为pending_review；逐条对照expected_facets核对侧重点和事实保持后，写人工复核文件，例如：

```json
{"X01":{"query_signature":"复制报告中的64位query_signature","facets_ok":true,"facts_preserved":true,"reason":"核对退货条件与运费两个侧重点，未添加订单事实"}}
```

每个需要扩写的样例均需复核；结果按实际queries的指纹绑定，输出变化后旧审批不复用。未复核不允许quality_status=passed。通用FAQ原样不扩写，无需扩写内容复核。规范原问第一条最多4096字符，补充检索问每条512字符；不截断用户的合法详细说明。累计调用/token/活动时间上限保持不变。

前端证据：data/ch06/reports/browser-local.json、browser-local.png，真实Edge、HTTP演示替身，7项交互通过；不能替代真实模型端到端验收。

最终真实验收（2026-10-02）：eval-live-final.json为passed，40/40意图、JSON合规100%、28/28对话用户轮、8/8扩写、3/3生产Agent资格样例。X01首个最终输出遗漏运费侧重点，人工拒绝后仅重试该题，重试通过；eval-live-final-attempt1.json保留首次结果，不将重试结果冒充首次稳定性。早期eval-live/v2/v3/v4报告保留返工证据。

真实网页证据：browser-live.json、browser-live.png，实际DeepSeek/SiliconFlow/MySQL/Milvus，无HTTP替身。验证无订单号申请弹卡片、刷新恢复、点选续跑、未知商品完好条件明确无法确认且不发退款按钮、补充条件后表单提交成功。记录的退款申请is_mock=true，不执行真实退款。最终回归625 passed、1第三方DeprecationWarning。

范围与剩余事项见ch06-finish.md；这是本章演示架构验收，不代表真实电商身份鉴权、多进程部署或全量真实场景已验收。

过程记录：dev-notes/ch06.md。设计与计划：docs/superpowers/specs/2026-10-02-ch06-routing-design.md、docs/superpowers/plans/2026-10-02-ch06-routing.md。
