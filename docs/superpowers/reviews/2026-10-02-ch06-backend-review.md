# Ch06后端最终审查与修复

审查范围：f04ca62..560d2fa，独立只读审查；前端按用户Vibe例外排除。原结论With fixes，Critical为0、Important为4。

## 重要问题：一次修复已验证

| 问题 | 修复 | RED→GREEN证据 |
|---|---|---|
| 合法长原问被512字符检索上限拒绝 | 第一条忠实规范原问沿4096契约，补充查询512；模型无需复述长原问 | test_long_original_question_is_valid_retrieval_input |
| 退款评估与实际Agent不一致、无效引用可通过 | 运行生产build_agent_graph，核实际refund_offer_defaults、actions与有效引用 | test_assessment_uses_production_agent_and_rejects_invalid_reference；另验证合法引用但未建议入口不通过 |
| 扩写内容未验收也能总通过、快照订单错配 | 匹配样例订单快照；内容人工复核按实际查询指纹绑定，未核则pending_review | test_unreviewed_expansion_never_passes_quality_gate；复核与输出绑定测试 |
| 变更后旧质量结果可能继续复用 | 签名覆盖生产workflow/knowledge、工具、退款校验、评估器、数据、相关非敏感配置与库版本 | test_signature_changes_with_code_or_threshold |

首轮4项RED失败，修复及补充覆盖目标17项GREEN。一次修复后完整回归：622 passed、1第三方warning（43.30秒）。未派二次审查，按Native流程由覆盖测试和全仓回归验证。

## 延后的小问题

1. workflow日志缺规范问题/来源状态/置信度/模型层级与升级等完整诊断字段。
2. 其他意图与知识证据不足共用兜底文案。
3. 三个旧测试文件末尾空行导致原审查范围diff --check格式提示。

## 审查未评价的边界与执行裁决

- 前端：用户明确Vibe例外，交互已用真实Edge/HTTP替身验证；不加正式前端审查。
- 真实模型质量：新数据外发授权待回复，必须保持未验收；替身和清单不代表准确率。
- 真实模型端到端浏览器：当前只有HTTP替身与真实本地零模型问候，不声称云端端到端通过。
- 多进程互斥：沿批准的单应用实例设计，已保留8001并停止核对后的旧8000重复服务。
- 身份与订单归属：沿批准的匿名演示身份及固定服务端候选，未扩展到真实电商认证。
- 正式语义充分性闸：沿设计复用现有分数闸，正式版留后续章。
- Native工作区与PowerShell：沿已批准的直接实施方式和Windows记录流程。

## 尚未完成的交付门槛

真实意图JSON合规率/准确率、多轮切换与指代、扩写内容复核、实际Agent退款资格样例及真实模型网页验收仍待授权后运行。此审查修复记录不等于Ch06整章finish。
