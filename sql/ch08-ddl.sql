-- ch08: 工具调用审计。不挂外键，审计失败不能阻断工具执行。
SET NAMES utf8mb4;
CREATE TABLE tool_audit_logs (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '审计主键',
  conversation_id BIGINT UNSIGNED NULL COMMENT '所属会话,无会话上下文的调用为 NULL',
  tool_call_id VARCHAR(64) NULL COMMENT '模型申请单 id,可对回 messages 流水',
  tool_name VARCHAR(128) NOT NULL COMMENT '工具名',
  tool_source ENUM('builtin','mcp') NOT NULL COMMENT '工具来源:内置 / MCP 接入',
  mcp_server VARCHAR(64) NULL COMMENT '来源 MCP Server 名,内置工具为 NULL',
  arguments JSON NULL COMMENT '调用参数',
  result_summary TEXT NULL COMMENT '返回结果,过长截断存摘要',
  status ENUM('成功','失败','超时','校验拦下','权限拒绝') NOT NULL COMMENT '本次调用结局',
  error_message VARCHAR(512) NULL COMMENT '失败 / 拦下时的原因说明',
  retry_count TINYINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '实际重试次数,写操作默认不重试恒为 0',
  duration_ms INT UNSIGNED NULL COMMENT '耗时毫秒',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '调用时间',
  PRIMARY KEY (id),
  KEY idx_conversation_id (conversation_id),
  KEY idx_tool_name (tool_name),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='工具调用审计留痕';
