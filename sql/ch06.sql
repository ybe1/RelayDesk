SET NAMES utf8mb4;
ALTER TABLE chat_action_offers MODIFY COLUMN action ENUM('handoff','ticket','refund') NOT NULL;
CREATE TABLE refund_requests (
 id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
 application_no VARCHAR(32) NOT NULL,
 offer_id VARCHAR(64) NOT NULL,
 conversation_id BIGINT UNSIGNED NOT NULL,
 order_id VARCHAR(64) NOT NULL,
 reason ENUM('不喜欢/不需要','商品质量问题','商品与描述不符','错发/漏发','其他') NOT NULL,
 status ENUM('submitted') NOT NULL DEFAULT 'submitted',
 is_mock TINYINT(1) NOT NULL DEFAULT 1,
 snapshot JSON NOT NULL,
 created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
 PRIMARY KEY(id),
 UNIQUE KEY uk_refund_application_no(application_no),
 UNIQUE KEY uk_refund_offer(offer_id),
 KEY idx_refund_conversation(conversation_id),
 KEY idx_refund_created_at(created_at),
 CONSTRAINT fk_refund_offer FOREIGN KEY(offer_id) REFERENCES chat_action_offers(offer_id),
 CONSTRAINT fk_refund_conversation FOREIGN KEY(conversation_id) REFERENCES conversations(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='演示退款申请，不执行资金动作';
