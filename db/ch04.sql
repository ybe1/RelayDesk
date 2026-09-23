SET NAMES utf8mb4;

CREATE TABLE low_confidence_questions (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  conversation_id BIGINT UNSIGNED NULL                    COMMENT '来源会话',
  raw_question    TEXT            NOT NULL                COMMENT '用户原话,带情绪口语',
  source          ENUM('retrieval_low_conf','self_check','user_feedback') NOT NULL COMMENT '入池入口:检索证据低 / 生成自评不足 / 用户反馈未解决',
  reason          TEXT            NULL                    COMMENT '判不能的原因,留作复盘',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '入池时间',
  PRIMARY KEY (id),
  KEY idx_source (source),
  KEY idx_created_at (created_at),
  CONSTRAINT fk_lcq_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='低置信度问题池';

CREATE TABLE faith_cases (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  eval_id       VARCHAR(16)     NOT NULL                COMMENT '评估集题号,如 A43;一题一行',
  bucket        VARCHAR(24)     NOT NULL                COMMENT '题目所属桶:A_policy / B_model / C_colloquial / E_multi',
  query         VARCHAR(512)    NOT NULL                COMMENT '用户问题原文',
  strategy      VARCHAR(24)     NOT NULL DEFAULT 'hybrid_rerank' COMMENT '产出这条答案的检索策略',
  answer        TEXT            NOT NULL                COMMENT '被判编造的那版生成答案原文',
  reason        TEXT            NOT NULL                COMMENT '裁判给的理由:编在哪一句',
  citations     JSON            NULL                    COMMENT '这一轮喂给模型的 Top-K 证据全集快照:[{n,chunk_id,section_path,question,answer}];答案里的角标 [n] 就是这份列表的序号,答案通常只引用其中两三条;老数据没记为 NULL',
  judge_model   VARCHAR(64)     NULL                    COMMENT '判这条的裁判模型',
  status        ENUM('未解决','已解决','无需解决') NOT NULL DEFAULT '未解决' COMMENT '处置状态,人工点按钮改',
  seen_count    INT UNSIGNED    NOT NULL DEFAULT 1      COMMENT '被判编造的累计次数(跨轮)',
  first_seen_at DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '第一次被判编造的时间',
  last_seen_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '最近一次被判编造的时间',
  resolution    VARCHAR(300)    NULL                    COMMENT '处置说明:标已解决要写清怎么解决的,标无需解决要写清为什么不用改;退回未解决时清空。空着的处置在台账上等于没有交代',
  resolved_at   DATETIME        NULL                    COMMENT '最近一次被标为已解决/无需解决的时间;复发后仍保留,用来标「复发」',
  PRIMARY KEY (id),
  UNIQUE KEY uk_eval_id (eval_id),
  KEY idx_status (status),
  KEY idx_last_seen_at (last_seen_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ch04 忠实度编造个案台账';
