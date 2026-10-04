from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, JSON, String, Text, UniqueConstraint, text
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ToolAuditLog(Base):
    __tablename__='tool_audit_logs'
    __table_args__=(Index('idx_conversation_id','conversation_id'),Index('idx_tool_name','tool_name'),Index('idx_status','status'),
        {'mysql_engine':'InnoDB','mysql_charset':'utf8mb4','comment':'工具调用审计留痕'})
    id:Mapped[int]=mapped_column(mysql.BIGINT(unsigned=True),primary_key=True,autoincrement=True,comment='审计主键')
    conversation_id:Mapped[int|None]=mapped_column(mysql.BIGINT(unsigned=True),nullable=True,comment='所属会话,无会话上下文的调用为 NULL')
    tool_call_id:Mapped[str|None]=mapped_column(String(64),nullable=True,comment='模型申请单 id,可对回 messages 流水')
    tool_name:Mapped[str]=mapped_column(String(128),nullable=False,comment='工具名')
    tool_source:Mapped[str]=mapped_column(mysql.ENUM('builtin','mcp'),nullable=False,comment='工具来源:内置 / MCP 接入')
    mcp_server:Mapped[str|None]=mapped_column(String(64),nullable=True,comment='来源 MCP Server 名,内置工具为 NULL')
    arguments:Mapped[dict|None]=mapped_column(JSON,nullable=True,comment='调用参数')
    result_summary:Mapped[str|None]=mapped_column(Text,nullable=True,comment='返回结果,过长截断存摘要')
    status:Mapped[str]=mapped_column(mysql.ENUM('成功','失败','超时','校验拦下','权限拒绝'),nullable=False,comment='本次调用结局')
    error_message:Mapped[str|None]=mapped_column(String(512),nullable=True,comment='失败 / 拦下时的原因说明')
    retry_count:Mapped[int]=mapped_column(mysql.TINYINT(unsigned=True),nullable=False,server_default=text('0'),comment='实际重试次数,写操作默认不重试恒为 0')
    duration_ms:Mapped[int|None]=mapped_column(mysql.INTEGER(unsigned=True),nullable=True,comment='耗时毫秒')
    created_at:Mapped[datetime]=mapped_column(DateTime,nullable=False,server_default=text('CURRENT_TIMESTAMP'),comment='调用时间')


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("idx_user_id", "user_id"), {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})

    id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(mysql.ENUM("进行中", "已转人工", "已结束"), nullable=False, server_default=text("'进行中'"))
    summary: Mapped[str | None] = mapped_column(Text, nullable=True, comment='最近几段梗概拼成的投影,拼装时跟证据一起挂在用户那句之后')
    summary_upto_msg_id: Mapped[int | None] = mapped_column(mysql.BIGINT(unsigned=True),nullable=True,comment='摘要已覆盖到哪条消息,滑窗从其后接原文')
    layer1_from_msg_id: Mapped[int | None] = mapped_column(mysql.BIGINT(unsigned=True),nullable=True,comment='层1(原文)起点;此 id 之后原样,之前渲染成半压形态')
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))


class ConversationSummary(Base):
    __tablename__ = 'conversation_summaries'
    __table_args__ = (UniqueConstraint('conversation_id','seq',name='uk_conv_seq'),
        Index('idx_conv_upto','conversation_id','upto_msg_id'),
        {'mysql_engine':'InnoDB','mysql_charset':'utf8mb4','comment':'分段摘要,一段一行只追加'})
    id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True),primary_key=True,autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True),nullable=False)
    seq: Mapped[int] = mapped_column(mysql.INTEGER,nullable=False,comment='第几段,从 1 开始')
    from_msg_id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True),nullable=False,comment='这段覆盖的消息区间,闭区间')
    upto_msg_id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True),nullable=False)
    content: Mapped[str] = mapped_column(Text,nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime,nullable=False,server_default=text('CURRENT_TIMESTAMP'))


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (Index("idx_conversation_id", "conversation_id"), Index("uk_message_key", "message_key", unique=True), {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})

    id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), ForeignKey("conversations.id", name="fk_messages_conversation"), nullable=False)
    role: Mapped[str] = mapped_column(mysql.ENUM("user", "assistant", "tool"), nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))


class ChatActionOffer(Base):
    __tablename__ = "chat_action_offers"
    __table_args__ = (Index("uk_offer_turn_action", "conversation_id", "turn_id", "action", unique=True),
                      {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})
    offer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), ForeignKey("conversations.id"), nullable=False)
    turn_id: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(mysql.ENUM("handoff", "ticket", "refund"), nullable=False)
    defaults: Mapped[dict] = mapped_column(JSON, nullable=False)
    confirmed_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ticket_no: Mapped[str | None] = mapped_column(String(32), ForeignKey("tickets.ticket_no"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))


class RefundRequest(Base):
    __tablename__ = 'refund_requests'
    __table_args__ = (Index('uk_refund_application_no','application_no',unique=True),
        Index('uk_refund_offer','offer_id',unique=True), Index('idx_refund_conversation','conversation_id'),
        Index('idx_refund_created_at','created_at'),
        {'mysql_engine':'InnoDB','mysql_charset':'utf8mb4','comment':'演示退款申请，不执行资金动作'})
    id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True),primary_key=True,autoincrement=True)
    application_no: Mapped[str] = mapped_column(String(32),nullable=False)
    offer_id: Mapped[str] = mapped_column(String(64),ForeignKey('chat_action_offers.offer_id',name='fk_refund_offer'),nullable=False)
    conversation_id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True),ForeignKey('conversations.id',name='fk_refund_conversation'),nullable=False)
    order_id: Mapped[str] = mapped_column(String(64),nullable=False)
    reason: Mapped[str] = mapped_column(mysql.ENUM('不喜欢/不需要','商品质量问题','商品与描述不符','错发/漏发','其他'),nullable=False)
    status: Mapped[str] = mapped_column(mysql.ENUM('submitted'),nullable=False,server_default=text("'submitted'"))
    is_mock: Mapped[bool] = mapped_column(mysql.TINYINT(display_width=1),nullable=False,server_default=text('1'))
    snapshot: Mapped[dict] = mapped_column(JSON,nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime,nullable=False,server_default=text('CURRENT_TIMESTAMP'))


class FAQ(Base):
    __tablename__ = "faq"
    __table_args__ = (Index("idx_category", "category"), {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})

    id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (Index("idx_conversation_id", "conversation_id"), {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), ForeignKey("conversations.id", name="fk_tickets_conversation"), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_type: Mapped[str] = mapped_column(mysql.ENUM("售后", "投诉", "咨询"), nullable=False)
    status: Mapped[str] = mapped_column(mysql.ENUM("待处理", "已处理"), nullable=False, server_default=text("'待处理'"))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        Index("idx_category", "category"),
        Index("idx_vectorize_status", "vectorize_status"),
        {
            "mysql_engine": "InnoDB",
            "mysql_charset": "utf8mb4",
            "comment": "知识库 chunk 原文权威源",
        },
    )

    id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="chunk 主键,与 Milvus 集合主键对齐")
    category: Mapped[str] = mapped_column(String(255), nullable=False, comment="分类 / 上级标题路径,进向量化文本")
    questions: Mapped[str] = mapped_column(Text, nullable=False, comment="问法或本节标题,多个问法换行分隔,进向量化文本")
    answer: Mapped[str] = mapped_column(Text, nullable=False, comment="正文答案,进向量化文本")
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True, comment="章节路径,元数据,溯源用,不进向量")
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True, comment="内容类型:faq / policy / manual 等,元数据")
    is_key_clause: Mapped[int] = mapped_column(mysql.TINYINT(1), nullable=False, server_default=text("0"), comment="是否关键条款,0 否 1 是,元数据")
    prev_chunk_id: Mapped[int | None] = mapped_column(mysql.BIGINT(unsigned=True), ForeignKey("knowledge_chunks.id", name="fk_chunks_prev", ondelete="SET NULL"), nullable=True, comment="前一块指针,元数据")
    next_chunk_id: Mapped[int | None] = mapped_column(mysql.BIGINT(unsigned=True), ForeignKey("knowledge_chunks.id", name="fk_chunks_next", ondelete="SET NULL"), nullable=True, comment="后一块指针,元数据")
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="Milvus 集合 knowledge 里的主键,写入后回填")
    vectorize_status: Mapped[str] = mapped_column(mysql.ENUM("pending", "done"), nullable=False, server_default=text("'pending'"), comment="待向量化 / 已向量化,双写幂等靠它")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"), comment="更新时间")


class QAExtractionStaging(Base):
    __tablename__ = "qa_extraction_staging"
    __table_args__ = (
        Index("idx_batch_no", "batch_no"),
        Index("idx_status", "status"),
        {
            "mysql_engine": "InnoDB",
            "mysql_charset": "utf8mb4",
            "comment": "历史对话抽 QA 的离线中转暂存表:分批抽取、整体去重,保留项入 knowledge_chunks,建库完成可清空",
        },
    )

    id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="暂存行主键")
    batch_no: Mapped[str] = mapped_column(String(64), nullable=False, comment="抽取批次号,一批几十个会话跑一次,分批防串味、按批追溯")
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True, comment="来源会话 / 导出文件标识,溯源用,不入最终知识库")
    question: Mapped[str] = mapped_column(Text, nullable=False, comment="LLM 从会话抽出的用户问法")
    answer: Mapped[str] = mapped_column(Text, nullable=False, comment="LLM 从会话抽出的客服答案")
    status: Mapped[str] = mapped_column(mysql.ENUM("extracted", "kept", "discarded"), nullable=False, server_default=text("'extracted'"), comment="已抽出待去重 / 去重保留 / 去重丢弃")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment="抽取写入时间")


class LowConfidenceQuestion(Base):
    __tablename__ = "low_confidence_questions"
    __table_args__ = (
        Index("idx_source", "source"),
        Index("idx_created_at", "created_at"),
        {
            "mysql_engine": "InnoDB",
            "mysql_charset": "utf8mb4",
            "comment": "低置信度问题池",
        },
    )

    id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="主键")
    conversation_id: Mapped[int | None] = mapped_column(mysql.BIGINT(unsigned=True), ForeignKey("conversations.id", name="fk_lcq_conversation"), nullable=True, comment="来源会话")
    raw_question: Mapped[str] = mapped_column(Text, nullable=False, comment="用户原话,带情绪口语")
    source: Mapped[str] = mapped_column(mysql.ENUM("retrieval_low_conf", "self_check", "user_feedback"), nullable=False, comment="入池入口:检索证据低 / 生成自评不足 / 用户反馈未解决")
    reason: Mapped[str | None] = mapped_column(Text, nullable=True, comment="判不能的原因,留作复盘")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment="入池时间")


class FaithCase(Base):
    __tablename__ = "faith_cases"
    __table_args__ = (
        Index("uk_eval_id", "eval_id", unique=True),
        Index("idx_status", "status"),
        Index("idx_last_seen_at", "last_seen_at"),
        {
            "mysql_engine": "InnoDB",
            "mysql_charset": "utf8mb4",
            "comment": "ch04 忠实度编造个案台账",
        },
    )

    id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True, comment="主键")
    eval_id: Mapped[str] = mapped_column(String(16), nullable=False, comment="评估集题号,如 A43;一题一行")
    bucket: Mapped[str] = mapped_column(String(24), nullable=False, comment="题目所属桶:A_policy / B_model / C_colloquial / E_multi")
    query: Mapped[str] = mapped_column(String(512), nullable=False, comment="用户问题原文")
    strategy: Mapped[str] = mapped_column(String(24), nullable=False, server_default=text("'hybrid_rerank'"), comment="产出这条答案的检索策略")
    answer: Mapped[str] = mapped_column(Text, nullable=False, comment="被判编造的那版生成答案原文")
    reason: Mapped[str] = mapped_column(Text, nullable=False, comment="裁判给的理由:编在哪一句")
    citations: Mapped[list | None] = mapped_column(JSON(none_as_null=True), nullable=True, comment="这一轮喂给模型的 Top-K 证据全集快照:[{n,chunk_id,section_path,question,answer}];答案里的角标 [n] 就是这份列表的序号,答案通常只引用其中两三条;老数据没记为 NULL")
    judge_model: Mapped[str | None] = mapped_column(String(64), nullable=True, comment="判这条的裁判模型")
    status: Mapped[str] = mapped_column(mysql.ENUM("未解决", "已解决", "无需解决"), nullable=False, server_default=text("'未解决'"), comment="处置状态,人工点按钮改")
    seen_count: Mapped[int] = mapped_column(mysql.INTEGER(unsigned=True), nullable=False, server_default=text("1"), comment="被判编造的累计次数(跨轮)")
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment="第一次被判编造的时间")
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"), comment="最近一次被判编造的时间")
    resolution: Mapped[str | None] = mapped_column(String(300), nullable=True, comment="处置说明:标已解决要写清怎么解决的,标无需解决要写清为什么不用改;退回未解决时清空。空着的处置在台账上等于没有交代")
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, comment="最近一次被标为已解决/无需解决的时间;复发后仍保留,用来标「复发」")
