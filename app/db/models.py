from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, JSON, String, Text, text
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("idx_user_id", "user_id"), {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})

    id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(mysql.ENUM("进行中", "已转人工", "已结束"), nullable=False, server_default=text("'进行中'"))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (Index("idx_conversation_id", "conversation_id"), {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"})

    id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(mysql.BIGINT(unsigned=True), ForeignKey("conversations.id", name="fk_messages_conversation"), nullable=False)
    role: Mapped[str] = mapped_column(mysql.ENUM("user", "assistant", "tool"), nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))


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
