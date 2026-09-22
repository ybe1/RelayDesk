from app.knowledge.chunks import parse_markdown


def test_parse_markdown_keeps_paths_key_clauses_sentences_and_table_headers():
    rows = "\n".join(f"| 商品{i} | 说明{i}。 |" for i in range(180))
    text = f"""# 退货政策
## 时限
> [!IMPORTANT]
> 演示规则：签收后七日内可申请退货。

{'这是一句足够长且必须完整保留的句子。' * 50}

## 明细
| 商品 | 说明 |
| --- | --- |
{rows}
"""

    drafts = parse_markdown(text, source="returns.md", content_type="policy")

    assert {draft.section_path for draft in drafts if "时限" in draft.section_path} == {"returns.md / 退货政策 / 时限"}
    assert any(draft.is_key_clause for draft in drafts)
    sentence = "这是一句足够长且必须完整保留的句子。"
    assert sum(draft.answer.count(sentence) for draft in drafts) >= 50
    assert all(not draft.answer.endswith(sentence[:-1]) for draft in drafts)
    table_drafts = [draft for draft in drafts if draft.is_table]
    assert len(table_drafts) >= 2
    assert all(draft.answer.startswith("| 商品 | 说明 |\n| --- | --- |") for draft in table_drafts)
    assert all(draft.answer.rstrip().endswith("|") for draft in table_drafts)


def test_parse_markdown_keeps_all_chunks_from_one_section_with_sentence_overlap():
    body = " ".join(f"第{i}句内容完整。" for i in range(300))
    drafts = parse_markdown(f"# 商品 FAQ\n## 配送\n{body}", source="products.md", content_type="faq")

    assert len(drafts) > 1
    assert all(draft.section_path == "products.md / 商品 FAQ / 配送" for draft in drafts)
    assert all(draft.answer.endswith("。") for draft in drafts)
    assert "第0句内容完整。" in drafts[0].answer
    assert "第299句内容完整。" in drafts[-1].answer
    assert "第" in drafts[0].answer and "第" in drafts[1].answer


def test_oversize_sentence_is_kept_once_and_never_becomes_overlap():
    oversize = "超长句" * 450 + "。"
    tail = " ".join(f"后续第{i}句。" for i in range(200))
    drafts = parse_markdown(f"# 手册\n## 说明\n{oversize}\n{tail}", "after_sales.md", "manual")

    assert oversize in drafts[0].answer
    assert all(oversize not in draft.answer for draft in drafts[1:])
    assert all(len(draft.answer) <= 950 or draft.answer == oversize for draft in drafts)


def test_oversize_sentence_is_standalone_after_a_short_sentence():
    oversize = "超长句" * 450 + "。"
    drafts = parse_markdown(
        f"# 手册\n## 说明\n简短前句。\n{oversize}\n后续正常句。",
        "after_sales.md",
        "manual",
    )

    oversize_draft = next(draft for draft in drafts if oversize in draft.answer)
    assert oversize_draft.answer == oversize
    assert "简短前句。" in drafts[0].answer
    assert "后续正常句。" in drafts[-1].answer


def test_faq_uses_question_lines_and_question_headings_without_making_them_categories():
    text = """# 商品 FAQ
## 商品咨询
运费怎么算？
演示规则：该问法来自 FAQ 正文。

### 能开发票吗？
订单完成后可在订单页申请发票。
"""
    drafts = parse_markdown(text, "products.md", "faq")

    assert drafts[0].category == "商品 FAQ / 商品咨询"
    assert drafts[0].questions == "运费怎么算？"
    assert "运费怎么算？" not in drafts[0].answer
    assert drafts[1].category == "商品 FAQ / 商品咨询"
    assert drafts[1].questions == "能开发票吗？"


def test_faq_question_persists_across_blank_lines_and_table_blocks():
    text = """# 商品 FAQ
## 商品咨询
运费怎么算？

演示规则：满 99 元包邮。

退货需要什么？
| 材料 | 要求 |
| --- | --- |
| 订单号 | 必填 |
"""
    drafts = parse_markdown(text, "products.md", "faq")

    shipping = [draft for draft in drafts if "满 99" in draft.answer]
    returns = [draft for draft in drafts if draft.is_table]
    assert shipping and all(draft.questions == "运费怎么算？" for draft in shipping)
    assert returns and all(draft.questions == "退货需要什么？" for draft in returns)


def test_important_only_marks_the_callout_chunk():
    drafts = parse_markdown("""# 政策
## 时限
> [!IMPORTANT]
> 关键条款：七日内申请。

普通说明：包装应当完整。
""", "returns.md", "policy")

    marked = [draft for draft in drafts if "关键条款" in draft.answer]
    ordinary = [draft for draft in drafts if "普通说明" in draft.answer]
    assert marked and all(draft.is_key_clause for draft in marked)
    assert ordinary and all(not draft.is_key_clause for draft in ordinary)
