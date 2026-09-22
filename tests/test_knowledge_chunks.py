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
