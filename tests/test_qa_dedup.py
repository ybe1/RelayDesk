from app.knowledge.dedup import qa_fingerprint


def test_fingerprint_uses_answer_and_normalization():
    assert qa_fingerprint("运费？", " 满 99 包邮。") == qa_fingerprint("运费", "满99包邮")
    assert qa_fingerprint("运费", "满99包邮") != qa_fingerprint("运费", "满199包邮")
    assert qa_fingerprint("邮费多少", "满99包邮") != qa_fingerprint("运费", "满99包邮")
