import hashlib
import json
import unicodedata


def normalize(text: str) -> str:
    """Return the deterministic comparison form used for every QA source."""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        character
        for character in text
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def qa_fingerprint(question: str, answer: str) -> str:
    payload = json.dumps(
        [normalize(question), normalize(answer)], ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
