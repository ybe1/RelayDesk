from pathlib import Path


SOURCE_TYPES: dict[str, str] = {
    "returns.md": "policy",
    "shipping.md": "policy",
    "products.md": "faq",
    "after_sales.md": "manual",
}


def read_demo_source(name: str, root: Path | None = None) -> tuple[str, str]:
    """Read one approved demo source, rejecting paths outside the fixed whitelist."""
    if name not in SOURCE_TYPES:
        raise ValueError("unknown knowledge source")
    directory = root or Path("data/kb")
    return (directory / name).read_text(encoding="utf-8"), SOURCE_TYPES[name]
