"""Guard the legacy reset against deleting another collection's SQL source."""


def require_legacy_reset(collection_name: str) -> None:
    if collection_name != "knowledge":
        raise ValueError(
            f"kb-reset is disabled while MILVUS_COLLECTION={collection_name}; "
            "Ch04 uses the shared SQL source. A complete Ch04 reset requires a "
            "separate maintenance action, which is not provided in this chapter."
        )
