.PHONY: dev test eval kb-preview kb-build kb-vectorize kb-mine seed-conv kb-reset eval-retrieval eval-mining kb-ch04-rebuild

dev:
	./scripts/dev.sh

test:
	uv run pytest -v

eval:
	uv run python scripts/eval_extract.py

kb-preview:
	uv run python -m scripts.kb preview

kb-build:
	uv run python -m scripts.kb build

kb-vectorize:
	uv run python -m scripts.kb vectorize

kb-ch04-rebuild:
	uv run python -m scripts.kb ch04-rebuild

kb-mine:
	uv run python -m scripts.kb mine

seed-conv:
	uv run python -m scripts.kb seed-conv

kb-reset:
	uv run python -m scripts.kb reset --confirmed

eval-retrieval:
	uv run python -m scripts.eval_ch03 retrieval

eval-mining:
	uv run python -m scripts.eval_ch03 mining
