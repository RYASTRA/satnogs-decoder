# All commands run inside the container (python:3.14-slim). No host venv, no host pip.
build: ; docker compose build
test:  ; docker compose run --rm app pytest
shell: ; docker compose run --rm app bash
lint:  ; docker compose run --rm app sh -c "ruff check . && pyright satnogs_decoder"
run:   ; docker compose run --rm app python scripts/build_corpus.py

.PHONY: build test shell lint run
