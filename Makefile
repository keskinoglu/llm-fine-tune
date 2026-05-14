.PHONY: help lint lf commit cz bump

help:
	@echo "Available commands:"
	@echo "  make lint       Check code with ruff linting and format checks"
	@echo "  make lf         Fix code with ruff linting and formatting"
	@echo "  make commit     Run lint checks, then create a commitizen commit"
	@echo "  make cz         Alias for 'make commit'"
	@echo "  make bump       Bump the project version using commitizen"

lint:
	uv run ruff check .
	uv run ruff format --check .

lf:
	uv run ruff check --fix .
	uv run ruff format .

commit: lint
	uv run cz commit

cz: commit

bump:
	uv run cz bump
