-include .env
export

.PHONY: help lint lf commit cz bump base instruct dataset clean-data upload publish

help:
	@echo "Available commands:"
	@echo "  make lint     Check code with ruff linting and format checks"
	@echo "  make lf       Fix code with ruff linting and formatting"
	@echo "  make commit   Run lint checks, then create a commitizen commit"
	@echo "  make cz       Alias for 'make commit'"
	@echo "  make bump     Bump the project version using commitizen"
	@echo "  make base     Clone walkccc/LeetCode (if needed) and build the base Parquet dataset"
	@echo "  make instruct Build the instruct Parquet dataset from the base dataset"
	@echo "  make dataset  Build both the base and instruct datasets"
	@echo "  make clean-data Remove the cloned source repo and generated output"
	@echo "  make upload   Upload existing Parquet files + dataset card to HuggingFace"
	@echo "  make publish  Build both datasets, then upload them (dataset + upload)"

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

base:
	uv run python -m llm_fine_tune.build_base_dataset

instruct:
	uv run python -m llm_fine_tune.build_instruct_dataset

dataset: base instruct

clean-data:
	rm -rf data/ output/

upload:
	uv run python -m llm_fine_tune.upload_dataset

publish: dataset upload
