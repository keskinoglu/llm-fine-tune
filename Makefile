-include .env
export

.PHONY: help lint lf commit cz bump base instruct dataset clean-data upload publish fertility finetune-sync test

help:
	@echo "Available commands:"
	@echo "  make lint            Check code with ruff linting and format checks"
	@echo "  make lf              Fix code with ruff linting and formatting"
	@echo "  make commit          Run lint checks, then create a commitizen commit"
	@echo "  make cz              Alias for 'make commit'"
	@echo "  make bump            Bump the project version using commitizen"
	@echo "  make base            Build and enrich the base Parquet dataset (--pull to update walkccc, --refresh to re-download HF sources)"
	@echo "  make instruct        Build instruct-train.parquet and instruct-test.parquet (70/30 split) from base"
	@echo "  make dataset         Build both the base and instruct datasets"
	@echo "  make clean-data      Remove the cloned source repo and generated output"
	@echo "  make upload          Upload base + instruct-train + instruct-test + dataset card to HuggingFace"
	@echo "  make publish         Build both datasets, then upload them (dataset + upload)"
	@echo "  make test            Run unit tests (splits, newfacade matching)"
	@echo "  make fertility       Compute tokenizer fertility for sources in tokenizer-sources.txt"
	@echo "  make finetune-sync   Rsync the finetune/ configs and scripts to the cluster (requires CLUSTER_HOST and CLUSTER_REPO_DIR in .env)"

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
	git push origin --tags

base:
	uv run python -m llm_fine_tune.dataset.build_base_dataset

instruct:
	uv run python -m llm_fine_tune.dataset.build_instruct_dataset

dataset: base instruct

clean-data:
	rm -rf data/ output/

upload:
	uv run python -m llm_fine_tune.dataset.upload_dataset

publish: dataset upload

test:
	uv run pytest

fertility:
	uv run python -m llm_fine_tune.tokenizer.analyze_tokenizer_fertility

finetune-sync:
	@if [ -z "$(CLUSTER_HOST)" ] || [ -z "$(CLUSTER_REPO_DIR)" ]; then \
		echo "ERROR: CLUSTER_HOST and CLUSTER_REPO_DIR must be set in .env"; \
		echo "  CLUSTER_HOST=$(CLUSTER_HOST)"; \
		echo "  CLUSTER_REPO_DIR=$(CLUSTER_REPO_DIR)"; \
		exit 1; \
	fi
	rsync -av --delete --mkpath \
		$(if $(CLUSTER_SSH_KEY),-e "ssh -i $(CLUSTER_SSH_KEY)") \
		src/llm_fine_tune/finetune/ \
		$(CLUSTER_HOST):$(CLUSTER_REPO_DIR)/src/llm_fine_tune/finetune/
