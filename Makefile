-include .env
export

.PHONY: help lint lf commit cz bump base instruct evaluation datasets clean-data upload publish fertility finetune-sync test docker-build verify-engines-docker

help:
	@echo "Available commands:"
	@echo "  make lint            Check code with ruff linting and format checks"
	@echo "  make lf              Fix code with ruff linting and formatting"
	@echo "  make commit          Run lint checks, then create a commitizen commit"
	@echo "  make cz              Alias for 'make commit'"
	@echo "  make bump            Bump the project version using commitizen"
	@echo "  make base            Build the base Parquet dataset (--pull updates walkccc, --refresh re-downloads HF sources)"
	@echo "  make instruct        Build instruct-train/test Parquet (70/30 split) from base"
	@echo "  make evaluation      Build leetcode-evaluation Parquet (held-out bigcode_task_payloads) from base"
	@echo "  make datasets        Build all three datasets"
	@echo "  make clean-data      Remove the cloned source repo and generated output"
	@echo "  make upload          Upload to HuggingFace (default: all). DATASET=base|instruct|evaluation to upload one"
	@echo "  make publish         Build all datasets then upload all"
	@echo "  make publish DATASET=base|instruct|evaluation  Build and upload one dataset"
	@echo "  make test            Run unit tests"
	@echo "  make fertility       Compute tokenizer fertility for sources in tokenizer-sources.txt"
	@echo "  make finetune-sync   Rsync the finetune/ configs and scripts to the cluster (requires CLUSTER_HOST and CLUSTER_REPO_DIR in .env)"
	@echo "  make docker-build           Build the execution harness image (Python 3.11 + g++ + openjdk-17; ~500MB)"
	@echo "  make verify-engines-docker  Validate expected code snippet translations (30-row sample). Use scripts/verify-engines for custom flags."

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

evaluation:
	uv run python -m llm_fine_tune.dataset.build_evaluation_dataset

datasets: base instruct evaluation

clean-data:
	rm -rf data/ output/

upload:
	uv run python -m llm_fine_tune.dataset.upload_dataset $(if $(DATASET),--datasets $(DATASET))

publish:
	$(MAKE) $(if $(DATASET),$(DATASET),datasets)
	$(MAKE) upload $(if $(DATASET),DATASET=$(DATASET))

test:
	uv run pytest

EXEC_IMAGE := llm-fine-tune-exec
EXEC_DOCKERFILE := docker/execution-harness/Dockerfile

docker-build:
	docker build -t $(EXEC_IMAGE) -f $(EXEC_DOCKERFILE) .

verify-engines-docker:
	scripts/verify-engines --sample 30

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
