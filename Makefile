# DeepVQE-GGML Makefile
# =====================
#
# Targets:
#   build          Build the Docker training image
#   test           Quick smoke test (dummy data, 2 epochs)
#   train-minimal  Train on DNS5 minimal subset
#   train-full     Train on full dataset (mount custom data dir)
#   eval           Evaluate a checkpoint
#   export         Export checkpoint to GGUF
#   download-data  Download DNS5 minimal dataset
#   test-model     Run model unit tests
#   test-data      Run data pipeline tests
#   tensorboard    Launch TensorBoard viewer
#   shell          Open a shell in the container
#   clean          Remove checkpoints, logs, eval output

IMAGE        := deepvqe
CONTAINER    := deepvqe-train
GPU          := all
CONFIG       := configs/default.yaml

# Host directories to mount
DATA_DIR     := $(CURDIR)/datasets_fullband
CKPT_DIR     := $(CURDIR)/checkpoints
LOG_DIR      := $(CURDIR)/logs
EVAL_DIR     := $(CURDIR)/eval_output
CACHE_DIR    := $(CURDIR)/.cache/torch_inductor

# Training overrides
EPOCHS       ?=
BATCH_SIZE   ?=
EXTRA_ARGS   ?=

DOCKER_RUN := docker run --rm --device nvidia.com/gpu=$(GPU) \
	--name $(CONTAINER) \
	--shm-size=4g \
	--device /dev/fuse --cap-add SYS_ADMIN \
	-e TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
	-e TORCHINDUCTOR_CACHE_DIR=/cache/torch_inductor \
	-v $(CURDIR):/workspace/deepvqe \
	-v $(DATA_DIR):/workspace/deepvqe/datasets_fullband \
	-v $(CKPT_DIR):/workspace/deepvqe/checkpoints \
	-v $(LOG_DIR):/workspace/deepvqe/logs \
	-v $(EVAL_DIR):/workspace/deepvqe/eval_output \
	-v $(CACHE_DIR):/cache/torch_inductor

# ── Build ────────────────────────────────────────────────────────────────────

.PHONY: build
build: ## Build Docker training image
	docker build -t $(IMAGE) .

# ── Training ─────────────────────────────────────────────────────────────────

.PHONY: test
test: build ## Smoke test: dummy data, 2 epochs, batch 4
	$(DOCKER_RUN) $(IMAGE) \
		python train.py --config $(CONFIG) --dummy $(EXTRA_ARGS)

.PHONY: train-minimal
train-minimal: build ## Train on DNS5 minimal subset
	$(DOCKER_RUN) $(IMAGE) \
		python train.py --config $(CONFIG) $(EXTRA_ARGS)

.PHONY: train-full
train-full: build ## Train on full dataset (set DATA_DIR= to override)
	$(DOCKER_RUN) $(IMAGE) \
		python train.py --config $(CONFIG) $(EXTRA_ARGS)

# ── Evaluation & Export ──────────────────────────────────────────────────────

CHECKPOINT ?= checkpoints/best.pt

.PHONY: eval
eval: build ## Evaluate a checkpoint (set CHECKPOINT=path)
	$(DOCKER_RUN) $(IMAGE) \
		python eval.py --config $(CONFIG) --checkpoint $(CHECKPOINT) $(EXTRA_ARGS)

.PHONY: export
export: build ## Export checkpoint to GGUF (set CHECKPOINT=path)
	$(DOCKER_RUN) \
		-v $(CURDIR):/workspace/deepvqe/output \
		$(IMAGE) \
		python export_ggml.py --config $(CONFIG) --checkpoint $(CHECKPOINT) \
			--output output/deepvqe.gguf $(EXTRA_ARGS)

# ── Tests ────────────────────────────────────────────────────────────────────

.PHONY: test-model
test-model: build ## Run model unit tests
	$(DOCKER_RUN) $(IMAGE) python test_model.py

.PHONY: test-data
test-data: build ## Run data pipeline tests
	$(DOCKER_RUN) $(IMAGE) python test_data.py

.PHONY: tests
tests: test-model test-data ## Run all unit tests

# ── Data ─────────────────────────────────────────────────────────────────────

.PHONY: download-data
download-data: ## Download DNS5 minimal dataset
	bash scripts/download_dns5_minimal.sh $(DATA_DIR)

# ── Utilities ────────────────────────────────────────────────────────────────

.PHONY: test-erle
test-erle: build ## Test ERLE and delay estimation with real speech
	$(DOCKER_RUN) $(IMAGE) \
		python scripts/test_erle.py $(EXTRA_ARGS)

.PHONY: check
check: build ## Check training progress (loss, entropy, grad norms)
	$(DOCKER_RUN) $(IMAGE) \
		python scripts/check_training.py $(EXTRA_ARGS)

.PHONY: tensorboard
tensorboard: ## Launch TensorBoard on port 6006
	docker run --rm -p 6006:6006 \
		-v $(LOG_DIR):/logs \
		tensorflow/tensorflow \
		tensorboard --logdir /logs --bind_all

.PHONY: shell
shell: build ## Open interactive shell in the container
	$(DOCKER_RUN) -it $(IMAGE) bash

.PHONY: clean
clean: ## Remove checkpoints, logs, and eval output
	rm -rf checkpoints/ logs/ eval_output/

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
