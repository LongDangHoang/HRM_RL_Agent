# Variables
PYTHON_VERSION=3.11
VENV_DIR=.venv

CUDA_HOME=/usr/local/cuda-124
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

.PHONY: all uv venv check_cuda torch deps jupyter

# Top-level target
all: uv venv deps jupyter

uv:
	which uv > /dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

venv: uv
	[ -d "$(VENV_DIR)" ] || uv venv $(VENV_DIR) --python $(PYTHON_VERSION) --seed

torch: venv
	@(	\
		. $(VENV_DIR)/bin/activate; \
		echo "Installing torch with $$(which pip3)"; \
		pip3 install torch==2.6.0 torchvision --index-url $(PYTORCH_INDEX_URL); \
		pip3 install packaging ninja wheel setuptools setuptools-scm; \
	)

check_cuda:
	@(	\
		if [ -d "$(CUDA_HOME)" ]; then \
			echo "CUDA found at $(CUDA_HOME)"; \
		else \
			echo "CUDA not found at $(CUDA_HOME). Please install CUDA 12.4"; \
			exit 1; \
		fi \
	)

deps: torch check_cuda
	@(	\
		. $(VENV_DIR)/bin/activate; \
		export CUDA_HOME=$(CUDA_HOME); \
		pip3 install flash-attn==2.7.4post1; \
		pip3 install -r requirements.txt; \
	)

jupyter: deps
	@(	\
		. $(VENV_DIR)/bin/activate; \
		python -m ipykernel install --name=hrm-agent --display-name "HRM Agent (uv-env)"; \
		jupyter notebook password; \
	)

.PHONY: remove_env
remove_env:
	uv run --no-sync jupyter kernelspec uninstall hrm_agent -f || true
	rm -rf $(VENV_DIR)
