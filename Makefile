.PHONY: setup test

setup:
	python3 -m venv .venv
	@echo "run 'source .venv/bin/activate' to activate the virtual environment"

install:
	pip install --upgrade pip && \
	pip install -r requirements.txt && \
	pip install -r test_requirements.txt

test:
	python -m pytest

