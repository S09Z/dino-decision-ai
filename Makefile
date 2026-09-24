.PHONY: help setup install train eval profile dashboard test clean lint format

help:
	@echo "Chrome Dino AI - Development Commands"
	@echo "setup      : Initialize project"
	@echo "install    : Install dependencies"
	@echo "train      : Train agents"
	@echo "eval       : Evaluate agents"
	@echo "profile    : Profile performance"
	@echo "dashboard  : Start dashboard"
	@echo "test       : Run tests"
	@echo "lint       : Code quality checks"
	@echo "format     : Auto-format code"
	@echo "clean      : Clean up files"

setup:
	poetry install
	cp .env.example .env.local

install:
	poetry install
	uv pip install -r requirements.txt

train:
	python -m src.cli.main train --agent both

eval:
	python -m src.cli.main eval

profile:
	python -m src.cli.main profile --duration 60

dashboard:
	uvicorn src.dashboard.api:app --reload --host 0.0.0.0

test:
	pytest tests/ -v --cov=src

lint:
	black --check src tests
	mypy src

format:
	black src tests
	isort src tests

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov
	rm -f *.db *.log
