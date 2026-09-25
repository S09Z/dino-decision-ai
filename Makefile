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
	poetry run python -m src.cli.main train --agent dqn

eval:
	poetry run python -m src.cli.main eval

profile:
	poetry run python -m src.cli.main profile --steps 500

dashboard:
	poetry run uvicorn src.dashboard.api:app --reload --host 0.0.0.0

test:
	poetry run pytest tests/ -v --cov=src

lint:
	poetry run black --check src tests
	poetry run mypy src

format:
	poetry run black src tests
	poetry run isort src tests

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov
	rm -f *.db *.log
