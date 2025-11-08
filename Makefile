.PHONY: help install install-dev test lint format check clean run

help:
	@echo "Available commands:"
	@echo "  install     Install production dependencies"
	@echo "  install-dev Install development dependencies"
	@echo "  test        Run all tests"
	@echo "  lint        Run linting checks"
	@echo "  format      Format code with black and isort"
	@echo "  check       Run all quality checks (lint + test)"
	@echo "  clean       Clean up cache and temp files"
	@echo "  run         Run the Streamlit app"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

test:
	python -m pytest -v

lint:
	flake8 *.py
	mypy *.py
	black --check *.py
	isort --check-only *.py

format:
	black *.py
	isort *.py

check: lint test
	@echo "All quality checks passed!"

clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/

run:
	streamlit run app.py