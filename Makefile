.PHONY: all
all: install

.PHONY: install
install: pyproject.lock

pyproject.lock: pyproject.toml
	poetry update
	poetry install --remove-untracked
	touch pyproject.lock

.PHONY: test
test: pyproject.lock
	@echo "running tests..."
	poetry run python -m pytest --cov=src --cov-report term-missing:skip-covered --cov-report html

.PHONY: lint
lint: pyproject.lock
	@echo "running linter..."
	poetry run python -m flake8

.PHONY: type
type: pyproject.lock
	@echo "running type checker..."
	poetry run python -m mypy . --check-untyped-defs

.PHONY: check
check:
	make test type lint --keep-going

.PHONY: doc
doc: pyproject.lock
	poetry run python -m pdoc -o ./docs src
