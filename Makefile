.PHONY: check
check:
	make test type lint --keep-going

.PHONY: install
install: pyproject.lock

.PHONY: install-python
install-python:
	wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
	~/bin/micromamba shell init -s bash -p ~/micromamba
	echo "micromamba activate"
	source ~/.bashrc
	micromamba install python=3.9

.PHONY: install-poetry
install-poetry:
	python -m pip install --user -U pipx
	python -m pipx ensurepath
	python -m pipx install poetry

pyproject.lock: pyproject.toml
	poetry update
	poetry install --remove-untracked
	touch pyproject.lock

.PHONY: test
test: pyproject.lock
	@echo "running tests..."
	poetry run python -m pytest --cov=src --cov-report term-missing:skip-covered --cov-report html --codeblocks

.PHONY: lint
lint: pyproject.lock
	@echo "running linter..."
	poetry run python -m flake8

.PHONY: type
type: pyproject.lock
	@echo "running type checker..."
	poetry run python -m mypy . --check-untyped-defs

.PHONY: profile
profile: pyproject.lock
	@echo "running profiler..."
	# poetry run py-spy record -f speedscope --full-filenames --rate 200 -n -- python examples/unigram.py
	cd examples; poetry run python -m scalene unigram.py --profile-all

.PHONY: doc
doc: pyproject.lock
	poetry run python -m pdoc -o ./docs src
