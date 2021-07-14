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

.PHONY: test
test-one: pyproject.lock
	@echo "running tests..."
	poetry run python -m pytest -x -s -vvv

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


examples/spr/protoroles_eng_ud1.2_11082016.tsv:
	mkdir -p examples/spr
	cd examples/spr; wget http://decomp.io/projects/semantic-proto-roles/protoroles_eng_udewt.tar.gz
	cd examples/spr; tar xvf protoroles_eng_udewt.tar.gz

.PHONY: spr-example
spr-example: examples/spr/protoroles_eng_ud1.2_11082016.tsv pyproject.lock
	poetry run python examples/spr/spr.py examples/spr/protoroles_eng_ud1.2_11082016.tsv

# .PHONY: profile-spr
# profile-spr: examples/spr/protoroles_eng_ud1.2_11082016.tsv pyproject.lock
# 	poetry run python examples/spr/spr.py examples/spr/protoroles_eng_ud1.2_11082016.tsv &; pid=`top | head -n 10 | grep python | head -n 1 | cut -d" " -f2`; poetry run py-spy
# 	poetry run python examples/spr/spr.py examples/spr/protoroles_eng_ud1.2_11082016.tsv
