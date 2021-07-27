.PHONY: check
check:
	make lint type test

.PHONY: check-all
check-all:
	make lint type test --keep-going

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
	poetry run python -m pytest --cov=src --cov-branch --cov-report term-missing:skip-covered --cov-report html --codeblocks

.PHONY: test-one
test-one: pyproject.lock
	@echo "running test..."
	poetry run python -m pytest -x -s -vvv

t := "."
.PHONY: test1
test1: pyproject.lock
	@echo "running test..."
	poetry run python -m pytest -x -s -vvv -k $t

.PHONY: lint
lint: pyproject.lock
	@echo "running linter..."
	poetry run python -m flake8

.PHONY: type
type: pyproject.lock
	@echo "running type checker..."
	poetry run python -m mypy . --check-untyped-defs

# .PHONY: profile
# profile: pyproject.lock
# 	@echo "running profiler..."
# 	# poetry run py-spy record -f speedscope --full-filenames --rate 200 -n -- python examples/unigram.py
# 	cd examples; poetry run python -m scalene unigram.py --profile-all

.PHONY: doc
doc: pyproject.lock
	poetry run python -m pdoc -o ./docs src


examples/spr/protoroles_eng_ud1.2_11082016.tsv:
	mkdir -p examples/spr
	cd examples/spr; wget http://decomp.io/projects/semantic-proto-roles/protoroles_eng_udewt.tar.gz
	cd examples/spr; tar xvf protoroles_eng_udewt.tar.gz

eargs := --batch_size 100
edeps := examples/spr/protoroles_eng_ud1.2_11082016.tsv
e := examples/sprlit.py $(eargs) --path examples/spr/protoroles_eng_ud1.2_11082016.tsv
.PHONY: example
example: $(edeps) pyproject.lock
	poetry run python $e

.PHONY: dexample
dexample: $(edeps) pyproject.lock
	poetry run python -m pdb $e

profile_args := --full-filenames --rate 25 -n
.PHONY: profile
.ONESHELL:
profile: $(edeps) pyproject.lock
	tmux new-session -d -s torchfactors_profile_example
	# remap prefix from 'C-b' to 'C-a'
	tmux unbind C-b
	tmux set-option -g prefix \`
	tmux bind \` send-prefix
	# tmux bind \`\` \`
	tmux send -t torchfactors_profile_example:0 '# Starting profiling...' ENTER
	tmux send -t torchfactors_profile_example:0 '# ` is the prefix key' ENTER
	tmux send -t torchfactors_profile_example:0 '# Ctrl-C Ctrl-D Ctrl-D Ctrl-D to exit' ENTER
	tmux send -t torchfactors_profile_example:0 "poetry run python $e" ENTER
	sleep 1
	pid=`top -bn1 | head -n 10 | grep python | grep -oE "[0-9]+" | head -n 1`
	echo $${pid}
	tmux new-window
	tmux send -t torchfactors_profile_example:1 "poetry run py-spy record $(profile_args) -o torchfactors_profile_example.speedscope --pid $${pid}" ENTER
	tmux new-window
	tmux send -t torchfactors_profile_example:2 "poetry run py-spy top $(profile_args) --pid $${pid}" ENTER
	tmux select-window -t torchfactors_profile_example:0
	tmux a
