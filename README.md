# Overview
This is a sample python project.

## Install
```bash
python -m pip install git+ssh://git@github.com/<username>/<repo>
```

## Development with `poetry`
Prereq: install [poetry](https://python-poetry.org/docs/#installation):
```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

Note that you will need to restart your terminal before poetry will be found on your path.

```bash
make test
make lint
make type
make check # does the above three
make doc # creates html documentation
```

## Style
Unless otherwise specified, let's follow [this guide](https://luminousmen.com/post/the-ultimate-python-style-guidelines).

