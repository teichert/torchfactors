# Overview
differentiable generalized belief propagation with pytorch

# Getting Started

## Install
<!--pytest-codeblocks:skip-->
```bash
python -m pip install git+ssh://git@github.com/teichert/torchfactors
```

## Basic Usage

### Imports

```python
import torch
import torchfactors as tx

from dataclasses import dataclass
```

### Describe the "Subject" of the Model
<!--pytest-codeblocks:cont-->
```python
@dataclass
class Bits(tx.Subject):
    bits: tx.Var = tx.VarField(tx.Range(2), tx.ANNOTATED)
    hidden: tx.Var = tx.VarField(tx.Range(10), tx.LATENT, shape=bits)

```

### Specify the Variables and Factors

<!--pytest-codeblocks:cont-->
```python

class BitsModel(tx.Model[TrueCaseSubject]):
    def factors(x: Bits):
        length = x.hidden.shape[-1]
        yield tx.LinearFactor(self.namespace('start'), x.hidden[..., 0])
        yield tx.LinearFactor(self.namespace('end'), x.hidden[..., -1])
        for i in range(length):
            yield tx.LinearFactor(self.namespace('emission'), x.hidden[..., i], x.bits[..., i])
            if i > 0:
                yield tx.LinearFactor(self.namespace('transition'), x.hidden[..., i - 1], x.hidden[..., i])
```

### Load Data
<!--pytest-codeblocks:cont-->
```python

bit_sequences = [
    [True, False, False, True],
    [False, False, True, True],
    [True, True, False, False],
    [True, True, False, False, True],
    [True, False, False, True, True],
    [False, False, True, True, False, False],
]

data = [Bits(tx.vtensor(bits)) for bits in bit_sequences]
```

### Train a Model
<!--pytest-codeblocks:cont-->
```python

system = tx.System(BitsModel(), tx.BP())
system.prime(data)

# train
for _ in range(10):
    for x in data:
        optimizer.zero_grad()
        system.marginal
        partition_free, = log_einsum(factor_graph, [()])
        partition_gold, = log_einsum(factor_graph, [()], clamp=tfs.OBSERVED)
        loss = partition_gold - partition_free
        loss.backward()
        optimizer.step()

# # test
# for x in val:
#     factor_graph = model(s)
#     out = tx.log_einsum(factor_graph, [x.is_upper])
#     print(x.case(torch.argmax(out, -1)))

```

# Contributing
## Development with `poetry`
Prereq: install [poetry](https://python-poetry.org/docs/#installation):

<!--pytest-codeblocks:skip-->
```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

Note that you will need to restart your terminal before poetry will be found on your path.

<!--pytest-codeblocks:skip-->
```bash
make test
make lint
make type
make check # does the above three
make doc # creates html documentation
```

## Style
Unless otherwise specified, let's follow [this guide](https://luminousmen.com/post/the-ultimate-python-style-guidelines).


# Using the Library

## Writing new Factors
We recommend that the paramters to your factor initialization adhere to the following:
- do not have positional arguments that are sequences
- have the last positional argument be a varargs for the variables that will be involved in the factore
- have additional information be specified by name listed after the varargs

