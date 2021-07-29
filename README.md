# Overview
differentiable generalized belief propagation with pytorch

# Getting Started

## Prereqs
- Python `>= 3.8` (if you have `make` and checkout the repo, you can use `make install-python`)

## Install
<!--pytest-codeblocks:skip-->
```bash
python -m pip install git+ssh://git@github.com/teichert/torchfactors
```

## Basic Usage


```python
import logging
from dataclasses import dataclass

import torch
import torchfactors as tx


# Describe the "Subject" of the Model
@dataclass
class Bits(tx.Subject):
    bits: tx.Var = tx.VarField(tx.Range(2), tx.ANNOTATED)


# Specify the Variables and Factors
class BitsModel(tx.Model[Bits]):
    def factors(self, x: Bits):
        length = x.bits.shape[-1]
        for i in range(length):
            if i > 0:
                yield tx.LinearFactor(self.namespace('transition'),
                                      x.bits[..., i - 1], x.bits[..., i])


# Load Data
bit_sequences = [
    [True, False, True, False],
    [True, False, True, False, True],
    [True, False],
    [False, True, False],
    [True, False, True],
    [True, False, True, False],
    [False, True],
    [False],
    [True],
]

data = [Bits(tx.vtensor(bits)) for bits in bit_sequences]

train_data = Bits.stack(data)

# Build a System and Train the Model
system = tx.System(BitsModel(), tx.BP())
system.prime(data)

optimizer = torch.optim.Adam(system.model.parameters(), lr=1.0)


# train
for i in range(5):
    optimizer.zero_grad()
    train_data.clamp_annotated()
    partition_gold = system.product_marginal(train_data).sum()
    train_data.unclamp_annotated()
    partition_free = system.product_marginal(train_data).sum()
    loss = partition_free - partition_gold
    loss.backward()
    optimizer.step()

    logging.info(f'iteration: {i}, loss: {loss} = {partition_free} - {partition_gold}')

# use the model on held out data
held_out = Bits(tx.vtensor([False] * 10))
held_out.bits.usage[1] = tx.OBSERVED
predicted = system.predict(held_out)
logging.info(predicted.bits.tensor.tolist())

# # save the model to disk
# import torch
# torch.save(model.state_dict(), "model.pt")
# model2 = BitsModel()
# model2.load_state_dict(torch.load("model.pt"), strict=False)

# held_out2 = Bits(tx.vtensor([False] * 10))
# held_out2.bits.usage[1] = tx.OBSERVED
# predicted2 = system.predict(held_out2)
# logging.info(predicted2.bits.tensor.tolist())

```

# Contributing
## Development with `poetry`
Prereq: install [poetry](https://python-poetry.org/docs/#installation):

### Option 1
`make install-poetry` will install `poetry` via `pipx`


### Option 2

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


# Running on a GPU

<!-- source /home/gqin2/scripts/acquire-gpu
echo $CUDA_VISIBLE_DEVICES
poetry run python /home/adamteichert/projects/torchfactors/examples/bits_lightning_gpu.py
# poetry run python -m pdb /home/adamteichert/projects/torchfactors/examples/mini_gpu.py -->
