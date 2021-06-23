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
import torchfactors as tx
from dataclasses import dataclass
```

### Describe the "Subject" of the Model
<!--pytest-codeblocks:cont-->
```python
# @dataclass
# class TrueCaseSubject:
#     lower_cased: torch.Tensor # TensorType["example": ..., "character", int]
#     hidden: torch.Tensor # TensorType["example": ..., "character", int]
#     is_upper: torch.Tensor # TensorType["example": ..., "character", bool]

#     @staticmethod
#     def make(input: str):
#         padded = [-1, *[ord(ch) for ch in sentence.lower()], -1]
#         return TrueCaseSubject(
#             lower_cased=torch.tensor(padded),
#             hidden=torch.zeros(len(padded)),
#             is_upper=torch.tensor([False, *[ch.isupper() for ch in sentence], False]))
    
#     def apply(self, upper):
#         chars = [chr(ch) for ch in self.lower_cased[1:-2]]
#         return ''.join(ch.uppder() if ui else ch for ch, ui in zip(chars, upper))
```

### Specify the Variables and Factors
Here is one way (there is also a decorator verison that saves a little typing):
<!--pytest-codeblocks:cont-->
```python

# class TrueCaseModel(tx.Model)
#     def factors(data):
#         hidden = tfs.Variable(data.hidden, tfs.Integer(5), tfs.LATENT)
#         is_upper = tfs.Variable(data.is_upper, tfs.Boolean)
#         for i in range(0, len(data.is_upper)):
#             # current character is used to predict current state
#             yield tfs.LinearFactor(
#                 [hidden[...,]],
#                 torch.nn.functional.one_hot(data.lower_cased[...,i]),
#                 model.params('hidden'))
#             # ... along with the next hidden state
#             yield tfs.LinearFactor(
#                 [hidden[...,i-1], hidden[...,i]],
#                 torch.tensor(1),
#                 model.params('transition'))
#             if i > 0:
#                 # ... current state predicts label
#                 yield tfs.LinearFactor(
#                     [hidden[...,i], is_upper[...,i]],
#                     model.params('emmission'))
```

### Load Data, Train, and Use the Model
<!--pytest-codeblocks:cont-->
```python

# with open(__file__) as f:
#     code = list(TrueCaseSubject.make(line) for line in f.readlines())

# train = code[:len(code)//2]
# val = code[len(code)//2:]

# optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

# # train
# for _ in range(100):
#     for x in train:
#         optimizer.zero_grad()
#         factor_graph = model(x)
#         partition_free, = log_einsum(factor_graph, [()])
#         partition_gold, = log_einsum(factor_graph, [()], clamp=tfs.OBSERVED)
#         loss = partition_gold - partition_free
#         loss.backward()
#         optimizer.step()

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

