from dataclasses import dataclass
from typing import Iterable

from torchfactors import LATENT, Factor, Model, Range, Subject, Var, VarField
from torchfactors.components.tensor_factor import TensorFactor


@dataclass
class Seq(Subject):
    items: Var = VarField(Range(5), LATENT, shape=(4,))


class Chain(Model[Seq]):
    def factors(self, subject: Seq) -> Iterable[Factor]:
        for index in range(subject.items.shape[-1]):
            yield TensorFactor(subject.items[..., index])
        for index in range(subject.items.shape[-1] - 1):
            yield TensorFactor([subject.items[..., index], subject.items[..., index + 1]])


def test_model():
    model = Chain()
    data = Seq()
    factors = list(model(data))
    assert len(factors) == 4 + 3


test_model()
