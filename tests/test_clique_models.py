from typing import Iterable

import torch
import torchfactors as tx


@tx.dataclass
class Seq(tx.Subject):
    items: tx.Var = tx.VarField(tx.Range(5), usage=torch.tensor(
        [tx.ANNOTATED, tx.LATENT, tx.CLAMPED, tx.OBSERVED, tx.PADDING]
    ))


class Chain(tx.Model[Seq]):
    def __init__(self, clique_model: tx.CliqueModel):
        super().__init__()
        self.clique_model = clique_model

    def factors(self, subject: Seq) -> Iterable[tx.Factor]:
        for index in range(subject.items.shape[-1]):
            yield from self.clique_model.factors(
                subject.environment, self.namespace('unaries'),
                subject.items[..., index])
        length = subject.items.shape[-1]
        for index in range(1, length):
            yield from self.clique_model.factors(
                subject.environment, self.namespace('pairs'),
                subject.items[..., index - 1], subject.items[..., index])


def test_prop_odds():

    model = Chain()
    x = Seq(tx.vtensor([1, 2, 3, 4, 5]))
    factors = list(model(x))
    assert len(factors) == 5 + 4
