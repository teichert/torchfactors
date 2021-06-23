from dataclasses import dataclass

import torch

import torchfactors as tx


@dataclass
class Utterance(tx.Subject):
    items: tx.TensorVar = tx.TensorVar(tx.Range(5))  # TensorType['batch': ..., 'index', int]


class MyModel(tx.Model[Utterance]):

    def factors(self, subject: Utterance):
        items = subject.items
        hidden = tx.TensorVar(items.tensor, tx.VarUsage.LATENT, tx.Range(100))
        for i in range(len(items.tensor)):
            if i > 0:
                yield tx.LinearFactor(self.namespace('transition'), hidden[i], hidden[i-1])
            yield tx.LinearFactor(self.namespace('emission'), hidden[i], items[i])


loader = Utterance.data_loader(batch_size=3, data=[
    Utterance(items=tx.TensorVar(torch.ones(n)))
    for n in range(4, 10)
])
model = MyModel()
for u in loader:
    grounded = tx.FactorGraph(model(u))
    # u.items[[3, 4, 5]].usage = tx.VarUsage.CLAMPED
    # u.items[[3, 4, 5]].usage = tx.VarUsage.ANNOTATED
    u.clamp_annotated()
    u.unclamp_annotated()
    logz = tx.product_marginal(grounded)
    print(logz)
# logz = log_einsum(grounded, [()])
# all_log_probs = log_einsum(grounded, u.items)
# one_log_probs = log_einsum(grounded, u.items[0])
# print(list(model.parameters()))


# class MyClass:
#     def __init__(self, a: str):
#         self.a = a

#     @property
#     def a(self) -> int:
#         return self._a

#     @a.setter
#     def a(self, s: str):
#         self._a = int(s)


# a = MyClass("78")
# assert a.a == 78
