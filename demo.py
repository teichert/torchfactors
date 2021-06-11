from dataclasses import dataclass

import torch

import torchfactors as tx


@dataclass
class Utterance(tx.Subject):
    items: tx.Var = tx.Var(tx.Range[5])  # TensorType['batch': ..., 'index', int]


class MyModel(tx.Model[Utterance]):

    def factors(self, subject: Utterance):
        items = subject.items
        hidden = tx.Var(items.tensor, tx.VarUsage.LATENT, tx.Range[100])
        for i in range(len(items.tensor)):
            if i > 0:
                yield tx.LinearFactor([hidden[i], hidden[i-1]], self.namespace('transition'))
            yield tx.LinearFactor([hidden[i], items[i]], self.namespace('emission'))


u = Utterance(items=tx.Var(torch.ones(10)))
model = MyModel()
grounded = tx.FactorGraph(model(u))
u.items[[3, 4, 5]].usage = tx.VarUsage.CLAMPED
u.items[[3, 4, 5]].usage = tx.VarUsage.ANNOTATED
logz = grounded.query()
print(logz)
# logz = log_einsum(grounded, [()])
# all_log_probs = log_einsum(grounded, u.items)
# one_log_probs = log_einsum(grounded, u.items[0])
# print(list(model.parameters()))
