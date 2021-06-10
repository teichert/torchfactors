from dataclasses import dataclass

import torch

import torchfactors as tx


@dataclass
class Utterance(tx.Subject):
    items: tx.Var = tx.Var(tx.Range[5])  # TensorType['batch': ..., 'index', int]

# @tx.subject
# class Utterance2:
#     items: tx.VarTensor = tx.VarTensor(tx.Range[5])  # TensorType['batch': ..., 'index', int]


class MyModel(tx.Model[Utterance]):

    def factors(self, subject: Utterance):
        items = subject.items
        hidden = tx.Var(items.tensor, tx.VarUsage.LATENT, tx.Range[100])
        for i in range(len(items.tensor)):
            if i > 0:
                yield tx.LinearFactor([hidden[i], hidden[i-1]], self.params('transition'))
            yield tx.LinearFactor([hidden[i], items[i]], self.params('emission'))


u = Utterance(items=tx.Var(torch.ones(10)))
print(u.items)
# u2 = Utterance2(items=tx.VarTensor(torch.ones(10)))
# print(u2.items)
