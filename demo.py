import math
from dataclasses import dataclass

import torch

import torchfactors as tx


def unigram():
    @dataclass
    class Characters(tx.Subject):
        char: tx.Var = tx.VarField(tx.Range(255), tx.ANNOTATED)

        @property
        def view(self) -> str:
            return ''.join(map(chr, self.char.tensor.tolist()))

        @staticmethod
        def from_string(text: str) -> 'Characters':
            text_nums = list(map(ord, text))
            return Characters(tx.TensorVar(torch.tensor(text_nums)))

    class Unigrams(tx.Model[Characters]):
        def factors(self, x: Characters):
            yield tx.LinearFactor(self.namespace('unigram'), x.char)

    model = Unigrams()
    system = tx.System(model, tx.BP())
    single = Characters(tx.TensorVar(torch.tensor([0])))
    # torch.autograd.anomaly_mode.set_detect_anomaly(True)
    with open(__file__) as f:
        text = f.read()
        text_nums = list(map(ord, text))
        x = Characters(tx.TensorVar(torch.tensor(text_nums)))
        system.prime(x)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        for i in range(100):
            optimizer.zero_grad()
            x.unclamp_annotated()
            logz_free = system.product_marginal(x)
            x.clamp_annotated()
            logz_clamped = system.product_marginal(x)
            loss = (logz_free - logz_clamped).sum()
            print(loss)
            loss.backward()
            optimizer.step()
            print(system.predict(single).view)

    total = 0.0
    for k, v in reversed(sorted(
            enumerate(system.product_marginal(single, single.char).tolist()[0]),
            key=lambda p: p[1])):
        print(f'{chr(k)}: {math.exp(v)*100:2.2}')
        total += math.exp(v)
    print(total)


unigram()


# @dataclass
# class TrueCaseExample(tx.Subject):
#     """
#     Model where a latent state is uses the lower_case letter to predict
#     whether or not it should be upper cased
#     """
#     lower_cased: tx.Var = tx.VarField(tx.OBSERVED, tx.Range(255))
#     is_upper: tx.Var = tx.VarField(tx.ANNOTATED, tx.Range(2))
#     hidden: tx.Var = tx.VarField(tx.LATENT, tx.Range(10), shape=lower_cased)

#     @classmethod
#     def from_str(cls, input: str):
#         with_start_and_end = [-1, *[ord(ch) for ch in input.lower()], -1]
#         return cls(
#             lower_cased=tx.TensorVar(torch.tensor(with_start_and_end)),
#             is_upper=tx.TensorVar(torch.tensor([False, *[ch.isupper() for ch in input], False])))

#     @property
#     def true_cased(self) -> str:
#         chars = [chr(ch) for ch in self.lower_cased.tensor[1:-1]]
#         return ''.join(ch.upper() if ui else ch
#                        for ch, ui in zip(chars, self.is_upper.tensor[1:-1]))

#     def __len__(self) -> int:
#         return self.lower_cased.shape[-1]


# x = TrueCaseExample.from_str("This is a test! Yes! Yes!")


# class TrueCaser(tx.Model[TrueCaseExample]):
#     def factors(self, x: TrueCaseExample) -> Iterable[Factor]:
#         for i in range(len(x)):
#             yield LinearFactor(
#                 self.namespace('emission'),
#                 x.hidden[..., i], x.is_upper[..., i],
#                 input=x.lower_cased.tensor[..., i])
#         for i in range(len(x) - 1):
#             yield LinearFactor(
#                 self.namespace('transition'),
#                 x.hidden[..., i], x.hidden[..., i + 1])


# true_caser = TrueCaser()
# bp = System(true_caser, BP())
# print(bp.product_marginal(x))
# print(bp.product_marginals(x))
# print(bp.predict(x).true_cased)
