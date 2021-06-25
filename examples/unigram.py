from dataclasses import dataclass
from itertools import cycle
from textwrap import wrap
from typing import List

import torch
import torchfactors as tx


@dataclass
class Characters(tx.Subject):
    char: tx.Var = tx.VarField(tx.Range(255), tx.ANNOTATED)

    @property
    def as_string(self) -> str:
        return ''.join(map(chr, self.char.tensor.tolist()))

    @staticmethod
    def from_string(text: str) -> 'Characters':
        text_nums = list(map(ord, text))
        return Characters(tx.TensorVar(torch.tensor(text_nums)))


class Unigrams(tx.Model[Characters]):
    def factors(self, x: Characters):
        yield tx.LinearFactor(self.namespace('unigram'), x.char)


def top_chars(character_log_probs: List[float], k: int):
    pairs = list(reversed(sorted(enumerate(character_log_probs), key=lambda p: p[1])))[:k]
    for k, v in pairs:
        print(f'{chr(k)}: {v:2.2}')


if __name__ == '__main__':
    model = Unigrams()
    system = tx.System(model, tx.BP())
    n = 40

    with open(__file__) as f:
        dataloader = Characters.data_loader([
            Characters.from_string(text)
            for text in wrap(f.read(), 1000)])
        system.prime(next(iter(dataloader)))
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        for i, data in zip(range(n), cycle(dataloader)):
            optimizer.zero_grad()

            data.unclamp_annotated_()
            logz_free = system.product_marginal(data)

            data.clamp_annotated_()
            logz_clamped = system.product_marginal(data)

            loss = (logz_free - logz_clamped).sum()

            print(loss)
            loss.backward()
            optimizer.step()

    single_char = Characters.from_string(' ')
    character_probs = system.product_marginal(single_char, single_char.char)[0].exp().tolist()
    print(top_chars(character_probs, 10))

# grep -o -E . examples/unigram.py | sort | uniq -c | sort -nr | head -n 10
# 694
# 202 r
# 201 e
# 188 t
# 180 a
# 154 s
# 112 o
#  98 i
#  91 l
#  90 c
