from dataclasses import dataclass

import pytorch_lightning as pl
import torchfactors as tx
from torchfactors.lightning import LitSystem


# Describe the "Subject" of the Model
@dataclass
class Bits(tx.Subject):
    bits: tx.Var = tx.VarField(tx.Range(2), tx.ANNOTATED)
    hidden: tx.Var = tx.VarField(tx.Range(10), tx.LATENT, shape=bits)
    length: int = -1

    def __post_init__(self):
        super().__post_init__()
        self.length = self.bits.shape[-1]

# Specify the Variables and Factors


class BitsModel(tx.Model[Bits]):
    def factors(self, x: Bits):
        max_length = x.bits.shape[-1]
        # lasts = [cast(int, length) - 1 for length in x.list('length')]
        yield tx.LinearFactor(self.namespace('start'), x.hidden[..., 0])
        # yield tx.LinearFactor(self.namespace('start-end'),
        #                       x.hidden[..., 0],
        #                       x.hidden[..., tx.gdrop(lasts)])
        for i in range(max_length):
            yield tx.LinearFactor(self.namespace('emission'), x.hidden[..., i], x.bits[..., i])
            if i > 0:
                yield tx.LinearFactor(self.namespace('transition'),
                                      x.hidden[..., i - 1], x.hidden[..., i])
            # if i > 1:
            #     yield tx.LinearFactor(self.namespace('transition'),
            #                           x.hidden[..., i - 2], x.hidden[..., i])


# Load Data
bit_sequences = [
    [True, False, False, True, True, False],
    [True, False],
    [False, False, True, True],
    [True, True, False, False],
    [True, True, False, False, True, True, False],
    [True, False, False, True, True, False],
    [False, False, True, True, False, False, True],
    [False, False, True],
    [False, True],
]

data = [Bits(tx.vtensor(bits)) for bits in bit_sequences]


# class LitBitsSystem(LitSystem[Bits]):

#     def configure_model(self) -> BitsModel:
#         return BitsModel()

#     def train_dataloader(self):
#         return Bits.data_loader(data)

#     def configure_inferencer(self) -> Inferencer:
#         return BP(passes=self.passes)


trainer = pl.Trainer()
system = LitSystem(BitsModel())
trainer.fit(system, Bits.data_loader(data))
