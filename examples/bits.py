import logging
from dataclasses import dataclass
from typing import cast

import torch
import torchfactors as tx


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
        lasts = [cast(int, length) - 1 for length in x.list('length')]
        yield tx.LinearFactor(self.namespace('start'), x.hidden[..., 0])
        yield tx.LinearFactor(self.namespace('start-end'),
                              x.hidden[..., 0],
                              x.hidden[..., tx.gdrop(lasts)])
        for i in range(max_length):
            yield tx.LinearFactor(self.namespace('emission'), x.hidden[..., i], x.bits[..., i])
            if i > 0:
                yield tx.LinearFactor(self.namespace('transition'),
                                      x.hidden[..., i - 1], x.hidden[..., i])
            if i > 1:
                yield tx.LinearFactor(self.namespace('transition'),
                                      x.hidden[..., i - 2], x.hidden[..., i])


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

train_data = Bits.stack(data)

# Build a System and Train the Model
model = BitsModel()


log_info = {}


def each_epoch(system, data_loader, data):
    # Use the model on held out data
    held_out = Bits(tx.vtensor([False] * 10))
    held_out.bits.usage[1] = tx.OBSERVED
    predicted = system.predict(held_out)
    # expecting 1001100110
    log_info['pred'] = ''.join(map(str, predicted.bits.tensor.int().tolist()))


system = tx.learning.example_fit_model(
    model, data, iterations=1000, lr=0.01, passes=10, penalty_coeff=1,
    log_info=log_info,
    each_epoch=each_epoch)

# # Use the model on held out data
# held_out = Bits(tx.vtensor([False] * 10))
# held_out.bits.usage[1] = tx.OBSERVED
# predicted = system.predict(held_out)
# logging.info(predicted.bits.tensor.tolist())

# Save the model parameters
torch.save(model.state_dict(), "model.pt")

# Make a new model and load the saved parameters
model2 = BitsModel()
model2.load_state_dict(torch.load("model.pt"), strict=False)

# Use the new model on held-out data
held_out2 = Bits(tx.vtensor([False] * 10))
held_out2.bits.usage[1] = tx.OBSERVED
predicted2 = system.predict(held_out2)
logging.info(predicted2.bits.tensor.tolist())
