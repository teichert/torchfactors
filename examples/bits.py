import logging
from dataclasses import dataclass

import torch
import torchfactors as tx


# Describe the "Subject" of the Model
@dataclass
class Bits(tx.Subject):
    bits: tx.Var = tx.VarField(tx.Range(2), tx.ANNOTATED)


# Specify the Variables and Factors
class BitsModel(tx.Model[Bits]):
    def factors(self, x: Bits):
        length = x.bits.shape[-1]
        for i in range(length):
            if i > 0:
                yield tx.LinearFactor(self.namespace('transition'),
                                      x.bits[..., i - 1], x.bits[..., i])


# Load Data
bit_sequences = [
    [True, False, True, False],
    [True, False, True, False, True],
    [True, False],
    [False, True, False],
    [True, False, True],
    [True, False, True, False],
    [False, True],
]

data = [Bits(tx.vtensor(bits)) for bits in bit_sequences]

train_data = Bits.stack(data)

# Build a System and Train the Model
system = tx.System(BitsModel(), tx.BP())
system.prime(data)

optimizer = torch.optim.Adam(system.model.parameters(), lr=1.0)


# train
for i in range(5):
    optimizer.zero_grad()
    system.product_marginal(train_data)
    partition_gold = system.product_marginal(train_data.clamp_annotated_()).sum()
    partition_free = system.product_marginal(train_data.unclamp_annotated_()).sum()
    loss = partition_free - partition_gold
    loss.backward()
    optimizer.step()

    logging.info(f'iteration: {i}, loss: {loss} = {partition_free} - {partition_gold}')

# use the model on held out data
held_out = Bits(tx.vtensor([False] * 10))
held_out.bits.usage[1] = tx.OBSERVED
predicted = system.predict(held_out)
logging.info(predicted.bits.tensor.tolist())


# alternative training
system2 = tx.learning.example_fit_model(BitsModel(), data, iterations=5, lr=1.0)
predicted2 = system2.predict(held_out)
logging.info(predicted2.bits.tensor.tolist())
