import logging
from dataclasses import dataclass

import torch
import torchfactors as tx


# Describe the "Subject" of the Model
@dataclass
class Bits(tx.Subject):
    bits: tx.Var = tx.VarField(tx.Range(2), tx.ANNOTATED)
    hidden: tx.Var = tx.VarField(tx.Range(2), tx.LATENT, shape=bits)


# Specify the Variables and Factors
class BitsModel(tx.Model[Bits]):
    def factors(self, x: Bits):
        length = x.hidden.shape[-1]
        yield tx.LinearFactor(self.namespace('start'), x.hidden[..., 0])
        yield tx.LinearFactor(self.namespace('end'), x.hidden[..., -1])
        # yield tx.LinearFactor(self.namespace(('start-end', length > 1)),
        #   x.hidden[..., 0], x.hidden[..., -1])
        for i in range(length):
            yield tx.LinearFactor(self.namespace('emission'), x.hidden[..., i], x.bits[..., i])
            # yield tx.LinearFactor(self.namespace('emission'), x.bits[..., i])
            if i > 0:
                yield tx.LinearFactor(self.namespace('transition'),
                                      x.hidden[..., i - 1], x.hidden[..., i])
            # if i > 1:
            #     yield tx.LinearFactor(self.namespace('transition_skip'),
            # x.hidden[..., i - 2], x.hidden[..., i])


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
]

data = [Bits(tx.vtensor(bits)) for bits in bit_sequences]

train_data = Bits.stack(data)

# Build a System and Train the Model
system = tx.System(BitsModel(), tx.BP())
system.prime(data)

optimizer = torch.optim.Adam(system.model.parameters(), lr=1.0)

# use the model
held_out = Bits(tx.vtensor([False] * 10))
held_out.bits.usage[1:3] = tx.OBSERVED

expected_held_out = Bits(tx.vtensor(
    [True, False, False, True, True, False, False, True, True, False]))
expected_held_out.bits.usage[..., :] = tx.OBSERVED

# train
for i in range(100):
    optimizer.zero_grad()
    system.product_marginal(train_data)
    partition_gold = system.product_marginal(train_data.clamp_annotated_()).sum()
    logging.info(f'iteration: {i}, gold: {partition_gold}')
    partition_free = system.product_marginal(train_data.unclamp_annotated_()).sum()
    logging.info(f'iteration: {i}, free: {partition_free}')
    loss = partition_free - partition_gold
    logging.info(f'iteration: {i}, loss: {loss}')
    loss.backward()
    optimizer.step()
    print(f'expected: {system.product_marginal(expected_held_out).sum()}')
    predicted = system.predict(held_out)
    predicted.bits.usage[..., :] = tx.OBSERVED
    print(f'achieved: {system.product_marginal(predicted).sum()}')
    print(predicted.bits.tensor.tolist())
