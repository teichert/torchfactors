import logging
from dataclasses import dataclass

import torchfactors as tx


# Describe the "Subject" of the Model
@dataclass
class Bits(tx.Subject):
    bits: tx.Var = tx.VarField(tx.Range(2), tx.ANNOTATED)
    hidden: tx.Var = tx.VarField(tx.Range(10), tx.LATENT, shape=bits)


# Specify the Variables and Factors
class BitsModel(tx.Model[Bits]):
    def factors(self, x: Bits):
        length = x.bits.shape[-1]
        yield tx.LinearFactor(self.namespace('start'), x.hidden[..., 0])
        # yield tx.LinearFactor(self.namespace(('start-end', length > 1)),
        #                       x.hidden[..., 0], x.hidden[..., -1])
        # yield tx.LinearFactor(self.namespace('emission'), x.hidden[..., 0], x.bits[..., 0])
        # yield tx.LinearFactor(self.namespace('emission'), x.hidden[..., 2], x.bits[..., 2])
        for i in range(length):
            yield tx.LinearFactor(self.namespace('emission'), x.hidden[..., i], x.bits[..., i])
        #     # yield tx.LinearFactor(self.namespace('emission'), x.bits[..., i])
        #     # yield tx.LinearFactor(self.namespace('transition'),
        #     #                       x.hidden[..., i], x.bits[..., i])
            if i > 0:
                yield tx.LinearFactor(self.namespace('transition'),
                                      x.hidden[..., i - 1], x.hidden[..., i])
        #     if i > 1:
        #         yield tx.LinearFactor(self.namespace('transition_skip'),
        #                               x.hidden[..., i - 2], x.hidden[..., i])

# this gives larger clamped than unclamped with the following single data and transition/emmission
# factors (why?!):
#        [-7.021953582763672, 3.808997869491577]
#        [4.331912040710449, -17.0089054107666]


# Load Data
bit_sequences = [
    [True, False, False, True, True, False],
    [True, False],
    # [False, False, True, True],
    [True, True, False, False],
    [True, True, False, False, True, True, False],
    [True, False, False, True, True, False],
    # [False, False, True, True, False, False, True],
    # [False, False, True],
    # [False, True],
]

data = [Bits(tx.vtensor(bits)) for bits in bit_sequences]

train_data = Bits.stack(data)

# Build a System and Train the Model
# system = tx.System(BitsModel(), tx.BP())
# system.prime(data)

# optimizer = torch.optim.Adam(system.model.parameters(), lr=1.0)


# train the model
model = BitsModel()
system = tx.learning.example_fit_model(
    model, data, iterations=200, lr=0.01, passes=4, penalty_coeff=2)

# use the model on held out data
held_out = Bits(tx.vtensor([False] * 10))
held_out.bits.usage[1] = tx.OBSERVED
predicted = system.predict(held_out)
logging.info(predicted.bits.tensor.tolist())

# # train
# for i in range(100):
#     optimizer.zero_grad()
#     partition_gold = system.product_marginal(train_data.clamp_annotated_()).sum()
#     partition_free = system.product_marginal(train_data.unclamp_annotated_()).sum()
#     loss = partition_free - partition_gold
#     if loss < 0:
#         print()
#     loss.backward()
#     optimizer.step()

#     logging.info(f'iteration: {i}, loss: {loss} = {partition_free} - {partition_gold}')
