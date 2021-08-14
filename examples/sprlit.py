from __future__ import annotations

import os

import torch
import torchfactors as tx
from pytorch_lightning.trainer.trainer import Trainer

from spr.data import SPRL, SPRLData_v1_0

# class SPRLModel(tx.Model[SPRL]):

#     def factors(self, x: SPRL):
#         n = x.property.shape[-1]
#         for i in range(n):
#             yield tx.LinearFactor(self.namespace('rating-property'),
#                                   x.rating[..., i], x.property[..., i])
#             # if i > 1:
#             #     yield tx.LinearFactor(self.namespace('rating-pair'),
#             #                           x.rating[..., i - 1], x.rating[..., i],
#             #                           x.property[..., i - 1], x.property[..., i])
#             for j in range(0, i):
#                 yield tx.LinearFactor(self.namespace('rating-pair'),
#                                       x.rating[..., j], x.rating[..., i],
#                                       x.property[..., j], x.property[..., i])


# class SPRLModel(tx.Model[SPRL]):

#     def factors(self, x: SPRL):
#         # n = x.property.shape[-1]
#         # x.add_factor(
#         #     tx.LinearFactor(self.namespace('rating-property'),
#         #     x.rating[..., tx.gslice()], x.property))
#         # yield tx.LinearFactor(self.namespace('rating-property'),
#         #                       x.rating, x.property)
#         # yield tx.LinearFactor(
#         #     self.namespace('rating-pair'),
#         #     x.rating[..., :-1], x.rating[..., 1:],
#         #     x.property[..., :-1], x.property[..., 1:])

#         # yield from [tx.LinearFactor(self.namespace('rating-property'),
#         #                             x.rating, x.property)] * 2
#         yield from [tx.LinearFactor(self.namespace('rating-pair'),
#                                     x.rating[..., :-1], x.rating[..., 1:],
#                                     x.property[..., :-1], x.property[..., 1:])] * 2

# #         firsts, seconds = zip(*itertools.combinations(range(n), 2))
# #         # yield tx.LinearFactor(
# #         #     self.namespace('rating-pair'),
# #         #     x.rating[..., tx.gslice(firsts)], x.rating[..., tx.gslice(seconds)],
# #         #     x.property[..., tx.gslice(firsts)], x.property[..., tx.gslice(seconds)],
# #         #     graph_dims=1)
# for i in range(n):
#     if i > 1:
#         yield tx.LinearFactor(self.namespace('rating-pair'),
#                               x.rating[..., i - 1], x.rating[..., i],
#                               x.property[..., i - 1], x.property[..., i])
# #         #     yield tx.LinearFactor(self.namespace('rating-pair'),
# #         #                           x.rating[..., tx.gslice([i] * (n - 1))], x.rating[..., i],
# #         #                           x.property[..., tx.gslice([i] * (n - 1))],
# x.property[..., i])


class SPRLModel(tx.Model[SPRL]):

    def factors(self, x: SPRL):
        n = x.property.shape[-1]
        # x.add_factor(
        #     tx.LinearFactor(self.namespace('rating-property'),
        #     x.rating[..., tx.gslice()], x.property))
        # yield tx.LinearFactor(self.namespace('rating-property'),
        #                       x.rating, x.property)
        # yield tx.LinearFactor(
        #     self.namespace('rating-pair'),
        #     x.rating[..., :-1], x.rating[..., 1:],
        #     x.property[..., :-1], x.property[..., 1:])

        # for i in range(n):
        #     yield tx.LinearFactor(self.namespace('rating-property'),
        #                           x.rating[..., i], x.property[..., i])


# #         firsts, seconds = zip(*itertools.combinations(range(n), 2))
# #         # yield tx.LinearFactor(
# #         #     self.namespace('rating-pair'),
# #         #     x.rating[..., tx.gslice(firsts)], x.rating[..., tx.gslice(seconds)],
# #         #     x.property[..., tx.gslice(firsts)], x.property[..., tx.gslice(seconds)],
# #         #     graph_dims=1)
        for i in range(n):
            # if i > 1:
            #     yield tx.LinearFactor(self.namespace('rating-pair'),
            #                           x.rating[..., i - 1], x.rating[..., i],
            #                           x.property[..., i - 1], x.property[..., i])
            for j in range(i):
                yield tx.LinearFactor(self.namespace('rating-pair'),
                                      x.rating[..., j], x.rating[..., i],
                                      x.property[..., j], x.property[..., i])
# x.property[..., i])


if __name__ == '__main__':
    try:
        print(f"Available GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
        torch.tensor(0.0).to("cuda")
    except KeyError:
        pass
    torch.set_num_threads(1)
    config = tx.Config(SPRLModel, SPRLData_v1_0, torch.optim.AdamW, tx.BP, Trainer,
                       defaults=dict(path='./examples/spr/protoroles_eng_ud1.2_11082016.tsv'))
    model = SPRLModel()
    data = SPRLData_v1_0(model=model)
    system = config.create(tx.LitSystem, model=model, data=data)
    trainer = config.create(Trainer)
    # def eval(dataloader: DataLoader[SPRL], gold: SPRL):
    #     predicted = system.predict(train)
    #     logging.info(torchmetrics.functional.f1(
    #         predicted.rating.flatten() > 3,
    #         train.rating.flatten() > 3,
    #         num_classes=len(predicted.rating.domain)))
    trainer.tune(system, lr_find_kwargs=dict(
        update_attr=True,
        num_training=2,
    ))
    print(system.hparams)
    trainer.fit(system)
