from __future__ import annotations

import argparse
import os

import pytorch_lightning as pl
import torch
import torchfactors as tx

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
    parser = argparse.ArgumentParser(add_help=False)
    parser = tx.LitSystem.add_argparse_args(parser)
    torch.set_num_threads(1)
    args = pl.Trainer.parse_argparser(parser.parse_args())
    # args = pl.Trainer.parse_argparser(parser.parse_args(
    #     "--batch_size 3 --auto_lr_find True".split()))
    trainer = pl.Trainer.from_argparse_args(args)
    model = SPRLModel()
    system = tx.LitSystem.from_args(
        model=model, data=SPRLData_v1_0(model=model),
        args=args,
        defaults=dict(path='./examples/spr/protoroles_eng_ud1.2_11082016.tsv'))

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
