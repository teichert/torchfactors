from __future__ import annotations

import argparse
import os
from typing import cast

import pytorch_lightning as pl
import torch
import torchfactors as tx
from torchfactors.domain import FlexDomain

from sprl import SPR, SPR1DataModule


class SPRLModel(tx.Model[SPR]):

    def factors(self, x: SPR):
        # n = x.property.shape[-1]
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
        domain = cast(FlexDomain, x.properties.domain)
        for property_id, property in enumerate(domain):
            yield tx.LinearFactor(self.namespace(f'label-{property}'),
                                  x.binary_labels[..., property_id],
                                  input=x.features.tensor)
        # for i in range(n):
        #     # if i > 1:
        #     #     yield tx.LinearFactor(self.namespace('rating-pair'),
        #     #                           x.rating[..., i - 1], x.rating[..., i],
        #     #                           x.property[..., i - 1], x.property[..., i])
        #     # for j in range(i):
        #         yield tx.LinearFactor(self.namespace('rating-pair'),
        #                               x.b[..., j], x.rating[..., i])
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
    data = SPR1DataModule(model=model)
    system = tx.LitSystem.from_args(
        model, data, defaults=dict(
            # path="./data/notxt.spr1.tar.gz",
            path="./data/notxt.mini10.spr1.tar.gz",
            # split_max_count=100,
            batch_size=-1))

    trainer.tune(system, lr_find_kwargs=dict(
        update_attr=True,
        num_training=10,
    ))
    print(system.hparams)
    trainer.fit(system)
