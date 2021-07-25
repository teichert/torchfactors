from __future__ import annotations

import argparse

import pytorch_lightning as pl
import torch
import torchfactors as tx

from spr.data import SPRL, SPRLData_v1_0


class SPRLModel(tx.Model[SPRL]):

    def factors(self, x: SPRL):
        n = x.property.shape[-1]
        for i in range(n):
            yield tx.LinearFactor(self.namespace('rating-property'),
                                  x.rating[..., i], x.property[..., i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser = tx.LitSystem.add_argparse_args(parser)
    torch.set_num_threads(1)
    args = pl.Trainer.parse_argparser(parser.parse_args())
    trainer = pl.Trainer.from_argparse_args(args)
    model = SPRLModel()
    system = tx.LitSystem.from_args(
        model, data=SPRLData_v1_0(model=model),
        args=args,
        defaults=dict(path='./examples/spr/protoroles_eng_ud1.2_11082016.tsv'))

    # def eval(dataloader: DataLoader[SPRL], gold: SPRL):
    #     predicted = system.predict(train)
    #     logging.info(torchmetrics.functional.f1(
    #         predicted.rating.flatten() > 3,
    #         train.rating.flatten() > 3,
    #         num_classes=len(predicted.rating.domain)))

    trainer.fit(system)
