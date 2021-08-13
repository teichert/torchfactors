from __future__ import annotations

import os
import sys
from typing import cast

import mlflow  # type: ignore
import pytorch_lightning as pl
import torch
import torchfactors as tx
from torchfactors.domain import FlexDomain
from torchfactors.model import Model

import myhydra
from sprl import SPR, SPR1DataModule, SPRSystem


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


base_config = tx.Config(Model, SPR1DataModule, SPRSystem, pl.Trainer)

path_to_data = "/home/adam/projects/torchfactors/data/notxt.mini10.spr1.tar.gz"
# path_to_checkpoint = "/home/adam/projects/torchfactors/outputs/2021-08-12/00-00-12/tb_logs/default/version_0/checkpoints/epoch=4-step=4.ckpt"

checkpoint_path = '/home/adam/projects/torchfactors/system.pt'
model_path = '/home/adam/projects/torchfactors/model.pt'


@myhydra.main(config_path=None, use_mlflow=False)
def main(cfg):
    # print(os.getcwd())
    # args = argparse.Namespace()
    # vars(args).update(cfg)
    # parser = argparse.ArgumentParser(add_help=False)
    # parser = SPRSystem.add_argparse_args(parser)
    torch.set_num_threads(1)
    # args = pl.Trainer.parse_argparser(parser.parse_args())
    # args = pl.Trainer.parse_argparser(parser.parse_args(
    #     "--batch_size 3 --auto_lr_find True".split()))
    config = base_config.child(parse_args=None, args_dict=cfg,
                               defaults=dict(
                                   path=path_to_data,
                                   #    model_state_dict_path=model_path,
                                   # checkpoint_path=path_to_checkpoint
                               ))
    model = config.create(SPRLModel)
    data = config.create(SPR1DataModule, model=model)
    system = config.create(SPRSystem, model=model, data=data)
    system.setup(None)
    # trainer = pl.Trainer.from_argparse_args(config.namespace)
    model_d = model.state_dict()
    print(model_d)
    model2 = SPRLModel()
    model2.load_state_dict(model_d)
    # torch.save(model_d, model_path)
    system_d = system.state_dict()
    print(system_d)
    # torch.save(dict(state_dict=system_d), checkpoint_path)
    # torch.save(dict(state_dict=system.state_dict()), "system.pt")
    # pritn([k for k in d if k.startswith('model.')]
    # print(d)

    # # # args = config.parser().parse_args()
    # # config.
    # # config = tx.Config(args, defaults)
    # # model = SPRLModel.from_args()
    # # # ppppppgppp
    # # system = SPRSystem.from_some_args(
    # #     model_class=SPRLModel,
    # #     data_class=SPR1DataModule,
    # #     args=args, defaults=dict(
    # #         # path="./data/notxt.spr1.tar.gz",
    # #         path="/home/adam/projects/torchfactors/data/notxt.mini10.spr1.tar.gz",
    # #         # split_max_count=100,
    # #         batch_size=-1))

    # if system.in_model is None:
    #     timed_checkpoint = ModelCheckpoint(save_top_k=0, save_last=True,
    #                                        train_time_interval=timedelta(minutes=30))
    #     best_model = ModelCheckpoint(save_top_k=1,
    #                                  monitor='data.val.training-objective',
    #                                  save_weights_only=True)
    #     best_model.FILE_EXTENSION = ".pt"
    #     early_stopping = EarlyStopping(monitor='data.val.training-objective',
    #                                    patience=cfg.get('patience', 3))
    #     trainer.callbacks.extend([
    #         timed_checkpoint,
    #         best_model,
    #         early_stopping
    #     ])
    #     trainer.logger = TensorBoardLogger('tb_logs')
    #     # trainer.tune(system, lr_find_kwargs=dict(
    #     #     update_attr=True,
    #     #     num_training=10,
    #     # ))
    #     # print(system.hparams, checkpoint)
    #     trainer.fit(system)
    #     best_model_path = Path(best_model.best_model_path).resolve()
    #     # best_model_path_str = str(best_model_path).replace('=', '\\=')
    #     print(f'path to best model: {best_model_path}')
    #     mlflow.log_artifact(f'{best_model_path}', 'best_model')
    # else:
    #     # system.load_from_checkpoint(args.in_model)
    #     system.eval()
    # trainer.test(system)


if __name__ == '__main__':
    if '--help' in sys.argv:
        base_config.parser.print_help()
    try:
        print(f"Available GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
        torch.tensor(0.0).to("cuda")
    except KeyError:
        pass
    main()
