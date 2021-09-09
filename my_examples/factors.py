from __future__ import annotations

import pathlib
from collections import ChainMap
from datetime import timedelta

# import mlflow
import mlflow  # type: ignore
import pytorch_lightning as pl
import torch
import torchfactors as tx
import wrap
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchfactors.model import Model

from sprl import SPR1DataModule, SPRSystem
from sprl.data import SPR
from sprl.model import SPRLModelChoice

# example usage: python my_examples/factors.py \
# -cp /home/adamteichert/projects/torchfactors/my_examples/conf \
# --config-name=sge_gpu \
# -m +lr=0.001 +weight_decay=1.0,2.0 +penalty_coeff=1.0 \
# hydra.sweep.dir=/export/c06/adamteichert/experiments/tx/pairs5 exp.run_with=qsub \
# +model_name=sprl.model.AllPairsSPRLModel


base_config = tx.Config(Model, SPR1DataModule, SPRSystem, pl.Trainer,
                        EarlyStopping, SPRLModelChoice)

# path_to_data = "/home/adam/projects/torchfactors/data/notxt.mini10.spr1.tar.gz"
# path_to_data = "/export/fs03/a09/adamteichert/data/thesis/notxt.mini10.spr1.tar.gz"
path_to_data = "/export/fs03/a09/adamteichert/data/thesis/notxt.spr1.tar.gz"
# # path_to_checkpoint =
# "/home/adam/projects/torchfactors/outputs/2021-08-12/00-00-12/"
# "tb_logs/default/version_0/checkpoints/epoch=4-step=4.ckpt"

# checkpoint_path = '/home/adam/projects/torchfactors/system.pt'
# model_path = '/home/adam/projects/torchfactors/model.pt'


@wrap.main(config_path='.', use_mlflow=True, grab_gpu=True)
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
    config = base_config.child(parse_args=None,
                               defaults=ChainMap(cfg, dict(
                                   path=path_to_data,
                                   batch_size=-1,
                                   train_batch_size=50,
                                   val_batch_size=-1,
                                   test_batch_size=-1,
                                   patience=3,
                                   #    model_state_dict_path=model_path,
                                   # checkpoint_path=path_to_checkpoint
                               )))
    model_config = config.create(SPRLModelChoice)
    model: tx.Model[SPR] = config.create_from_name(model_config.model_name)
    data = config.create(SPR1DataModule, model=model)
    system = config.create(SPRSystem, model=model, data=data)
    trainer = config.create(pl.Trainer)

    if config.args.checkpoint_path is None:
        timed_checkpoint = ModelCheckpoint(save_top_k=0, save_last=True,
                                           train_time_interval=timedelta(minutes=30))
        best_model = ModelCheckpoint(save_top_k=1,
                                     monitor='data.val.training-objective',
                                     save_weights_only=True)
        best_model.FILE_EXTENSION = ".pt"
        early_stopping = EarlyStopping(monitor='data.val.training-objective',
                                       patience=config.args.patience)
        trainer.callbacks.extend([
            timed_checkpoint,
            best_model,
            early_stopping
        ])
        trainer.logger = TensorBoardLogger('tb_logs')
        trainer.fit(system)
        best_model_path = pathlib.Path(best_model.best_model_path).resolve()
        print(f'path to best model: {best_model_path}')
        # mlflow.log_artifact(f'{best_model_path}', 'best_model')
        mlflow.log_param('best_model_path', str(best_model))
    else:
        system.eval()
    trainer.test(system)


if __name__ == '__main__':
    main()
