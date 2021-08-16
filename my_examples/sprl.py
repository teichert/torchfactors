from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, cast

import pandas as pd
import torch
import torchfactors as tx
# from mlflow import log_artifact  # type: ignore
from mlflow.tracking.fluent import log_metrics  # type: ignore
from torchfactors.model import Model
from torchmetrics import functional
from tqdm import tqdm  # type: ignore

import thesis_utils as tu


@tx.dataclass
class SPR(tx.Subject):
    # property_domain = tx.FlexDomain('spr-properties')

    labels: tx.Var = tx.VarField(tx.ANNOTATED, tx.Range(5))
    binary_labels: tx.Var = tx.VarField(tx.ANNOTATED, tx.Range(2))
    properties: tx.Var = tx.VarField(tx.OBSERVED)
    features: tx.Var = tx.VarField(tx.OBSERVED)

    @classmethod
    def load_spr1(cls, model: Model[SPR], inputs: pd.DataFrame, labels: pd.DataFrame,
                  split: str, maxn: int):
        property_domain = model.domain('spr-properties')
        inputs = inputs[inputs['Split'] == split]  # type: ignore
        labels = labels[labels['Split'] == split]  # type: ignore
        labels = labels.sort_values('Variable.ID').reset_index()
        # combine NA with 1 and then subtract 1
        labels['Response'] = (cast(pd.Series, labels['Response'].clip(lower=1.0)) - 1.0).astype(int)
        labels_by_input_id = dict(list(labels.groupby('Input.ID')))
        num_properties = len(labels['Variable.ID'].unique())

        def examples():
            for input, input_df in inputs.groupby('Input.ID'):
                these_labels = labels_by_input_id[input]
                if len(these_labels) != num_properties:
                    raise RuntimeError("skipping an example becuase not "
                                       "all of the properties there; did you expect that?")
                properties = model.domain_ids(property_domain, these_labels['Variable.ID'])
                # domain = model._domains[SPR.property_domain.name]
                embed = input_df['embed'].values[0]
                # embed = embed.unsqueeze(-1).expand(-1, num_properties)
                labels = torch.tensor(these_labels['Response'].values)
                yield SPR(
                    labels=tx.TensorVar(labels),
                    binary_labels=tx.TensorVar((labels >= 3).int()),
                    properties=tx.TensorVar(properties, domain=property_domain),
                    features=tx.TensorVar(embed)
                )
        out = list(itertools.islice(examples(), 0, maxn))
        return out

    @staticmethod
    def evaluate_binary(pred: SPR, gold: SPR):
        all_gold = []
        all_pred = []
        domain = gold.properties.domain
        entries = []
        metrics = dict(
            f1=functional.f1,
            precision=functional.precision,
            recall=functional.recall,
        )
        for prop_id, property in enumerate(tqdm(domain, leave=False, desc="Properties")):
            y_gold = gold.binary_labels[..., prop_id].tensor
            y_pred = pred.binary_labels[..., prop_id].tensor
            all_pred.extend(list(y_pred * (prop_id + 1)))
            all_gold.extend(list(y_gold * (prop_id + 1)))
            for metric_name, metric in metrics.items():
                entry = dict(property=property)
                entry['metric'] = metric_name
                entry['value'] = float(metric(  # type: ignore
                    preds=y_pred,
                    target=y_gold,
                    ignore_index=0))
                entries.append(entry)

        results = pd.DataFrame(entries)

        # micros
        micros_preharmonic = {
            metric_name: metric(  # type: ignore
                preds=torch.tensor(all_pred),
                target=torch.tensor(all_gold),
                average="micro",
                num_classes=len(domain) + 1,
                ignore_index=0)
            for metric_name, metric in metrics.items()}

        entries.append(dict(
            property='total-00',
            metric='total-count',
            value=min(len(all_gold), len(all_pred)),
        ))

        for metric_name, value in micros_preharmonic.items():
            entry = dict(property='total-01-micro-preharmonic')
            entry['metric'] = metric_name
            entry['value'] = float(value)
            entries.append(entry)

        averages = {
            metric: results[results['metric'] == metric]['value'].mean()
            for metric in metrics
        }

        # macro pre f1
        entries.append(dict(
            property='total-02-macro-preharmonic',
            metric='f1',
            value=float((2 * averages['precision'] * averages['recall'] /
                         (averages['precision'] + averages['recall'])))
        ))

        # macro post f1
        entries.append(dict(
            property='total-03-macro-preharmonic',
            metric='f1',
            value=float(averages['f1'])))

        # macro precision and recall
        for metric_name in ['precision', 'recall']:
            for property in ['total-02-macro-preharmonic', 'total-02-macro-postharmonic']:
                entries.append(dict(
                    property=property,
                    metric=metric_name,
                    value=float(averages[metric_name])))

        return entries


def missing_required_field(name):
    def raise_error():
        raise ValueError(f"Missing required field: {name}")
    return raise_error


@ dataclass
class SPRDataModule(tx.DataModule[SPR]):
    model: tx.Model[SPR] = field(default_factory=missing_required_field("model in SPRDataModule"))
    combine_train_and_val: bool = False


class SPR1DataModule(SPRDataModule):

    def setup(self, stage=None):
        # self, pkl_path, splits,
        # max_props=None, max_n=None):
        # binarize=True):
        """
        pkl_path:
        splits: tuple of split names, e.g. ('train', 'val', 'dev', 'test')
        """
        inputs, labels = tu.load_tgz_input_labels(self.path)
        if self.model is None:
            raise TypeError("should have a model by now")
        if stage in (None, 'fit'):
            self.train = SPR.load_spr1(self.model, inputs, labels, 'train', maxn=self.train_limit)
            self.val = SPR.load_spr1(self.model, inputs, labels, 'val', maxn=self.val_limit)
            if self.combine_train_and_val:
                self.add_val_to_train()
        if stage in (None, 'test'):
            if self.test_mode:
                self.test = tx.ListDataset(SPR.load_spr1(
                    self.model, inputs, labels, 'test', maxn=self.test_limit))
            else:
                self.dev = tx.ListDataset(SPR.load_spr1(
                    self.model, inputs, labels, 'dev', maxn=self.test_limit))


class SPRSystem(tx.LitSystem[SPR]):

    # @ classmethod
    # def from_some_args(cls, model_class: Type[Model[SPR]],
    #                    data_class: Type[SPRDataModule],
    #                    args: Namespace,
    #                    defaults: Mapping[str, Any], **kwargs) -> SPRSystem:
    #     all_args = ChainMap(vars(args), kwargs, defaults)
    #     model = all_args.get('in_model', None)
    #     loaded_model = model_class()
    #     if model is not None:
    #         saved = torch.load(all_args['in_model'])
    #         # loaded_model.load_state_dict(saved['just-model'], strict=False)
    #         system = LitSystem(loaded_model)
    #         system.load_state_dict(saved['state_dict'], strict=False)
    #         loaded_model = system.model
    #     data = data_class(model=loaded_model)
    #     system = cast(SPRSystem,
    #                   super().from_args(loaded_model, data, args=args,
    #                   defaults=defaults, **kwargs))
    #     return system

    def log_evaluation(self, x: SPR, data_name: str, step: int | None = None
                       ) -> Dict[str, float]:
        with torch.set_grad_enabled(False):
            filename = f'{data_name}.binary.csv'
            out = self(x)
            entries = SPR.evaluate_binary(pred=out, gold=x)
            df = pd.DataFrame(entries)
            df.to_csv(filename)
            # log_artifact(filename, filename)
            metrics = {
                f"{e['property']}.{data_name}.{e['metric']}": e['value']
                for e in entries
            }
            metrics[f'data.{data_name}.training-objective'] = float(
                self.training_loss(x, data_name))
            log_metrics(metrics, step=self.current_epoch if step is None else step)
            self.log_dict(metrics)
            return metrics

    # def training_step(self, *args, **kwargs) -> Union[torch._tensor.Tensor, Dict[str, Any]]:
    #     out = super().training_step(*args, **kwargs)
    #     print(out)
    #     self.log_dict(dict(
    #         train_loss=out,
    #         hi=3.0), on_epoch=True)
    #     # self.log('train-loss', float(out))
    #     # self.log('hi', 3.0)
    #     return out

    def validation_step(self, *args, **kwargs):
        x = cast(SPR, args[0])
        if len(x):
            log = self.log_evaluation(x, 'val')
            return next(iter(reversed(log.values())))

    def test_step(self, *args, **kwargs):
        datasets = {}
        if self.txdata.val:
            datasets['val'] = self.txdata.val_dataloader()
        if self.txdata.test_mode and self.txdata.test:
            datasets['test'] = self.txdata.test_dataloader()
        elif not self.txdata.test_mode and self.txdata.dev:
            datasets['dev'] = self.txdata.test_dataloader()
        for data_name, dataloader in datasets.items():
            x: SPR = cast(SPR, next(iter(dataloader)))
            self.log_evaluation(x.to_device(self.device), data_name=data_name, step=-1)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint['just-model'] = self.model.state_dict()
