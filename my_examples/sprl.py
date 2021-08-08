from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import cast

import pandas as pd
import torch
import torchfactors as tx
from torchmetrics import functional
from tqdm import tqdm  # type: ignore

import thesis_utils as tu


@tx.dataclass
class SPR(tx.Subject):
    property_domain = tx.FlexDomain('spr-properties')

    labels: tx.Var = tx.VarField(tx.ANNOTATED, tx.Range(5))
    binary_labels: tx.Var = tx.VarField(tx.ANNOTATED, tx.Range(2))
    properties: tx.Var = tx.VarField(tx.OBSERVED, property_domain)
    features: tx.Var = tx.VarField(tx.OBSERVED)

    @staticmethod
    def load_spr1(model: tx.Model[SPR], inputs: pd.DataFrame, labels: pd.DataFrame,
                  split: str, maxn: int):
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
                properties = model.domain_ids(SPR.property_domain, these_labels['Variable.ID'])
                embed = input_df['embed'].values[0]
                # embed = embed.unsqueeze(-1).expand(-1, num_properties)
                labels = torch.tensor(these_labels['Response'].values)
                yield SPR(
                    labels=tx.TensorVar(labels),
                    binary_labels=tx.TensorVar((labels >= 3).int()),
                    properties=tx.TensorVar(properties),
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
        for prop_id, property in enumerate(tqdm(domain)):
            y_gold = gold.binary_labels[..., prop_id].tensor
            y_pred = pred.binary_labels[..., prop_id].tensor
            all_pred.extend(list(y_pred * (prop_id + 1)))
            all_gold.extend(list(y_gold * (prop_id + 1)))
            for metric_name, metric in metrics.items():
                entry = dict(property=property)
                entry['metric'] = metric_name
                entry['value'] = float(metric(  # type: ignore
                    preds=torch.tensor(y_pred),
                    target=torch.tensor(y_gold),
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
        for metric_name, value in micros_preharmonic.items():
            entry = dict(property='total-01-micro-preharmonic')
            entry['metric'] = metric_name
            entry['value'] = value
            entries.append(entry)

        averages = {
            metric: results[results['metric'] == metric].mean()
            for metric in metrics
        }

        # macro pre f1
        entries.append(dict(
            property='total-02-macro-preharmonic',
            metric='f1',
            value=(2 * averages['precision'] * averages['recall'] /
                   (averages['precision'] + averages['recall']))
        ))

        # macro post f1
        entries.append(dict(
            property='total-03-macro-preharmonic',
            metric='f1',
            value=averages['f1']))

        # macro precision and recall
        for metric_name in ['precision', 'recall']:
            for property in ['total-02-macro-preharmonic', 'total-02-macro-postharmonic']:
                entries.append(dict(
                    property=property,
                    metric=metric_name,
                    value=averages[metric_name]))

        return entries


@dataclass
class SPR1DataModule(tx.DataModule[SPR]):
    model: tx.Model[SPR] | None = None
    # after preprocessing
    # RECOMMENDED_BIN_THRESHOLD = 3
    # MAX_LABEL = 4

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
            self.val = SPR.load_spr1(self.model, inputs, labels, 'val',
                                     maxn=self.val_limit if self.train_limit is None else 0)
            self.add_val_to_train()
        if stage in (None, 'test'):
            if self.test_mode:
                self.test = tx.ListDataset(SPR.load_spr1(
                    self.model, inputs, labels, 'test', maxn=self.test_limit))
            else:
                self.dev = tx.ListDataset(SPR.load_spr1(
                    self.model, inputs, labels, 'dev', maxn=self.test_limit))
