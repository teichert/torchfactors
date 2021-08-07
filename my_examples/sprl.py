from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import cast

import pandas as pd
import torch
import torchfactors as tx

import thesis_utils as tu


@tx.dataclass
class SPR(tx.Subject):
    property_domain = tx.FlexDomain('spr-properties')

    labels: tx.Var = tx.VarField(tx.ANNOTATED, tx.Range(5), ndims=1)
    properties: tx.Var = tx.VarField(tx.OBSERVED, property_domain, ndims=1)
    features: tx.Var = tx.VarField(tx.OBSERVED, ndims=1)

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
                yield SPR(
                    labels=tx.TensorVar(torch.tensor(these_labels['Response'].values)),
                    properties=tx.TensorVar(properties),
                    features=tx.TensorVar(embed)
                )
        out = list(itertools.islice(examples(), 0, maxn))
        return out


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
