from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Optional

import pandas as pd
import torch
import torchfactors as tx
from pandas import DataFrame
from torchfactors.model import Model
from torchfactors.subject import ListDataset
from tqdm import tqdm  # type: ignore


@tx.dataclass
class SPRL(tx.Subject):
    property_domain = tx.FlexDomain('property')
    # annotator_domain = tx.FlexDomain('annotator')
    # predicate_domain = tx.FlexDomain('predicate')

    # TensorType["batch": ..., "instance"]
    # features: tx.Var = tx.VarField(tx.OBSERVED)
    rating: tx.Var = tx.VarField(tx.Range(5), tx.ANNOTATED)
    # applicable: tx.Var = tx.VarField(tx.Range(2), tx.ANNOTATED, shape=rating)
    property: tx.Var = tx.VarField(property_domain, tx.OBSERVED, shape=rating)
    # annotator: tx.Var = tx.VarField(annotator_domain, tx.OBSERVED, shape=rating)
    # predicate: tx.Var = tx.VarField(predicate_domain, tx.OBSERVED, shape=rating)
    # bin_rating: tx.Var = tx.VarField(tx.Range(2), tx.LATENT, shape=rating)

    @staticmethod
    def from_data_frame(data: DataFrame, model: tx.Model[SPRL], max_count: Optional[int] = None
                        ) -> ListDataset:
        iter = (SPRL(
            rating=tx.TensorVar(torch.tensor(pair_df['Response'].values).int() - 1),
            property=tx.TensorVar(model.domain_ids(
                SPRL.property_domain, pair_df['Property'].values)))
                for pair_df in data.groupby(['Sentence.ID', 'Pred.Token', 'Arg.Tokens.Begin',
                                             'Arg.Tokens.End', 'Annotator.ID'])
                if not pair_df['Response'].isna().any()
                )
        return ListDataset(list(tqdm(islice(iter, max_count))))

# a regular factor directly specifies as score for each configuration;
# the point of a factor group reduces the number of parameters;
# each variable corresponds to some collection of variables (itself or latent variables)
# and sub-groups of these variables are what generate the actual factors;
# Cliques


@dataclass
class SPRLData_v1_0(tx.lightning.DataModule[SPRL]):
    model: Optional[Model[SPRL]] = None
    _data_splits: dict[str, DataFrame] | None = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            raise ValueError("need a model to make sprl data (to hold the domain)")
        if self._data_splits is None:
            data = pd.read_csv(self.path, sep='\t')
            self._data_splits = dict(list(data.groupby('Split')))
        if stage in (None, 'fit'):
            self.train = SPRL.from_data_frame(self._data_splits['Train'],
                                              self.model, self.train_limit)
            super().setup_val()
        if stage in (None, 'test'):
            test_split = 'Test' if self.test_mode else 'Dev'
            self.train = SPRL.from_data_frame(self._data_splits[test_split],
                                              self.model, self.test_limit)
