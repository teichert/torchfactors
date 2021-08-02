from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import islice
from typing import Any, Optional

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
    info: Any = None
    # annotator: tx.Var = tx.VarField(annotator_domain, tx.OBSERVED, shape=rating)
    # predicate: tx.Var = tx.VarField(predicate_domain, tx.OBSERVED, shape=rating)
    # bin_rating: tx.Var = tx.VarField(tx.Range(2), tx.LATENT, shape=rating)

    @classmethod
    def from_data_frame(cls, data: DataFrame, model: tx.Model[SPRL], max_count: Optional[int] = None
                        ) -> ListDataset:
        def examples():
            for info, pair_df in data.groupby(['Sentence.ID', 'Pred.Token', 'Arg.Tokens.Begin',
                                               'Arg.Tokens.End', 'Annotator.ID']):
                for property, prop_df in pair_df.groupby('Property'):
                    if len(prop_df) > 1:
                        logging.warning(f'multiple annotations: {info}, {property}')
                    if pair_df['Response'].isna().any():
                        logging.warning(f'nan response: {info}, {property}')
                if not pair_df['Response'].isna().any():
                    property_values = model.domain_ids(
                        SPRL.property_domain, pair_df['Property'].values)
                    properties = tx.TensorVar(property_values)
                    example = SPRL(
                        rating=tx.TensorVar(torch.tensor(pair_df['Response'].values).int() - 1),
                        property=properties,
                        info=info)
                    yield example
        all_examples = list(islice(examples(), max_count))
        out = ListDataset[SPRL](list(tqdm(all_examples, desc="Loading data...")))
        logging.info(f"Loaded: {len(out)} examples covering {len(cls.property_domain)} properties: "
                     f"{cls.property_domain.values}")
        first = out.examples[0]
        logging.info(f"First example shapes: rating: {first.rating.marginal_shape}; "
                     f"property: {first.property.marginal_shape}")
        return out

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
            self.train = SPRL.from_data_frame(self._data_splits['train'],
                                              self.model, self.train_limit)
            super().setup_val()
        if stage in (None, 'test'):
            test_split = 'test' if self.test_mode else 'dev'
            self.train = SPRL.from_data_frame(self._data_splits[test_split],
                                              self.model, self.test_limit)
