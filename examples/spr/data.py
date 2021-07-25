from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, cast

import pandas as pd
import torch
import torchfactors as tx
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset
from torchfactors.model import Model
from torchfactors.subject import ListDataset, SubjectType
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
    def from_tsv(path: str, model: tx.Model[SPRL], split_max_sizes: Mapping[str, Optional[int]],
                 test_mode=False) -> Mapping[str, Dataset[SPRL]]:
        df = pd.read_csv(path, sep='\t')
        splits = dict(list(df.groupby('Split')))
        outs: Mapping[str, Dataset[SPRL]] = {}
        for split, max_size in split_max_sizes.items():
            if test_mode and split == 'test':
                split = 'dev'
            if split not in splits:
                continue
            pairs = list(dict(list(splits[split].groupby(
                ['Sentence.ID', 'Pred.Token', 'Arg.Tokens.Begin',
                 'Arg.Tokens.End', 'Annotator.ID']))).values())
            outs[split] = ListDataset([
                SPRL(
                    rating=tx.TensorVar(torch.tensor(pair_df['Response'].values).int() - 1),
                    # applicable=tx.TensorVar(torch.tensor(pair_df['Response'].values) == 'yes'),
                    # the domain mapping needs to be stored in the model so that the parameters
                    # are relevent; it makes sense to declare in the varfield (if you want)
                    # that the variable domain is flexible and which domain it is;
                    # model.domain_ids(domain, )
                    property=tx.TensorVar(model.domain_ids(
                        SPRL.property_domain, pair_df['Property'].values)),
                    # annotator=tx.TensorVar(
                    #     model.domain_ids(annotator_domain, pair_df['Annotator.ID'].values)),
                    # predicate=tx.TensorVar(
                    #     model.domain_ids(predicate_domain, pair_df['Pred.Lemma'].values)),
                )
                for pair_df in tqdm(pairs[:max_size])
                if not pair_df['Response'].isna().any()
            ])
        return outs


# a regular factor directly specifies as score for each configuration;
# the point of a factor group reduces the number of parameters;
# each variable corresponds to some collection of variables (itself or latent variables)
# and sub-groups of these variables are what generate the actual factors;
# Cliques


@dataclass
class SPRLData_v1_0(tx.lightning.DataModule[SubjectType]):
    model: Optional[Model[SPRL]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.model is None:
            raise TypeError("need a model in order to use flex domains")
        else:
            split_counts = self.split_max_counts(stage)
            del split_counts['val']
            loaded_splits: Mapping[str, Dataset[SPRL]] = SPRL.from_tsv(
                self.path, self.model, split_counts, test_mode=self.test_mode)
            # TODO: fix this to make more sense
            train = loaded_splits['train']
            num_train = int(len(train) * 0.8)
            loaded_splits['train'], loaded_splits['val'] = random_split(
                cast(Dataset[SubjectType], train), [num_train, len(train) - num_train],
                generator=torch.Generator().manual_seed(42))
            for split, data in loaded_splits.items():
                self.set_split(split, data)
