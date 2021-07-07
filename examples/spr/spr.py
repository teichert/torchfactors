import argparse
import logging
from typing import Iterable

import pandas as pd  # type: ignore
import torch
import torchfactors as tx
import torchmetrics
from torch.utils.data.dataloader import DataLoader
from torchfactors.inferencers.bp import BP
from torchfactors.learning import example_fit_model
from torchfactors.model_inferencer import System

property_domain = tx.FlexDomain('property')
annotator_domain = tx.FlexDomain('annotator')


@tx.dataclass
class SPRL(tx.Subject):

    # TensorType["batch": ..., "instance"]
    # features: tx.Var = tx.VarField(tx.OBSERVED)
    rating: tx.Var = tx.VarField(tx.Range(5), tx.ANNOTATED)
    applicable: tx.Var = tx.VarField(tx.Range(2), tx.ANNOTATED, shape=rating)
    property: tx.Var = tx.VarField(property_domain, tx.OBSERVED, shape=rating)
    annotator: tx.Var = tx.VarField(annotator_domain, tx.OBSERVED, shape=rating)
    bin_rating: tx.Var = tx.VarField(tx.Range(2), tx.LATENT, shape=rating)

    @classmethod
    def from_tsv(cls, path: str, model: tx.Model['SPRL']) -> Iterable['SPRL']:
        df = pd.read_csv(path, sep='\t')
        pairs = list(dict(list(df[df['Split'] == 'train'].groupby(
            ['Sentence.ID', 'Pred.Token', 'Arg.Tokens.Begin',
             'Arg.Tokens.End', 'Annotator.ID']))).values())
        # ['Sentence.ID', 'Pred.Token', 'Arg.Tokens.Begin', 'Arg.Tokens.End']))).values())
        for pair_df in pairs:
            yield SPRL(
                rating=tx.TensorVar(torch.tensor(pair_df['Response'].values).int() - 1),
                applicable=tx.TensorVar(torch.tensor(pair_df['Response'].values) == 'yes'),
                # the domain mapping needs to be stored in the model so that the parameters
                # are relevent; it makes sense to declare in the varfield (if you want)
                # that the variable domain is flexible and which domain it is;
                # model.domain_ids(domain, )
                property=tx.TensorVar(
                    model.domain_ids(property_domain, pair_df['Property'].values)),
                annotator=tx.TensorVar(
                    model.domain_ids(annotator_domain, pair_df['Annotator.ID'].values)))


class SPRLModel(tx.Model[SPRL]):

    # def factors(self, x: SPRL):
    #     return tqdm(self._factors(x))

    def factors(self, x: SPRL):
        n = x.property.shape[-1]
        for i in range(n):
            yield tx.LinearFactor(self.namespace('bin-from-5'),
                                  x.bin_rating[..., i], x.applicable[..., i],
                                  x.rating[..., i])
            # yield tx.LinearFactor(self.namespace('joint'),
            #                       x.property[..., i], x.applicable[..., i],
            #                       x.rating[..., i])
            for j in range(i + 1, n):
                yield tx.LinearFactor(self.namespace('property-pair'),
                                      x.property[..., i], x.property[..., j],
                                      x.bin_rating[..., i], x.bin_rating[..., j])


# class SPRLModel2(tx.Model[SPRL]):

#     def __init__(self, group_factorizer: tx.VarGroupFactorizer):
#         self.grouper = group_factorizer

#     def factors(self, x: SPRL):
#         n = x.property.shape[-1]
#         for i in range(n):
#             yield from self.grouper(self.namespace('joint'), x.property[..., i],
#                                     x.applicable[..., i], x.rating[..., i])
#             for j in range(i + 1, n):
#                 yield from self.grouper(self.namespace('property-pair'),
#                                         x.property[..., i], x.property[..., j],
#                                         x.rating[..., i], x.rating[..., j])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tsv',
                        nargs="?",
                        default='./protoroles_eng_ud1.2_11082016.tsv')
    args = parser.parse_args()
    torch.set_anomaly_enabled(True)
    # print(os.cpu_count())
    torch.set_num_threads(1)
    model = SPRLModel()
    # file available: http://decomp.io/projects/semantic-proto-roles/protoroles_eng_udewt.tar.gz
    examples = list(SPRL.from_tsv(
        args.tsv,
        model=model))[:1]
    system = System(model, BP())
    train = SPRL.stack(examples)

    def eval(dataloader: DataLoader[SPRL], gold: SPRL):
        predicted = system.predict(train)
        logging.info(torchmetrics.functional.f1(
            predicted.rating.flatten() > 3,
            train.rating.flatten() > 3,
            num_classes=len(predicted.rating.domain)))

    # example_fit_model(model, examples=examples, each_step=eval,
    #                   lr=0.01, batch_size=1, passes=10, penalty_coeff=2)
    example_fit_model(model, examples=examples,
                      lr=0.01, batch_size=1, passes=10, penalty_coeff=2)
    # system = tx.System(model, tx.BP())
    # system.prime(example)
    # print(list(model.parameters()))
