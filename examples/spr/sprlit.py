import argparse
from typing import Iterable

import pandas as pd  # type: ignore
import torch
import torchfactors as tx
from tqdm import tqdm  # type: ignore

property_domain = tx.FlexDomain('property')
annotator_domain = tx.FlexDomain('annotator')
predicate_domain = tx.FlexDomain('predicate')


@tx.dataclass
class SPRL(tx.Subject):

    # TensorType["batch": ..., "instance"]
    # features: tx.Var = tx.VarField(tx.OBSERVED)
    rating: tx.Var = tx.VarField(tx.Range(5), tx.ANNOTATED)
    applicable: tx.Var = tx.VarField(tx.Range(2), tx.ANNOTATED, shape=rating)
    property: tx.Var = tx.VarField(property_domain, tx.OBSERVED, shape=rating)
    annotator: tx.Var = tx.VarField(annotator_domain, tx.OBSERVED, shape=rating)
    predicate: tx.Var = tx.VarField(predicate_domain, tx.OBSERVED, shape=rating)
    bin_rating: tx.Var = tx.VarField(tx.Range(2), tx.LATENT, shape=rating)

    @classmethod
    def from_tsv(cls, path: str, model: tx.Model['SPRL']) -> Iterable['SPRL']:
        df = pd.read_csv(path, sep='\t')
        pairs = list(dict(list(df[df['Split'] == 'train'].groupby(
            ['Sentence.ID', 'Pred.Token', 'Arg.Tokens.Begin',
             'Arg.Tokens.End', 'Annotator.ID']))).values())
        # ['Sentence.ID', 'Pred.Token', 'Arg.Tokens.Begin', 'Arg.Tokens.End']))).values())
        for pair_df in tqdm(pairs):
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
                    model.domain_ids(annotator_domain, pair_df['Annotator.ID'].values)),
                predicate=tx.TensorVar(
                    model.domain_ids(predicate_domain, pair_df['Pred.Lemma'].values)),
            )

# a regular factor directly specifies as score for each configuration;
# the point of a factor group reduces the number of parameters;
# each variable corresponds to some collection of variables (itself or latent variables)
# and sub-groups of these variables are what generate the actual factors;
# Cliques


class SPRLModel(tx.Model[SPRL]):

    # def factors(self, x: SPRL):
    #     return tqdm(self._factors(x))

    def factors(self, x: SPRL):
        n = x.property.shape[-1]
        for i in range(n):
            yield tx.LinearFactor(self.namespace('bin-from-5'),
                                  x.bin_rating[..., i], x.applicable[..., i],
                                  x.rating[..., i])
            # yield tx.LinearFactor(self.namespace('bin-from-5'),
            #                       x.bin_rating[..., i], x.applicable[..., i],
            #                       x.rating[..., i])
            # yield from tx.Clique(
            #     self.namespace('bin-from-5'),
            #     x.bin_rating[..., i], x.applicable[..., i],
            #     x.rating[..., i])

            # yield tx.LinearFactor(self.namespace('joint'),
            #                       x.property[..., i], x.applicable[..., i],
            #                       x.rating[..., i])
            property_domain.freeze()
            annotator_domain.freeze()
            predicate_domain.freeze()
            num_properties = len(x.property.domain)
            num_predicates = len(x.predicate.domain)
            embed_size = 5
            for j in range(i + 1, n):
                def embed_property_constructor():
                    return torch.nn.Bilinear(num_properties, num_properties, embed_size)
                embed_property = self.namespace('embed_property').module(embed_property_constructor)
                embeded_property = embed_property(
                    tx.one_hot(x.property[..., i].tensor.long(), num_properties).float(),
                    tx.one_hot(x.property[..., j].tensor.long(), num_properties).float()
                )

                def embed_predicate_constructor():
                    return torch.nn.Bilinear(num_predicates, num_predicates, embed_size)
                embed_predicate = self.namespace(
                    'embed_predicate').module(embed_predicate_constructor)
                embeded_predicate = embed_predicate(
                    tx.one_hot(x.predicate[..., i].tensor.long(), num_predicates).float(),
                    tx.one_hot(x.predicate[..., j].tensor.long(), num_predicates).float()
                )
                yield tx.LinearFactor(self.namespace('property-pair'),
                                      x.bin_rating[..., i], x.bin_rating[..., j],
                                      input=torch.cat([embeded_property, embeded_predicate], -1))


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
                        # default='./protoroles_eng_ud1.2_11082016.tsv')
                        default='./examples/spr/protoroles_eng_ud1.2_11082016.tsv')
    args = parser.parse_args()
    # torch.set_anomaly_enabled(True)
    # print(os.cpu_count())
    torch.set_num_threads(1)
    model = SPRLModel()
    # file available: http://decomp.io/projects/semantic-proto-roles/protoroles_eng_udewt.tar.gz
    examples = list(SPRL.from_tsv(
        args.tsv,
        model=model))[:50]
    # system = System(model, BP())
    # train = SPRL.stack(examples)

    # def eval(dataloader: DataLoader[SPRL], gold: SPRL):
    #     predicted = system.predict(train)
    #     logging.info(torchmetrics.functional.f1(
    #         predicted.rating.flatten() > 3,
    #         train.rating.flatten() > 3,
    #         num_classes=len(predicted.rating.domain)))

    import pytorch_lightning as pl
    trainer = pl.Trainer()
    system = tx.LitSystem(SPRLModel(), inference_kwargs=dict(passes=0))
    trainer.fit(system, SPRL.data_loader(examples, batch_size=3))

    # example_fit_model(model, examples=examples, each_step=eval,
    #                   lr=0.01, batch_size=1, passes=10, penalty_coeff=2)
    # example_fit_model(model, examples=examples,
    #                   lr=0.01, batch_size=1, passes=10, penalty_coeff=2)
    # system = tx.System(model, tx.BP())
    # system.prime(example)
    # print(list(model.parameters()))
