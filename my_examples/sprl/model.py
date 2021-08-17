import torchfactors as tx

from .data import SPR


class UnarySPRLModel(tx.Model[SPR]):

    def factors(self, x: SPR):
        for property_id, property in enumerate(x.properties.flex_domain):
            yield tx.LinearFactor(self.namespace(f'label-{property}'),
                                  x.binary_labels[..., property_id],
                                  input=x.features.tensor)


class AllPairsSPRLModel(tx.Model[SPR]):

    def factors(self, x: SPR):
        properties = list(x.properties.flex_domain)
        for property_id, property in enumerate(properties):
            yield tx.LinearFactor(self.namespace(f'label-{property}'),
                                  x.binary_labels[..., property_id],
                                  input=x.features.tensor)
            for property2_id, property2 in enumerate(properties[:property_id]):
                yield tx.LinearFactor(self.namespace(f'label-{property}-{property2}'),
                                      x.binary_labels[..., property_id],
                                      x.binary_labels[..., property2_id],
                                      input=x.features.tensor)


@tx.dataclass
class SPRLModelChoice:
    model_name: str = 'sprl.model.UnarySPRLModel'
