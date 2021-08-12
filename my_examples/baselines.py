from __future__ import annotations

from typing import Dict, Iterable, cast

import torchfactors as tx
from pytorch_lightning.trainer.trainer import Trainer
from sklearn.linear_model import RidgeClassifierCV  # type: ignore
from tqdm import tqdm  # type: ignore

from sprl import SPR, SPR1DataModule, SPRSystem


class SPRDummyModel(tx.Model[SPR]):

    def factors(self, x: SPR) -> Iterable[tx.Factor]:
        n_properties = x.labels.shape[-1]
        domain = x.properties.flex_domain
        for i in range(n_properties):
            self.namespace(f'unary-{domain.get_value(i)}')
        return []


# def baseline(data: tx.DataModule[SPR], at_least: int = 3):
#     train_dl = data.train_dataloader()
#     train: SPR = list(train_dl)[0]
#     dev = list(data.test_dataloader())[0]
#     print(train.features.shape)
#     print(dev.features.shape)
#     domain = cast(tx.FlexDomain, train.properties.domain)
#     y_eval_all_preds = []
#     y_eval_all_golds = []
#     x_train = train.features.tensor.numpy()
#     x_eval = dev.features.tensor.numpy()
#     run = 2
#     precisions = []
#     recalls = []
#     f1s = []
#     print('run,property,metric,value')

#     def format(num):
#         return f'{num * 100:3.1f}'

class BaselineSystem(SPRSystem):

    # def __init__(self, model, data):
    #     super().__init__(model, data)

    def training_step(self, batch, batch_idx):
        pass

    def on_fit_start(self) -> None:
        self.automatic_optimization = False
        self.property_to_model: Dict[str, RidgeClassifierCV] = {}
        train_dl = self.train_dataloader()
        train: SPR = list(train_dl)[0]
        domain = cast(tx.FlexDomain, train.properties.domain)
        x_train = train.features.tensor.numpy()
        for prop_id, property in enumerate(tqdm(domain, leave=False, desc="Properties")):
            y_train = train.binary_labels[..., prop_id].tensor.numpy()
            self.property_to_model[property] = RidgeClassifierCV(
                alphas=[1e-3, 1e-2, 1e-1, 1],
                # store_cv_values=True
            ).fit(x_train, y_train)

    def configure_optimizers(self):
        return None

    def forward_(self, x: SPR) -> SPR:
        out = x.clone()
        domain = cast(tx.FlexDomain, x.properties.domain)
        input_tensor = x.features.tensor
        input = input_tensor.numpy()
        for prop_id, property in enumerate(tqdm(domain, leave=False, desc="Properties")):
            model = self.property_to_model[property]
            out.binary_labels[..., prop_id].tensor = input_tensor.new_tensor(
                model.predict(input))
        return out


if __name__ == '__main__':
    model = SPRDummyModel()
    data = SPR1DataModule(model=model, combine_train_and_val=True)
    system = BaselineSystem.from_args(
        model, data, defaults=dict(
            # path="./data/notxt.spr1.tar.gz",
            path="./data/notxt.mini10.spr1.tar.gz",
            # split_max_count=100,
            batch_size=-1))
    trainer = Trainer(max_epochs=0)
    trainer.fit(system)
    trainer.test(system)
