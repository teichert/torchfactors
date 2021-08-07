from __future__ import annotations

from typing import Iterable, cast

import torch
import torchfactors as tx
from sklearn.linear_model import RidgeClassifierCV  # type: ignore
from torchmetrics import functional
from tqdm import trange  # type: ignore

from sprl import SPR, SPR1DataModule


class SPRDummyModel(tx.Model[SPR]):

    def factors(self, x: SPR) -> Iterable[tx.Factor]:
        n_properties = x.labels.shape[-1]
        for i in range(n_properties):
            self.namespace(f'unary-{SPR.property_domain.get_value(i)}')
        return []


def baseline(data: tx.DataModule[SPR], at_least: int = 3):
    train_dl = data.train_dataloader()
    train: SPR = list(train_dl)[0]
    dev = list(data.test_dataloader())[0]
    print(train.features.shape)
    print(dev.features.shape)
    domain = cast(tx.FlexDomain, train.properties.domain)
    y_eval_all_preds = []
    y_eval_all_golds = []
    x_train = train.features.tensor.numpy()
    x_eval = dev.features.tensor.numpy()
    run = 2
    precisions = []
    recalls = []
    f1s = []
    print('run,property,metric,value')

    def format(num):
        return f'{num * 100:3.1f}'

    for prop_id in trange(len(domain)):
        property = domain.get_value(prop_id)
        y_train = (train.labels[..., prop_id].tensor >= at_least).int().numpy()
        y_eval_gold = (dev.labels[..., prop_id].tensor >= at_least).int().numpy()
        clf = RidgeClassifierCV(
            alphas=[1e-3, 1e-2, 1e-1, 1],
            # store_cv_values=True
        ).fit(x_train, y_train)
        y_eval_pred = clf.predict(x_eval)
        y_eval_all_preds.extend(list(y_eval_pred * (prop_id + 1)))
        y_eval_all_golds.extend(list(y_eval_gold * (prop_id + 1)))
        precision = functional.precision(preds=torch.tensor(y_eval_pred),
                                         target=torch.tensor(y_eval_gold),
                                         ignore_index=0)
        recall = functional.recall(preds=torch.tensor(y_eval_pred),
                                   target=torch.tensor(y_eval_gold),
                                   ignore_index=0)
        f1 = functional.f1(preds=torch.tensor(y_eval_pred),
                           target=torch.tensor(y_eval_gold),
                           ignore_index=0)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        # print(f'{run},{property},{format(precision)},{format(recall)},{format(f1)}')
        print(f'{run},{property},precision,{format(precision)}')
        print(f'{run},{property},recall,{format(recall)}')
        print(f'{run},{property},f1,{format(f1)}')

    y_eval_all_pred = torch.tensor(y_eval_all_preds)
    y_eval_all_gold = torch.tensor(y_eval_all_golds)

    micro_precision = functional.precision(preds=y_eval_all_pred, target=y_eval_all_gold,
                                           average="micro",
                                           num_classes=len(domain) + 1, ignore_index=0)
    micro_recall = functional.recall(preds=y_eval_all_pred, target=y_eval_all_gold, average="micro",
                                     num_classes=len(domain) + 1, ignore_index=0)
    micro_f1 = functional.f1(preds=y_eval_all_pred, target=y_eval_all_gold, average="micro",
                             num_classes=len(domain) + 1, ignore_index=0)
    macro_precision_tm = functional.precision(preds=y_eval_all_pred, target=y_eval_all_gold,
                                              average="macro",
                                              num_classes=len(domain) + 1, ignore_index=0)
    macro_recall_tm = functional.recall(preds=y_eval_all_pred, target=y_eval_all_gold,
                                        average="macro",
                                        num_classes=len(domain) + 1, ignore_index=0)
    macro_postharmonic_f1_tm = functional.f1(preds=y_eval_all_pred, target=y_eval_all_gold,
                                             average="macro",
                                             num_classes=len(domain) + 1, ignore_index=0)
    macro_precision = torch.tensor(precisions).mean()
    macro_recall = torch.tensor(recalls).mean()
    macro_postharmonic_f1 = torch.tensor(f1s).mean()
    macro_preharmonic_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)
    print(f'{run},all-micro-preharmonic,precision,{format(micro_precision)}')
    print(f'{run},all-micro-preharmonic,recall,{format(micro_recall)}')
    print(f'{run},all-micro-preharmonic,f1,{format(micro_f1)}')
    print(f'{run},all-macro-preharmonic,precision,{format(macro_precision)}')
    print(f'{run},all-macro-preharmonic,recall,{format(macro_recall)}')
    print(f'{run},all-macro-preharmonic,f1,{format(macro_preharmonic_f1)}')
    print(f'{run},all-macro-postharmonic,precision,{format(macro_precision)}')
    print(f'{run},all-macro-postharmonic,recall,{format(macro_recall)}')
    print(f'{run},all-macro-postharmonic,f1,{format(macro_postharmonic_f1)}')
    print(f'{run},all-macro-postharmonic_tm,precision,{format(macro_precision_tm)}')
    print(f'{run},all-macro-postharmonic_tm,recall,{format(macro_recall_tm)}')
    print(f'{run},all-macro-postharmonic_tm,f1,{format(macro_postharmonic_f1_tm)}')


if __name__ == '__main__':
    model = SPRDummyModel()
    data = SPR1DataModule(model=model)
    system = tx.LitSystem.from_args(
        model, data, defaults=dict(
            # path="./data/notxt.spr1.tar.gz",
            path="./data/notxt.mini10.spr1.tar.gz",
            # split_max_count=100,
            batch_size=-1))
    baseline(data)
