

from typing import Iterable, cast

import torch
import torchfactors_lightning as tx
from pytorch_lightning.trainer.trainer import Trainer
from torchfactors.factor import Factor


@tx.dataclass
class IncompleteText(tx.Subject):
    letter_domain = tx.FlexDomain("letters")
    text: tx.Var = tx.VarField(letter_domain)


class MyModel(tx.Model[IncompleteText]):

    def factors(self, subject: IncompleteText) -> Iterable[Factor]:
        # yield from [  # two copies to make the single factor accept messages from itself
        #     tx.LinearFactor(self.namespace('pair'),
        #                     subject.text[..., :-1], subject.text[..., 1:])] * 2
        for i in range(2, subject.text.shape[-1]):
            yield tx.LinearFactor(self.namespace('triple'),
                                  subject.text[..., i - 2],
                                  subject.text[..., i - 1],
                                  subject.text[..., i])
        for i in range(1, subject.text.shape[-1]):
            yield tx.LinearFactor(self.namespace('pair'),
                                  subject.text[..., i - 1], subject.text[..., i])
        for i in range(2, subject.text.shape[-1]):
            yield tx.LinearFactor(self.namespace('skip-pair'),
                                  subject.text[..., i - 2], subject.text[..., i])
        for i in range(3, subject.text.shape[-1]):
            yield tx.LinearFactor(self.namespace('3-skip-pair'),
                                  subject.text[..., i - 2], subject.text[..., i])


class MyData(tx.lightning.DataModule[IncompleteText]):

    # train_text = ("# This program will read in this text and "
    #               "then try to fill in ---- goes in the dashes.")
    # train_text = "12312--23"
    train_text = "aabbbaa---a-b-b"
    # train_text = "cat rat sat m-t p-- vat"

    def __init__(self, model: tx.Model[IncompleteText]):
        super().__init__()
        self.model = model

    def subject_from_str(self, text: str) -> IncompleteText:
        ch = text.replace('-', '')[0]
        ids = self.model.domain_ids(IncompleteText.letter_domain,
                                    text.replace('-', ch))
        is_latent = torch.tensor([c == '-' for c in text])
        usage = (torch.full_like(ids, tx.VarUsage.ANNOTATED)
                      .masked_fill(is_latent, tx.VarUsage.LATENT))
        return IncompleteText(tx.TensorVar(tensor=ids, usage=usage))

    def setup(self, stage=None):
        self.train = tx.ListDataset([self.subject_from_str(MyData.train_text)])


class MySystem(tx.LitSystem[IncompleteText]):

    # def training_step(self, *args, **kwargs):
    #     loss = super().training_step(*args, **kwargs)
    #     self.log_info['orig'] = float(loss)
    #     size = sum(p.norm(1) for p in self.model.parameters())
    #     adjusted = loss + 100.0 * size
    #     # print(loss, adjusted)
    #     self.log_info['adjusted'] = float(adjusted)
    #     return adjusted

    def on_train_epoch_end(self, unused=None):
        x = cast(MyData, self.data).subject_from_str(data.train_text)
        params = ''.join(f'{min(9, int(abs(f)*100))*(-1 if f < 0 else 1)}'
                         for p in self.model.parameters()
                         for f in p.tolist())
        # self.log_info['prediction1'] = ''.join(map(str, out.text.tensor.tolist()))
        # self.log_info['usage'] = ''.join(map(str, out.text.usage.tolist()))
        self.log_info['params'] = params

        x.unclamp_annotated()
        out = system.system.predict(x)
        self.log_info['free_prediction'] = ''.join(
            map(str, map(IncompleteText.letter_domain.get_value, out.text.tensor.tolist())))

        x.clamp_annotated()
        out = system.system.predict(x)
        # self.log_info['prediction1'] = ''.join(map(str, out.text.tensor.tolist()))
        # self.log_info['usage'] = ''.join(map(str, out.text.usage.tolist()))
        self.log_info['prediction'] = ''.join(
            map(str, map(IncompleteText.letter_domain.get_value, out.text.tensor.tolist())))


model = MyModel()
data = MyData(model)
config = tx.Config(MySystem, MyData, torch.optim.AdamW, tx.BP, Trainer,
                   defaults=dict(lr=0.1, passes=1,
                                 penalty_coeff=100.0,
                                 weight_decay=1.0,
                                 optimizer='AdamW'))
system = config.create(MySystem, model=model, data=data)
trainer = Trainer()
trainer.fit(system)
