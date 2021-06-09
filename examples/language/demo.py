from dataclasses import dataclass
import torchfactors as tx
from torchfactors import Range as myrange


@dataclass
class Utterance:
    items: tx.VariableTensor #[tx.Range[5]] # TODO: make this work


class MyModel(tx.Model[Utterance]):
    
    def factors(self, subject: Utterance):
        items = subject.items
        hidden = tx.Variable(items.tensor, tx.VariableType.LATENT, tx.Range[100])
        for i in range(len(items.tensor)):
            if i > 0:
                yield tx.LinearFactor([hidden[i], hidden[i-1]], self.params('transition'))
            yield tx.LinearFactor([hidden[i], items[i]], self.params('emission'))

