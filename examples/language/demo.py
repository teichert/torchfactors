import torchfactors as tx


@tx.subject
class Utterance:
    items = tx.VarTensor(tx.Range[5])  # TensorType['batch': ..., 'index', int]


class MyModel(tx.Model[Utterance]):

    def factors(self, subject: Utterance):
        items = subject.items
        hidden = tx.VarTensor(items.tensor, tx.VarUsage.LATENT, tx.Range[100])
        for i in range(len(items.tensor)):
            if i > 0:
                yield tx.LinearFactor([hidden[i], hidden[i-1]], self.params('transition'))
            yield tx.LinearFactor([hidden[i], items[i]], self.params('emission'))
