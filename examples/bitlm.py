import torch
import torchfactors as tx


@tx.dataclass
class BitLanguageSubject(tx.Subject):
    bits: tx.Var = tx.VarField(tx.Range(2), tx.ANNOTATED)  # TensorType[..., len, 8]
    cases: tx.Var = tx.VarField(tx.Range(2), tx.ANNOTATED)  # TensorType[..., len]
    states: tx.Var = tx.VarField(tx.Range(30), tx.LATENT, shape=cases)  # TensorType[..., len]

    @classmethod
    def from_string(cls, x: str) -> 'BitLanguageSubject':
        return BitLanguageSubject(
            bits=tx.vtensor(
                [
                    [bit == '1' for bit in format(
                        ord(ch) % (2 ** 8),
                        '08b')]
                    for ch in x]),
            cases=tx.vtensor([ch.isupper() for ch in x]))

    def to_string(self) -> str:
        n = self.cases.shape[-1]

        def to_binstring(x: torch.Tensor):
            return ''.join(['1' if bit else '0' for bit in x.tolist()])
        bin_strings = [to_binstring(self.bits.tensor[..., i, :]) for i in range(n)]
        return ''.join(chr(int(bin_string, 2)) for bin_string in bin_strings)


class BitLanguageModel(tx.Model[BitLanguageSubject]):
    def factors(self, x: BitLanguageSubject):
        length, bits = x.bits.shape[-2:]
        for i in range(length):
            for bit in range(bits):
                yield tx.LinearFactor(self.namespace(('bit', bit)), x.bits[..., i, bit])
            #     yield tx.LinearFactor(self.namespace(('cased-bit', bit)),
            #                           x.bits[..., i, bit], x.cases[..., i], x.states[..., i])
            # yield tx.LinearFactor(self.namespace('case-unary'), x.cases[..., i])
            # yield tx.LinearFactor(self.namespace('state-unary'), x.states[..., i])
            # if i > 0:
            #     # yield tx.LinearFactor(self.namespace('case-pairwise'),
            #     #                       x.cases[..., i], x.cases[..., i])
            #     yield tx.LinearFactor(self.namespace('state-pairwise'),
            #                           x.states[..., i - 1], x.states[..., i])


if __name__ == '__main__':
    text = [
        "T",
        # "This is a test. Yes!",
        # "Another test."
    ]
    subjects = list(map(BitLanguageSubject.from_string, text))
    data = subjects[0]
    # data = BitLanguageSubject.stack(subjects)
    model = BitLanguageModel()
    system = tx.System(model, tx.BP())
    list(map(system.prime, subjects))

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    n, m = data.bits.shape[-2:]
    for i in range(100):
        optimizer.zero_grad()

        logz_free = system.product_marginal(data)
        print(logz_free)

        data.clamp_annotated_()
        logz_clamped = system.product_marginal(data)
        print(logz_clamped)

        loss = (logz_free - logz_clamped).sum()
        print(loss)

        loss.backward()
        optimizer.step()
        data.unclamp_annotated_()

        marginals = system.product_marginals(data, *[(data.bits[..., i, j],)
                                                     for i in range(n)
                                                     for j in range(m)])
        print(marginals)
        predicted = system.predict(data)
        print()
        print('data')
        print(data.to_string())
        print(data.bits.tensor)
        print(data.cases.tensor)
        print()
        print('predicted:')
        print(predicted.to_string())
        print(predicted.bits.tensor)
        print(predicted.cases.tensor)
