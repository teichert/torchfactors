from abc import ABC, abstractclassmethod
from typing import Iterable, Sequence, TypeVar, Union

from torch.functional import Tensor

from .factor import Factor, check_queries
from .variable import Var

T = TypeVar('T')


class Inferencer(ABC):
    @abstractclassmethod
    def product_marginals_(self, factors: Sequence[Factor], *queries: Sequence[Var],
                           normalize=True) -> Sequence[Tensor]: ...

    def product_marginal(self, factors: Iterable[Factor],
                         query: Union[Sequence[Var], Var, None] = None,
                         normalize=True) -> Tensor:
        r"""
        convenience method for a single product_marginal query
        """
        if query is None:
            query = ()
        elif isinstance(query, Var):
            query = (query,)
        out, = self.product_marginals(factors, query,
                                      normalize=normalize)
        return out

    def product_marginals(self,
                          factors: Iterable[Factor],
                          *queries: Union[Var, Sequence[Var]],
                          normalize=True
                          ) -> Sequence[Tensor]:
        r"""
        Returns marginals corresponding to the specified queries.
        If `normalize` is specified as True (note, this is usually more efficient
        than not normalizing), then each marginal tensor will logsumexp to 0
        with the exception of a query for () which is used to represent a request
        for the log partition function. If normalize is False,
        then the partition function will be used to unnormalize
        the normalized belief.
        The idea is that the normalized belief came from dividing the unnormalized by Z,
        so I get the unnormalized back by multiplying by Z:
        b = m / z => m = b * z
        """
        check_queries(queries)
        wrapped_queries = tuple([(q,) if isinstance(q, Var) else q for q in queries])
        return self.product_marginals_(list(factors), *wrapped_queries, normalize=normalize)

    def predict_(self, factors: Sequence[Factor], variables: Sequence[Var]) -> None:
        marginals = self.product_marginals_(factors, variables)
        for marginal, variable in zip(marginals, variables):
            variable.tensor = marginal.argmax()
