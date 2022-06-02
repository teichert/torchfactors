from abc import ABC, abstractmethod
from typing import Iterable, Sequence, Tuple, TypeVar, Union

from torch import Tensor

from .factor import Factor, check_queries
from .variable import Var

T = TypeVar('T')


class Inferencer(ABC):
    @abstractmethod
    def product_marginals_(self, factors: Sequence[Factor], *queries: Sequence[Var],
                           normalize: bool = True, append_total_change: bool = False
                           ) -> Sequence[Tensor]: ...  # pragma: no cover

    def partition_with_change(self, factors: Iterable[Factor]) -> Tuple[Tensor, Tensor]:
        r"""
        convenience method for the log partition and the
        total kl between the prior messages and the current messages
        """
        logz, total_change = self.product_marginals(factors, (),
                                                    normalize=False,
                                                    append_total_change=True)
        return logz, total_change

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
                          normalize=True,
                          append_total_change=False,
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
        # print(queries)
        wrapped_queries = tuple([(q,) if isinstance(q, Var) else q for q in queries])
        # print(wrapped_queries)
        factors_list = list(factors)
        out = self.product_marginals_(factors_list, *wrapped_queries, normalize=normalize,
                                      append_total_change=append_total_change)
        return out

    def predict(self, factors: Iterable[Factor]) -> None:
        wrapped_factors = list(factors)
        # all_variables = list(set(list(v.origin for f in wrapped_factors for v in f)))
        all_variables = list(set(v for f in wrapped_factors for v in f))
        self.predict_(wrapped_factors, all_variables)

    # TODO: would be nice to have this do max-product inference rather than just independently
    # pick the max of each variable
    def predict_(self, factors: Sequence[Factor], variables: Sequence[Var]) -> None:
        queries = [(v,) for v in variables]
        marginals = self.product_marginals_(factors, *queries)
        for marginal, variable in zip(marginals, variables):
            variable.tensor = marginal.argmax(-1)
