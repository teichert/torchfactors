from __future__ import annotations

# from functools import lru_cache
from typing import List, Sequence

from torch import Tensor

from ..factor import Factor
from ..inferencer import Inferencer
from ..variable import Var


class BruteForce(Inferencer):

    def product_marginals_(self, factors: Sequence[Factor], *queries: Sequence[Var],
                           normalize: bool = True, append_total_change: bool = False
                           ) -> Sequence[Tensor]:
        first, *others = factors
        marginals: List[Tensor] = [*first.product_marginals(*queries, other_factors=others)]
        if normalize:
            logz = first.product_marginal(other_factors=others)
            # since it is the first dimension that matches rather than the last,
            # we transpose first and then untranspose
            marginals = [m if q == () else (m.T - logz.T).T for m, q in zip(marginals, queries)]
        if append_total_change:
            marginals.append(Tensor(0.0))
        return marginals
