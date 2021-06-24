import copy
from typing import Generic, Sequence, Union

from torch.functional import Tensor

from .inferencer import Inferencer
from .model import Model
from .subject import SubjectType
from .variable import Var


class ModelInferencer(Generic[SubjectType]):
    r"""
    Knows how to get marginals of variables in the given subject
    and/or how to predict an annotated copy given an input subject
    """

    def __init__(self, model: Model[SubjectType], inferencer: Inferencer):
        self.model = model
        self.inferencer = inferencer

    def predict(self, x: SubjectType) -> SubjectType:
        r"""
        Returns a copy of the subject x with the values of the variables replaced with
        the prediction of this inferencer
        """
        x = copy.deepcopy(x)
        factors = list(self.model(x))
        self.inferencer.predict_(factors, x.variables)
        return x

    def product_marginal(self, x: SubjectType, query: Union[Sequence[Var], Var, None] = None,
                         normalize=True) -> Tensor:
        r"""
        convenience method for a single product_marginal query
        """
        return self.inferencer.product_marginal(self.model(x), query, normalize=normalize)

    def product_marginals(self, x: SubjectType,
                          *queries: Union[Var, Sequence[Var]],
                          normalize=True) -> Sequence[Tensor]:
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
        return self.inferencer.product_marginals(self.model(x), *queries,
                                                 normalize=normalize)
