import copy
from typing import Generic, Iterable, Sequence, Tuple, Union, cast

from torch.functional import Tensor
from tqdm import tqdm  # type: ignore

from .inferencer import Inferencer
from .model import Model
from .subject import Subject, SubjectType
from .variable import Var


class System(Generic[SubjectType]):
    r"""
    Knows how to get marginals of variables in the given subject
    and/or how to predict an annotated copy given an input subject
    """

    def __init__(self, model: Model[SubjectType], inferencer: Inferencer):
        self.model = model
        self.inferencer = inferencer

    def prime(self, x: Union[SubjectType, Iterable[SubjectType]]) -> None:
        r"""
        Applies the model to the given subject without reporting anything.
        This can be useful for initializing all paramters prior to building
        an optimizer.
        """
        if isinstance(x, Subject):
            for f in self.model(cast(SubjectType, x)):
                f.dense
                f.clear_cache()
        else:
            for subject in tqdm(cast(Iterable[SubjectType], x)):
                self.prime(subject)

    def predict(self, x: SubjectType) -> SubjectType:
        r"""
        Returns a copy of the subject x with the values of the variables replaced with
        the prediction of this inferencer
        """
        x = copy.deepcopy(x)
        factors = self.model(x)
        self.inferencer.predict(factors)
        return x

    def partition_with_change(self, x: SubjectType) -> Tuple[Tensor, Tensor]:
        r"""
        convenience method for the log partition and the
        total kl between the prior messages and the current messages
        """
        factors = self.model(x)
        out = self.inferencer.partition_with_change(factors)
        return out

    def product_marginal(self, x: SubjectType, query: Union[Sequence[Var], Var, None] = None,
                         normalize=True) -> Tensor:
        r"""
        convenience method for a single product_marginal query
        """
        factors = self.model(x)
        marginal = self.inferencer.product_marginal(factors, query, normalize=normalize)
        return marginal

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
