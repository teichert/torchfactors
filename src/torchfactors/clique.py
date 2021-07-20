
from abc import ABC, abstractmethod
from typing import Dict, Hashable, Optional, Sequence, Tuple, cast

from torch.functional import Tensor

from torchfactors.types import ShapeType

from .components.linear_factor import ShapedLinear
from .domain import Range
from .model import ParamNamespace
from .subject import Environment
from .variable import TensorVar, Var, VarUsage


# TODO: might be good to have sub environments like paramnamespaces
class CliqueModel(ABC):

    @abstractmethod
    def factors(self, env: Environment, params: ParamNamespace, *
                variables: Var, input: Tensor): ...  # pragma: no cover
    # TODO: when needed later, will accept an additional (optional) input map from
    # variable subset (frozenset?) to tensor and the ability to specify `cat` or `add`
    # for the variables subsets that match a subset of interest.  This
    # will allow sufficient generalization as to simulate separate keys per variable
    # and per element of the same batch.


def make_binary_label_variables(env: Environment, *variables: Var, key: Hashable = None,
                                latent: bool = False, only_equals: bool = False
                                ) -> Dict[Tuple[Var, int], Var]:
    r"""
    Returns a dictionary from each (`orig` variable, `label`) to a binary
    variable that is 1 when `orig.tensor >= label` (or, if `only_equals`, then
    the value of the binary variable is 1 when `orig.tensor == label`) only
    strictly positive labels are modeled.

    """
    def binary_variable(v: Var, label: int):
        if latent:
            usage = v.usage.clone().masked_fill(v.usage != VarUsage.PADDING, VarUsage.LATENT)
        else:
            usage = v.usage.clone()
        if only_equals:
            labels = v.tensor == label
        else:
            labels = v.tensor >= label
        out = TensorVar(
            domain=Range(2),
            usage=usage,
            tensor=labels.int())
        return out
    binary_variables: Dict[Tuple[Var, int], Var] = {
        (v, label): env.variable((v, label, key), lambda: binary_variable(v, label))
        for v in variables
        for label in range(1, len(v.domain))
    }
    return binary_variables


def make_binary_threshold_variables(env: Environment, *variables: Var, key: Hashable = None,
                                    latent: bool = False
                                    ) -> Dict[Var, Var]:
    r"""
    Returns a dictionary from original variable to a binary version;
    if observed then the value is assigned so that
    roughly half of the labels are negative and the half are positive
    (with one more negative label than positive if there is a tie)
    """
    # TODO: allow different ways of thresholding
    def binary_variable(v: Var):
        if latent:
            usage = v.usage.clone().masked_fill(v.usage != VarUsage.PADDING, VarUsage.LATENT)
        else:
            usage = v.usage.clone()
        out = TensorVar(
            domain=Range(2),
            usage=usage,
            tensor=(v.tensor >= len(v.domain) / 2).int())
        return out
    binary_variables: Dict[Var, Var] = {
        v: env.variable((v, key), lambda: binary_variable(v))
        for v in variables
    }
    return binary_variables


def BinaryScoresModule(params: ParamNamespace, variables: Sequence[Var],
                       input_shape: Optional[ShapeType] = None,
                       input: Optional[Tensor] = None,
                       bias: bool = False) -> ShapedLinear:
    r"""
    Returns a module that will create a score for each binary configuration of
    the given variables (i.e. if there are 7 variables, will produce 2**7
    scores) as a function of the given input_shape (if input in provided, the
    batch_dimensions are removed from it).
    """
    if input_shape is None and input is not None:
        num_batch_dims = len(variables[0].shape)
        input_shape = input.shape[num_batch_dims:]

    def factory():
        out = ShapedLinear(input_shape=input_shape,
                           output_shape=(2,) * len(variables),
                           bias=bias)
        return out
    out = params.module(factory)
    return cast(ShapedLinear, out)
