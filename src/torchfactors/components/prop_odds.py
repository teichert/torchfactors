import itertools
from typing import Optional

from torch import Tensor
from torchfactors.components.at_least import KIsAtLeastJ
from torchfactors.components.linear_factor import LinearFactor, LinearTensor

from ..clique import (CliqueModel, make_binary_label_variables)
from ..components.tensor_factor import TensorFactor
from ..model import ParamNamespace
from ..subject import Environment
from ..variable import Var


class ProportionalOdds(CliqueModel):
    """
    Models a group of ordinal variables via a separate binary variable for
    each value of each variable (except the zero): the probability
    that the variable gets a label at least that large.
    """

    def factors(self, env: Environment, params: ParamNamespace,
                *variables: Var, input: Optional[Tensor] = None):

        # make a binary variable for each label of each variable (except 0)
        binary_variables = make_binary_label_variables(env, *variables)
        # weights are the same for all values (but different for different variable subsets)
        weight_tensor = LinearTensor(params.namespace(
            'binary-config-weight'), *[binary_variables[v, 1] for v in variables],
            minimal=True, bias=False)(input)
        # bias is different for each configuration
        for configuration in itertools.product(*[range(1, len(v.domain)) for v in variables]):
            config_vars = [binary_variables[v, value] for v, value in zip(variables, configuration)]
            yield LinearFactor(params.namespace(f'config-bias:{configuration}'),
                               *config_vars, input=None, bias=True, minimal=True)
            yield TensorFactor(*config_vars, tensor=weight_tensor)

        if len(variables) == 1:
            v = variables[0]
            for label in range(1, len(v.domain)):
                yield KIsAtLeastJ(v, binary_variables[v, label], label)
