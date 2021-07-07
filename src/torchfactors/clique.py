
from abc import ABC, abstractmethod

from torchfactors.subject import Environment

from .model import ParamNamespace
from .variable import Var


# TODO: might be good to have sub environments like paramnamespaces
class CliqueModel(ABC):

    @abstractmethod
    def factors(self, env: Environment, params: ParamNamespace, *variables: Var, **kwargs): ...
