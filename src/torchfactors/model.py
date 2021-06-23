from __future__ import annotations

from typing import (Callable, Dict, Generic, Hashable, Iterable, List,
                    Optional, TypeVar)

import torch
from torch import Tensor
from torch.nn import Module, ModuleDict, ParameterDict
from torch.nn.init import xavier_uniform_, zeros_
from torch.nn.parameter import Parameter

from .domain import Domain
from .factor import Factor
from .types import ShapeType

T = TypeVar('T')


class ParamNamespace:
    """
    Corresponds to a particular model parameter or module which is associated
    with a unique key and the Model that actually stores everything
    """

    def __init__(self, model: Model, key: Hashable):
        self.model = model
        self.key = key

    def namespace(self, key: Hashable) -> ParamNamespace:
        r"""
        Returns a new namespace branch labeled with the given key
        """
        return ParamNamespace(
            model=self.model, key=(self.key, key))

    def parameter(self, shape: ShapeType,
                  initialization: Optional[Callable[[Tensor], None]
                                           ] = None
                  ) -> Tensor:
        if initialization is None:
            if len([d for d in list(torch.Size(shape)) if d > 1]) < 2:
                def initialization(t): return zeros_(t)
            else:
                def initialization(t): return xavier_uniform_(t)

        def gen_param():
            tensor = torch.zeros(shape)
            if initialization is not None:
                initialization(tensor)
            return Parameter(tensor)
        return self.model._get_param(self.key, check_shape=shape, default_factory=gen_param)

    def module(self, constructor: Optional[Callable[[], torch.nn.Module]] = None):
        return self.model._get_module(self.key, default_factory=constructor)


class Model(torch.nn.Module, Generic[T]):

    def __init__(self):
        super(Model, self).__init__()
        self._model_factor_generators: List[Callable[[T], Iterable[Factor]]] = []
        self._model_domains: Dict[Hashable, Domain] = {}
        self._model_parameters = ParameterDict()
        self._model_modules = ModuleDict()

    def domain(self, key: Hashable) -> Domain:
        return self._model_domains.setdefault(key, Domain.OPEN)

    def namespace(self, key) -> ParamNamespace:
        return ParamNamespace(self, key)

    def factors_from(self, factor_generator: Callable[[T], Iterable[Factor]]) -> None:
        self._model_factor_generators.append(factor_generator)

    def factors(self, subject: T) -> Iterable[Factor]:
        for gen in self._model_factor_generators:
            yield from gen(subject)

    def _get_param(self, key: Hashable, check_shape: Optional[ShapeType] = None,
                   default_factory: Optional[Callable[[], Parameter]] = None
                   ) -> Parameter:
        repr = f'{key}:{hash(key)}'
        if repr in self._model_modules:
            raise KeyError(
                "trying to get a parameter with a key "
                f"already used for a module: {repr}")
        if repr not in self._model_parameters:
            if default_factory is not None:
                param = default_factory()
                self._model_parameters[repr] = param
                return param
            else:
                raise KeyError("no param at that key and no default factory given")
        else:
            param = self._model_parameters[repr]
            if check_shape is not None and check_shape != param.shape:
                raise ValueError(
                    f"This key has already been used with different shape: "
                    f"{check_shape} vs {param.shape}")
            return param

    # def set_param(self, key: Hashable, value: Parameter, first=True) -> None:
    #     repr = f'{key}:{hash(key)}'
    #     if first and repr in self._model_parameters:
    #         raise ValueError(f"This key has already been used!: {repr}")
    #     self._model_parameters[repr] = value

    def _get_module(self, key: Hashable,
                    default_factory: Optional[Callable[[], Module]] = None
                    ) -> Module:
        repr = f'{key}:{hash(key)}'
        if repr in self._model_parameters:
            raise KeyError(
                "trying to get a module with a key "
                f"already used for a paramter: {repr}")
        if repr not in self._model_modules:
            if default_factory is not None:
                module = default_factory()
                self._model_modules[repr] = module
                return module
            else:
                raise KeyError("no module at that key and no default factory given")
        else:
            return self._model_modules[repr]

    def __call__(self, subject: T) -> List[Factor]:
        return list(self.factors(subject))
