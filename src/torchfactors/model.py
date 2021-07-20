from __future__ import annotations

from typing import (Callable, Dict, Generic, Hashable, Iterable, List,
                    Optional, Sequence, cast, overload)

import torch
from multimethod import multimethod
from torch import Tensor
from torch.nn import Module, ModuleDict, ParameterDict
from torch.nn.init import xavier_uniform_, zeros_
from torch.nn.parameter import Parameter

from .domain import FlexDomain
from .factor import Factor
from .subject import Environment, SubjectType
from .types import ShapeType


class ParamNamespace:
    """
    Corresponds to a particular model parameter or module which is associated
    with a unique key and the Model that actually stores everything
    """

    def __init__(self, model: 'Model', key: Hashable):
        self.model = model
        self.key = key

    def namespace(self, key: Hashable) -> ParamNamespace:
        r"""
        Returns a new namespace branch labeled with the given key
        """
        return ParamNamespace(
            model=self.model, key=(self.key, key))

    @multimethod
    def _parameter(self, shape: ShapeType,
                   initialization: Optional[Callable[[Tensor], None]
                                            ] = None
                   ) -> Parameter:
        if initialization is None:
            if len([d for d in list(torch.Size(shape)) if d > 1]) < 2:
                def initialization(t): return zeros_(t)
            else:
                def initialization(t): return xavier_uniform_(t)
        non_none_initialization = cast(Callable[[Tensor], None], initialization)

        def gen_param():
            tensor = torch.empty(shape)
            non_none_initialization(tensor)
            return Parameter(tensor)
        return self.model._get_param(self.key, check_shape=shape, default_factory=gen_param)

    @_parameter.register
    def get_saved_parameter(self) -> Parameter:
        return self.model._get_param(self.key)

    # # extra stubs are in here to help vscode intellisense work without making mypy mad
    @overload
    def parameter(self, shape: ShapeType,
                  initialization: Optional[Callable[[Tensor], None]] = None
                  ) -> Parameter: ...  # pragma: no cover

    @overload
    def parameter(self) -> Parameter: ...  # pragma: no cover

    def parameter(self, *args, **kwargs):
        return self._parameter(*args, **kwargs)

    def module(self, constructor: Optional[Callable[[], torch.nn.Module]] = None) -> Module:
        return self.model._get_module(self.key, default_factory=constructor)


class Model(torch.nn.Module, Generic[SubjectType]):

    def __init__(self):
        super(Model, self).__init__()
        self._model_parameters = ParameterDict()
        self._model_modules = ModuleDict()
        # TODO: these domains are not actually saved anywhere to file
        self._domains: Dict[str, FlexDomain] = {}

    def domain_ids(self, domain: FlexDomain, values: Sequence[Hashable]) -> torch.Tensor:
        domain = self._domains.setdefault(domain.name, domain)
        return torch.tensor([domain.get_id(value) for value in values])

    def domain_values(self, domain: FlexDomain, ids: torch.Tensor) -> Sequence[Hashable]:
        domain = self._domains.setdefault(domain.name, domain)
        return [domain.get_value(id) for id in ids.tolist()]

    def namespace(self, key: Hashable) -> ParamNamespace:
        return ParamNamespace(self, key)

    def factors(self, subject: SubjectType) -> Iterable[Factor]:
        return []

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
                raise KeyError(
                    f"This key has already been used with different shape: "
                    f"{check_shape} vs {param.shape}")
            return param

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

    def __call__(self, subject: SubjectType) -> List[Factor]:
        subject.environment = Environment()
        return list(self.factors(subject))
