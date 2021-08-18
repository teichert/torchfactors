from __future__ import annotations

import json
from collections import OrderedDict
from typing import (Any, Callable, Dict, Generic, Hashable, Iterable, List,
                    Optional, Sequence, Tuple, cast, overload)

import torch
from config import build_module
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

    def __init__(self, model: Model, key: Hashable):
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

    @property
    def _key_repr(self) -> str:
        return self.model.get_key_repr(self.key)

    def module(self, cls: type | None = None, **kwargs) -> Module:
        if cls is None:
            return self.model._get_module(self.key)
        else:
            non_none_cls = cls

            def setup_new_module():
                self.model._module_constructors[self._key_repr] = (
                    '.'.join([non_none_cls.__module__, non_none_cls.__name__]), kwargs)
                return non_none_cls(**kwargs)
            return self.model._get_module(self.key, setup_new_module)


class Model(torch.nn.Module, Generic[SubjectType]):

    def __init__(self,
                 model_state_dict_path: str | None = None,
                 checkpoint_path: str | None = None,
                 ):
        super(Model, self).__init__()
        self._model_parameters = ParameterDict()
        self._model_modules = ModuleDict()
        self._domains: Dict[str, FlexDomain] = {}
        self._module_constructors: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        if model_state_dict_path is not None:
            self.load_state_dict(torch.load(model_state_dict_path))
        elif checkpoint_path is not None:
            check_point = torch.load(checkpoint_path)
            state_dict = check_point['state_dict']
            only_model_state_dict_items: List[Tuple[str, Any]] = []
            for k, v in state_dict.items():
                starting = 'model.'
                if k.startswith(starting):
                    k = k[len(starting):]
                only_model_state_dict_items.append((k, v))
            self.load_state_dict(OrderedDict(only_model_state_dict_items))  # type: ignore

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + '_domains'] = [d.to_list() for _, d in self._domains.items()]
        destination[prefix + '_module_constructors'] = self._module_constructors.copy()
        return destination

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._model_parameters = ParameterDict()
        self._model_modules = ModuleDict()
        domains = state_dict.pop(prefix + '_domains')
        self._domains = {name: FlexDomain.from_list((name, unk, values))
                         for name, unk, values in domains}
        self._module_constructors = state_dict.pop(prefix + '_module_constructors')
        for key, (name, kwargs) in self._module_constructors.items():
            self._model_modules[key] = build_module(name, **kwargs)
        for key in state_dict:
            param_start = prefix + '_model_parameters.'
            if key.startswith(param_start):
                name = key[len(param_start):].split('.')[0]
                self._model_parameters[name] = torch.nn.parameter.Parameter(state_dict[key])
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                             missing_keys, unexpected_keys, error_msgs)

    def domain(self, name: str) -> FlexDomain:
        domain = self._domains.get(name, None)
        if domain is None:
            domain = self._domains.setdefault(name, FlexDomain(name))
        return domain

    def domain_ids(self, domain: FlexDomain, values: Sequence[Hashable], warn=True) -> torch.Tensor:
        domain = self._domains.setdefault(domain.name, domain)
        ids = torch.tensor([domain.get_id(value, warn=warn) for value in values])
        return ids

    def domain_values(self, domain: FlexDomain, ids: torch.Tensor) -> Sequence[Hashable]:
        domain = self._domains.setdefault(domain.name, domain)
        return [domain.get_value(id) for id in ids.tolist()]

    def namespace(self, key: Hashable) -> ParamNamespace:
        return ParamNamespace(self, key)

    def factors(self, subject: SubjectType) -> Iterable[Factor]:
        return []

    def get_key_repr(self, key: Any) -> str:
        return json.dumps(key)

    def _get_param(self, key: Hashable, check_shape: Optional[ShapeType] = None,
                   default_factory: Optional[Callable[[], Parameter]] = None
                   ) -> Parameter:
        repr = self.get_key_repr(key)
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
        repr = self.get_key_repr(key)
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
