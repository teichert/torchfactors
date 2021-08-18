from __future__ import annotations

import argparse
import importlib
import inspect
import re
import sys
from argparse import ArgumentParser, Namespace
from inspect import Signature
from typing import (Any, Callable, Counter, Dict, Iterable, Mapping, Sequence,
                    Tuple, Type, TypeVar, cast)

from torch.nn import Module

T = TypeVar('T')


def get_class(name: str) -> type:
    r"""
    Returns the class with the given name:
    e.g.
    assert get_class('torchfactors.inferencers.bp.BP') is torchfactors.inferencers.bp.BP
    """
    module, bare_name = name.rsplit('.', 1)
    cls = getattr(importlib.import_module(module), bare_name)
    return cls


def str_to_bool(s: str) -> bool:
    return s.lower() == 'true'


legal_arg_types: Dict[Any, Callable] = {
    str: str,
    float: float,
    int: int,
    bool: str_to_bool,
    'str': str,
    'float': float,
    'int': int,
    'bool': str_to_bool
}


def DuplicateEntry(orig_name: str, duplicate_name: str):
    def throw_error(x):
        raise RuntimeError(f"--{duplicate_name} is just to help you see "
                           "that the argument belongs to multiple groups. "
                           f"Please use --{orig_name} instead to set the "
                           "argument.")
    return throw_error


def simple_arguments_and_info(f) -> Iterable[Tuple[str, Callable, Any]]:
    for arg, param in inspect.signature(f).parameters.items():
        possible_types = [param.annotation, type(param.default),
                          *re.split('[^a-zA-Z]+', str(param.annotation))]
        try:
            type_id = next(iter([t for t in possible_types if t in legal_arg_types]))
            # default = None if param.default is Signature.empty else param.default
            yield arg, legal_arg_types[type_id], param.default
        except StopIteration:
            pass


def simple_arguments(f) -> Iterable[str]:
    for arg, *_ in simple_arguments_and_info(f):
        yield arg


class _DoNothingAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default = argparse.SUPPRESS

    def __call__(self, parser, namespace, values, option_string=None):  # pragma: no cover
        pass


def add_arguments(cls: type, parser: ArgumentParser, arg_counts: Counter | None = None):
    # make sure that the parser knows how to ignore options
    parser.register('action', 'nothing', _DoNothingAction)
    if arg_counts is None:
        arg_counts = Counter()
    group = parser.add_argument_group(cls.__name__)
    for arg, use_type, default in simple_arguments_and_info(cls.__init__):  # type: ignore
        if arg in arg_counts:
            duplicate_name = f'{arg}{arg_counts[arg]}'
            group.add_argument(f'--{duplicate_name}', type=DuplicateEntry(arg, duplicate_name),
                               action='nothing',
                               help=f"(Note --{duplicate_name} is just place-holder "
                               f"to show relevance to this group. Please use --{arg} instead.)")
        else:
            used_default = None if default is Signature.empty else default
            group.add_argument(f'--{arg}', type=use_type, default=used_default,
                               required=default is Signature.empty)
        arg_counts[arg] += 1


class Config:
    # TODO: change interface: make classes be named param and defaults be first param?
    def __init__(self, *classes: type, parent_parser: ArgumentParser | None = None,
                 parse_args: Sequence[str] | str | None = None,
                 #  args_dict: Mapping[str, Any] | None = None,
                 defaults: Mapping[str, Any] | None = None):
        r"""
        Parameters:
            parent_parser: parser to attach things to and use for parsing
            parse_args: sequence of arguments to use as if commandline args
                - 'sys' means to get them from sys
                - None means to not use them
                - anything else means to use that instead
            defaults: values to use if value for given param not
                specified on the commandline (takes precedence
                over any defaults specified in the parser
                and can add vars not recognized by the parser)
        """
        self.classes = classes
        if parent_parser is None:
            parent_parser = ArgumentParser()
        self._parser = parent_parser
        arg_counts = Counter[str]()
        for cls in classes:
            add_arguments(cls, self.parser, arg_counts=arg_counts)
        if parse_args is None:
            self.parse_args = cast(Sequence[str], [])
        elif parse_args == 'sys':
            self.parse_args = sys.argv
        else:
            self.parse_args = parse_args
        # self.args_dict = args_dict
        self.defaults = {} if defaults is None else defaults
        self.parsed_args: Namespace | None = None

    def child(self, *args, **kwargs) -> Config:
        return Config(*self.classes, *args, **kwargs)

    @property
    def parser(self):
        return self._parser

    @property
    def args(self) -> Namespace:
        r"""
        Returns an argparse namespace with the contents of the config;
        1) override values given by `defaults` with arguments parsed from `parser_args`
        """
        if self.parsed_args is None:
            # back_off_args: Mapping[str, Any] = ChainMap(*[d for d in [
            #     self.args_dict, self.defaults
            # ] if d is not None])

            # Note: we need to do the default thing twice: once for those that
            # are in the parser (in order to override the parser defaluts) and
            # one for things that aren't even registered with the parser
            # (since they won't be included automatically)
            self.parser.set_defaults(**self.defaults)
            self.parsed_args = self.parser.parse_args(self.parse_args)
            unused_defaults = {k: v for k, v in self.defaults.items()
                               if k not in vars(self.parsed_args)}
            vars(self.parsed_args).update(unused_defaults)
        return self.parsed_args

    @property
    def dict(self) -> dict[str, Any]:
        return vars(self.args)

    # def create_with_help(self, cls: Type[T], **kwargs) -> T:
    #     try:
    #         return self.create(cls, **kwargs)
    #     except Exception as e:
    #         print(e, file=sys.stderr)
    #         self.parser.print_help()
    #         # sys.exit(1)
    #         raise SystemExit(1)

    # def maybe_add_config(self, cls: Type[T], kwargs: Dict[str, Any]):

    def create(self, cls: Type[T], **kwargs) -> T:
        r"""
        returns the instantiated class with the relevant simple settings
        overridden by the settings of this config;
        if config is a non-simple argument of the class initializer,
        then this config will be passed in (unless overridden by kwargs),
        finally kwargs override
        """
        d = self.dict
        simple_params = set(simple_arguments(cls.__init__))
        known_params = simple_params.intersection(d.keys())
        params = {k: d[k] for k in known_params}
        config_param = 'config'
        all_params = inspect.signature(cls.__init__).parameters.keys()
        if (config_param in all_params and
                config_param not in simple_params):
            params[config_param] = self
        params.update(kwargs)
        return cls(**params)  # type: ignore

    def create_from_name(self, classname: str, **kwargs) -> T:
        cls = get_class(classname)
        return self.create(cls, **kwargs)


__known_modules: Dict[str, Type[Module]] = {}


def register_module(cls: Type[Module] | None = None, name: str | None = None
                    ) -> Callable[[Type[Module]], Type[Module]] | Type[Module]:
    def decorate(nested: Type[Module]) -> Type[Module]:
        full_name = name if name is not None else '.'.join([nested.__module__, nested.__name__])
        __known_modules.setdefault(full_name, nested)
        return nested
    # if no class given,
    if cls is None:
        return decorate
    else:
        return decorate(nested=cls)


def build_module(name: str, **kwargs) -> Module:
    cls: type
    try:
        cls = __known_modules[name]
    except KeyError:
        cls = get_class(name)
    built = cls(**kwargs)
    return built
