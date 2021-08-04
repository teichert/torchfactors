from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Hashable, Sequence, Tuple, TypeVar, overload

from multimethod import multimethod

T = TypeVar('T')


class Domain(ABC):
    r"""
    The possible values (and corresponding indices) that a variable can be
    assigned.
    """

    OPEN: ClassVar[Domain]

    @abstractmethod
    def __iter__(self): ...  # pragma: no cover

    @abstractmethod
    def __len__(self): ...  # pragma: no cover


class __OpenDomain(Domain):
    r"""
    represents that any value is okay (cannot be used for factor arguments)
    """

    def __iter__(self):
        raise ValueError("cannot iterate over open domain")

    def __len__(self):
        raise ValueError("no size for open domain")


Domain.OPEN = __OpenDomain()


class FlexDomain(Domain):
    r"""
    Represents a set of labels that can grow (until it it frozen).
    Useful for open-ended label-sets.  In order for parameters
    to still be meaningful, the flexdomain would need to be
    saved along with the model since encountering the labels in
    a different order could change their id.
    """

    def __init__(self, name: str, unk: bool = False):
        self.name = name
        self.frozen = False
        if unk:
            self.unk = object()
            self.unk_id = 0
            self.values = [self.unk]
            self.value_to_id = {self.unk: 0}
        else:
            self.unk = None
            self.unk_id = -1
            self.values = []
            self.value_to_id = {}

    def freeze(self):
        self.frozen = True

    def get_id(self, value: Hashable, warn=True) -> int:
        default = self.unk_id if self.frozen else len(self.values)
        id = self.value_to_id.setdefault(value, default)
        if warn and id == self.unk_id:
            warnings.warn("unknown value on frozen flex domain", RuntimeWarning)
        if id >= len(self.values):
            self.values.append(value)
        return id

    def get_value(self, id: int) -> Hashable:
        if id < len(self.values) and id >= 0:
            return self.values[id]
        else:
            warnings.warn("id out of range for domain, returning unk value", RuntimeWarning)
            return self.unk

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def to_list(self) -> Tuple[str, bool, Sequence[Hashable]]:
        return self.name, self.unk is not None, self.values[(1 + self.unk_id):]

    @staticmethod
    def from_list(input: Tuple[str, bool, Sequence[Hashable]]) -> FlexDomain:
        r"""
        returns a FlexDomain with the given string as the domain name,
        the bool indicates if the domain should include an unknown (unk)
        value
        """
        name, unk, values = input
        domain = FlexDomain(name, unk=unk)
        for value in values:
            domain.get_id(value)
        domain.freeze()
        return domain


@dataclass(frozen=True)
class SeqDomain(Domain):
    r"""Wraps an underlying sequence of possible values"""

    range: Sequence

    def __iter__(self):
        return iter(self.range)

    def __len__(self):
        return len(self.range)

    def __repr__(self):
        return f"SeqDomain{self.range}"


class _Range:
    r"""
    Factory for an integer domain backed by a python range object.

    e.g.
    > Range(10)
    Domain(range(0, 10))
    > Range(5, 10)
    Domain(range(5, 10))
    > Range(10)
    Domain(range(0, 10))
    > Range[5:10]
    Domain(range(5, 10))
    """

    @staticmethod
    def __getitem__(key: slice) -> SeqDomain:
        return SeqDomain(range(
            key.start if key.start is not None else 0,
            key.stop if key.stop is not None else 0,
            key.step if key.step is not None else 1))

    @multimethod
    def ___call__(self, start: int, stop: int, step: int = 1) -> Domain:
        return SeqDomain(range(start, stop, step))

    @___call__.register
    def _(self, stop: int) -> Domain:
        return SeqDomain(range(stop))

    @overload
    def __call__(self, start: int, stop: int, step: int = 1) -> Domain: ...  # pragma: no cover

    @overload
    def __call__(self, stop: int) -> Domain: ...  # pragma: no cover

    def __call__(self, *args, **kwargs):
        return self.___call__(*args, **kwargs)


Range = _Range()
