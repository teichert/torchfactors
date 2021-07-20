from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Hashable, Sequence, TypeVar, overload

from multimethod import multimethod

T = TypeVar('T')


class Domain(ABC):
    r"""
    The possible values (and corresponding indices) that a variable can be
    assigned.
    """

    OPEN: ClassVar[Domain]

    @abstractmethod
    def __iter__(self): ...

    @abstractmethod
    def __len__(self): ...


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

    def __init__(self, name: str):
        self.name = name
        self.frozen = False
        self.unk = object()
        self.values = [self.unk]
        self.value_to_id = {self.unk: 0}

    def freeze(self):
        self.frozen = True

    def get_id(self, value: Hashable) -> int:
        default = 0 if self.frozen else len(self.values)
        id = self.value_to_id.setdefault(value, default)
        if id >= len(self.values):
            self.values.append(value)
        return id

    def get_value(self, id: int) -> Hashable:
        if id < len(self.values):
            return self.values[id]
        else:
            return self.unk

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


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
        # if isinstance(key, int):
        #     return SeqDomain(range(key))
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
    def __call__(self, start: int, stop: int, step: int = 1) -> Domain: ...

    @overload
    def __call__(self, stop: int) -> Domain: ...

    def __call__(self, *args, **kwargs):
        return self.___call__(*args, **kwargs)


Range = _Range()
