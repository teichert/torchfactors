from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Sequence, TypeVar

from multimethod import multidispatch as overload

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

    # for torchtyping.details if we every want to support that
    # @classmethod
    # def tensor_repr(cls, tensor: Tensor):
    #     return "??"

    # def check(self, tensor: Tensor) -> bool:
    #     return bool((tensor < len(self.range)).all() and (tensor >= 0).all())


# should have been able to do @object.__new__ but mypy doesn't get it


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

    @overload
    def __call__(self, start: int, stop: int, step: int = 1) -> Domain:
        return SeqDomain(range(start, stop, step))

    @__call__.register
    def __call_with_one(self, stop: int) -> Domain:
        return SeqDomain(range(stop))


Range = _Range()
