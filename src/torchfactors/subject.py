from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from typing import (Callable, Dict, FrozenSet, Generic, Hashable, List,
                    Optional, Sequence, Sized, Tuple, TypeVar, Union, cast)

import torch
from torch._C import Size
from torch.utils.data import DataLoader, Dataset

from .domain import Domain
from .factor import Factor
from .types import ShapeType
from .variable import TensorVar, Var, VarField, VarUsage

SubjectType = TypeVar('SubjectType', bound='Subject')


ExampleType = TypeVar('ExampleType')


@ dataclass
class ListDataset(Dataset, Generic[ExampleType]):
    examples: List[ExampleType]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> ExampleType:
        return self.examples[index]


class Environment(object):
    """
    allows additional variables and factors to be
    added without duplicates
    """

    def __init__(self):
        self.variables: Dict[Hashable, Var] = {}
        self.factors: Dict[Hashable, Factor] = {}

    def variable(self, key: Hashable, factory: Optional[Callable[[], Var]] = None) -> Var:
        try:
            out = self.variables[key]
            return out
        except KeyError:
            if factory is None:
                raise KeyError(f"key {key} not found and no factory provided")
            else:
                out = factory()
                return self.variables.setdefault(key, out)

    def factor(self, key: Hashable, factory: Optional[Callable[[], Factor]] = None) -> Factor:
        try:
            out = self.factors[key]
            return out
        except KeyError:
            if factory is None:
                raise KeyError(f"key {key} not found and no factory provided")
            else:
                out = factory()
                return self.factors.setdefault(key, out)


@ dataclass
class Subject:
    r"""
    Represents something to be modeled.
    Subjects can automatically be stacked in a way that pads tensor variables.
    Subclasses of Subject should also be decorated with `@dataclass`.
    VarFields can be used to give variable information that will not
    have to be repeated for each instance.

    """
    # TODO: [ ] a model should be agnostic whether (and how many times) this
    # subject has been stacked the idea is that a model should be a model of a
    # single one of these; [ ] there should be a canonical way to slice with
    # respect to the original boundary; [ ] the subject should know its batch
    # dimensions; when it comes to variables, non batch dimensions except the
    # last one should be a short-hand for separate factors (hopefully for which
    # messages can be computed and sent in parallel?)
    is_stacked: bool = field(init=False, default=False)
    __length: int = field(init=False, default=1)
    __lists: Dict[str, List[object]] = field(init=False, default_factory=dict)
    __varset: FrozenSet = field(init=False, default=frozenset())
    environment: Environment = field(init=False, default_factory=Environment)

    def __len__(self) -> int:
        return self.__length

    def list(self, key: str) -> List[object]:
        if self.is_stacked:
            return self.__lists[key]
        else:
            return[getattr(self, key)]

    def init_variables(self):
        self.__variables: List[Var] = []
        cls = type(self)
        vars = []
        cls_attr_id_to_var: Dict[int, TensorVar] = {}
        # basic idea: for each field, check if the class_variable has more information
        # than in this particular object; if it does, then copy over.
        # class version should never have tensor-like usage nor an actual tensor.
        # instance version should always have a tensor-based usage and an actual tensor.
        for f in fields(cls):
            attr_name = f.name
            if not hasattr(cls, attr_name):
                continue
            var_field = getattr(cls, attr_name)
            if isinstance(var_field, VarField):
                var_instance = getattr(self, attr_name)
                if var_instance is not None and not isinstance(var_instance, Var):
                    raise TypeError(
                        f"Your subject value corresponding to the variable field: {attr_name} "
                        f"was not a variable, instead it was a {type(var_instance)}. "
                        "Did you forget to wrap with a TensorVar?")
                if not isinstance(var_instance, TensorVar):
                    if var_field._usage != VarUsage.LATENT:
                        raise ValueError("only latent variables can be left implicit; "
                                         "make sure that you pass in all required variables "
                                         "to the subject constructor")
                    var_instance = TensorVar()
                    setattr(self, attr_name, var_instance)
                cls_attr_id_to_var[id(var_field)] = var_instance
                if var_field._shape is not None:
                    specified_shape: Union[Size, Tuple[int, ...]]
                    if isinstance(var_field._shape, Var):
                        source_var = cls_attr_id_to_var[id(var_field._shape)]
                        specified_shape = source_var.shape
                    else:
                        specified_shape = var_field._shape
                    if (var_instance._tensor is not None and
                            specified_shape != var_instance._tensor.shape):
                        raise ValueError("the shape of the tensor you provided "
                                         "does not match the pre-specified shape: "
                                         f"found: {var_instance._tensor.shape}, "
                                         f"expected: {specified_shape}")
                    # shape can be delegated to earlier field
                    elif isinstance(var_field._shape, VarField):
                        source_var = cls_attr_id_to_var[id(var_field._shape)]
                        # source should already have tensor with shape by now
                        var_instance._tensor = var_field._init(source_var.shape)
                    else:
                        var_instance._tensor = var_field._init(cast(ShapeType, var_field._shape))
                elif var_instance._tensor is None:
                    raise ValueError(
                        "need to specify an actual tensor (or a shape)"
                        "for every variable in the subject")
                if var_field._domain is not Domain.OPEN:
                    var_instance._domain = var_field._domain
                if var_field._usage is not None:
                    var_instance.set_usage(var_field._usage)
                vars.append(attr_name)
                self.__variables.append(var_instance)
        self.__varset = frozenset(vars)

    @property
    def variables(self):
        r"""returns a list of all variable fields defined for this subjec"""
        return self.__variables

    @staticmethod
    def stack(subjects: Sequence[SubjectType]) -> SubjectType:
        if not subjects:
            raise ValueError(
                "Your list of subjects needs to have at least one in it")
        first = subjects[0]
        if first.is_stacked:
            raise ValueError(
                "Not allowed to stack already stacked subjects")
        out = copy.deepcopy(first)
        out.is_stacked = True
        # cls = type(out)
        generic_fields = set(field.name for field in fields(Subject))
        my_fields = set(field.name for field in fields(first)) - first.__varset - generic_fields
        for attr_name in first.__varset:
            # attr = cast(TensorVar, getattr(out, attr_name))
            stacked = TensorVar.pad_and_stack([
                cast(TensorVar, getattr(subject, attr_name))
                for subject in subjects])
            setattr(out, attr_name, stacked)
        for attr_name in my_fields:
            # attr = getattr(out, attr_name)
            out.__lists[attr_name] = [
                getattr(subject, attr_name)
                for subject in subjects]
        out.__length = len(subjects)
        return out

    def to_device(self: SubjectType, device: torch.device) -> SubjectType:
        if self.variables and self.variables[0].tensor.device != device:
            return self.clone(device=device)
        else:
            return self

    def unstack(self: SubjectType) -> List[SubjectType]:
        generic_fields = set(field.name for field in fields(Subject))
        my_fields = set(field.name for field in fields(self)) - self.__varset - generic_fields
        if self.__varset:
            first_var_fieldname = next(iter(self.__varset))
            first_var = cast(Var, getattr(self, first_var_fieldname))
            batch_size = first_var.shape[0]
        elif my_fields:
            first_fieldname = next(iter(my_fields))
            first_field = cast(list, self.list(first_fieldname))
            batch_size = len(first_field)
        else:
            raise ValueError("You need to have at least one field to unstack")

        out = [copy.copy(self) for _ in range(batch_size)]
        for obj in out:
            obj.is_stacked = False
            obj.__lists = {}
            obj.__varset = frozenset()
            obj.__length = 1
        for var_fieldname in self.__varset:
            joined = cast(TensorVar, getattr(self, var_fieldname))
            split = joined.unstack()
            for obj, val in zip(out, split):
                setattr(obj, var_fieldname, val)
        for other_fieldname in my_fields:
            split = cast(list, self.list(other_fieldname))
            for obj, val in zip(out, split):
                setattr(obj, other_fieldname, val)
        return out

    @staticmethod
    def data_loader(data: Union[List[ExampleType], Dataset[ExampleType]],
                    batch_size: int = -1,
                    **kwargs) -> DataLoader[ExampleType]:
        """
        Given a set of examples, returns a dataloader with the appropriate `collate_fn`
        """
        if not isinstance(data, Dataset):
            data = ListDataset(data)
        if batch_size == -1:
            batch_size = len(cast(Sized, data))
        return DataLoader(cast(Dataset, data), collate_fn=Subject.stack, batch_size=batch_size,
                          **kwargs)

    def clamp_annotated(self: SubjectType) -> SubjectType:
        r"""Returns a clone of the input example with annotated variable cells
        being marked as clamped"""
        out = self.clone()
        for attr_name in out.__varset:
            cast(TensorVar, getattr(out, attr_name)).clamp_annotated()
        return out

    def unclamp_annotated(self: SubjectType) -> SubjectType:
        r"""Returns a clone of the input example with clamped variable cells
        being marked as annotated"""
        out = self.clone()
        for attr_name in out.__varset:
            cast(TensorVar, getattr(out, attr_name)).unclamp_annotated()
        return out

    def __post_init__(self):
        self.init_variables()

    def clone(self: SubjectType, device=None) -> SubjectType:
        out = copy.copy(self)
        for attr_name in out.__varset:
            old = cast(TensorVar, getattr(out, attr_name))
            setattr(out, attr_name, old.clone(device=device))
        return out

    def __init__(self, *args, **kwargs):
        raise ValueError(
            "you shouldn't call this initializer: make a subclass Subject and "
            "use the @dataclass decorator on it")
