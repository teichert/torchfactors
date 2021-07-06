from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from typing import (Dict, FrozenSet, Generic, List, Sequence, Sized, TypeVar,
                    Union, cast)

from torch.utils.data import DataLoader, Dataset

from .domain import Domain
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


@ dataclass
class Subject:
    is_stacked: bool = field(init=False, default=False)
    __lists: Dict[str, List[object]] = field(init=False, default_factory=dict)
    __varset: FrozenSet = field(init=False, default=frozenset())

    def list(self, key: str) -> List[object]:
        return self.__lists[key]

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
                    # shape can be delegated to earlier field
                    if isinstance(var_field._shape, VarField):
                        source_var = cls_attr_id_to_var[id(var_field._shape)]
                        # source should already have tensor with shape by now
                        var_instance._tensor = var_field._init(source_var.shape)
                    elif var_field._shape is not None:
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

    # if this object has been stacked, then:
    # 1) (it will know it and not allow stacking again for now)
    # 2) all variable fields will be replaced with stacked and padded variables
    # 3) other values will take the value of the first object, but
    #    --- the full list will be accessible via stacked.list(stacked.item)
    #

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
        return out

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
            # obj.__lists = {}
            # obj.__vars = frozenset()
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
        if not isinstance(data, Dataset):
            data = ListDataset(data)
        if batch_size == -1:
            batch_size = len(cast(Sized, data))
        return DataLoader(cast(Dataset, data), collate_fn=Subject.stack, batch_size=batch_size,
                          **kwargs)
        # def shapes(self):
        #     cls = type(obj)
        #     for attr_name in dir(cls):
        #         attr = getattr(cls, attr_name)
        #         if isinstance(attr, Var):
        #             property = cast(Var, getattr(obj, attr_name))
        #             if property.tensor is None:
        #                 raise ValueError(
        #                     "need to specify an actual tensor for every variable in the subject")
        #             if property.domain is Domain.OPEN:
        #                 property._domain = attr.domain
        #             if property.usage is None:
        #                 property.usage = torch.full_like(property.tensor, attr.usage.value)

        # @staticmethod
        # def collate(subjects: Sequence[SubjectType]) -> SubjectType:

        #     return

    def clamp_annotated(self: SubjectType) -> SubjectType:
        out = self.clone()
        for attr_name in out.__varset:
            cast(TensorVar, getattr(out, attr_name)).clamp_annotated()
        return out

    def unclamp_annotated(self: SubjectType) -> SubjectType:
        out = self.clone()
        for attr_name in out.__varset:
            cast(TensorVar, getattr(out, attr_name)).unclamp_annotated()
        return out

    def __post_init__(self):
        self.init_variables()

    def clone(self: SubjectType) -> SubjectType:
        out = copy.copy(self)
        for attr_name in out.__varset:
            old = cast(TensorVar, getattr(out, attr_name))
            setattr(out, attr_name, old.clone())
        return out

    def __init__(self, *args, **kwargs):
        raise ValueError(
            "you shouldn't call this initializer: make a subclass Subject and "
            "use the @dataclass decorator on it")


# def subject(cls):
#     def post_init(self):
#         for k, v in get_type_hints(self, include_extras=True).items():
#             print(v)
#             if isinstance(v, VariableTensor):
#                 if v.__metadata__:
#                     meta = v.__metadata__[0]
#                     for detail in meta.get('details', []):
#                         if isinstance(detail, Domain):
#                             self.domain = detail
#                 # if hasattr(v, '__metadata__'):
#                 #     for detail in v.__metadata__[0]['details']:
#                 #         if isinstance(detail, Domain):
#                 #             return detail

#     cls.__post_init__ = post_init
#     cls = dataclass(cls)
#     return cls


# class ExactlyOneFactor(Factor):

#     def __init__(self, variables):
#         pass
