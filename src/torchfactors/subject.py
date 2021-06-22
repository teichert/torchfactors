from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from typing import (Dict, FrozenSet, Generic, List, Sequence, TypeVar, Union,
                    cast)

from torch.utils.data import DataLoader, Dataset

from .domain import Domain
from .variable import Var, VarField, VarUsage

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
    __lists: Dict[object, List[object]] = field(init=False, default_factory=dict)
    __vars: FrozenSet = field(init=False, default=frozenset())

    def init_variables(self):
        cls = type(self)
        vars = []
        cls_attr_id_to_var: Dict[int, Var] = {}
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
                if not isinstance(var_instance, Var):
                    if var_field._usage != VarUsage.LATENT:
                        raise ValueError("only latent variables can be left implicit; "
                                         "make sure that you pass in all required variables "
                                         "to the subject constructor")
                    var_instance = Var()
                    setattr(self, attr_name, var_instance)
                cls_attr_id_to_var[id(var_field)] = var_instance
                if var_instance._tensor is None:
                    if isinstance(var_field._shape, VarField):
                        source_var = cls_attr_id_to_var[id(var_field._shape)]
                        # source should already have tensor with shape by now
                        var_instance._tensor = var_field._init(source_var.shape)
                    elif var_field._shape is not None:
                        var_instance._tensor = var_field._init(var_field._shape)
                    else:
                        raise ValueError(
                            "need to specify an actual tensor (or a shape)"
                            "for every variable in the subject")
                if var_instance.domain is Domain.OPEN:
                    var_instance._domain = var_field._domain
                if var_instance.usage is None:
                    var_instance.usage = var_field._usage
                vars.append(attr_name)
        self.__vars = frozenset(vars)

    # if this object has been stacked, then:
    # 1) (it will know it and not allow stacking again for now)
    # 2) all variable fields will be replaced with stacked and padded variables
    # 3) other values will take the value of the first object, but
    #    --- the full list will be accessible via stacked.list(stacked.item)
    #
    @ staticmethod
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
        my_fields = set(field.name for field in fields(first)) - first.__vars
        for attr_name in first.__vars:
            attr = cast(Var, getattr(out, attr_name))
            stacked = Var.pad_and_stack([
                cast(Var, getattr(subject, attr_name))
                for subject in subjects])
            setattr(out, attr_name, stacked)
        for attr_name in my_fields:
            attr = getattr(out, attr_name)
            out.__lists[attr] = [
                getattr(subject, attr_name)
                for subject in subjects]
        return out

    @ staticmethod
    def data_loader(data: Union[List[ExampleType], Dataset], **kwargs) -> DataLoader:
        if not isinstance(data, Dataset):
            data = ListDataset(data)
        return DataLoader(cast(Dataset, data), collate_fn=Subject.stack, **kwargs)
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

    def clamp_annotated(self) -> None:
        for attr_name in self.__vars:
            cast(Var, getattr(self, attr_name)).clamp_annotated()

    def unclamp_annotated(self) -> None:
        for attr_name in self.__vars:
            cast(Var, getattr(self, attr_name)).unclamp_annotated()

    def __post_init__(self):
        self.init_variables()

    def __init__(self, *args, **kwargs):
        raise ValueError(
            "shouldn't call this initializer: subclass Subject and use @dataclass decorator")


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
