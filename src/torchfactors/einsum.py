
from typing import Dict, List, Tuple

import torch
# import torch_semiring_einsum as tse  # type: ignore
from torch import Tensor

from torchfactors.utils import min_tensors, sum_tensors

# class MultiEquation(object):
#     r"""
#     Represents multiple einsum equations to operate on the same input
#     """

#     def __init__(self, equations: Sequence[tse.equation.Equation]):
#         self.equations = list(equations)


# @singledispatch
# def log_einsum(compiled_equation: tse.equation.Equation, *tensors: Tensor, block_size: int = 100):
#     return log_einsum2(compiled_equation, *[t for t in tensors],
#                        block_size=block_size)


# @log_einsum.register
# def _from_multi(compiled_equation: MultiEquation,
#                 *tensors: Tensor, block_size: int = 100):
#     return tuple(log_einsum(eq, *tensors, block_size=block_size)
#                  for eq in compiled_equation.equations)


# def compile_equation(equation: str, force_multi: bool = False
#                      ) -> tse.equation.Equation:
#     args_str, outputs_str = equation.split('->', 1)
#     arg_strs = args_str.split(',')
#     out_strs = outputs_str.split(',')
#     return compile_obj_equation(arg_strs, out_strs, repr=equation, force_multi=force_multi)


# def compile_obj_equation(arg_strs: Sequence[Sequence[Hashable]],
#                          out_strs: Sequence[Sequence[Hashable]],
#                          repr: str = '', force_multi: bool = False
#                          ) -> Union[tse.equation.Equation, MultiEquation]:
#     r"""modified from: https://github.com/bdusell/semiring-einsum/blob/"""
#     r"""7fbebdddc70aab81ede5e7c086719bff700b3936/torch_semiring_einsum/equation.py#L63-L92
#
#     Pre-compile an einsum equation for use with the einsum functions in
#     this package.

#     :return: A pre-compiled equation.
#     """  # noqa: E501
#     char_to_int: Dict[Hashable, int] = {}
#     int_to_arg_dims: List[List[Tuple[int, int]]] = []
#     args_dims: List[List[int]] = []
#     for arg_no, arg_str in enumerate(arg_strs):
#         arg_dims = []
#         for dim_no, dim_char in enumerate(arg_str):
#             dim_int = char_to_int.get(dim_char)
#             if dim_int is None:
#                 dim_int = char_to_int[dim_char] = len(char_to_int)
#                 int_to_arg_dims.append([])
#             int_to_arg_dims[dim_int].append((arg_no, dim_no))
#             arg_dims.append(dim_int)
#         args_dims.append(arg_dims)
#     num_variables = len(char_to_int)
#     equations = []
#     for out_str in out_strs:
#         output_dims = [char_to_int[c] for c in out_str]
#         equations.append(tse.equation.Equation(
#             repr,
#             int_to_arg_dims,
#             args_dims,
#             output_dims,
#             num_variables))
#     if len(equations) != 1 or force_multi:
#         return MultiEquation(equations)
#     else:
#         return equations[0]


# def nonan_logsumexp(a: Tensor, dims):
#     if dims:
#         return torch.logsumexp(a.nan_to_num(), dim=dims)
#     else:
#         return a


# def log_einsum2(
#         equation: tse.equation.Equation,
#         *args: torch.Tensor,
#         block_size: int) -> torch.Tensor:
#     def callback(compute_sum):
#         return compute_sum(
#             # (lambda a, b: torch.logaddexp(a.nan_to_num(), b.nan_to_num(), out=a)),
#             None,
#             nonan_logsumexp,
#             tse.utils.add_in_place)

#     writeable_args = [t.as_subclass(torch.Tensor) for t in args]  # type: ignore
#     out = tse.semiring_einsum_forward(equation, writeable_args, block_size, callback)
#     return out


# # def ids(values: Iterable[object]) -> List[int]:
# #     return list(map(id, values))


@torch.jit.script
def _map_and_order_tensor(t: Tensor, name_to_id_and_size: Dict[str, Tuple[int, int]],
                          names: List[str]) -> Tuple[Tensor, List[int]]:  # pragma: no cover
    # store mapping from name to id and size
    current_order = [name_to_id_and_size.setdefault(name, (len(name_to_id_and_size),
                                                           size))
                     for name, size in zip(names, t.shape)]
    sorted_names = sorted([(name, i) for i, (name, _) in enumerate(current_order)])
    permutation = [i for _, i in sorted_names]
    out_names = [name for name, _ in sorted_names]
    return t.permute(permutation), out_names


@torch.jit.script
def _map_and_order_names(name_to_id_and_size: Dict[str, Tuple[int, int]], names: List[str]
                         ) -> Tuple[List[int], List[int]]:  # pragma: no cover
    r"""
    Returns the mapped names from the query in order, as well as the order that will
    permute a tensor with those dimensions back to the original order
    """
    current_order = [name_to_id_and_size[name] for name in names]
    sorted_names = sorted([(name, i) for i, (name, _) in enumerate(current_order)])
    permutation = [i for _, i in sorted_names]
    out_names = [name for name, _ in sorted_names]
    unsorted_indexes = sorted([(sorted_index, i) for i, sorted_index in enumerate(permutation)])
    unpermutation = [i for _, i in unsorted_indexes]
    return out_names, unpermutation


@torch.jit.script
def _map_order_and_invert_query(name_to_ix: Dict[str, Tuple[int, int]], names: List[str]
                                ) -> Tuple[List[int], List[int]]:  # pragma: no cover
    r"""
    given a dictionary from orig id to compact id and an unordered list of original ids,
    returns the ordered list of indexs not listed as well as a permutation on the
    indexes that would remain after summing out those dimensions

    """
    ordered_query, unpermutation = _map_and_order_names(name_to_ix, names)
    num_consumed = 0
    out_query: List[int] = []
    for name, _ in enumerate(name_to_ix.values()):
        if num_consumed < len(names) and ordered_query[num_consumed] == name:
            num_consumed += 1
        else:
            out_query.append(name)
    return out_query, unpermutation


@torch.jit.script
def expand_to(t: Tensor, names: List[int], shape: List[int]):  # pragma: no cover
    """
    returns an expanded view of t with shape matching that given
    (the names say which of the dimensions are already used)
    """
    names_consumed = 0
    # for each dimension, if we have that dimension, then skip it
    for i in range(len(shape)):
        if names_consumed < len(names) and names[names_consumed] == i:
            names_consumed += 1
        else:
            t = t.unsqueeze(i)

    t = t.expand(shape)
    return t


@torch.jit.script
def _log_dot(tensors: List[Tuple[Tensor, List[int]]],
             inverse_queries: List[Tuple[List[int], List[int]]],
             full_shape: List[int],
             nan_to_num: bool = False) -> List[Tensor]:  # pragma: no cover
    r"""
    This version assumes that all ids are integers between 0 and len(full_shape),
    and that all tensors and queries are in increasing order of those ids
    """
    aligned = [expand_to(t, names, full_shape) for t, names in tensors]
    product = sum_tensors(aligned)
    minimum = min_tensors(aligned)
    mask = minimum == float('-inf')
    ninf = float('-inf')
    product = product.masked_fill(mask, ninf)
    answers = [product.logsumexp(dim=iq).permute(unpermute)
               if len(iq) > 0 else product
               for iq, unpermute in inverse_queries]
    return answers


@torch.jit.script
def log_dot(tensors: List[Tuple[Tensor, List[str]]],
            queries: List[List[str]],
            nan_to_num: bool = False) -> List[Tensor]:  # pragma: no cover
    """
    Carries out a generalized tensor dot product across multiple named
    tensors.
    """
    # make sure that the ints are all in range and in order
    name_to_ix: Dict[str, Tuple[int, int]] = {}
    out_tensors = [_map_and_order_tensor(t, name_to_ix, names) for t, names in tensors]
    shape = [size for _, size in name_to_ix.values()]
    out_inverse_queries = [_map_order_and_invert_query(name_to_ix, names) for names in queries]
    return _log_dot(out_tensors, out_inverse_queries, shape, nan_to_num)


# def _log_dot2(tensors: List[Tuple[Tensor, List[int]]],
#               queries: List[List[int]]) -> List[Tensor]:
#     """
#     see log_dot which squelches warning about named tensors
#     """
#     def make_names(ids: List[int]) -> List[str]:
#         return [f'v{i}' for i in ids]
#     named = [t.rename(*make_names(names)) for t, names in tensors]
#     all_names, all_shapes = list(zip(*set((name, shape) for t in named
#                                           for name, shape in zip(t.names, t.shape))))

#     aligned = [t.align_to(*all_names).expand(*all_shapes).rename(None) for t in named]
#     stacked = torch.stack(aligned, dim=0)
#     # "multiply" across tensors
#     product = torch.sum(stacked, dim=0).nan_to_num().rename(*all_names)

#     names_set = set(all_names)

#     def logsumexp_to(t: Tensor, q: List[str]):
#         name_diff = names_set.difference(q)
#         if name_diff:
#             return product.logsumexp(dim=list(name_diff)).align_to(*q).rename(None)
#         else:
#             return product.align_to(*q).rename(None)

#     named_queries = list(map(make_names, queries))
#     # "sum" over dimensions not in the query
#     answers = [logsumexp_to(product, q) for q in named_queries]
#     return answers
