
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from torchfactors.utils import min_tensors, sum_tensors


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
def _slow_log_dot(tensors: List[Tuple[Tensor, List[int]]],
                  ordered_queries: List[Tuple[List[int], List[int]]],
                  full_shape: List[int],
                  nan_to_num: bool = False) -> List[Tensor]:  # pragma: no cover
    r"""
    This version assumes that all ids are integers between 0 and len(full_shape),
    and that all tensors and queries are in increasing order of those ids
    """
    permuted_answers = [torch.zeros([full_shape[d] for d in q], dtype=torch.float)
                        for q, _ in ordered_queries]
    config = [0 for _ in full_shape]
    config[-1] = -1  # make it so we can advance on the first one
    num_configs = 0
    while True:
        # move to the next config by incrementing the last possible dimension
        # and then filling with zeros
        while config:
            dim = len(config) - 1
            if config[dim] < full_shape[dim] - 1:
                config[dim] += 1
                while len(config) < len(full_shape):
                    config.append(0)
                break
            else:
                config.pop()
        if not config:
            break
        num_configs += 1
        if num_configs % 1000 == 0:
            print(num_configs)
        # get the score for this config
        cell_score = torch.tensor(0.0)
        for factor, dims in tensors:
            sub_factor = factor
            factor_dimensions_used = 0
            for Y, y in enumerate(config):
                if factor_dimensions_used == len(dims):
                    break
                if Y == dims[factor_dimensions_used]:
                    sub_factor = sub_factor[y]
                    factor_dimensions_used += 1
            cell_score += sub_factor

        # add the score to all relevant queries
        for answer, (q, _) in zip(permuted_answers, ordered_queries):
            # get the query-specific config by
            sub_config = [config[ix] for ix in q]
            answer[sub_config] = torch.logaddexp(answer[sub_config], cell_score)
    answers = [orig.permute(unpermute)
               if len(q) > 0 else orig
               for orig, (q, unpermute) in zip(permuted_answers, ordered_queries)]
    return answers


@torch.jit.script
def slow_log_dot(tensors: List[Tuple[Tensor, List[str]]],
                 queries: List[List[str]],
                 nan_to_num: bool = False) -> List[Tensor]:  # pragma: no cover
    """
    Carries out a generalized tensor dot product across multiple named
    tensors.
    """
    name_to_ix: Dict[str, Tuple[int, int]] = {}
    out_tensors = [_map_and_order_tensor(t, name_to_ix, names) for t, names in tensors]
    shape = [size for _, size in name_to_ix.values()]
    out_queries = [_map_and_order_names(name_to_ix, names) for names in queries]
    return _slow_log_dot(out_tensors, out_queries, shape, nan_to_num)


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
