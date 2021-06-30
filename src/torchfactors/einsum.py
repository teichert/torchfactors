
from functools import singledispatch
from typing import Dict, Hashable, List, Sequence, Tuple, Union

import torch
import torch_semiring_einsum as tse  # type: ignore
from torch import Tensor


class MultiEquation(object):
    def __init__(self, equations: Sequence[tse.equation.Equation]):
        self.equations = list(equations)


@singledispatch
def log_einsum(compiled_equation: tse.equation.Equation, *tensors: Tensor, block_size: int = 100):
    return log_einsum2(compiled_equation, *[t.nan_to_num() for t in tensors],
                       block_size=block_size)
    # return tse.log_einsum(compiled_equation, *[t.nan_to_num() for t in tensors],
    #                       block_size=block_size)


@log_einsum.register
def _from_multi(compiled_equation: MultiEquation,
                *tensors: Tensor, block_size: int = 100):
    return tuple(log_einsum(eq, *tensors, block_size=block_size)
                 for eq in compiled_equation.equations)


# def einsum(compiled_equation: tse.equation.Equation, *tensors: Tensor, block_size: int = 50):
#     # return log_einsum2(compiled_equation, *tensors, block_size=block_size)
#     return tse.einsum(compiled_equation, *tensors, block_size=block_size)


def compile_equation(equation: str, force_multi: bool = False
                     ) -> tse.equation.Equation:
    args_str, outputs_str = equation.split('->', 1)
    arg_strs = args_str.split(',')
    out_strs = outputs_str.split(',')
    return compile_obj_equation(arg_strs, out_strs, repr=equation, force_multi=force_multi)


def compile_obj_equation(arg_strs: Sequence[Sequence[Hashable]],
                         out_strs: Sequence[Sequence[Hashable]],
                         repr: str = '', force_multi: bool = False
                         ) -> Union[tse.equation.Equation, MultiEquation]:
    r"""modified from: https://github.com/bdusell/semiring-einsum/blob/7fbebdddc70aab81ede5e7c086719bff700b3936/torch_semiring_einsum/equation.py#L63-L92

    Pre-compile an einsum equation for use with the einsum functions in
    this package.

    :return: A pre-compiled equation.
    """  # noqa: E501
    char_to_int: Dict[Hashable, int] = {}
    int_to_arg_dims: List[List[Tuple[int, int]]] = []
    args_dims: List[List[int]] = []
    for arg_no, arg_str in enumerate(arg_strs):
        arg_dims = []
        for dim_no, dim_char in enumerate(arg_str):
            dim_int = char_to_int.get(dim_char)
            if dim_int is None:
                dim_int = char_to_int[dim_char] = len(char_to_int)
                int_to_arg_dims.append([])
            int_to_arg_dims[dim_int].append((arg_no, dim_no))
            arg_dims.append(dim_int)
        args_dims.append(arg_dims)
    num_variables = len(char_to_int)
    equations = []
    for out_str in out_strs:
        output_dims = [char_to_int[c] for c in out_str]
        equations.append(tse.equation.Equation(
            repr,
            int_to_arg_dims,
            args_dims,
            output_dims,
            num_variables))
    if len(equations) != 1 or force_multi:
        return MultiEquation(equations)
    else:
        return equations[0]


@torch.jit.script
def logsumexp(a: Tensor, dims: List[int]):
    if len(dims) > 0:
        return a.nan_to_num().logsumexp(dim=dims)
    else:
        return a


# def logsumadd_(a: Tensor, b: Tensor):
#     torch.logaddexp(a.nan_to_num(), b.nan_to_num(), out=a)

def log_dot(tensors_with_names: List[Tuple[torch.Tensor, List[object]]],
            queries: List[List[object]]) -> List[torch.Tensor]:
    named: List[Tensor] = [
        t.rename(*[f'_{id(name)}' for name in names])
        for t, names in tensors_with_names]
    return _log_dot(named, [
        [f'_{id(name)}' for name in query]
        for query in queries])


@torch.jit.script
def _log_dot(tensors: Dict[torch.Tensor, List[str]],
             queries: List[List[str]]) -> List[torch.Tensor]:
    """
    Carries out a generalized tensor dot product across multiple named
    tensors. The dimensions listed in `keep` but not in `exclude` will
    have a dimension in the output: `None` for no dimensions in
    output. (default) `"all"` for all dimensions in output. tuple or
    list of dimensions to explicitly list `keeps` may alternatively be
    used to specify multiple output dimension collections (this can be
    more efficient than calling `dot` multiple times.)

    The resulting tensor is constructed by first generating a tensor
    `FULL` whose dimensions include the union of all input tensor
    dimensions (dimensions with the same name must match in size). If
    `multiplication` is specified to be, e.g., `torch.prod`, then the
    value at a particular index in `FULL` is computed as a product of
    terms, one for each input tensor, where the term is the value of
    that tensor at the given index (only using the dimensions
    available in each respective tensor). Finally, `TMP` is reduced
    along all but the specified output dimensions using the given
    `addition` function.

    If ts is a dictionary, then the values are used.
    """
    # settle on an order for union of names and get corresponding sizes
    # name_to_ix: Dict[int, int]
    # dom = {}
    # for t in ts:
    #     if t is None:
    #         continue
    #     for name, size in zip(t.names, t.shape):
    #         dom.setdefault(name, size)
    # names, sizes = unzip(dom.items(), dw=2)
    # named.append(t.rename(*str_names))
    used_names: Dict[str, int] = {}
    all_names: List[str] = []
    all_sizes: List[int] = []
    for t, names in tensors.items():
        for name, size in zip(names, t.shape):
            if name not in used_names:
                used_names[name] = size
                all_names.append(name)
                all_sizes.append(size)
    # uniques: Set[Tuple[str, int]] = set()
    # name_and_size: List[Tuple[str, int]] = list(uniques.union(*[list(
    #     zip(t.names, t.shape))
    #     for t in tensors]))
    # all_names, all_sizes = zip(*name_and_size)
    # get all tensors in the same shape:
    #   match dimensions, make copies to match sizes,
    #   and remove names since names aren't yet supported for most ops
    aligned: List[Tensor] = [
        t.align_to(all_names)
        .expand(all_sizes)
        .rename(None) for t in tensors]
    stacked = torch.stack(aligned)
    # "multiply" across tensors
    product = torch.sum(stacked, dim=0).rename(all_names)
    answers: List[Tensor] = []
    for query in queries:
        reduce_along = dict(used_names)
        for name in query:
            del reduce_along[name]
        if len(reduce_along) > 0:
            reduced = product.logsumexp(dim=list(reduce_along.keys()))
        else:
            reduced = product
        answers.append(reduced.rename(None))
    return answers


def log_einsum2(
        equation: tse.equation.Equation,
        *args: torch.Tensor,
        block_size: int) -> torch.Tensor:

    def callback(compute_sum):
        return compute_sum(
            # (lambda a, b: torch.logaddexp(a.nan_to_num(), b.nan_to_num(), out=a)),
            None,
            logsumexp,
            torch.jit.script(tse.utils.add_in_place))

    return tse.semiring_einsum_forward(equation, args, block_size, callback)


# # def log_expectation_exp_einsum2(
# #         equation: tse.equation.Equation,
# #         *args: torch.Tensor,
# #         block_size: int) -> torch.Tensor:
# #     def callback(compute_sum):
# #         return compute_sum(logsumadd_, logsumexp, tse.utils.add_in_place)

# #     return tse.semiring_einsum_forward(equation, args, block_size, callback)

# def log_sum_xlogy(logx: Tensor, logy: Tensor, dim: Sequence[int]):
#     slog1 = torch.ones_like(logx, dtype=torch.bool)
#     llog1 = logx
#     slog2 = (logy >= 0)
#     llog2 = logy.abs().log()
#     order = llog1 >= llog2
#     sloga = slog1.where(order, slog2)
#     lloga = llog1.where(order, llog2)
#     slogb = slog1.where(order.logical_not(), slog2)
#     llogb = llog1.where(order.logical_not(), llog2)
#     sprods = sloga == slogb
#     lprods = lloga + llogb
#     zeros = torch.zeros_like(lprods).log()
#     lpositive = torch.where(sprods, lprods, zeros).logsumexp(dim=dim)
#     spositive = torch.ones_like(lpositive, dtype=torch.bool)
#     lnegative = torch.where(sprods.logical_not(), lprods, zeros).logsumexp(dim=dim)
#     snegative = torch.ones_like(lnegative, dtype=torch.bool).logical_not()
#     order = lpositive >= lnegative
#     sloga = spositive.where(order, snegative)
#     lloga = lpositive.where(order, lnegative)
#     slogb = spositive.where(order.logical_not(), snegative)
#     llogb = lpositive.where(order.logical_not(), lnegative)
#     ssum = sloga
#     expdiff = (llogb - lloga).exp()
#     add = sloga == slogb
#     lsum = lloga + torch.log1p(expdiff.where(add, -expdiff))
#     out = lsum.exp()
#     return out.where(ssum, -out)


# def sum_xlogx_minus_sum_xlogy(logx: Tensor, logy: Tensor, dim: Sequence[int]):
#     log_entropy =

# def expectation_forward_sl(equation, *args: torch.Tensor, block_size: int):
#     r"""
#     follows tables 1 and 3 from https://cs.jhu.edu/~jason/papers/li+eisner.emnlp09.pdf
#     The input is assumed to have an extra last dimension of size 2:
#     since inputs are already in log space, s_p and s_r are always both +
#     0: log |p|
#     1: log |r|

#     <p1,r1>x<p2,r2> = <p1.r2, p1.r2 + p2.r1>
#     <p1,r1>+<p2,r2> = <p1+p2, r1+r2>
#     <p,r>* = <p*, p*.p*.r>
#     0 = <0,0>
#     1 = <1,0>
#     (assumes la >= lb)!!
#     <sa=+, la> + <sb=+, lb> = <+, la + log1p(e^(lb-la))>
#     # <sa=+, la> + <sb=-, lb> = <+, la + log1p(-e^(lb-la))>
#     # <sa=-, la> + <sb=+, lb> = <-, la + log1p(-e^(lb-la))>
#     # <sa=-, la> + <sb=-, lb> = <-, la + log1p(e^(lb-la))>

#     <sa=+, la> . <sb=+, lb> = <+, la + lb>
#     # <sa=+, la> . <sb=-, lb> = <-, la + lb>
#     # <sa=-, la> . <sb=+, lb> = <-, la + lb>
#     # <sa=-, la> . <sb=-, lb> = <+, la + lb>
#     """

#     # def lplus(sa, la, sb, lb):
#     #     return sa, torch.log1p(sa*sb*torch.exp(lb - la))

#     # def ltimes(sa, la, sb, lb):
#     #     return sa*sb, la+lb

#     def lplus(a, b):
#         sa, la = a
#         sb, lb = b
#         return sa, torch.log1p(sa*sb*torch.exp(lb - la))

#     def ltimes(a, b):
#         sa, la = a
#         sb, lb = b
#         return sa*sb, la+lb

#     def times(a, b):
#         """
#         a, b -> o
#         <pa,ra>, <pb,rb> -> <po, ro>
#         <<spa, lpa>, <sra, lra>>, <<spb, lpb>, <srb, lrb>> -> <<spo, lpo>, <sro, lro>>
#         a is first arg, b is second arg
#         o is output
#         s for sign, l for log of absolute value
#         """
#         spa, lpa, sra, lra = a.unbind(-1)
#         spb, lpb, srb, lrb = b.unbind(-1)
#         pa, ra = (spa, lpa), (sra, lra)
#         pb, rb = (spb, lpb), (srb, lrb)
#         po = ltimes(pa, pb)
#         ro1 = ltimes(pa, rb)
#         ro2 = ltimes(pb, ra)
#         ro = lplus(ro1, ro2)
#         return torch.stack(tuple(*po, *ro), -1)

#     def add(a, b):
#         spa, lpa, sra, lra = a.unbind(-1)
#         spb, lpb, srb, lrb = b.unbind(-1)
#         pa, ra = (spa, lpa), (sra, lra)
#         pb, rb = (spb, lpb), (srb, lrb)
#         po = lplus(pa, pb)
#         ro = lplus(ra, rb)
#         return torch.stack(tuple(*po, *ro), -1)

#     def add_in_place(a, b):
#         a.data = add(a, b)

#     def mul_in_place(a, b):
#         a.data = times(a, b)
#         # lpa, lra = a.unbind(-1)
#         # lpb, lrb = b.unbind(-1)
#         # outp = lpa + lpb
#         # parb = (lpa + lrb)
#         # pbra = (lpb + lra)
#         # outr = parb + torch.log1p(torch.exp(pbra - parb))
#         # a.data = torch.stack((outp, outr), -1)

#     # def mul_in_place(a, b):

#     # def sum_block(a, dims):
#     #     lpa, lra = a.unbind()
#     #     lpb, lrb = b.unbind()
#     #     outp = lpa + torch.log1p(torch.exp(lpb-lpa))
#     #     outr = lra + torch.log1p(torch.exp(lrb-lra))
#     #     a.data = torch.stack((outp, outr))

#     # def max_in_place(a, b):
#     #     torch.max(a, b, out=a)

#     def add_block(a, dims):
#         if dims:
#             return torch.sum(a, dim=dims)
#         else:
#             return a
#         # result = a
#         # for dim in reversed(sorted(dims)):
#         #     ts = result.unbind(dim)
#         #     result = reduce(add_in_place, ts)
#         # return result

#     def callback(compute_sum):
#         return compute_sum(add_in_place, add_block, mul_in_place)

#     return tse.semiring_einsum_forward(equation, args, block_size, callback)


# def plogp(logp: Tensor):
#     sp = torch.ones_like(logp).float()
#     lp = logp
#     sr = torch.where(logp >= 0, 1., -1.)
#     lr = logp.abs().log()
#     return torch.stack((sp, lp, sr, lr), -1)


# def free_energy(equation, logps):
#     p_plogps = tuple(plogp(logp) for logp in logps)
#     out = expectation_forward_sl(equation, *p_plogps, block_size=1)
#     sp, lp, splogp, lplogp = torch.unbind(out, -1)
#     print(sp, lp, splogp, lplogp)


# ts = [torch.rand(3, 3) for _ in range(3)]
# eq = tse.compile_equation('ij,jk,kl->')
# free_energy(eq, ts)
