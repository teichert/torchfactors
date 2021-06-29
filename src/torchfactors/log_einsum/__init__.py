import torch
from torch._VF import tensordot as base_tensordot  # type: ignore

# from opt_einsum.parser import convert_to_valid_einsum_chars


def transpose(a, axes):
    """Normal torch transpose is only valid for 2D matrices.
    """
    return a.permute(*axes)


def tensordot(a: torch.Tensor, b: torch.Tensor, axes=2):
    a = a.exp().nan_to_num()
    b = b.exp().nan_to_num()
    if isinstance(axes, (tuple, list)) and axes and not axes[0]:
        # return torch.tensordot(a[None], b[..., None], dims=len(a.shape))
        # return torch.stack(torch.meshgrid(a, b), 0).prod(0)
        return base_tensordot(a, b, (), ()).log()
    else:
        return torch.tensordot(a, b, dims=axes).log()


# def _get_torch_and_device():
#     global _TORCH_DEVICE
#     global _TORCH_HAS_TENSORDOT

#     if _TORCH_DEVICE is None:
#         import torch
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         _TORCH_DEVICE = torch, device
#         _TORCH_HAS_TENSORDOT = hasattr(torch, 'tensordot')

#     return _TORCH_DEVICE


# @to_backend_cache_wrap
# def to_logtorch(array):
#     torch, device = _get_torch_and_device()

#     if isinstance(array, np.ndarray):
#         return torch.from_numpy(array).to(device)

#     return array


# def build_expression(_, expr):  # pragma: no cover
#     """Build a torch function based on ``arrays`` and ``expr``.
#     """
#     def torch_contract(*arrays):
#         torch_arrays = [to_logtorch(x) for x in arrays]
#         torch_out = expr._contract(torch_arrays, backend='torchfactors.log_einsum')

#         if torch_out.device.type == 'cpu':
#             return torch_out.numpy()

#         return torch_out.cpu().numpy()

#     return torch_contract


# def evaluate_constants(const_arrays, expr):
#     """Convert constant arguments to torch, and perform any possible constant
#     contractions.
#     """
#     const_arrays = [to_logtorch(x) for x in const_arrays]
#     return expr(*const_arrays, backend='torchfactors.log_einsum', evaluate_constants=True)


__all__ = [
    'transpose',
    'tensordot',
    # "to_logtorch",
    # "build_expression",
    # "evaluate_constants"
]
