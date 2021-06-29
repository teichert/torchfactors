import torch
from torch._VF import tensordot as base_tensordot  # type: ignore


def transpose(a, axes):
    """Normal torch transpose is only valid for 2D matrices.
    # https://github.com/dgasmith/opt_einsum/blob/be2d3dcb9792016d3b7ed0afacfed70024698c9e/opt_einsum/backends/torch.py#L31
    """
    return a.permute(*axes)


def tensordot(a: torch.Tensor, b: torch.Tensor, axes=2):
    a = a.exp().nan_to_num()
    b = b.exp().nan_to_num()
    # this type of input worked in 1.8 but needs special handling in 1.9:
    if isinstance(axes, (tuple, list)) and axes and not axes[0]:
        # return torch.tensordot(a[None], b[..., None], dims=len(a.shape))
        # return torch.stack(torch.meshgrid(a, b), 0).prod(0)
        return base_tensordot(a, b, (), ()).log()
    else:
        return torch.tensordot(a, b, dims=axes).log()


__all__ = [
    'transpose',
    'tensordot',
]
