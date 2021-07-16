import pytest
import torch
from torch import arange
from torchfactors.utils import (as_ndrange, compose, compose_single, ndarange,
                                ndslices_cat, ndslices_overlap, outer,
                                stereotype)


def test_compose_single():
    assert compose_single(slice(3, 14, 2), slice(2, 6, 3), 100) == slice(7, 14, 6)


def test_compose():
    shape = (4, 10, 12, 7)
    first = (slice(None), 3, slice(3, 9))
    second = (2, slice(1, 3), slice(5))
    expected_combined = (2, 3, slice(4, 6, 1), slice(0, 5, 1))
    out = compose(shape, first, second)
    assert out == expected_combined

    other_first = (slice(2, 4), slice(None), slice(2, 7))
    other_second = (0, 3, slice(2, -1), slice(5))
    other_out = compose(shape, other_first, other_second)
    assert other_out == expected_combined


def test_compose2():
    shape = (4, 5)
    first = (slice(1, 4), slice(None))
    second = ([2, 0, 1], 1)
    expected_combined = ([3, 1, 2], 1)

    assert compose(shape, first, second) == expected_combined


def test_compose3():
    shape = (4, 5, 6)
    first = (slice(1, 4), slice(None), [2, 1, 1])
    second = ([2, 0, 1], 1, [0, 2])
    expected_combined = ([3, 1, 2], 1, slice(2, 0, -1))

    assert compose(shape, first, second) == expected_combined


def test_ndrange():
    t = ndarange(2, 3, 4)
    assert (t == arange(2*3*4).reshape(2, 3, 4)).all()


def test_ndslices_overlap():
    shape = (10, 23, 8, 10, 15)
    t = ((3, slice(None), slice(3, 5)))
    assert ndslices_overlap(t, (3, 5, slice(4, 8)), shape)
    assert not ndslices_overlap(t, (4, 5, slice(4, 8)), shape)
    assert ndslices_overlap(t, (slice(None), slice(None), slice(4, 8)), shape)
    assert not ndslices_overlap(t, (slice(None), slice(None), 5), shape)
    assert not ndslices_overlap(t, (slice(None), slice(
        None), slice(None), slice(None), slice(3, 3)), shape)


def test_ndslice_bad():
    with pytest.raises(NotImplementedError):
        as_ndrange(8, shape=(10,))


def test_asndrange_dots():
    shape = (10, 11, 12, 13, 14)
    out = as_ndrange([slice(3, 6), ..., 5], shape=shape)
    assert out == (range(3, 6), range(11), range(12), range(13), 5)


def test_asndrange_bad_dots():
    shape = (10, 11, 12, 13, 14)
    with pytest.raises(ValueError):
        as_ndrange([slice(3, 6), ..., 2, ..., 5], shape=shape)


def test_ndslices_cat():
    assert ndslices_cat(1, 3) == (1, 3)
    assert ndslices_cat(1, (slice(None), 3)) == (1, slice(None), 3)
    assert ndslices_cat((1, 3), 2) == (1, 3, 2)
    assert ndslices_cat((1, 3), (9, 2)) == (1, 3, 9, 2)


def test_outer2():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([2, 3])
    out = outer(a, b)
    assert out.allclose(torch.tensor([
        [2, 3],
        [4, 6],
        [6, 9]
    ]))


def test_outer3():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([2, 3])
    c = torch.tensor([7, 9])
    out = outer(a, b, c)
    assert out.allclose(torch.tensor([
        [[14, 18], [21, 27]],
        [[28, 36], [42, 54]],
        [[42, 54], [63, 81]]
    ]))


def test_outer_batch():
    a = torch.tensor([[1, 2, 3], [1, 2, 6]])
    b = torch.tensor([[2, 3], [2, 3]])
    c = torch.tensor([[7, 9], [7, 9]])
    out = outer(a, b, c, num_batch_dims=1)
    expected = torch.tensor([[
        [[14, 18], [21, 27]],
        [[28, 36], [42, 54]],
        [[42, 54], [63, 81]]
    ], [
        [[14, 18], [21, 27]],
        [[28, 36], [42, 54]],
        [[84, 108], [126, 162]]
    ]])
    assert out.allclose(expected)


def test_stereotype():
    scales = torch.tensor([2, 3, 4])
    binary = torch.tensor([5, 6])
    out = stereotype(scales, binary)
    # total = scales.sum() * binary[0]
    # full = scales * binary[1] + total
    # for each coordinate, iterate over the 4
    expected = torch.tensor([2*6+9*5, 3*6+9*5, 4*6+9*5])
    assert out.allclose(expected)


def test_stereotype2():
    scales = torch.tensor([
        [1, 2, 3],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11]
    ]).float()
    binary = torch.tensor([
        [12, 13],
        [14, 15]
    ])
    out = stereotype(scales, binary)
    total = scales.sum() * binary[0, 0]
    full = scales * binary[1, 1] + total
    cols = scales.sum(0) * binary[0, 1]
    rows = scales.sum(1) * binary[1, 0]
    # for each coordinate, iterate over the 4
    expected = torch.tensor(
        [
            [
                full[0, 0] + rows[0] + cols[0],
                full[0, 1] + rows[0] + cols[1],
                full[0, 2] + rows[0] + cols[2],
            ],
            [
                full[1, 0] + rows[1] + cols[0],
                full[1, 1] + rows[1] + cols[1],
                full[1, 2] + rows[1] + cols[2],
            ],
            [
                full[2, 0] + rows[2] + cols[0],
                full[2, 1] + rows[2] + cols[1],
                full[2, 2] + rows[2] + cols[2],
            ],
            [
                full[3, 0] + rows[3] + cols[0],
                full[3, 1] + rows[3] + cols[1],
                full[3, 2] + rows[3] + cols[2],
            ],
        ])
    assert out.allclose(expected)
