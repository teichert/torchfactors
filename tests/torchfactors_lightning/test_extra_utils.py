import pandas as pd

from torchfactors_lightning import with_rep_number


def test_with_rep_number():
    orig = dict(
        sentence=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        property=[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        annotator=[1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
        label=[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2]
    )
    df = pd.DataFrame(orig)
    with_rep = with_rep_number(
        df, group_key=['sentence', 'property'], name='my_rep')
    expected = dict(
        sentence=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        property=[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        annotator=[1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1],
        label=[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
        my_rep=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    out = with_rep.to_dict(orient='list')
    assert out == expected
    assert df.to_dict(orient='list') == orig
