from typing import Sequence

from pandas.core.frame import DataFrame


def with_rep_number(data: DataFrame, group_key: Sequence[str],
                    name: str = 'rep',
                    ) -> DataFrame:
    r"""
    Returns a copy of the given data frame augmented by an aditional column
    specifying how many examples with the same group_key came before it (i.e.
    what row it would be on if grouped by the given key)

    Parameters:

        data: the dataframe

        group_key: the key to group on

        name: the name of the column to be added
    """
    return data.assign(**{name: data.groupby(group_key).cumcount()})
