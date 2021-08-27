import pickle
import tarfile

import pandas as pd

# for easier short-term maintenance I will develop this here
# to avoid having a separate project while making it easy
# to factor out later.


def load_tgz_df(path: str, filt=None, **kwargs):
    with tarfile.open(path, 'r:*') as tar:
        # get the last one that matches
        paths = list(filter(filt, tar.getnames()))

        def read_one(sub_path: str) -> pd.DataFrame:
            extracted = tar.extractfile(sub_path)
            if extracted is None:
                raise FileNotFoundError("unable to read member file '{sub_path}' "
                                        "from tar file '{path}'")
            if sub_path.endswith('.csv'):
                return pd.read_csv(extracted, **kwargs)
            else:
                return pickle.load(extracted)
        return list(read_one(sub_path) for sub_path in paths)


def load_tgz_input_labels(path):
    inputs = load_tgz_df(path, lambda p: p.endswith('input.df.pkl'))[0]
    labels = load_tgz_df(path, lambda p: p.endswith('labels.df.pkl'))[0]
    return inputs, labels
