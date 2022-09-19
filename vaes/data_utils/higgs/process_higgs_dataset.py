import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

ORIGINAL_COLNAMES = [
    "label",
    "lepton pT",
    "lepton eta",
    "lepton phi",
    "missing energy",
    "missing energy phi",
    "jet1 pT",
    "jet1 eta",
    "jet1 phi",
    "jet1 btag",
    "jet2 pT",
    "jet2 eta",
    "jet2 phi",
    "jet2 btag",
    "jet3 pT",
    "jet3 eta",
    "jet3 phi",
    "jet3 btag",
    "jet4 pT",
    "jet4 eta",
    "jet4 phi",
    "jet4 btag",
    "m jj",
    "m jjj",
    "m lv",
    "m jlv",
    "m bb",
    "m wbb",
    "m wwbb",
]

COLNAMES = [
    "label",
    "lepton pT",
    "lepton eta",
    "missing energy",
    "jet1 pT",
    "jet1 eta",
    "jet2 pT",
    "jet2 eta",
    "jet3 pT",
    "jet3 eta",
    "jet4 pT",
    "jet4 eta",
    "m jj",
    "m jjj",
    "m lv",
    "m jlv",
    "m bb",
    "m wbb",
    "m wwbb",
]


def download_higgs_dataset(data_dir):
    """Downloads Higgs dataset from https://archive.ics.uci.edu/ml/datasets/HIGGS. If already downloaded returns file path.

    Parameters
    ----------
    data_dir : str
        Downloaded in this directory (needs to exist).

    Returns
    -------
    str
        Data path name.

    References
    ----------
    Searching for exotic particles in high-energy physics with deep learning: https://www.nature.com/articles/ncomms5308

    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
    response = requests.get(url, stream=True)

    file_name = data_dir + url.split("/")[-1]

    if Path(file_name).is_file() is not True:
        with open(file_name, "wb") as handle:
            for data in tqdm(response.iter_content(chunk_size=1024 * 1024)):  # in MB
                if data:
                    handle.write(data)
    else:
        logging.warning("already downloaded!")

    return file_name


def _read_gzip_df(file_name):
    """Read .gz file using pandas (used in get_higgs_dataset function). Can be slow."""
    logging.warning("started reading zip!")
    df = pd.read_csv(file_name)
    logging.warning("done reading zip!")
    return df


def get_higgs_dataset(data_dir="data/"):
    """Creates higgs dataset dataframe. Runs download_higgs_dataset and _read_gzip_df.

    Parameters
    ----------
    data_dir : str, optional
        Path to higgs data, by default "data/".

    Returns
    -------
    pd.DataFrame
        29 dim dataframe of all downloaded data.

    """
    file_name = download_higgs_dataset(data_dir)
    df = _read_gzip_df(file_name)
    df.columns = ORIGINAL_COLNAMES
    return df


def process_higgs_dataset(df, drop_colnames=None, keep_ratio=1.0, shuffle=True, save_path=None, file_name="HIGGS_new"):
    """Process higgs dataset:

    - drop columns
    - delete fraction of data
    - shuffle
    - rename

    Parameters
    ----------
    df : pd.DataFrame
        See get_higgs_dataset.
    drop_colnames : list of str, optional
        Drop these columns, by default None.
    keep_ratio : float, optional
        Fraction of data to keep, by default 1.
    shuffle : bool, optional
        Shuffle data, by default True.
    save_path : str, optional
        Save path, by default None. If not None return df that already exists.
    file_name : str, optional
        Processed file name, by default "HIGGS_new".

    Returns
    -------
    pd.DataFrame
        Processed dataframe.

    """
    logging.warning("started preprocessing...")

    if drop_colnames:
        df = df.drop(columns=drop_colnames)

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    df = df[: int(len(df) * keep_ratio)]

    if save_path:
        sp = save_path + file_name + ".csv"
        if not Path(sp).is_file():
            logging.warning("saving...")
            df.to_csv(sp, index=False)
            logging.warning(f"saved to {sp}.")
        else:
            logging.warning(f"{sp} already exists, returning df!")

    return df


def make_npy_files(file_path, subsets, sizes, bkg_only=False, sig_only=False, df=None):
    """Make .npy files from higgs.csv file. Used for faster loading times.

    Parameters
    ----------
    file_path: str
        File path to csv.
    subsets: list of str
        Name of .npy files: ["train", "val", "test"].
    sizes: list of ints
        Number of rows to keep in each npy file.
    bkg_only: bool, optional
        If True keep only background, by default False.
    sig_only: bool, optional
        If True keep only signal events, by default False.
    df: pd.DataFrame, optional
        None to load the df else df instance.

    """
    if df is not None:
        X = df
    else:
        X = pd.read_csv(file_path)

    path = Path(file_path)
    name = str(path).rstrip("".join(path.suffixes))
    sig_type = "bkg" if bkg_only else ("sig" if sig_only else "")

    if type(X) == pd.DataFrame:
        X = X.to_numpy()
    elif type(X) == np.ndarray:
        pass
    else:
        raise ValueError

    if bkg_only:
        idx = X[:, 0] == 0
        X = X[idx]

    if sig_only:
        idx = X[:, 0] == 1
        X = X[idx]

    c = 0
    for set, size in zip(subsets, sizes):
        save_path = f"{name}_{sig_type}_{set}.npy"
        logging.warning(f"saved {size} rows to {save_path}")
        np.save(save_path, X[c : c + size])
        c += size


if __name__ == "__main__":
    df = get_higgs_dataset("data/higgs/")

    df = process_higgs_dataset(
        df,
        drop_colnames=[
            "jet1 btag",
            "jet2 btag",
            "jet3 btag",
            "jet4 btag",
            "lepton phi",
            "missing energy phi",
            "jet1 phi",
            "jet2 phi",
            "jet3 phi",
            "jet4 phi",
        ],
        keep_ratio=1,
        shuffle=True,
        save_path="data/higgs/",
        file_name="HIGGS_18_features",
    )

    make_npy_files(
        "data/higgs/HIGGS_18_features.csv",
        subsets=["train", "val", "test"],
        sizes=[10 ** 6, 10 ** 6, 10 ** 6],
        bkg_only=True,
        df=df,
    )
