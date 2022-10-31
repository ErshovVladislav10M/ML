import pandas as pd


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    res = (df - df.mean()) / df.std()
    return res.fillna(1)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    res = (df - df.min()) / (df.max() - df.min())
    return res.fillna(1)
