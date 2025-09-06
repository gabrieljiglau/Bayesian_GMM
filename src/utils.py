import numpy as np
import pandas as pd

def init_cov_matrix(dim: int, small: bool=True):

    if small:
        return np.diag(np.random.rand(dim) + 0.1)
    else:
        sigma = np.random.rand(dim, dim)
        return sigma * sigma.transpose()

def preprocess_data(df: pd.DataFrame, out_name: str):

    df['species'], original_string = pd.factorize(df['species'])
    df.drop_duplicates()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna()
    df.to_csv(out_name, index=False)

def split_data(df: pd.DataFrame):

    print(f"df.shape[0] = {df.shape[0]}")
    size = int(0.9 * df.shape[0])
    print(f"train size = {size}")
    x_train, y_train = df.iloc[:size, :-1], df.iloc[:size, -1:]
    x_test, y_test = df.iloc[size:, :-1], df.iloc[size:, -1:]
    return x_train, y_train, x_test, y_test
