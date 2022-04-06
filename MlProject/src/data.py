from sklearn.model_selection import train_test_split
import pandas as pd
import typing as t


def split_data(df_x: pd.DataFrame, df_y: pd.DataFrame, size=0.2) -> t.Tuple[t.List]:
    """
    The function split data into 4 lists (Ex: x_train, x_test, y_train, y_test) based on given size
    df: Description of df Ex: pd.DataFrame("/path")
    size: Splitting size for given data Ex: size=0.2
    """
    return train_test_split(df_x.values, df_y.values, test_size=size)
