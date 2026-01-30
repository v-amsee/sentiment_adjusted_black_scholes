import pandas as pd
import numpy as np


def compute_historical_volatility(
    df: pd.DataFrame,
    window: int = 30,
    trading_days: int = 252
) -> pd.DataFrame:
    """
    Compute rolling historical volatility.
    """
    df = df.copy()
    df["log_return"] = np.log(df["stock_price"] / df["stock_price"].shift(1))

    df["historical_volatility"] = (
        df["log_return"]
        .rolling(window)
        .std()
        * np.sqrt(trading_days)
    )

    df.dropna(inplace=True)
    return df
