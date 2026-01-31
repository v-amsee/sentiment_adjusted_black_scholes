import pandas as pd
import csv


def load_nvda_option_chains(
    path: str,
    min_dte: int = 20,
    max_dte: int = 40,
    moneyness_tol: float = 0.05,
):
    """
    Load NVDA option chains and select near-the-money calls and puts
    using LAST price + IMPLIED volatility.
    """

    csv.field_size_limit(10**7)
    records = []

    for chunk in pd.read_csv(
        path,
        engine="python",
        encoding="utf-8",
        chunksize=100_000,
    ):

        # Clean column names
        chunk.columns = (
            chunk.columns
            .str.strip()
            .str.replace("[", "", regex=False)
            .str.replace("]", "", regex=False)
        )

        chunk["QUOTE_DATE"] = pd.to_datetime(
            chunk["QUOTE_DATE"], errors="coerce"
        ).dt.date

        chunk = chunk.dropna(subset=["QUOTE_DATE"])

        chunk = chunk[
            (chunk["DTE"] >= min_dte) &
            (chunk["DTE"] <= max_dte)
        ]

        chunk["moneyness"] = (
            abs(chunk["STRIKE"] - chunk["UNDERLYING_LAST"])
            / chunk["UNDERLYING_LAST"]
        )

        chunk = chunk[chunk["moneyness"] <= moneyness_tol]

        for date, day_df in chunk.groupby("QUOTE_DATE"):

            spot = day_df["UNDERLYING_LAST"].iloc[0]

            # CALL
            calls = day_df.dropna(subset=["C_LAST", "C_IV"])

            if not calls.empty:

                call = calls.iloc[
                    (calls["STRIKE"] - spot).abs().argmin()
                ]

                records.append({
                    "date": date,
                    "type": "call",
                    "market_price": call["C_LAST"],
                    "implied_vol": call["C_IV"],
                    "strike": call["STRIKE"],
                    "spot": spot,
                    "time_to_maturity": call["DTE"] / 365.0,
                })

            # PUT
            puts = day_df.dropna(subset=["P_LAST", "P_IV"])

            if not puts.empty:

                put = puts.iloc[
                    (puts["STRIKE"] - spot).abs().argmin()
                ]

                records.append({
                    "date": date,
                    "type": "put",
                    "market_price": put["P_LAST"],
                    "implied_vol": put["P_IV"],
                    "strike": put["STRIKE"],
                    "spot": spot,
                    "time_to_maturity": put["DTE"] / 365.0,
                })

    df = pd.DataFrame(records)

    numeric_cols = [
        "market_price",
        "implied_vol",
        "strike",
        "spot",
        "time_to_maturity",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    # Convert IV if dataset uses %
    if df["implied_vol"].mean() > 2:
        df["implied_vol"] /= 100

    print("\nLoaded options:", len(df))
    print("Average IV:", round(df["implied_vol"].mean(), 3))

    return df
