import pandas as pd
import matplotlib.pyplot as plt

from data.load_data import load_stock_data, load_fnspid_news
from data.load_option_data import load_nvda_option_chains
from data.preprocess import compute_historical_volatility
from sentiment.sentiment_score import compute_daily_sentiment
from models.black_scholes import black_scholes_call, black_scholes_put
from models.sentiment_adjusted_bs import adjust_volatility
from config import (
    RISK_FREE_RATE,
    SENTIMENT_ALPHA,
)
from evaluation.metrics import mean_absolute_error, root_mean_squared_error


def run_pipeline():

    # ==================================================
    # 1. LOAD STOCK + SENTIMENT DATA
    # ==================================================
    stock_df = load_stock_data(
        ticker="NVDA",
        start="2020-01-01",
        end="2022-12-31",
    )

    stock_df = compute_historical_volatility(stock_df)
    stock_df["date"] = stock_df["Date"].dt.date

    # Load news
    news_df = load_fnspid_news(
        path="data/raw/nasdaq_external_data.csv",
        ticker="NVDA",
        start_date="2020-01-01",
        end_date="2022-12-31",
    )

    print("\nFNSPID news rows:", len(news_df))

    daily_sentiment = compute_daily_sentiment(news_df)

    stock_df = stock_df.merge(
        daily_sentiment,
        how="left",
        on="date",
    )

    stock_df["daily_sentiment"] = stock_df["daily_sentiment"].fillna(0.0)

    # Sentiment-adjusted historical volatility (kept for research comparison)
    stock_df["adjusted_volatility"] = stock_df.apply(
        lambda row: adjust_volatility(
            base_volatility=row["historical_volatility"],
            sentiment=row["daily_sentiment"],
            alpha=SENTIMENT_ALPHA,
        ),
        axis=1,
    )

    # ==================================================
    # 2. LOAD REAL OPTION DATA (WITH IMPLIED VOL)
    # ==================================================
    print("\nLoading real option market data...")

    options_df = load_nvda_option_chains(
        path="data/raw/nvda_2020_2022.csv"
    )

    # Merge sentiment only (NOT historical vol anymore)
    options_df = options_df.merge(
        stock_df[["date", "daily_sentiment"]],
        on="date",
        how="left",
    ).dropna()

    print("Options ready for pricing:", len(options_df))

    # Safety check
    print("\nImplied Vol Summary:")
    print(options_df["implied_vol"].describe())

    # ==================================================
    # 3. BASELINE — BLACK SCHOLES USING IMPLIED VOL
    # ==================================================
    bs_prices = []

    for row in options_df.itertuples(index=False):

        if row.type == "call":
            price = black_scholes_call(
                S=row.spot,
                K=row.strike,
                T=row.time_to_maturity,
                r=RISK_FREE_RATE,
                sigma=row.implied_vol,  # ⭐ KEY CHANGE
            )
        else:
            price = black_scholes_put(
                S=row.spot,
                K=row.strike,
                T=row.time_to_maturity,
                r=RISK_FREE_RATE,
                sigma=row.implied_vol,  # ⭐ KEY CHANGE
            )

        bs_prices.append(price)

    options_df["bs_implied_price"] = bs_prices

    # ==================================================
    # 4. SENTIMENT-ADJUSTED IMPLIED VOL
    # ==================================================
    adj_prices = []

    for row in options_df.itertuples(index=False):

        adj_vol = adjust_volatility(
            base_volatility=row.implied_vol,
            sentiment=row.daily_sentiment,
            alpha=SENTIMENT_ALPHA,
        )

        if row.type == "call":
            price = black_scholes_call(
                S=row.spot,
                K=row.strike,
                T=row.time_to_maturity,
                r=RISK_FREE_RATE,
                sigma=adj_vol,
            )
        else:
            price = black_scholes_put(
                S=row.spot,
                K=row.strike,
                T=row.time_to_maturity,
                r=RISK_FREE_RATE,
                sigma=adj_vol,
            )

        adj_prices.append(price)

    options_df["sentiment_price"] = adj_prices

    # ==================================================
    # 5. NUMERIC SAFETY (real datasets are messy)
    # ==================================================
    numeric_cols = [
        "market_price",
        "bs_implied_price",
        "sentiment_price",
    ]

    for col in numeric_cols:
        options_df[col] = pd.to_numeric(options_df[col], errors="coerce")

    options_df = options_df.dropna(subset=numeric_cols)

    # ==================================================
    # 6. REAL MARKET EVALUATION
    # ==================================================
    print("\n=========== IMPLIED VOL VALIDATION ===========")

    for opt_type in ["call", "put"]:

        subset = options_df[options_df["type"] == opt_type]

        mae_bs = mean_absolute_error(
            subset["market_price"],
            subset["bs_implied_price"],
        )

        mae_adj = mean_absolute_error(
            subset["market_price"],
            subset["sentiment_price"],
        )

        rmse_bs = root_mean_squared_error(
            subset["market_price"],
            subset["bs_implied_price"],
        )

        rmse_adj = root_mean_squared_error(
            subset["market_price"],
            subset["sentiment_price"],
        )

        print(f"\n{opt_type.upper()} OPTIONS")
        print(f"BS (Implied)  → MAE: {mae_bs:.4f}, RMSE: {rmse_bs:.4f}")
        print(f"Sentiment IV  → MAE: {mae_adj:.4f}, RMSE: {rmse_adj:.4f}")

    # ==================================================
    # 7. OPTIONAL — VISUAL CHECK
    # ==================================================
    sample_calls = options_df[options_df["type"] == "call"].head(150)

    plt.figure()

    plt.plot(
        sample_calls["market_price"].values,
        label="Market Price",
    )

    plt.plot(
        sample_calls["bs_implied_price"].values,
        label="BS Implied",
    )

    plt.plot(
        sample_calls["sentiment_price"].values,
        label="Sentiment IV",
    )

    plt.legend()
    plt.title("Market vs Model Prices (Sample)")
    plt.tight_layout()
    plt.show()

    return options_df


if __name__ == "__main__":
    run_pipeline()
