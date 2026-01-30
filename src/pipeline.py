from data.load_data import load_stock_data,load_fnspid_news
from data.preprocess import compute_historical_volatility
from sentiment.sentiment_score import compute_daily_sentiment
from models.black_scholes import black_scholes_call, black_scholes_put
from models.sentiment_adjusted_bs import adjust_volatility
from config import (
    RISK_FREE_RATE,
    TIME_TO_MATURITY_YEARS,
    STRIKE_MULTIPLIER,
    SENTIMENT_ALPHA,
)
from evaluation.metrics import mean_absolute_error, root_mean_squared_error


def run_pipeline():
    # --------------------------------------------------
    # 1. Load & preprocess stock data
    # --------------------------------------------------
    stock_df = load_stock_data(
        ticker="NVDA",
        start="2020-01-01",
        end="2022-12-31",
    )

    stock_df = compute_historical_volatility(stock_df)

    stock_df["strike_price"] = stock_df["stock_price"] * STRIKE_MULTIPLIER
    stock_df["time_to_maturity"] = TIME_TO_MATURITY_YEARS

    # --------------------------------------------------
    # 2. Baseline Black–Scholes pricing
    # --------------------------------------------------
    call_prices = []
    put_prices = []

    for row in stock_df.itertuples(index=False):
        call_prices.append(
            black_scholes_call(
                S=row.stock_price,
                K=row.strike_price,
                T=row.time_to_maturity,
                r=RISK_FREE_RATE,
                sigma=row.historical_volatility,
            )
        )

        put_prices.append(
            black_scholes_put(
                S=row.stock_price,
                K=row.strike_price,
                T=row.time_to_maturity,
                r=RISK_FREE_RATE,
                sigma=row.historical_volatility,
            )
        )

    stock_df["bs_call_price"] = call_prices
    stock_df["bs_put_price"] = put_prices

    # --------------------------------------------------
    # 3. Sentiment pipeline
    # --------------------------------------------------
    news_df = load_fnspid_news(
        path="data/raw/nasdaq_external_data.csv",
        ticker="NVDA",
        start_date="2020-01-01",
        end_date="2022-12-31",
        )

    print("\nFNSPID news rows:", len(news_df))
    print(news_df.head())

    daily_sentiment = compute_daily_sentiment(news_df)


    stock_df["date"] = stock_df["Date"].dt.date

    stock_df = stock_df.merge(
        daily_sentiment,
        how="left",
        left_on="date",
        right_on="date",
    )

    # Treat missing sentiment as neutral
    stock_df["daily_sentiment"] = stock_df["daily_sentiment"].fillna(0.0)

    # --------------------------------------------------
    # 4. Sentiment-adjusted volatility
    # --------------------------------------------------
    stock_df["adjusted_volatility"] = stock_df.apply(
        lambda row: adjust_volatility(
            base_volatility=row["historical_volatility"],
            sentiment=row["daily_sentiment"],
            alpha=SENTIMENT_ALPHA,
        ),
        axis=1,
    )

    # --------------------------------------------------
    # 5. Sentiment-adjusted Black–Scholes pricing
    # --------------------------------------------------
    adj_call_prices = []
    adj_put_prices = []

    for row in stock_df.itertuples(index=False):
        adj_call_prices.append(
            black_scholes_call(
                S=row.stock_price,
                K=row.strike_price,
                T=row.time_to_maturity,
                r=RISK_FREE_RATE,
                sigma=row.adjusted_volatility,
            )
        )

        adj_put_prices.append(
            black_scholes_put(
                S=row.stock_price,
                K=row.strike_price,
                T=row.time_to_maturity,
                r=RISK_FREE_RATE,
                sigma=row.adjusted_volatility,
            )
        )

    stock_df["adj_call_price"] = adj_call_prices
    stock_df["adj_put_price"] = adj_put_prices

    # --------------------------------------------------
    # 6. Sanity check output
    # --------------------------------------------------
    print(
        stock_df[
            [
                "Date",
                "stock_price",
                "historical_volatility",
                "daily_sentiment",
                "adjusted_volatility",
                "bs_call_price",
                "adj_call_price",
                "bs_put_price",
                "adj_put_price",
            ]
        ].head(10)
    )

    # --------------------------------------------------
    # 7. Evaluation (baseline vs sentiment-adjusted)
    # --------------------------------------------------
    call_mae = mean_absolute_error(
        stock_df["bs_call_price"],
        stock_df["adj_call_price"],
    )

    call_rmse = root_mean_squared_error(
        stock_df["bs_call_price"],
        stock_df["adj_call_price"],
    )

    put_mae = mean_absolute_error(
        stock_df["bs_put_price"],
        stock_df["adj_put_price"],
    )

    put_rmse = root_mean_squared_error(
        stock_df["bs_put_price"],
        stock_df["adj_put_price"],
    )

    print("\n=== Evaluation Results ===")
    print(f"Call MAE  : {call_mae:.6f}")
    print(f"Call RMSE : {call_rmse:.6f}")
    print(f"Put MAE   : {put_mae:.6f}")
    print(f"Put RMSE  : {put_rmse:.6f}")

# --------------------------------------------------
# 8. Visualization: Baseline vs Sentiment-Adjusted Prices
# --------------------------------------------------
# This plot compares standard Black–Scholes call option prices
# with sentiment-adjusted Black–Scholes prices over time.
#
# The goal is to visually demonstrate how incorporating
# news-based sentiment into volatility affects option pricing,
# while keeping the original Black–Scholes framework intact.

    import matplotlib.pyplot as plt

    plt.figure()

# Baseline Black–Scholes call prices
    plt.plot(
        stock_df["Date"],
        stock_df["bs_call_price"],
        label="Baseline Black–Scholes",
    )

# Sentiment-adjusted Black–Scholes call prices
    plt.plot(
        stock_df["Date"],
        stock_df["adj_call_price"],
        label="Sentiment-Adjusted Black–Scholes",
        )

    plt.title("Baseline vs Sentiment-Adjusted Call Prices (NVDA)")
    plt.xlabel("Date")
    plt.ylabel("Option Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return stock_df


if __name__ == "__main__":
    run_pipeline()
