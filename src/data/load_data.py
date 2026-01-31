import yfinance as yf
import pandas as pd
from datetime import date
import csv


def load_fnspid_news(
    path: str,
    ticker: str,
    start_date: str,
    end_date: str,
):
    """
    Memory-safe loader for FNSPID.
    Only loads minimal columns required for sentiment.
    """

    csv.field_size_limit(10**7)
    records = []

    usecols = ["Date", "Article_title", "Stock_symbol"]

    for chunk in pd.read_csv(
        path,
        engine="python",
        encoding="utf-8",
        usecols=usecols,          # ⭐ KEY FIX
        chunksize=100_000,
        on_bad_lines="skip",
    ):

        # Parse date safely
        chunk["Date"] = pd.to_datetime(
            chunk["Date"], errors="coerce"
        ).dt.date

        chunk = chunk.dropna(subset=["Date"])

        # Filter early (VERY important for memory)
        chunk = chunk[
            (chunk["Stock_symbol"] == ticker) &
            (chunk["Date"] >= pd.to_datetime(start_date).date()) &
            (chunk["Date"] <= pd.to_datetime(end_date).date())
        ]

        if not chunk.empty:
            records.append(
                chunk[["Date", "Article_title"]]
            )

    if not records:
        return pd.DataFrame(columns=["date", "title"])

    news_df = pd.concat(records, ignore_index=True)

    news_df = news_df.rename(
        columns={
            "Date": "date",
            "Article_title": "title",
        }
    )

    return news_df



def load_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Load historical stock price data using yfinance.
    """
    stock = yf.download(ticker, start=start, end=end, progress=False)

    # Ensure clean columns
    stock = stock.reset_index()
    stock = stock.loc[:, ["Date", "Close"]].copy()

    stock.columns = ["Date", "stock_price"]
    stock["stock_price"] = stock["stock_price"].astype(float)

    return stock

def load_news_data(ticker: str) -> pd.DataFrame:
    """
    Load recent news headlines for a stock using yfinance.
    Handles nested Yahoo Finance news structure.
    """
    stock = yf.Ticker(ticker)
    news = stock.news

    if not news:
        return pd.DataFrame(columns=["date", "title"])

    records = []

    for item in news:
        content = item.get("content", {})

        title = content.get("title") or content.get("summary")
        pub_date = content.get("pubDate")

        if title and pub_date:
            records.append(
                {
                    "date": pd.to_datetime(pub_date).date(),
                    "title": title,
                }
            )

    return pd.DataFrame(records)
