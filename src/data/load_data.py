import yfinance as yf
import pandas as pd
from datetime import date


def load_fnspid_news(
    path: str,
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Load historical news from FNSPID (Nasdaq subset),
    filtered by ticker and date range.
    Uses chunked reading to handle very large text fields.
    """
    import csv
    csv.field_size_limit(10**7)  # increase field size limit safely

    chunks = []

    for chunk in pd.read_csv(
        path,
        usecols=["Date", "Article_title", "Stock_symbol"],
        engine="python",
        encoding="utf-8",
        chunksize=100_000,
    ):
        # Rename columns
        chunk = chunk.rename(
            columns={
                "Date": "date",
                "Article_title": "title",
                "Stock_symbol": "symbol",
            }
        )

        # Parse dates safely
        chunk["date"] = pd.to_datetime(
            chunk["date"], errors="coerce"
        ).dt.date
        chunk = chunk.dropna(subset=["date"])

        # Filter ticker
        chunk = chunk[chunk["symbol"] == ticker]

        # Filter date range
        chunk = chunk[
            (chunk["date"] >= pd.to_datetime(start_date).date()) &
            (chunk["date"] <= pd.to_datetime(end_date).date())
        ]

        if not chunk.empty:
            chunks.append(chunk[["date", "title"]])

    if not chunks:
        return pd.DataFrame(columns=["date", "title"])

    return pd.concat(chunks, ignore_index=True)


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
