import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def compute_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily sentiment scores from news headlines using VADER.
    """
    sia = SentimentIntensityAnalyzer()

    news_df = news_df.copy()
    news_df["sentiment"] = news_df["title"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )

    daily_sentiment = (
        news_df.groupby("date")["sentiment"]
        .mean()
        .reset_index()
        .rename(columns={"sentiment": "daily_sentiment"})
    )

    return daily_sentiment
