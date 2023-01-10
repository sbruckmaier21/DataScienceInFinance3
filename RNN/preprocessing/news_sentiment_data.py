import pandas as pd
import os
import matplotlib.pyplot as plt


def load_and_preprocess_news_sentiment_data(start_date, end_date):
    news_sentiment_path = r"../raw_data/news_sentiment_data.csv"
    news_sentiment_data = pd.read_csv(news_sentiment_path, sep=";")
    news_sentiment_data['date'] = pd.to_datetime(news_sentiment_data['date'], format='%d.%m.%Y').dt.date
    news_sentiment_data = news_sentiment_data[
        (news_sentiment_data['date'] >= start_date) & (news_sentiment_data['date'] <= end_date)]
    news_sentiment_data['News Sentiment'] = news_sentiment_data['News Sentiment'].str.replace(',', '.').astype(float)
    news_sentiment_data.to_csv('training_data/news_sentiment_data.csv', index=False)
    return news_sentiment_data