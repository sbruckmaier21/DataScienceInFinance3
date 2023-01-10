import datetime
import os
import pandas as pd
from preprocessing.tsla_hist import load_and_preprocess_tsla_hist_data
from preprocessing.tweets import load_and_preprocess_tweets
from preprocessing.econ_data import load_and_preprocess_econ_data
from preprocessing.news_sentiment_data import load_and_preprocess_news_sentiment_data
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale


def get_train_data(start_date, end_date):
    if os.path.exists("training_data/tsla_hist.csv"):
        tsla_hist = pd.read_csv("training_data/tsla_hist.csv")
        tsla_hist['Date'] = pd.to_datetime(tsla_hist['Date']).dt.date
    else:
        tsla_hist = load_and_preprocess_tsla_hist_data(start_date, end_date)

    if os.path.exists("training_data/tweets.csv"):
        tweets_df = pd.read_csv("training_data/tweets.csv")
        tweets_df['date'] = pd.to_datetime(tweets_df['date']).dt.date
    else:
        tweets_df = load_and_preprocess_tweets(tweets_path)

    if os.path.exists("training_data/econ_data_oecd.csv"):
        econ_data = pd.read_csv("training_data/econ_data_oecd.csv")
    else:
        econ_data = load_and_preprocess_econ_data()

    if os.path.exists("training_data/news_sentiment_data.csv"):
        news_sentiment_data = pd.read_csv("training_data/news_sentiment_data.csv")
        news_sentiment_data['date'] = pd.to_datetime(news_sentiment_data['date']).dt.date
    else:
        news_sentiment_data = load_and_preprocess_news_sentiment_data(start_date, end_date)
    return tsla_hist, tweets_df, econ_data, news_sentiment_data


def plot_news_sentiment(tsla_hist, news_sentiment_data, save: bool = False):
    merged = tsla_hist[['Date', 'Close']].merge(news_sentiment_data[['date', 'News Sentiment']], left_on='Date',
                                                right_on='date')
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(merged['Date'], merged['News Sentiment'], label='News Sentiment')
    ax.plot(merged['Date'], merged['Close'], label='Close')
    ax.set_xlabel('Date')
    ax.legend()
    plt.xticks(rotation=45)
    if save:
        plt.savefig("sentiment_analysis_plot.png")
    else:
        plt.show()


def plot_tweets_data(tsla_hist, tweets_df):
    merged = tsla_hist[['Date', 'Close']].merge(tweets_df, left_on='Date',
                                                right_on='date')
    merged = merged.drop_duplicates()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(merged['Date'], merged['nlikes_sum'], label='Number of Likes')
    ax.plot(merged['Date'], merged['Close'], label='Close')
    ax.set_xlabel('Date')
    ax.legend()
    plt.xticks(rotation=45)
    plt.show()


if __name__ == "__main__":
    tweets_path = r"C:\Users\User\Documents\Uni\Master\3_Sem\Data Science in Finance\Project\tweets"
    start_date = datetime.datetime.strptime('2012-01-01', '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime('2018-08-31', '%Y-%m-%d').date()

    tsla_hist, tweets_df, econ_data, news_sentiment_data = get_train_data(start_date, end_date)
    tsla_hist = tsla_hist.loc[tsla_hist['Date'] < end_date,:]
    tsla_hist['Close'] = minmax_scale(tsla_hist['Close'], feature_range=(-1, 1))
    tweets_df = tweets_df.loc[tweets_df['date'] < end_date,:]
    tweets_df['nlikes_sum'] = minmax_scale(tweets_df['nlikes_sum'], feature_range=(-1, 1))
    plot_tweets_data(tsla_hist, tweets_df)

    #news_sentiment_data = news_sentiment_data.loc[news_sentiment_data['date'] < end_date,:]
    #tsla_hist['Close'] = minmax_scale(tsla_hist['Close'], feature_range=(-0.5, 0.5))
    #plot_news_sentiment(tsla_hist, news_sentiment_data, True)
    #print(tsla_hist)




