import pandas as pd
from textblob import TextBlob
import re
import ast
import datetime
import os


def get_sentiment_from_tweet(tweet: str):
    cleaned_tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])| (\w+:\ / \ / \S+)", " ", tweet).split())
    analysis = TextBlob(cleaned_tweet)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'


def get_nr_sentiments(x):
    pos = 0
    neg = 0
    neut = 0
    for sent in x:
        if sent == "positive":
            pos += 1
        elif sent == "negative":
            neg += 1
        elif sent=="neutral":
            neut += 1
    return pos, neg, neut


def clean_none_lists(x):
    if all(v is None for v in x) or all(pd.isna(v) for v in x):
        return None
    else:
        return x


def encode_time(x):
    night = 0 # 0-6
    morning = 0 # 6-12
    afternoon = 0 # 12-18
    evening = 0 # 18-24
    night_start = datetime.datetime.strptime("00:00:00", '%H:%M:%S').time()
    morning_start = datetime.datetime.strptime("06:00:00", '%H:%M:%S').time()
    afternoon_start = datetime.datetime.strptime("12:00:00", '%H:%M:%S').time()
    evening_start = datetime.datetime.strptime("18:00:00", '%H:%M:%S').time()
    if x:
        for time in x:
            dt_time = datetime.datetime.strptime(time, '%H:%M:%S').time()
            if night_start <= dt_time < morning_start:
                night += 1
            elif morning_start <= dt_time < afternoon_start:
                morning += 1
            elif afternoon_start <= dt_time < evening_start:
                afternoon += 1
            else:
                evening += 1
    return night, morning, afternoon, evening


def preprocess_tweets(tweets_df: pd.DataFrame):
    tweets_df = tweets_df.drop(
        columns=['id', 'conversation_id', 'created_at', 'cashtags', 'timezone', 'place', 'language', 'user_id',
                 'user_id_str','retweet',
                 'username', 'name', 'day', 'hour', 'link', 'urls', 'thumbnail', 'quote_url', 'search', 'near', 'geo',
                 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'retweet_date', 'translate', 'trans_src',
                 'trans_dest'])
    tweets_df = tweets_df[tweets_df.columns[1:]]
    tweets_df = tweets_df.drop_duplicates()
    tweets_df = tweets_df.reset_index().drop(columns=['index'])
    tweets_df['nlikes'] = tweets_df['nlikes'].fillna(tweets_df['likes_count'])
    tweets_df['nreplies'] = tweets_df['nreplies'].fillna(tweets_df['replies_count'])
    tweets_df['nretweets'] = tweets_df['nretweets'].fillna(tweets_df['retweets_count'])
    tweets_df = tweets_df.drop(columns=['likes_count', 'replies_count', 'retweets_count'])
    tweets_df['sentiment'] = tweets_df['tweet'].apply(lambda x: get_sentiment_from_tweet(x))
    tweets_df['date'] = pd.to_datetime(tweets_df['date']).dt.date
    tweets_df['sentiments'] = tweets_df.groupby('date')['sentiment'].transform(lambda x: [x.tolist()] * len(x))
    tweets_df['nlikes_sum'] = tweets_df.groupby('date')['nlikes'].transform(sum)
    tweets_df['nreplies_sum'] = tweets_df.groupby('date')['nreplies'].transform(sum)
    tweets_df['nretweets_sum'] = tweets_df.groupby('date')['nretweets'].transform(sum)
    tweets_df["mentions"] = tweets_df["mentions"].apply(lambda x: len(ast.literal_eval(x)) if not pd.isna(x) else 0)
    tweets_df["reply_to"] = tweets_df["reply_to"].apply(lambda x: len(ast.literal_eval(x)) if not pd.isna(x) else 0)
    tweets_df["mentions"] = tweets_df.groupby('date')['mentions'].transform(sum)
    tweets_df["reply_to"] = tweets_df.groupby('date')['reply_to'].transform(sum)
    tweets_df['hashtags'] = tweets_df['hashtags'].apply(lambda x: len(ast.literal_eval(x)))
    tweets_df['hashtags'] = tweets_df.groupby('date')['hashtags'].transform(sum)
    tweets_df['photos'] = tweets_df['photos'].apply(lambda x: 0 if x == '[]' else 1)
    tweets_df['photos_sum'] = tweets_df.groupby('date')['photos'].transform(sum)
    tweets_df['videos_sum'] = tweets_df.groupby('date')['video'].transform(sum)
    tweets_df['time_list'] = tweets_df.groupby('date')['time'].transform(lambda x: [x.tolist()] * len(x))
    tweets_df = tweets_df.drop(columns=['tweet', 'hashtags', 'photos', 'video', 'nlikes',
                                        'nreplies', 'nretweets', 'time', 'sentiment'])
    tweets_df["time_list"] = tweets_df["time_list"].apply(lambda x: clean_none_lists(x))
    tweets_df["time_encoded"] = tweets_df["time_list"].apply(lambda x: encode_time(x))
    tweets_df[['nr_night_posts', 'nr_morning_posts', 'nr_afternoon_posts', 'nr_evening_posts']] = pd.DataFrame(tweets_df['time_encoded'].tolist(), index=tweets_df.index)
    tweets_df['nr_sentiments'] = tweets_df['sentiments'].apply(lambda x: get_nr_sentiments(x))
    tweets_df[['nr_positive', 'nr_negative', 'nr_neutral']] = pd.DataFrame(tweets_df['nr_sentiments'].tolist(), index=tweets_df.index)
    tweets_df = tweets_df.drop(columns=['nr_sentiments', 'sentiments', 'time_encoded', 'time_list'])
    return tweets_df


def load_and_preprocess_tweets():
    tweets_path = r"../raw_data/tweets"
    tweets_df = pd.DataFrame()
    for file in os.listdir(tweets_path):
        file_path = os.path.join(tweets_path, file)
        year_x_tweets = pd.read_csv(file_path)
        tweets_df = pd.concat([tweets_df, year_x_tweets])

    tweets_df = preprocess_tweets(tweets_df)
    tweets_df.to_csv("training_data/tweets.csv", index=False)
    return tweets_df


