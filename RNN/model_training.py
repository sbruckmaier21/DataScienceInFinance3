import os
import pandas as pd
import numpy as np
import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from preprocessing.tsla_hist import load_and_preprocess_tsla_hist_data
from preprocessing.tweets import load_and_preprocess_tweets
from preprocessing.econ_data import load_and_preprocess_econ_data
from preprocessing.news_sentiment_data import load_and_preprocess_news_sentiment_data
import keras_tuner as kt

def get_train_data(start_date, end_date):
    if os.path.exists("training_data/tsla_hist.csv"):
        tsla_hist = pd.read_csv("training_data/tsla_hist.csv")
        tsla_hist['Date'] = pd.to_datetime(tsla_hist['Date']).dt.date
    else:
        tsla_hist = load_and_preprocess_tsla_hist_data(start_date, end_date)

    if os.path.exists("training_data/tweets.csv"):
        tweets_df = pd.read_csv("training_data/tweets.csv")
        tweets_df['date'] = pd.to_datetime(tweets_df['date']).dt.date
        tweets_df = tweets_df.drop(columns=['nr_night_posts', 'nr_morning_posts', 'nr_afternoon_posts', 'nr_evening_posts'])
    else:
        tweets_df = load_and_preprocess_tweets()

    if os.path.exists("training_data/econ_data_oecd.csv"):
        econ_data = pd.read_csv("training_data/econ_data_oecd.csv")
    else:
        econ_data = load_and_preprocess_econ_data()

    if os.path.exists("training_data/news_sentiment_data.csv"):
        news_sentiment_data = pd.read_csv("training_data/news_sentiment_data.csv")
        news_sentiment_data['date'] = pd.to_datetime(news_sentiment_data['date']).dt.date
    else:
        news_sentiment_data = load_and_preprocess_news_sentiment_data(start_date, end_date)

    df = tsla_hist.merge(tweets_df, left_on='Date', right_on='date', how='left').merge(econ_data,  left_on='month', right_on='time', how='left').merge(news_sentiment_data, left_on='Date', right_on='date', how='left')
    df = df.drop(columns=['time', 'month', 'date_x', 'date_y', df.columns[23]])
    df = df.drop_duplicates()
    df = df.set_index('Date')
    return df


def get_train_test_split(df: pd.DataFrame):
    train = df.loc[:datetime.datetime.strptime('2017-12-31', '%Y-%m-%d').date(),:]
    test = df.loc[datetime.datetime.strptime('2018-01-01', '%Y-%m-%d').date():datetime.datetime.strptime('2018-08-31', '%Y-%m-%d').date(),:]
    x_train = np.array(train.drop(columns='Close'))
    y_train = np.array(train['Close'])
    x_test = np.array(test.drop(columns='Close'))
    y_test = np.array(test['Close'])
    return x_train, y_train, x_test, y_test


def build_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    #model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(32))
    model.add(Dense(1))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model


if __name__ == "__main__":
    start_date = datetime.datetime.strptime('2012-01-01', '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime('2018-08-31', '%Y-%m-%d').date()

    df = get_train_data(start_date, end_date)

    x_train, y_train, x_test, y_test = get_train_test_split(df)

    # Scaling
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Imputation
    x_train = np.nan_to_num(x_train, nan=-1)
    x_test = np.nan_to_num(x_test, nan=-1)

    # Preparing for Tensor format
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # model training:
    model = build_model()
    model.fit(x_train, y_train, batch_size=1, epochs=50)
    # model evaluation
    score = model.evaluate(x_test, y_test, batch_size=32)
    predictions = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    results = pd.DataFrame(data=[predictions.flatten(), y_test]).T
    results = results.rename(columns={0: 'prediction', 1: 'actual'})
    results['FE'] = (results['actual'] - results['prediction']) / results['actual']
    results.index = df.loc[datetime.datetime.strptime('2018-01-01', '%Y-%m-%d').date():datetime.datetime.strptime('2018-08-31','%Y-%m-%d').date(),:].index
    #model.save(r"C:\Users\User\PycharmProjects\seminarproject\models\m1")
    print("RMSE: ", rmse)
    print(results)
    print(results['FE'].mean())
    print("Score: ", score)
    #results.to_csv(r"C:\Users\User\PycharmProjects\seminarproject\comparison\results.csv")
