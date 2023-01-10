import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_and_preprocess_tsla_hist_data(start_date, end_date):
    tsla_hist = yf.download(
        tickers="TSLA",
        period="max",
        interval="1d",
        group_by=" ticker",
        auto_adjust=False,
        prepost=False,
        threads=10
    )
    tsla_hist = tsla_hist.reset_index()
    tsla_hist['Date'] = pd.to_datetime(tsla_hist['Date']).dt.date
    tsla_hist = tsla_hist[tsla_hist['Date'] >= start_date]
    tsla_hist['month'] = tsla_hist['Date'].astype(str).str.split('-')
    tsla_hist['month'] = tsla_hist['month'].apply(lambda x: '-'.join(x[0:2]))
    tsla_hist = tsla_hist[(tsla_hist['Date'] >= start_date) & (tsla_hist['Date'] <= end_date)]
    tsla_hist.to_csv("training_data/tsla_hist.csv", index=False)
    return tsla_hist
