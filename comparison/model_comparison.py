import pandas as pd
import matplotlib.pyplot as plt
import datetime
import scipy.stats
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def create_plot(arima_pred, rnn_pred, actual, save: bool=False):
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot(dates, rnn_pred, label='RNN Prediction')
    ax.plot(dates, arima_pred, label='ARIMA Prediction')
    ax.plot(dates, actual, label='Actual')

    ax.set_xlabel('Date')
    ax.set_ylabel('Result')
    ax.set_ylim(15, 25)
    ax.legend()
    plt.xticks(rotation=45)
    if save:
        plt.savefig("comparison.png")
    else:
        plt.show()


def test_hypothesis(arima_pred, rnn_pred, actual):
    error_1 = np.mean(abs(arima_pred - actual))
    error_2 = np.mean(abs(rnn_pred - actual))

    se_1 = scipy.stats.sem(abs(arima_pred - actual))
    se_2 = scipy.stats.sem(abs(rnn_pred - actual))

    t, p = scipy.stats.ttest_ind_from_stats(error_1, se_1, len(arima_pred), error_2, se_2,
                                            len(rnn_pred))

    #p-value
    print(p)


def get_rmses(arima_pred, rnn_pred, actual):
    rmse_arima = np.sqrt(mean_squared_error(actual, arima_pred))
    rmse_rnn = np.sqrt(mean_squared_error(actual, rnn_pred))
    print("RMSE ARIMA: ", rmse_arima)
    print("RMSE RNN: ", rmse_rnn)


def get_r_squared(arima_pred, rnn_pred, actual):
    r2_arima = r2_score(actual, arima_pred)
    r2_rnn = r2_score(actual, rnn_pred)
    print("R2 ARIMA: ", r2_arima)
    print("R2 RNN: ", r2_rnn)




if __name__ == "__main__":
    arima_results = pd.read_excel("result_arima121.xlsx")['predicted_mean']
    rnn_results = pd.read_csv("results.csv")
    rnn_results['Date'] = pd.to_datetime(rnn_results['Date']).dt.date
    #rnn_results = rnn_results.loc[rnn_results['Date'] < datetime.datetime.strptime('2018-02-01', '%Y-%m-%d').date()]
    dates = rnn_results['Date']
    create_plot(arima_pred=arima_results, rnn_pred=rnn_results['prediction'], actual=rnn_results['actual'], save=True)
    #test_hypothesis(arima_pred=arima_results, rnn_pred=rnn_results['prediction'], actual=rnn_results['actual'])
    #get_rmses(arima_pred=arima_results, rnn_pred=rnn_results['prediction'], actual=rnn_results['actual'])
    #get_r_squared(arima_pred=arima_results, rnn_pred=rnn_results['prediction'], actual=rnn_results['actual'])
