import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from numpy import log

from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import statsmodels.api as smapi
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_predict
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



df = pd.read_csv("./TSLA (12).csv")

plt.plot(df["Date"], df["Close"])
plt.xticks(np.arange(0,1509, 200), df['Date'][0:1509:200])

plt.title("TESLA stock price over time")
plt.xlabel("time")
plt.ylabel("price")
plt.show()


result = adfuller(df.Close.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])






model2 = ARIMA(df.Close, order=(1,0,0))
model_fit2 = model2.fit()
print(model_fit2.summary())

forecast_value2 =model_fit2.forecast(167)
print(forecast_value2)


model3 = ARIMA(df.Close, order=(0,0,1))
model_fit3 = model3.fit()
print(model_fit3.summary())

forecast_value3 =model_fit3.forecast(167)
print(forecast_value3)

model4 = ARIMA(df.Close, order=(1,1,0))
model_fit4 = model4.fit()
print(model_fit4.summary())

forecast_value4 =model_fit4.forecast(167)
print(forecast_value4)

forecast_value4.to_csv('result110.txt', sep=' ', index=False)


model5 = ARIMA(df.Close, order=(1,0,1))
model_fit5 = model5.fit()
print(model_fit5.summary())

forecast_value5 =model_fit5.forecast(167)
print(forecast_value5)

model6 = ARIMA(df.Close, order=(0,1,1))
model_fit6 = model6.fit()
print(model_fit6.summary())

forecast_value6 =model_fit6.forecast(167)
print(forecast_value6)

model7 = ARIMA(df.Close, order=(1,1,1))
model_fit7 = model7.fit()
print(model_fit7.summary())

forecast_value7 =model_fit7.forecast(167)
print(forecast_value7)

model8 = ARIMA(df.Close, order=(1,1,2))
model_fit8 = model8.fit()
print(model_fit8.summary())

forecast_value8 =model_fit8.forecast(169)
print(forecast_value8)

forecast_value8.to_csv('result112.txt', sep=' ', index=False)

model9 = ARIMA(df.Close, order=(1,2,1))
model_fit9 = model9.fit()
print(model_fit9.summary())

forecast_value9 =model_fit9.forecast(169)
print(forecast_value9)

forecast_value9.to_csv('result121.txt', sep=' ', index=False)




model10 = ARIMA(df.Close, order=(2,1,1))
model_fit10 = model10.fit()
print(model_fit10.summary())

forecast_value10 =model_fit10.forecast(167)
print(forecast_value10)

model11 = ARIMA(df.Close, order=(2,0,0))
model_fit11 = model11.fit()
print(model_fit11.summary())

forecast_value11 =model_fit11.forecast(167)
print(forecast_value11)

model12 = ARIMA(df.Close, order=(0,2,0))
model_fit12 = model12.fit()
print(model_fit12.summary())

model13 = ARIMA(df.Close, order=(0,0,2))
model_fit13 = model13.fit()
print(model_fit13.summary())

model14 = ARIMA(df.Close, order=(2,2,0))
model_fit14 = model14.fit()
print(model_fit14.summary())

model15 = ARIMA(df.Close, order=(2,0,2))
model_fit15 = model15.fit()
print(model_fit15.summary())


model16 = ARIMA(df.Close, order=(0,2,2))
model_fit16 = model16.fit()
print(model_fit16.summary())

model17 = ARIMA(df.Close, order=(2,2,1))
model_fit17 = model17.fit()
print(model_fit17.summary())

forecast_value17 =model_fit17.forecast(167)
print(forecast_value17)


model18 = ARIMA(df.Close, order=(1,2,2))
model_fit18 = model18.fit()
print(model_fit18.summary())

model19 = ARIMA(df.Close, order=(2,1,2))
model_fit19 = model19.fit()
print(model_fit19.summary())

forecast_value19 =model_fit19.forecast(169)
print(forecast_value19)

forecast_value19.to_csv('result212.txt', sep=' ', index=False)


model20 = ARIMA(df.Close, order=(2,2,2))
model_fit20 = model20.fit()
print(model_fit20.summary())

model21 = ARIMA(df.Close, order=(2,2,2))
model_fit21 = model21.fit()
print(model_fit21.summary())