import numpy, requests, json
import yfinance as yf
import pandas as pd
from datetime import date
from datetime import timedelta
from datetime import datetime
import matplotlib.pyplot as plt

data = yf.download("ETH-USD", start="2017-11-09", end=datetime.now())
df = data.dropna()

tdate = date.today()

def obv_calc(dataframe):
    vccolumns = dataframe[["Close", "Volume"]]
    vc = vccolumns.iloc[-7:]

    obv = 0
    day = 6
    obvs = []
    obvys = [1,2,3,4,5,6]
    dates = []
    for i in range(7):
        dates.append(str(tdate - timedelta(days = i + 1)))

    for _ in range(6):
        volume = vc.at[dates[day-1], "Volume"]  
        closingcur = vc.at[dates[day-1], "Close"]
        closingprev = vc.at[dates[day], "Close"]

        if closingcur > closingprev:
            obv = obv + volume
            obvs.append(obv)
            day = day - 1   
        elif closingcur == closingprev:
            obv = obv
            obvs.append(obv)
            day = day - 1    
        elif closingcur < closingprev:
            obv = obv - volume
            obvs.append(obv)
            day = day - 1

    slope, intercept = numpy.polyfit(obvs, obvys, 1)
    # plt.figure()
    # plt.scatter(obvs, obvys) 
    # plt.plot(numpy.unique(obvs), numpy.poly1d(numpy.polyfit(obvs, obvys, 1))(numpy.unique(obvs)), color = 'k')
    # plt.show()
    return slope

def adl_calc(dataframe):
    vccolumns = dataframe[["High", "Low", "Close", "Volume"]]
    vc = vccolumns.iloc[-7:]

    mfv = []
    adlys = [1,2,3,4,5,6]
    dates = []
    day = 0
    for i in range(7):
        dates.append(str(tdate - timedelta(days = i + 1)))

    for _ in range(6):
        high = vc.at[dates[day-1], "High"]  
        low = vc.at[dates[day-1], "Low"]
        close = vc.at[dates[day-1], "Close"]
        volume = vc.at[dates[day-1], "Volume"]  

        mfm = ((close - low) - (high - close)) / (high - low)
        mfv.append(volume * mfm)

    slope, intercept = numpy.polyfit(mfv, adlys, 1)
    return slope

def adx_calc():
