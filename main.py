import requests, json
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import date
from datetime import timedelta
from datetime import datetime
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

data = yf.download("ETH-USD", start="2017-11-09", end=datetime.now())
df = data.dropna()

tdate = date.today()

def obv_calc(dataframe, period: int):
    vccolumns = dataframe[["Close", "Volume"]]
    vc = vccolumns.iloc[-abs(period):]

    obv = 0
    day = 6
    obvs = []
    obvys = [1,2,3,4,5,6]
    dates = []
    for i in range(period):
        dates.append(str(tdate - timedelta(days = i + 1)))

    for _ in range(period - 1):
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

    slope, intercept = np.polyfit(obvs, obvys, 1)
    return slope

def adl_calc(dataframe, period: int):
    vccolumns = dataframe[["High", "Low", "Close", "Volume"]]
    vc = vccolumns.iloc[-abs(period):]

    mfv = []
    adlys = [1,2,3,4,5,6]
    dates = []
    day = 0
    for i in range(period):
        dates.append(str(tdate - timedelta(days = i + 1)))

    for _ in range(period - 1):
        high = vc.at[dates[day-1], "High"]  
        low = vc.at[dates[day-1], "Low"]
        close = vc.at[dates[day-1], "Close"]
        volume = vc.at[dates[day-1], "Volume"]  

        mfm = ((close - low) - (high - close)) / (high - low)
        mfv.append(volume * mfm)

    slope, intercept = np.polyfit(mfv, adlys, 1)
    return slope

def adx_calc(dataframe, period: int):
    vccolumns = dataframe[["High", "Low", "Close"]]
    df = vccolumns.iloc[-abs(period + 1):]

    # True Range (TR)
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    del df['H-L'], df['H-C'], df['L-C']

    # Average True Range (ATR)
    alpha = 1 / period
    df['ATR'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()

    # +DI and -DI
    df['H-pH'] = df['High'] - df['High'].shift(1)
    df['pL-L'] = df['Low'].shift(1) - df['Low']
    df['+DX'] = np.where((df['H-pH'] > df['pL-L']) & (df['H-pH'] > 0), df['H-pH'], 0.0)
    df['-DX'] = np.where((df['H-pH'] < df['pL-L']) & (df['pL-L'] > 0), df['pL-L'], 0.0)
    del df['H-pH'], df['pL-L']

    # Smoothed +DMI and -DMI
    df['S+DM'] = df['+DX'].ewm(alpha=alpha, adjust=False).mean()
    df['S-DM'] = df['-DX'].ewm(alpha=alpha, adjust=False).mean()

    # +DMI and -DMI as percentages
    df['+DMI'] = (df['S+DM'] / df['ATR']) * 100
    df['-DMI'] = (df['S-DM'] / df['ATR']) * 100
    del df['S+DM'], df['S-DM']

    # ADX
    df['DX'] = (np.abs(df['+DMI'] - df['-DMI']) / (df['+DMI'] + df['-DMI'])) * 100
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    del df['DX'], df['ATR'], df['TR'], df['-DX'], df['+DX'], df['+DMI'], df['-DMI']
    adx = df[["ADX"]]
    adx = adx.dropna()
    return adx

def ema_calc(dataframe, period: int):
    vccolumns = dataframe[["Close"]]
    df = vccolumns.iloc[-abs(period):]

    sma = df["Close"].sum() / period
    multiplier = (2 / (period + 1))
    fclose = df.iat[0, 0]
    ema = (fclose - sma) * multiplier + sma
    for close in df["Close"][1:]:
        ema = (close - ema) * multiplier + ema
    return ema

def macd_calc(dataframe):
    macd = ema_calc(dataframe, period=12) - ema_calc(dataframe, period=26)
    return macd
print(macd_calc(df))