import requests, json
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import date
from datetime import timedelta
from datetime import datetime

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
    obvys = []
    dates = []

    for i in range(period):
        obvys.append(i)
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
    adlys = []
    dates = []
    day = 0

    for i in range(period):
        adlys.append(i)
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
    adx_df = df[["ADX"]]
    adx_df = adx_df.dropna()
    adx = adx_df.iat[-1,0]
    return adx

def ema_calc(dataframe, period: int, period_num: int, column_name: str):
    vccolumns = dataframe[[column_name]]
    df = vccolumns.iloc[-abs(period * period_num):]

    ema_list = []

    for i in range(period_num):
        temp_df = df.iloc[i * period:(i + 1) * period]
        sma = temp_df[column_name].mean()
        multiplier = 2 / (period + 1)
        ema = sma
        for value in temp_df[column_name][1:]:
            ema = (value - ema) * multiplier + ema
        ema_list.append(ema)
    if period_num == 1:
        return ema_list[0]
    else:
        return ema_list

def macd_calc(dataframe, num_macds: int):
    # num_macds should be a multiple of 9 and a minimum of 18 to ensure that a signal line can be calculated
    vccolumns = dataframe[["Close"]]
    df = vccolumns.iloc[-abs(num_macds * 26):]

    macd_list = []

    for i in range(num_macds):
        temp_df = df.iloc[i * 26:(i + 1) * 26]
        macd = ema_calc(temp_df, period=12, period_num=1, column_name="Close") - ema_calc(temp_df, period=26, period_num=1, column_name="Close")
        macd_list.append(macd)

    macd_df = pd.DataFrame(macd_list, columns=['MACD_values'])
    period_num = int(len(macd_df) / 9)
    signal_line_vals = ema_calc(macd_df, period=9, period_num=period_num, column_name="MACD_values")

    if macd_list[-1] > signal_line_vals[-1] and macd_list[-2] < signal_line_vals[-2]:
        return "bearish"
    elif macd_list[-1] < signal_line_vals[-1] and macd_list[-2] > signal_line_vals[-2]:
        return "bullish"
    else:
        return "no crosses"
    
def rsi_calc(dataframe, period=int):
    vccolumns = dataframe[["Close"]]
    df = vccolumns.iloc[-abs(period):]

    gains = []
    losses = []

    for i in range(1, period):
        price_diff = df.iat[i, 0] - df.iat[i - 1, 0]
        if price_diff > 0:
            gains.append(price_diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(price_diff))

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi