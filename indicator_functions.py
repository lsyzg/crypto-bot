import requests, json
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import date
from datetime import timedelta
from datetime import datetime
from indicator_functions import *

pd.options.mode.chained_assignment = None

data = yf.download("ETH-USD", start="2017-11-09", end=datetime.now())
df = data.dropna()

bullish = 0 #uptrend
bearish = 0 #downtrend
trend_strength = 0 #weight value
buy = True

if obv_calc(df, period=7) and adl_calc(df, period=7) > 0:
    bullish = bullish + 1
if obv_calc(df, period=7) and adl_calc(df, period=7) < 0: 
    bearish = bearish + 1
if obv_calc(df, period=7) > 0 and adl_calc(df, period=7) < 0:
    bullish = bullish + 1
if obv_calc(df, period=7) < 0 and adl_calc(df, period=7) > 0:
    bearish = bearish + 1

if 0 <= adx_calc(df, period=14) < 25:
    trend_strength = 0
if 25 <= adx_calc(df, period=14) < 50:
    trend_strength = 1
if 50 <= adx_calc(df, period=14) < 75:
    trend_strength = 2
if 75 <= adx_calc(df, period=14) < 100:
    trend_strength = 3

bull_bear, macd_gen_val = macd_calc(df, num_macds=18)

if bull_bear == "bullish":
    bullish = bullish + 1
if bull_bear == "bearish":
    bearish = bearish + 1
    
if macd_gen_val > 0:
    bullish = bullish + 1
if macd_gen_val < 0:
    bearish = bearish + 1

if rsi_calc(df, period=7) < 30:
    bullish = bullish + 1
if rsi_calc(df, period=7) > 70:
    bearish = bearish + 1

if bullish > bearish:
    buy = True
else:
    buy = False

print(bullish, bearish, buy)