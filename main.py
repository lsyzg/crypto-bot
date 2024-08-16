from indicator_functions import *

pd.options.mode.chained_assignment = None

data = yf.download("ETH-USD", start="2017-11-09", end=datetime.now())
df = data.dropna()


def trade_decision(df):    
    obv = obv_calc(df, period=7)
    adl = adl_calc(df, period=14)
    adx = adx_calc(df, period=14)
    bull_bear1, macd_gen_val = macd_calc(df, num_macds=18)
    rsi = rsi_calc(df, period=14)
    bull_bear2, reading = stoch_oscillator_calc(df, period=14, period_num=6)

    bullish = 0 #uptrend
    bearish = 0 #downtrend

    if obv and adl > 0:
        bullish += 1
    if obv and adl < 0: 
        bearish += 1
    if obv > 0 and adl < 0:
        bullish += 1
    if obv < 0 and adl > 0:
        bearish += 1

    if adx > 25:
        if adx > 50:
            if adx > 70:
                bullish += 2
            else:
                bullish += 1
        else:
            bearish += 1

    if bull_bear1 == "bullish":
        bullish += 1
    if bull_bear1 == "bearish":
        bearish += 1
        
    if macd_gen_val > 0:
        bullish += 1
    if macd_gen_val < 0:
        bearish += 1

    if rsi < 30:
        bullish += 1
    if rsi > 70:
        bearish += 1

    if bull_bear2 == "bullish":
        bullish += 1
    if bull_bear2 == "bearish":
        bearish += 1

    if reading > 80:
        bullish += 1
    if reading < 20:
        bearish += 1

    if bullish > bearish:
        return "Buy"
    elif bullish < bearish:
        return "Sell"
    else:
        return "Hold"
    
print(trade_decision(df))