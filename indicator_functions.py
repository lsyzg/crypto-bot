import numpy as np
import yfinance as yf
import pandas as pd
from datetime import date
from datetime import timedelta
from datetime import datetime

pd.options.mode.chained_assignment = None

# Original functions (keeping all your existing functions unchanged)
def ema_calc(dataframe, period: int, period_num: int, column_name: str):
    try:
        vccolumns = dataframe[[column_name]]
        df = vccolumns.iloc[-abs(period * period_num):].copy()
        
        if len(df) < period * period_num:
            # If we don't have enough data, return what we can calculate
            if len(df) >= period:
                # Calculate single EMA with available data
                temp_df = df.iloc[-period:]
                sma = temp_df[column_name].mean()
                multiplier = 2 / (period + 1)
                ema = sma
                for value in temp_df[column_name].iloc[1:]:
                    ema = (value - ema) * multiplier + ema
                return ema if period_num == 1 else [ema]
            else:
                return 0 if period_num == 1 else [0]

        ema_list = []

        for i in range(period_num):
            start_idx = i * period
            end_idx = (i + 1) * period
            
            if end_idx <= len(df):
                temp_df = df.iloc[start_idx:end_idx]
                sma = temp_df[column_name].mean()
                multiplier = 2 / (period + 1)
                ema = sma
                for value in temp_df[column_name].iloc[1:]:  # Use iloc instead of direct indexing
                    ema = (value - ema) * multiplier + ema
                ema_list.append(ema)
        
        if len(ema_list) == 0:
            return 0 if period_num == 1 else [0]
            
        if period_num == 1:
            return ema_list[0] if ema_list else 0
        else:
            return ema_list
            
    except Exception as e:
        print(f"Error in EMA calculation: {e}")
        return 0 if period_num == 1 else [0]

def signal_line_crossover(list: list, signal_line_vals: list, interpreter_val):
    try:
        if len(list) < 2 or len(signal_line_vals) < 2:
            return "None", interpreter_val
        
        # Ensure we have matching data points
        min_len = min(len(list), len(signal_line_vals))
        if min_len < 2:
            return "None", interpreter_val
            
        main_vals = list[-min_len:]
        signal_vals = signal_line_vals[-min_len:]
        
        if main_vals[-1] > signal_vals[-1] and main_vals[-2] < signal_vals[-2]:
            return "bearish", interpreter_val
        elif main_vals[-1] < signal_vals[-1] and main_vals[-2] > signal_vals[-2]:
            return "bullish", interpreter_val
        else:
            return "None", interpreter_val
    except Exception as e:
        print(f"Error in signal crossover: {e}")
        return "None", interpreter_val

def obv_calc(dataframe, period: int):
    # no standard period
    vccolumns = dataframe[["Close", "Volume"]]
    vc = vccolumns.iloc[-abs(period):].copy()
    
    if len(vc) < period:
        return 0  # Not enough data
    
    obv = 0
    obvs = []
    
    for i in range(1, len(vc)):
        volume = vc.iloc[i]["Volume"]
        closingcur = vc.iloc[i]["Close"]
        closingprev = vc.iloc[i-1]["Close"]

        if closingcur > closingprev:
            obv = obv + volume
        elif closingcur == closingprev:
            obv = obv
        elif closingcur < closingprev:
            obv = obv - volume
        
        obvs.append(obv)

    if len(obvs) < 2:
        return 0
    
    # Calculate slope of OBV trend
    x_vals = np.arange(len(obvs))
    slope, intercept = np.polyfit(x_vals, obvs, 1)
    return slope

def adl_calc(dataframe, period: int):
    # standard period 14
    vccolumns = dataframe[["High", "Low", "Close", "Volume"]]
    vc = vccolumns.iloc[-abs(period):].copy()
    
    if len(vc) < period:
        return 0  # Not enough data
    
    mfv = []
    
    for i in range(len(vc)):
        high = vc.iloc[i]["High"]  
        low = vc.iloc[i]["Low"]
        close = vc.iloc[i]["Close"]
        volume = vc.iloc[i]["Volume"]  

        if (high - low) != 0:  # Avoid division by zero
            mfm = ((close - low) - (high - close)) / (high - low)
            mfv.append(volume * mfm)
        else:
            mfv.append(0)

    if len(mfv) < 2:
        return 0
    
    # Calculate slope of Money Flow Volume trend
    x_vals = np.arange(len(mfv))
    slope, intercept = np.polyfit(x_vals, mfv, 1)
    return slope

def adx_calc(dataframe, period: int): 
    #standard period 14
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

def macd_calc(dataframe, num_macds: int):
    # num_macds should be a multiple of 9 and a minimum of 18 to ensure that a signal line can be calculated
    vccolumns = dataframe[["Close"]]
    df = vccolumns.iloc[-abs(num_macds * 26):].copy()
    
    if len(df) < num_macds * 26:
        print(f"Insufficient data for MACD: need {num_macds * 26}, have {len(df)}")
        return "None", 0

    macd_list = []

    try:
        for i in range(num_macds):
            start_idx = i * 26
            end_idx = (i + 1) * 26
            
            if end_idx <= len(df):
                temp_df = df.iloc[start_idx:end_idx]
                if len(temp_df) >= 26:
                    ema12 = ema_calc(temp_df, period=12, period_num=1, column_name="Close")
                    ema26 = ema_calc(temp_df, period=26, period_num=1, column_name="Close")
                    macd = ema12 - ema26
                    macd_list.append(macd)

        if len(macd_list) == 0:
            return "None", 0

        macd_gen_val = sum(macd_list) / len(macd_list)

        # Calculate signal line
        macd_df = pd.DataFrame(macd_list, columns=['MACD_values'])
        period_num = max(1, len(macd_df) // 9)  # Ensure at least 1
        
        if period_num > 0 and len(macd_df) >= 9:
            signal_line_vals = ema_calc(macd_df, period=9, period_num=period_num, column_name="MACD_values")
            if isinstance(signal_line_vals, list) and len(signal_line_vals) > 0:
                return signal_line_crossover(list=macd_list[-len(signal_line_vals):], signal_line_vals=signal_line_vals, interpreter_val=macd_gen_val)

        return "None", macd_gen_val
        
    except Exception as e:
        print(f"Error in MACD calculation: {e}")
        return "None", 0

def rsi_calc(dataframe, period: int):
    # standard period 14
    vccolumns = dataframe[["Close"]]
    df = vccolumns.iloc[-abs(period):].copy()
    
    if len(df) < period:
        return 50.0  # Return neutral RSI if insufficient data

    gains = []
    losses = []

    for i in range(1, len(df)):  # Use len(df) instead of period
        price_diff = df.iloc[i, 0] - df.iloc[i - 1, 0]  # Use iloc consistently
        if price_diff > 0:
            gains.append(price_diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(price_diff))

    if len(gains) == 0:
        return 50.0

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def stoch_oscillator_calc(dataframe, period: int, period_num: int):
    # period_num should be a multiple of 3 and more than or equal to 6 for the sake of SMA calculation
    # standard period 14
    vccolumns = dataframe[["Close", "Low", "High"]]
    df = vccolumns.iloc[-abs(period * period_num):].copy()
    
    if len(df) < period * period_num:
        return "None", 50  # Return neutral values if insufficient data

    k_list = []
    d_list = []
    
    try:
        for i in range(period_num):
            start_idx = i * period
            end_idx = (i + 1) * period
            
            if end_idx <= len(df):
                temp_df = df.iloc[start_idx:end_idx]
                lowest_low = temp_df["Low"].min()
                highest_high = temp_df["High"].max()
                closingcur = temp_df["Close"].iloc[-1]

                if highest_high != lowest_low:  # Avoid division by zero
                    k = (closingcur - lowest_low) / (highest_high - lowest_low) * 100
                else:
                    k = 50  # Neutral value
                k_list.append(k)

        if len(k_list) < 3:
            return "None", k_list[-1] if k_list else 50

        # Calculate D values (3-period SMA of K)
        for j in range(len(k_list) - 2):
            d = np.mean(k_list[j:j+3])
            d_list.append(d)

        if len(d_list) == 0:
            return "None", k_list[-1]

        return signal_line_crossover(list=k_list[-len(d_list):], signal_line_vals=d_list, interpreter_val=k_list[-1])
    
    except Exception as e:
        print(f"Error in stochastic calculation: {e}")
        return "None", 50

# New advanced indicator functions
def calculate_volatility_metrics(dataframe, period=20):
    """Calculate volatility-based metrics for risk assessment"""
    close_prices = dataframe['Close'].iloc[-period:]
    returns = close_prices.pct_change().dropna()
    
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    var_95 = np.percentile(returns, 5)  # Value at Risk (95% confidence)
    
    return volatility, var_95

def bollinger_bands_analysis(dataframe, period=20, std_dev=2):
    """Enhanced Bollinger Bands with squeeze detection"""
    close_prices = dataframe['Close'].iloc[-period:]
    sma = close_prices.mean()
    std = close_prices.std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    current_price = close_prices.iloc[-1]
    
    # Bollinger Band Width (squeeze indicator)
    bb_width = (upper_band - lower_band) / sma
    
    # Position within bands
    bb_position = (current_price - lower_band) / (upper_band - lower_band)
    
    signal = "neutral"
    if bb_width < 0.1:  # Squeeze condition
        signal = "squeeze"
    elif bb_position > 0.8:
        signal = "overbought"
    elif bb_position < 0.2:
        signal = "oversold"
        
    return signal, bb_position, bb_width

def momentum_confirmation(dataframe):
    """Multi-timeframe momentum analysis"""
    data_length = len(dataframe)
    
    # Check if we have enough data for each timeframe
    short_momentum = 0
    medium_momentum = 0
    long_momentum = 0
    
    try:
        # Short-term momentum (5 days) - need at least 6 days
        if data_length >= 6:
            short_momentum = (dataframe['Close'].iloc[-1] / dataframe['Close'].iloc[-6] - 1) * 100
        
        # Medium-term momentum (20 days) - need at least 21 days
        if data_length >= 21:
            medium_momentum = (dataframe['Close'].iloc[-1] / dataframe['Close'].iloc[-21] - 1) * 100
        
        # Long-term momentum (60 days) - need at least 61 days
        if data_length >= 61:
            long_momentum = (dataframe['Close'].iloc[-1] / dataframe['Close'].iloc[-61] - 1) * 100
        
    except Exception as e:
        print(f"Warning in momentum calculation: {e}")
        short_momentum = medium_momentum = long_momentum = 0
    
    # Momentum alignment score
    momentum_alignment = 0
    if short_momentum > 0: momentum_alignment += 1
    if medium_momentum > 0: momentum_alignment += 1
    if long_momentum > 0: momentum_alignment += 1
    
    return short_momentum, medium_momentum, long_momentum, momentum_alignment

def volume_analysis(dataframe, period=20):
    """Enhanced volume analysis with relative volume"""
    data_length = len(dataframe)
    
    # Ensure we have enough data
    if data_length < period + 20:  # Need extra data for comparison
        return 1.0, 0.0  # Default values
    
    try:
        recent_volume = dataframe['Volume'].iloc[-period:].mean()
        avg_volume = dataframe['Volume'].iloc[-60:-period].mean() if data_length >= 60 else dataframe['Volume'].iloc[:-period].mean()
        
        relative_volume = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # Price-Volume correlation
        price_changes = dataframe['Close'].iloc[-period:].pct_change().dropna()
        volume_changes = dataframe['Volume'].iloc[-period:].pct_change().dropna()
        
        # Ensure both series have the same length
        min_length = min(len(price_changes), len(volume_changes))
        if min_length > 1:
            price_changes = price_changes.iloc[-min_length:]
            volume_changes = volume_changes.iloc[-min_length:]
            pv_correlation = price_changes.corr(volume_changes)
            if pd.isna(pv_correlation):
                pv_correlation = 0.0
        else:
            pv_correlation = 0.0
        
    except Exception as e:
        print(f"Warning in volume analysis: {e}")
        relative_volume, pv_correlation = 1.0, 0.0
    
    return relative_volume, pv_correlation

def market_regime_detection(dataframe):
    """Detect current market regime (trending vs ranging)"""
    data_length = len(dataframe)
    
    # Ensure we have enough data
    if data_length < 50:
        return "ranging", 0.0, 0.0  # Default values for insufficient data
    
    try:
        # Calculate moving averages
        ma_20 = dataframe['Close'].iloc[-20:].mean()
        ma_50 = dataframe['Close'].iloc[-50:].mean()
        
        # Current price
        current_price = dataframe['Close'].iloc[-1]
        
        # Trend strength using linear regression on recent 20 periods
        prices = dataframe['Close'].iloc[-20:].values
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        # R-squared to measure trend reliability
        y_pred = slope * x + np.mean(prices)
        ss_res = np.sum((prices - y_pred) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Ensure r_squared is not negative
        r_squared = max(0, r_squared)
        
        regime = "ranging"
        if r_squared > 0.7:  # Strong trend
            if slope > 0:
                regime = "uptrend"
            else:
                regime = "downtrend"
        
    except Exception as e:
        print(f"Warning in regime detection: {e}")
        regime, slope, r_squared = "ranging", 0.0, 0.0
    
    return regime, slope, r_squared

def risk_adjusted_position_sizing(dataframe, account_value=10000, risk_per_trade=0.02):
    """Calculate position size based on volatility"""
    volatility, var_95 = calculate_volatility_metrics(dataframe)
    
    # Risk-adjusted position size
    max_loss = account_value * risk_per_trade
    position_size = max_loss / (abs(var_95) * dataframe['Close'].iloc[-1]) if var_95 != 0 else 0
    
    # Cap position size at 10% of account value
    max_position_value = account_value * 0.1
    max_shares = max_position_value / dataframe['Close'].iloc[-1]
    
    position_size = min(position_size, max_shares)
    
    return position_size, volatility