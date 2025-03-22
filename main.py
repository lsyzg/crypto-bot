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
    
    # Initialize weighted scores for bullish and bearish signals
    bullish_score = 0.0 #up trend
    bearish_score = 0.0 #down trend
    
    # --- OBV and ADL Analysis ---
    # Alignment of OBV and ADL indicates volume-based momentum.
    if obv > 0 and adl > 0:
        bullish_score += 1.5  # both indicators suggest upward pressure
    elif obv < 0 and adl < 0:
        bearish_score += 1.5  # both indicate downward pressure
    else:
        # When indicators are conflicting, assign smaller weights individually.
        if obv > 0:
            bullish_score += 0.5
        elif obv < 0:
            bearish_score += 0.5
        if adl > 0:
            bullish_score += 0.5
        elif adl < 0:
            bearish_score += 0.5

    # --- ADX Analysis (Trend Strength) ---
    # ADX values above 25 suggest a trending market.
    if adx > 25:
        if adx > 70:
            bullish_score += 2  # very strong trend favoring upward movement
        elif adx > 50:
            bullish_score += 1.5
        else:
            bullish_score += 1
    else:
        # A weak trend may hint at a lack of momentum.
        bearish_score += 0.5

    # --- MACD Analysis ---
    # MACD provides insight into momentum changes.
    if bull_bear1 == "bullish":
        bullish_score += 1
    elif bull_bear1 == "bearish":
        bearish_score += 1
        
    if macd_gen_val > 0:
        bullish_score += 1
    elif macd_gen_val < 0:
        bearish_score += 1

    # --- RSI Analysis (Overbought/Oversold Conditions) ---
    # RSI values <30 typically indicate oversold (potential buy) conditions,
    # while values >70 indicate overbought (potential sell) conditions.
    if rsi < 30:
        bullish_score += 1.5
    elif rsi > 70:
        bearish_score += 1.5
    else:
        # A neutral RSI slightly favors holding.
        bullish_score += 0.2
        bearish_score += 0.2

    # --- Stochastic Oscillator Analysis ---
    # This oscillator gives additional confirmation on momentum.
    if bull_bear2 == "bullish":
        bullish_score += 1
    elif bull_bear2 == "bearish":
        bearish_score += 1

    # Use the oscillator's reading for further detail.
    if reading > 80:
        bullish_score += 1
    elif reading < 20:
        bearish_score += 1

    # --- Final Decision Calculation ---
    # Compute the net score: a positive score leans bullish, while a negative score leans bearish.
    total_score = bullish_score - bearish_score
    
    # Adjust thresholds to allow for "Hold" when signals are mixed.
    if total_score > 1.5:
        decision = "Buy"
    elif total_score < -1.5:
        decision = "Sell"
    else:
        decision = "Hold"

    return decision
print(trade_decision(df))