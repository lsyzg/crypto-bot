import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime
import time

pd.options.mode.chained_assignment = None

def download_data_with_retry(symbol, start_date, max_retries=3):
    """Download data with retry mechanism and error handling"""
    for attempt in range(max_retries):
        try:
            print(f"Downloading {symbol} data... (attempt {attempt + 1})")
            
            # Try different yfinance approaches
            if attempt == 0:
                # Standard approach
                data = yf.download(symbol, start=start_date, end=datetime.now(), progress=False)
            elif attempt == 1:
                # Try with different parameters
                data = yf.download(symbol, start=start_date, progress=False, threads=False)
            else:
                # Try with Ticker object
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=datetime.now())
            
            if data.empty:
                print(f"No data returned for {symbol}")
                if attempt < max_retries - 1:
                    time.sleep(3)  # Longer wait
                    continue
                else:
                    return None
                    
            df = data.dropna()
            if len(df) < 100:  # Need minimum data for calculations
                print(f"Insufficient data for {symbol}: {len(df)} rows")
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                return None
                
            print(f"Successfully downloaded {len(df)} rows of data")
            return df
            
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting {3 * (attempt + 1)} seconds before retry...")
                time.sleep(3 * (attempt + 1))  # Progressive delay
                continue
            else:
                return None
    
    return None

def create_sample_data():
    """Create sample data for testing when yfinance fails"""
    print("Creating sample data for testing purposes...")
    
    # Create sample price data that resembles real market data
    np.random.seed(42)  # For reproducible results
    
    dates = pd.date_range(start='2023-01-01', end=datetime.now(), freq='D')
    n_days = len(dates)
    
    # Generate realistic price movement
    base_price = 2000
    returns = np.random.normal(0.001, 0.03, n_days)  # Mean return ~0.1% daily, 3% volatility
    returns[0] = 0
    
    prices = [base_price]
    for i in range(1, n_days):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 1))  # Ensure no negative prices
    
    # Create OHLCV data
    highs = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
    lows = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    volumes = [np.random.randint(1000000, 5000000) for _ in range(n_days)]
    
    # Ensure High >= Close >= Low
    for i in range(n_days):
        highs[i] = max(highs[i], prices[i])
        lows[i] = min(lows[i], prices[i])
    
    sample_data = pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volumes
    }, index=dates)
    
    # Remove weekends to simulate market data
    sample_data = sample_data[sample_data.index.dayofweek < 5]
    
    print(f"Created sample dataset with {len(sample_data)} trading days")
    return sample_data

# Try to download data with fallbacks
print("Initializing data download...")
df = None

# List of symbols to try
symbols_to_try = [
    ("ETH-USD", "2020-01-01"),
    ("BTC-USD", "2020-01-01"), 
    ("AAPL", "2020-01-01"),
    ("MSFT", "2020-01-01"),
    ("GOOGL", "2020-01-01"),
    ("TSLA", "2020-01-01")
]

for symbol, start_date in symbols_to_try:
    print(f"\nTrying {symbol}...")
    df = download_data_with_retry(symbol, start_date)
    if df is not None:
        current_symbol = symbol
        break
    else:
        print(f"{symbol} failed, trying next symbol...")

if df is None:
    print("\n" + "="*60)
    print("NETWORK ISSUES DETECTED - USING SAMPLE DATA")
    print("="*60)
    print("This commonly happens due to:")
    print("1. yfinance API rate limiting")
    print("2. Network connectivity issues") 
    print("3. Yahoo Finance server issues")
    print("\nUsing sample data to demonstrate the analysis system...")
    print("="*60)
    
    df = create_sample_data()
    current_symbol = "SAMPLE-DATA"

if df is None:
    print("ERROR: Could not create any data for analysis.")
    exit(1)

print(f"\nUsing {current_symbol} for analysis with {len(df)} data points")

def advanced_trade_decision(df, account_value=10000, risk_per_trade=0.02):
    """Enhanced trading decision with multiple strategies"""
    
    # Check if we have sufficient data
    if len(df) < 100:
        print("Insufficient data for advanced analysis")
        return {
            'decision': 'Hold',
            'confidence': 'Low',
            'net_score': 0.0,
            'position_size': 0.0,
            'volatility': 0.0,
            'regime': 'unknown',
            'trend_strength': 0.0,
            'momentum_alignment': '0/3',
            'relative_volume': 1.0,
            'rsi': 50.0,
            'bb_position': 0.5,
            'details': {
                'bullish_score': 0.0,
                'bearish_score': 0.0,
                'confidence_multiplier': 1.0
            }
        }
    
    try:
        # Original indicators
        obv = obv_calc(df, period=14)
        adl = adl_calc(df, period=14)
        adx = adx_calc(df, period=14)
        macd_signal, macd_gen_val = macd_calc(df, num_macds=18)
        rsi = rsi_calc(df, period=14)
        stoch_signal, stoch_reading = stoch_oscillator_calc(df, period=14, period_num=6)
        
        # New advanced indicators
        bb_signal, bb_position, bb_width = bollinger_bands_analysis(df)
        short_mom, medium_mom, long_mom, mom_alignment = momentum_confirmation(df)
        relative_volume, pv_correlation = volume_analysis(df)
        regime, trend_slope, trend_strength = market_regime_detection(df)
        position_size, volatility = risk_adjusted_position_sizing(df, account_value, risk_per_trade)
        
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return {
            'decision': 'Hold',
            'confidence': 'Low',
            'net_score': 0.0,
            'position_size': 0.0,
            'volatility': 0.0,
            'regime': 'error',
            'trend_strength': 0.0,
            'momentum_alignment': '0/3',
            'relative_volume': 1.0,
            'rsi': 50.0,
            'bb_position': 0.5,
            'details': {
                'bullish_score': 0.0,
                'bearish_score': 0.0,
                'confidence_multiplier': 1.0
            }
        }
    
    # Initialize scoring system with confidence levels
    bullish_score = 0.0
    bearish_score = 0.0
    confidence_multiplier = 1.0
    
    # === REGIME-BASED ANALYSIS ===
    if regime == "uptrend" and trend_strength > 0.7:
        bullish_score += 2.0
        confidence_multiplier += 0.3
    elif regime == "downtrend" and trend_strength > 0.7:
        bearish_score += 2.0
        confidence_multiplier += 0.3
    elif regime == "ranging":
        # In ranging markets, favor mean reversion
        confidence_multiplier -= 0.2
    
    # === MOMENTUM CONVERGENCE ===
    if mom_alignment >= 2:  # At least 2 timeframes agree
        if short_mom > 0:
            bullish_score += 1.5 * mom_alignment / 3
        else:
            bearish_score += 1.5 * mom_alignment / 3
    
    # === VOLUME CONFIRMATION ===
    if relative_volume > 1.3:  # Above average volume
        confidence_multiplier += 0.2
        if pv_correlation > 0.3:  # Price and volume moving together
            if short_mom > 0:
                bullish_score += 1.0
            else:
                bearish_score += 1.0
    
    # === BOLLINGER BANDS STRATEGY ===
    if bb_signal == "squeeze" and adx > 25:
        # Volatility contraction before expansion
        bullish_score += 0.5
        bearish_score += 0.5  # Prepare for breakout in either direction
    elif bb_signal == "oversold" and rsi < 35:
        bullish_score += 1.5
    elif bb_signal == "overbought" and rsi > 65:
        bearish_score += 1.5
    
    # === ORIGINAL INDICATOR ANALYSIS (REFINED) ===
    # OBV and ADL with volume confirmation
    if obv > 0 and adl > 0 and relative_volume > 1.0:
        bullish_score += 2.0
    elif obv < 0 and adl < 0 and relative_volume > 1.0:
        bearish_score += 2.0
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
    
    # ADX trend strength (refined)
    if adx > 30:
        strength_multiplier = min(adx / 50, 2.0)  # Cap at 2x
        if trend_slope > 0:
            bullish_score += 1.0 * strength_multiplier
        else:
            bearish_score += 1.0 * strength_multiplier
    elif adx < 25:
        # A weak trend may hint at a lack of momentum.
        bearish_score += 0.5
    
    # MACD with momentum confirmation
    if macd_signal == "bullish" and short_mom > 0:
        bullish_score += 1.5
    elif macd_signal == "bearish" and short_mom < 0:
        bearish_score += 1.5
    elif macd_signal == "bullish":
        bullish_score += 1.0
    elif macd_signal == "bearish":
        bearish_score += 1.0
    
    if macd_gen_val > 0:
        bullish_score += 1.0
    elif macd_gen_val < 0:
        bearish_score += 1.0
    
    # RSI with volatility adjustment
    if rsi < 25:  # Extremely oversold
        bullish_score += 2.0
    elif rsi < 35 and bb_signal == "oversold":
        bullish_score += 1.5
    elif rsi < 30:
        bullish_score += 1.5
    elif rsi > 75:  # Extremely overbought
        bearish_score += 2.0
    elif rsi > 65 and bb_signal == "overbought":
        bearish_score += 1.5
    elif rsi > 70:
        bearish_score += 1.5
    else:
        # A neutral RSI slightly favors holding.
        bullish_score += 0.2
        bearish_score += 0.2
    
    # Stochastic with momentum confirmation
    if stoch_signal == "bullish" and mom_alignment >= 1:
        bullish_score += 1.2
    elif stoch_signal == "bearish" and mom_alignment >= 1:
        bearish_score += 1.2
    elif stoch_signal == "bullish":
        bullish_score += 1.0
    elif stoch_signal == "bearish":
        bearish_score += 1.0
    
    # Use the oscillator's reading for further detail.
    if stoch_reading > 80:
        bullish_score += 1.0
    elif stoch_reading < 20:
        bearish_score += 1.0
    
    # === FINAL DECISION WITH CONFIDENCE ===
    net_score = (bullish_score - bearish_score) * confidence_multiplier
    
    # Dynamic thresholds based on volatility
    volatility_threshold = max(1.0, volatility * 2)
    
    decision = "Hold"
    confidence = "Low"
    
    if net_score > volatility_threshold:
        decision = "Buy"
        if net_score > volatility_threshold * 1.5:
            confidence = "High"
        else:
            confidence = "Medium"
    elif net_score < -volatility_threshold:
        decision = "Sell"
        if net_score < -volatility_threshold * 1.5:
            confidence = "High"
        else:
            confidence = "Medium"
    else:
        if abs(net_score) > volatility_threshold * 0.5:
            confidence = "Medium"
    
    # Risk management override
    if volatility > 0.8:  # Very high volatility
        if decision == "Buy":
            position_size *= 0.5  # Reduce position size
        confidence = "Low" if confidence == "High" else confidence
    
    return {
        'decision': decision,
        'confidence': confidence,
        'net_score': round(net_score, 2),
        'position_size': round(position_size, 2),
        'volatility': round(volatility, 3),
        'regime': regime,
        'trend_strength': round(trend_strength, 3),
        'momentum_alignment': f"{mom_alignment}/3",
        'relative_volume': round(relative_volume, 2),
        'rsi': round(rsi, 1),
        'bb_position': round(bb_position, 2),
        'details': {
            'bullish_score': round(bullish_score, 2),
            'bearish_score': round(bearish_score, 2),
            'confidence_multiplier': round(confidence_multiplier, 2)
        }
    }

def print_analysis_report(result, symbol="ETH-USD"):
    """Print detailed analysis report"""
    print("=" * 60)
    print(f"ADVANCED TRADING ANALYSIS FOR {symbol}")
    print("=" * 60)
    print(f"Decision: {result['decision']} (Confidence: {result['confidence']})")
    print(f"Net Score: {result['net_score']}")
    print(f"Recommended Position Size: {result['position_size']} shares")
    print()
    print("MARKET ANALYSIS:")
    print(f"  Market Regime: {result['regime']}")
    print(f"  Trend Strength: {result['trend_strength']}")
    print(f"  Volatility: {result['volatility']}")
    print(f"  Momentum Alignment: {result['momentum_alignment']}")
    print(f"  Relative Volume: {result['relative_volume']}")
    print(f"  RSI: {result['rsi']}")
    print(f"  Bollinger Band Position: {result['bb_position']}")
    print()
    print("SCORING BREAKDOWN:")
    print(f"  Bullish Score: {result['details']['bullish_score']}")
    print(f"  Bearish Score: {result['details']['bearish_score']}")
    print(f"  Confidence Multiplier: {result['details']['confidence_multiplier']}")
    print("=" * 60)

# Original simple decision function (keeping for compatibility)
def trade_decision(df):
    """Original simplified trade decision function"""
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

from indicator_functions import *
if __name__ == "__main__":
    if df is not None and len(df) > 100:
        try:
            # Run advanced analysis
            result = advanced_trade_decision(df)
            print_analysis_report(result, "ETH-USD" if "ETH" in str(df.index[0]) else "Market Data")
            
            print("\n" + "="*60)
            print("ORIGINAL SIMPLE ANALYSIS:")
            print("="*60)
            simple_decision = trade_decision(df)
            print(f"Original Decision: {simple_decision}")
            print("="*60)
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            print("This might be due to insufficient data or network issues.")
    else:
        print("No data available for analysis.")