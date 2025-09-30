import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def ema(series, window):
    return pd.Series(series).ewm(span=window, adjust=False).mean().iloc[-1]

def rsi(series, period=14):
    """RSI индикатор"""
    if len(series) < period:
        return 50
    delta = pd.Series(series).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-8)
    return (100 - (100 / (1 + rs))).iloc[-1]

def obv(close: List[float], volume: List[float]) -> List[float]:
    """On Balance Volume"""
    if len(close) == 0:
        return [0]
        
    obv_values = [0]
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv_values.append(obv_values[-1] + volume[i])
        elif close[i] < close[i-1]:
            obv_values.append(obv_values[-1] - volume[i])
        else:
            obv_values.append(obv_values[-1])
    return obv_values

def detect_volume_climax(volume: List[float], lookback: int = 20) -> bool:
    """Обнаружение климакса объема"""
    if len(volume) < lookback:
        return False
    avg_volume = np.mean(volume[-lookback:])
    std_volume = np.std(volume[-lookback:])
    current_volume = volume[-1]
    return current_volume > avg_volume + (2 * std_volume)

def calculate_quantity(equity: float, risk_percent: float, stop_percent: float, 
                      entry_price: float, leverage: int, position_multiplier: int):
    """СКАЛЬПЕРСКИЙ расчет - риск 1% от депозита"""
    risk_amount = equity * risk_percent  # $2
    # Базовая позиция по риску
    base_position = risk_amount * leverage  # $6
    # Умножаем на множитель для увеличения позиции
    position_size = base_position * position_multiplier  # $6 × 10 = $60
    # Вариант 1: Консервативный ($6 позиция)
    # position_size = risk_amount * leverage  # $2 × 3 = $6
    
    # ИЛИ Вариант 2: Компромиссный ($60 позиция)
    # position_size = risk_amount * leverage * 10  # $2 × 3 × 10 = $60
    
    # ИЛИ Вариант 3: Агрессивный ($120 позиция)  
    # position_size = risk_amount * leverage * 20  # $2 × 3 × 20 = $120
    
    # Пересчитываем фактический стоп
    actual_stop = risk_amount / position_size  # $2 / $60 = 3.33%
    
     # Ограничиваем стоп снизу (мин 0.8%) и сверху (макс 5%)
    actual_stop = max(stop_percent, min(actual_stop, 0.05))
    
    quantity = position_size / entry_price
    
    # Логируем
    print(f"💰 SCALPER CALC:")
    print(f"   Equity: ${equity} | Risk: ${risk_amount}")
    print(f"   Position: ${position_size} | Stop: {actual_stop*100:.1f}%")
    print(f"   Qty: {quantity:.6f}")
    
    return position_size, quantity, actual_stop

def calculate_atr(highs, lows, closes, period=14):
    """Average True Range"""
    if len(highs) < period + 1:
        return 0
    
    tr = []
    for i in range(1, len(highs)):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        tr.append(max(tr1, tr2, tr3))
    
    return np.mean(tr[-period:]) if tr else 0

def is_hammer(highs, lows, closes, lookback=3):
    """Определение молота"""
    if len(closes) < lookback + 1:
        return False
    
    body = abs(closes[-1] - closes[-2])
    lower_wick = min(closes[-1], closes[-2]) - lows[-1]
    upper_wick = highs[-1] - max(closes[-1], closes[-2])
    
    return lower_wick > body * 2 and upper_wick < body * 0.5

def is_shooting_star(highs, lows, closes, lookback=3):
    """Определение падающей звезды"""
    if len(closes) < lookback + 1:
        return False
    
    body = abs(closes[-1] - closes[-2])
    upper_wick = highs[-1] - max(closes[-1], closes[-2])
    lower_wick = min(closes[-1], closes[-2]) - lows[-1]
    
    return upper_wick > body * 2 and lower_wick < body * 0.5

def is_inside_bar(highs, lows):
    """Внутренний бар"""
    if len(highs) < 2:
        return False
    return highs[-1] <= highs[-2] and lows[-1] >= lows[-2]

def is_compression(highs, lows, period=5):
    """Сжатие волатильности"""
    if len(highs) < period:
        return False
    
    ranges = [highs[i] - lows[i] for i in range(-period, 0)]
    current_range = ranges[-1]
    avg_range = np.mean(ranges[:-1])
    
    return current_range < avg_range * 0.7

def find_liquidity_zones(highs, lows, closes, period=20):
    """Поиск зон ликвидности (поддержка/сопротивление)"""
    if len(closes) < period:
        return closes[-1] * 0.95, closes[-1] * 1.05
    
    # Простой метод - пики и впадины
    support = min(lows[-period:])
    resistance = max(highs[-period:])
    
    return support, resistance