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