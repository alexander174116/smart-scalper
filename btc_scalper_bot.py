"""
SMART SCALPER PRO - Продвинутая стратегия против толпы
Философия: ловим моменты когда алгоритмы ломают ожидания толпы
"""

import time, logging, os, csv
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from enum import Enum

import ccxt, pandas as pd, numpy as np, yaml
from dotenv import load_dotenv

class EntryStrategy(Enum):
    SMART_MOMENTUM = "smart_momentum"
    VOLUME_SURGE = "volume_surge" 
    BREAKOUT_FAKE = "breakout_fake"

# ----------------------------- CONFIG ---------------------------------
@dataclass
class Config:
    mode: str
    exchange_id: str
    symbols: List[str]
    timeframe: str
    initial_equity: float
    active_capital_pct: float
    leverage: int
    risk_per_trade_pct: float
    initial_stop_pct: float
    profit_target_pct: float
    min_profit_pct: float
    trailing_start: float
    trailing_step: float
    trailing_lock_pct: float
    rsi_period: int
    rsi_overbought: float
    rsi_oversold: float
    volume_multiplier: float
    max_open_trades: int
    trades_per_symbol: int
    trade_cooldown_sec: int
    max_daily_loss_pct: float
    log_trades: bool
    log_file: str
    verbose: bool
    # Новые параметры
    entry_strategy: str
    multi_timeframe: bool
    tf_primary: str
    tf_confirmation: str
    min_volume_ratio: float
    max_volatility: float
    trend_filter: bool

def load_config() -> Config:
    with open("config.yaml", "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)

CFG = load_config()
load_dotenv()

# -------------------------- Logging ----------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# -------------------------- Trade Logger -----------------------------
class TradeLogger:
    def __init__(self, filename):
        self.filename = filename
        self._init_file()
    
    def _init_file(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'entry_price', 'quantity',
                    'exit_price', 'exit_reason', 'pnl', 'pnl_pct', 'duration_sec',
                    'max_profit_pct', 'trailing_activated'
                ])
    
    def log_trade(self, trade_data):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trade_data.get('timestamp', datetime.now().isoformat()),
                trade_data.get('symbol', ''),
                trade_data.get('side', ''),
                trade_data.get('entry_price', 0),
                trade_data.get('quantity', 0),
                trade_data.get('exit_price', 0),
                trade_data.get('exit_reason', ''),
                trade_data.get('pnl', 0),
                trade_data.get('pnl_pct', 0),
                trade_data.get('duration_sec', 0),
                trade_data.get('max_profit_pct', 0),
                trade_data.get('trailing_activated', False)
            ])

# -------------------------- Advanced Indicators ----------------------
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

def heikin_ashi(df):
    """Heikin Ashi для фильтрации шума - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
    ha_df = df.copy()
    
    # Инициализация первых значений
    ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    ha_open = [0] * len(df)
    ha_open[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
    
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2
    
    ha_high = df[['high']].copy()
    ha_low = df[['low']].copy()
    
    ha_high['ha_open'] = ha_open
    ha_high['ha_close'] = ha_close
    ha_high['ha_high'] = ha_high[['high', 'ha_open', 'ha_close']].max(axis=1)
    
    ha_low['ha_open'] = ha_open  
    ha_low['ha_close'] = ha_close
    ha_low['ha_low'] = ha_low[['low', 'ha_open', 'ha_close']].min(axis=1)
    
    return {
        'ha_open': ha_open,
        'ha_high': ha_high['ha_high'].values,
        'ha_low': ha_low['ha_low'].values, 
        'ha_close': ha_close.values
    }

def obv(close, volume):
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

def detect_volume_climax(volume, lookback=20):
    """Обнаружение климакса объема"""
    if len(volume) < lookback:
        return False
    avg_volume = np.mean(volume[-lookback:])
    std_volume = np.std(volume[-lookback:])
    current_volume = volume[-1]
    return current_volume > avg_volume + (2 * std_volume)

def calculate_quantity(equity, risk_percent, stop_percent, entry_price, leverage):
    """ПРАВИЛЬНЫЙ расчет размера позиции"""
    # 1. Сумма риска в $
    risk_amount = equity * risk_percent  # 200 × 0.01 = $2
    
    # 2. Размер позиции на основе стопа
    position_size = risk_amount / stop_percent  # 2 / 0.008 = $250
    
    # 3. Максимальная позиция по марже (50% капитала)
    max_margin = equity * 0.5  # $100
    max_position_by_margin = max_margin * leverage  # $100 × 3 = $300
    
    # 4. Берем минимум
    final_size = min(position_size, max_position_by_margin)
    
    # 5. Пересчитываем фактический риск
    actual_risk = final_size * stop_percent
    actual_risk_percent = actual_risk / equity
    
    quantity = final_size / entry_price
    
    logger.info(f"💰 Position Calc:")
    logger.info(f"   Risk: ${risk_amount:.2f} | Position: ${final_size:.2f}")
    logger.info(f"   Margin: ${final_size/leverage:.2f} | Qty: {quantity:.6f}")
    logger.info(f"   Actual Risk: ${actual_risk:.2f} ({actual_risk_percent:.1%})")
    
    return final_size, quantity

# -------------------------- Exchange Client --------------------------
# -------------------------- Exchange Client --------------------------
class ScalperProClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.balance = cfg.initial_equity
        self.initial_balance = cfg.initial_equity
        self.open_positions = []
        self.last_trade_time = 0
        self.trade_logger = TradeLogger(cfg.log_file) if cfg.log_trades else None
        
        if cfg.mode == "live":
            self.exchange = getattr(ccxt, cfg.exchange_id)({
                'apiKey': os.getenv("API_KEY"),
                'secret': os.getenv("API_SECRET"),
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
        else:
            self.exchange = None

    def fetch_ohlcv(self, symbol, timeframe, limit=100):
        """Безопасное получение данных"""
        try:
            ex_cls = getattr(ccxt, self.cfg.exchange_id)
            ex = ex_cls({'enableRateLimit': True})
            data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            return data if data else []
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return []

    def get_current_price(self, symbol):
        """Текущая цена для paper-режима"""
        ohlcv = self.fetch_ohlcv(symbol, self.cfg.timeframe, limit=1)
        return float(ohlcv[-1][4]) if ohlcv else 0

    def can_trade_symbol(self, symbol):
        """Проверка лимита сделок на символ"""
        symbol_trades = [p for p in self.open_positions if p['symbol'] == symbol]
        return len(symbol_trades) < self.cfg.trades_per_symbol

    def place_market_order(self, symbol, side, quantity):
        """Размещение ордера с проверками"""
        current_price = self.get_current_price(symbol)
        position_value = quantity * current_price
        margin_required = position_value / self.cfg.leverage
        
        # Проверка маржи
        if margin_required > self.balance * 0.5:  # макс 50% депозита
            logger.warning(f"🚨 Margin too high: ${margin_required:.2f}")
            # Уменьшаем до лимита
            max_position_value = (self.balance * 0.5) * self.cfg.leverage
            quantity = max_position_value / current_price
            logger.info(f"   Adjusted quantity to: {quantity:.6f}")
        
        if self.cfg.mode == "paper":
            position = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price': current_price,
                'entry_time': time.time(),
                'initial_stop': current_price * (1 - self.cfg.initial_stop_pct) if side == 'long' else current_price * (1 + self.cfg.initial_stop_pct),
                'current_stop': None,
                'trailing_activated': False,
                'max_profit_pct': 0,
                'log_data': {
                    'symbol': symbol,
                    'side': side,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'entry_time': time.time()
                }
            }
            self.open_positions.append(position)
            logger.info(f"[PAPER] {side.upper()} {quantity:.6f} {symbol} @ {current_price:.2f}")
            return {'status': 'filled', 'price': current_price}
        
        else:
            # Live trading
            try:
                return self.exchange.create_order(
                    symbol, 'market', side.lower(), quantity
                )
            except Exception as e:
                logger.error(f"Live order failed: {e}")
                return None

    def update_positions(self):
        """Обновление позиций и трейлинг-стопов"""
        if not self.open_positions:
            return

        for position in list(self.open_positions):
            symbol = position['symbol']
            current_price = self.get_current_price(symbol)
            
            if current_price == 0:
                continue

            # Расчет текущего PnL
            if position['side'] == 'long':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                stop_hit = current_price <= (position['current_stop'] or position['initial_stop'])
            else:  # short
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                stop_hit = current_price >= (position['current_stop'] or position['initial_stop'])

            # Обновление максимальной прибыли
            position['max_profit_pct'] = max(position['max_profit_pct'], pnl_pct)

            # Активация трейлинг-стопа
            if not position['trailing_activated'] and pnl_pct >= self.cfg.profit_target_pct:
                position['trailing_activated'] = True
                if position['side'] == 'long':
                    position['current_stop'] = position['entry_price'] * (1 + self.cfg.trailing_lock_pct)
                else:
                    position['current_stop'] = position['entry_price'] * (1 - self.cfg.trailing_lock_pct)
                logger.info(f"🚀 Trailing activated for {symbol} @ {pnl_pct*100:.2f}%")

            # Подтягивание трейлинг-стопа
            elif position['trailing_activated']:
                if position['side'] == 'long':
                    new_stop = current_price * (1 - self.cfg.trailing_step)
                    if new_stop > position['current_stop']:
                        position['current_stop'] = new_stop
                else:  # short
                    new_stop = current_price * (1 + self.cfg.trailing_step)
                    if new_stop < position['current_stop']:
                        position['current_stop'] = new_stop

            # Закрытие позиции по стопу
            if stop_hit:
                self.close_position(position, current_price, "trailing_stop" if position['trailing_activated'] else "initial_stop")

    def close_position(self, position, exit_price, reason):
        """Закрытие позиции и логирование"""
        # Расчет PnL
        if position['side'] == 'long':
            pnl = (exit_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - exit_price) * position['quantity']
        
        pnl_pct = pnl / (position['entry_price'] * position['quantity']) * 100
        duration = time.time() - position['entry_time']

        # Обновление баланса
        self.balance += pnl

        # Логирование
        if self.trade_logger:
            log_data = position['log_data']
            log_data.update({
                'timestamp': datetime.now().isoformat(),
                'exit_price': exit_price,
                'exit_reason': reason,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'duration_sec': duration,
                'max_profit_pct': position['max_profit_pct'],
                'trailing_activated': position['trailing_activated']
            })
            self.trade_logger.log_trade(log_data)

        logger.info(f"🔴 CLOSE {position['symbol']} {position['side']}: {reason}, PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")

        self.open_positions.remove(position)

    def get_equity(self):
        return self.balance

    def close_all_positions(self):
        for position in list(self.open_positions):
            current_price = self.get_current_price(position['symbol'])
            self.close_position(position, current_price, "shutdown")

# -------------------------- Advanced Strategy ------------------------
class SmartAntiCrowdStrategy:
    def __init__(self, client, cfg):
        self.client = client
        self.cfg = cfg
        self.entry_strategy = EntryStrategy(cfg.entry_strategy)
        
    def multi_timeframe_analysis(self, symbol) -> Tuple[str, float]:
        """Мультитаймфрейм анализ для подтверждения тренда"""
        if not self.cfg.multi_timeframe:
            return "neutral", 0.5
            
        try:
            # Данные с большего таймфрейма для контекста
            confirmation_data = self.client.fetch_ohlcv(
                symbol, self.cfg.tf_confirmation, limit=50
            )
            if not confirmation_data:
                return "neutral", 0.5
                
            df_conf = pd.DataFrame(confirmation_data, 
                                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Простой анализ тренда на старшем ТФ
            ma_fast = df_conf['close'].rolling(8).mean()
            ma_slow = df_conf['close'].rolling(21).mean()
            
            current_fast = ma_fast.iloc[-1]
            current_slow = ma_slow.iloc[-1]
            prev_fast = ma_fast.iloc[-2]
            prev_slow = ma_slow.iloc[-2]
            
            # Сила тренда
            if current_fast > current_slow and prev_fast <= prev_slow:
                return "bullish", 0.8
            elif current_fast < current_slow and prev_fast >= prev_slow:
                return "bearish", 0.8
            elif current_fast > current_slow:
                return "bullish", 0.6
            elif current_fast < current_slow:
                return "bearish", 0.6
            else:
                return "neutral", 0.5
                
        except Exception as e:
            logger.error(f"Multi-timeframe analysis error for {symbol}: {e}")
            return "neutral", 0.5

    def analyze_smart_momentum(self, symbol, df) -> Optional[Dict]:
        """
        АГРЕССИВНАЯ стратегия для 1м таймфрейма
        """
        closes = df['close'].values
        volumes = df['volume'].values
        highs = df['high'].values
        lows = df['low'].values
        
        if len(closes) < 20:
            return None

        # 1. Базовые индикаторы
        current_rsi = rsi(closes, self.cfg.rsi_period)
        
        # 2. Volume анализ (менее строгий)
        volume_ok = volumes[-1] > np.mean(volumes[-10:]) * 1.2
        
        # 3. Простая MA для тренда
        ma_fast = np.mean(closes[-5:])   # 5-периодная MA
        ma_slow = np.mean(closes[-15:])  # 15-периодная MA
        
        # 4. Ценовое действие
        price_trend = "up" if closes[-1] > ma_fast else "down"
        ma_cross = "bullish" if ma_fast > ma_slow else "bearish"
        
        # 🎯 АГРЕССИВНЫЕ СИГНАЛЫ ДЛЯ 1М:
        
        # Сигнал 1: RSI перекупленность + медвежий тренд
        if (current_rsi > 65 and volume_ok and ma_cross == "bearish" and 
            closes[-1] < closes[-2]):  # цена падает
            return {
                'side': 'short', 
                'symbol': symbol, 
                'reason': f'1m Short: RSI {current_rsi:.1f} + MA bearish + price down',
                'confidence': 0.75
            }
        
        # Сигнал 2: RSI перепроданность + бычий тренд  
        elif (current_rsi < 35 and volume_ok and ma_cross == "bullish" and 
            closes[-1] > closes[-2]):  # цена растет
            return {
                'side': 'long', 
                'symbol': symbol, 
                'reason': f'1m Long: RSI {current_rsi:.1f} + MA bullish + price up', 
                'confidence': 0.75
            }
        
        # Сигнал 3: Отскок от экстремумов (самый частый на 1м)
        if current_rsi > 68 and closes[-1] < closes[-2] and closes[-2] < closes[-3]:
            return {
                'side': 'short',
                'symbol': symbol,
                'reason': f'1m Reversal Short: RSI {current_rsi:.1f} + 2 red candles',
                'confidence': 0.7
            }
        elif current_rsi < 32 and closes[-1] > closes[-2] and closes[-2] > closes[-3]:
            return {
                'side': 'long', 
                'symbol': symbol,
                'reason': f'1m Reversal Long: RSI {current_rsi:.1f} + 2 green candles',
                'confidence': 0.7
            }

        return None

    def detect_fake_breakout(self, df) -> bool:
        """Обнаружение ложных пробоев"""
        if len(df) < 15:
            return False
            
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # Ложный пробой вверх: новый хай, но закрытие ниже предыдущего
        if (highs[-1] > max(highs[-10:-1]) and closes[-1] < closes[-2]):
            return True
            
        # Ложный пробой вниз: новый лоу, но закрытие выше предыдущего  
        if (lows[-1] < min(lows[-10:-1]) and closes[-1] > closes[-2]):
            return True
            
        return False

    def analyze_volume_surge(self, symbol, df) -> Optional[Dict]:
        """Стратегия на основе аномальных объемов"""
        volumes = df['volume'].values
        closes = df['close'].values
        
        if len(volumes) < 20:
            return None
            
        volume_surge = detect_volume_climax(volumes)
        current_rsi = rsi(closes, self.cfg.rsi_period)
        
        if volume_surge:
            # Большой объем на падении - возможно паника продаж (покупаем)
            if closes[-1] < closes[-2] and current_rsi < 35:
                return {
                    'side': 'long',
                    'symbol': symbol, 
                    'reason': f'Volume Surge Long: Panic selling RSI {current_rsi:.1f}',
                    'confidence': 0.75
                }
            # Большой объем на росте - возможно FOMO (продаем)
            elif closes[-1] > closes[-2] and current_rsi > 65:
                return {
                    'side': 'short',
                    'symbol': symbol,
                    'reason': f'Volume Surge Short: FOMO buying RSI {current_rsi:.1f}',
                    'confidence': 0.75
                }
        
        return None

    def analyze_symbol(self, symbol) -> Optional[Dict]:
        """Главный анализ символа с выбранной стратегией"""
        data = self.client.fetch_ohlcv(symbol, self.cfg.timeframe, limit=100)
        if len(data) < 50:
            return None

        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Выбор стратегии входа
        if self.entry_strategy == EntryStrategy.SMART_MOMENTUM:
            signal = self.analyze_smart_momentum(symbol, df)
        elif self.entry_strategy == EntryStrategy.VOLUME_SURGE:
            signal = self.analyze_volume_surge(symbol, df)
        else:
            signal = self.analyze_smart_momentum(symbol, df)  # по умолчанию
            
        # Дополнительная фильтрация
        if signal and signal.get('confidence', 0) < 0.7:
            return None
            
        return signal

    def execute_trading(self):
        """Исполнение торговли с улучшенной логикой"""
        if len(self.client.open_positions) >= self.cfg.max_open_trades:
            return
            
        if time.time() - self.client.last_trade_time < self.cfg.trade_cooldown_sec:
            return

        # Проверка дневного убытка
        daily_pnl = self.client.balance - self.client.initial_balance
        if daily_pnl <= -self.client.initial_balance * self.cfg.max_daily_loss_pct:
            logger.warning("Daily loss limit reached! Stopping trading.")
            return

        # Анализ всех символов
        signals = []
        for symbol in self.cfg.symbols:
            if not self.client.can_trade_symbol(symbol):
                continue
                
            signal = self.analyze_symbol(symbol)
            if signal:
                signals.append(signal)
        
        # Сортируем по уверенности и берем лучший сигнал
        if signals:
            best_signal = max(signals, key=lambda x: x.get('confidence', 0))
            if best_signal.get('confidence', 0) >= 0.7:
                self.enter_trade(best_signal)

    def enter_trade(self, signal):
        """Вход в сделку с ПРАВИЛЬНЫМ MM"""
        equity = self.client.get_equity()
        current_price = self.client.get_current_price(signal['symbol'])
        
        # Базовый риск 1%
        base_risk = self.cfg.risk_per_trade_pct
        
        # Размер позиции
        notional, quantity = calculate_quantity(
            equity, 
            base_risk,
            self.cfg.initial_stop_pct,
            current_price,
            self.cfg.leverage
        )

        # Проверка минимального количества
        if quantity <= 0 or notional < 10:  # минимальная позиция $10
            logger.warning(f"❌ Position too small: ${notional:.2f}")
            return

        # Проверка превышения лимитов
        if notional > equity * self.cfg.leverage:
            logger.warning(f"❌ Position too large: ${notional:.2f} > ${equity * self.cfg.leverage:.2f}")
            return

        # Размещение ордера
        order = self.client.place_market_order(
            signal['symbol'], 
            signal['side'], 
            quantity
        )

        if order and order.get('status') in ('filled', None):
            self.client.last_trade_time = time.time()
            logger.info(f"🎯 ENTRY {signal['side'].upper()} {signal['symbol']}")
            logger.info(f"   💰 Size: ${notional:.2f} | Risk: ${notional * self.cfg.initial_stop_pct:.2f}")

    def run(self, runtime_hours=24):
        """Запуск бота"""
        logger.info(f"🚀 Starting SMART SCALPER PRO (Anti-Crowd) | {len(self.cfg.symbols)} symbols")
        logger.info(f"Strategy: {self.entry_strategy.value} | RSI {self.cfg.rsi_period} | Stop: {self.cfg.initial_stop_pct*100}% | Trail: {self.cfg.trailing_start*100}%")
        
        end_time = time.time() + runtime_hours * 3600
        
        while time.time() < end_time:
            try:
                # Обновляем открытые позиции
                self.client.update_positions()
                
                # Ищем новые входы
                self.execute_trading()
                
                # Статус каждую минуту
                if int(time.time()) % 60 == 0:
                    logger.info(f"📊 Status: {len(self.client.open_positions)} open | Equity: ${self.client.get_equity():.2f}")
                
                time.sleep(5)  # 5 секунд между проверками
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(10)

        logger.info("⏹️ Runtime finished")
        self.client.close_all_positions()

# ------------------------------ Main ---------------------------------
def main():
    client = ScalperProClient(CFG)
    strategy = SmartAntiCrowdStrategy(client, CFG)
    
    try:
        strategy.run(runtime_hours=24)
    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user")
        client.close_all_positions()

if __name__ == '__main__':
    main()