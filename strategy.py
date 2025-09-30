import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from config import Config
from exchange_client import ScalperProClient
from indicators import rsi, obv, detect_volume_climax, calculate_quantity

logger = logging.getLogger(__name__)

class EntryStrategy(Enum):
    SMART_MOMENTUM = "smart_momentum"
    VOLUME_SURGE = "volume_surge" 
    BREAKOUT_FAKE = "breakout_fake"
    ALGO_HUNTER = "algo_hunter"  # ← ДОБАВИТЬ

class SmartAntiCrowdStrategy:
    def __init__(self, client: ScalperProClient, cfg: Config):
        self.client = client
        self.cfg = cfg
        self.entry_strategy = EntryStrategy(cfg.entry_strategy)

    def analyze_algo_hunter(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """
        Охотник за алгоритмами - ловим моменты слома паттернов
        """
        if len(df) < 50:
            return None

        closes = df['close'].values
        highs = df['high'].values  
        lows = df['low'].values
        volumes = df['volume'].values
        
        # 1. Базовые индикаторы
        rsi_val = rsi(closes, 14)
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:])
        
        # 2. Volume анализ
        volume_surge = volumes[-1] > np.mean(volumes[-20:]) * 2
        volume_decline = volumes[-1] < np.mean(volumes[-20:]) * 0.5
        
        # 3. Волатильность
        atr_val = calculate_atr(highs, lows, closes, self.cfg.atr_period)
        current_range = highs[-1] - lows[-1]
        high_volatility = current_range > atr_val * 1.5 if atr_val > 0 else False
        
        # 4. Price Action паттерны
        hammer = is_hammer(highs, lows, closes)
        shooting_star = is_shooting_star(highs, lows, closes)
        inside_bar = is_inside_bar(highs, lows)
        
        # 5. Liquidity зоны
        support, resistance = find_liquidity_zones(highs, lows, closes)
        near_support = abs(closes[-1] - support) / closes[-1] < 0.005
        near_resistance = abs(closes[-1] - resistance) / closes[-1] < 0.005
        
        # 🎯 СИГНАЛЫ ПРОТИВ АЛГОРИТМОВ:
        
        # Сигнал 1: Ложный пробой + volume surge
        if (self.is_fake_breakout(highs, lows, closes) and volume_surge and 
            high_volatility and near_resistance):
            return {
                'side': 'short',
                'symbol': symbol, 
                'reason': 'Algo Short: Fake breakout + volume surge + resistance',
                'confidence': 0.8
            }
        
        # Сигнал 2: Панические продажи + поддержка
        if (rsi_val < 25 and volume_surge and near_support and 
            closes[-1] > sma_20 and hammer):
            return {
                'side': 'long',
                'symbol': symbol,
                'reason': 'Algo Long: Oversold panic + support + hammer',
                'confidence': 0.85
            }
        
        # Сигнал 3: Тихий накопление перед движением
        if (volume_decline and near_support and 
            is_compression(highs, lows, self.cfg.compression_period) and rsi_val < 40):
            return {
                'side': 'long', 
                'symbol': symbol,
                'reason': 'Algo Long: Accumulation + compression + support',
                'confidence': 0.75
            }
        
        # Сигнал 4: Distribution на вершине
        if (volume_decline and near_resistance and
            is_compression(highs, lows, self.cfg.compression_period) and rsi_val > 60 and shooting_star):
            return {
                'side': 'short',
                'symbol': symbol, 
                'reason': 'Algo Short: Distribution + compression + resistance',
                'confidence': 0.75
            }

        return None

    def is_fake_breakout(self, highs, lows, closes, lookback=10):
        """Определение ложного пробоя"""
        if len(highs) < lookback + 1:
            return False
            
        # Пробой вверх но закрытие внутри диапазона
        if highs[-1] > max(highs[-lookback:-1]) and closes[-1] < closes[-2]:
            return True
            
        # Пробой вниз но закрытие внутри диапазона  
        if lows[-1] < min(lows[-lookback:-1]) and closes[-1] > closes[-2]:
            return True
            
        return False

    def circus_arbitrage_signals(self, symbol: str, df: pd.DataFrame):
        """
        Ловим моменты когда алгоритмы создают цирк на рынке
        """
        closes = df['close'].values
        volumes = df['volume'].values
        
        # 1. "Паника" - резкий объем на небольшом движении
        recent_volume = np.mean(volumes[-3:])
        avg_volume = np.mean(volumes[-20:])
        price_change = abs(closes[-1] - closes[-3]) / closes[-3]
        
        panic_signal = (recent_volume > avg_volume * 3 and 
                       price_change < 0.02)  # большой объем на маленьком движении
        
        # 2. "Затишье перед бурей" - низкий объем перед движением
        calm_signal = (recent_volume < avg_volume * 0.7 and 
                      is_compression(df['high'].values, df['low'].values))
        
        # 3. "Отскок от ликвидности" - быстрая реакция от ключевых уровней
        support, resistance = find_liquidity_zones(df['high'].values, df['low'].values, closes)
        near_level = min(abs(closes[-1] - support), abs(closes[-1] - resistance)) / closes[-1] < 0.003
        
        if panic_signal and near_level:
            direction = 'long' if closes[-1] > closes[-2] else 'short'
            return {
                'side': direction,
                'symbol': symbol,
                'reason': f'Circus Arb: Panic {direction} near liquidity level',
                'confidence': 0.7
            }
        
        return None
        
    def multi_timeframe_analysis(self, symbol: str) -> Tuple[str, float]:
        """Мультитаймфрейм анализ"""
        if not self.cfg.multi_timeframe:
            return "neutral", 0.5
            
        try:
            confirmation_data = self.client.fetch_ohlcv(
                symbol, self.cfg.tf_confirmation, limit=50
            )
            if not confirmation_data:
                return "neutral", 0.5
                
            df_conf = pd.DataFrame(confirmation_data, 
                                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            ma_fast = df_conf['close'].rolling(8).mean()
            ma_slow = df_conf['close'].rolling(21).mean()
            
            current_fast = ma_fast.iloc[-1]
            current_slow = ma_slow.iloc[-1]
            prev_fast = ma_fast.iloc[-2]
            prev_slow = ma_slow.iloc[-2]
            
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

    def analyze_smart_momentum(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """АГРЕССИВНАЯ стратегия для 1м таймфрейма"""
        closes = df['close'].values
        volumes = df['volume'].values
        highs = df['high'].values
        lows = df['low'].values
        
        if len(closes) < 20:
            return None

        current_rsi = rsi(closes, self.cfg.rsi_period)
        volume_ok = volumes[-1] > np.mean(volumes[-10:]) * 1.2
        
        ma_fast = np.mean(closes[-5:])
        ma_slow = np.mean(closes[-15:])
        ma_cross = "bullish" if ma_fast > ma_slow else "bearish"
        
        # Сигналы
        if (current_rsi > 65 and volume_ok and ma_cross == "bearish" and 
            closes[-1] < closes[-2]):
            return {
                'side': 'short', 
                'symbol': symbol, 
                'reason': f'1m Short: RSI {current_rsi:.1f} + MA bearish + price down',
                'confidence': 0.75
            }
        elif (current_rsi < 35 and volume_ok and ma_cross == "bullish" and 
              closes[-1] > closes[-2]):
            return {
                'side': 'long', 
                'symbol': symbol, 
                'reason': f'1m Long: RSI {current_rsi:.1f} + MA bullish + price up', 
                'confidence': 0.75
            }
        
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

    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
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
        elif self.entry_strategy == EntryStrategy.ALGO_HUNTER:  # ← ДОБАВИТЬ
            signal = self.analyze_algo_hunter(symbol, df)
        else:
            signal = self.analyze_smart_momentum(symbol, df)
            
        if signal and signal.get('confidence', 0) < 0.7:
            return None
            
        return signal

    def execute_trading(self):
        """Исполнение торговли"""
        if len(self.client.open_positions) >= self.cfg.max_open_trades:
            return
            
        if time.time() - self.client.last_trade_time < self.cfg.trade_cooldown_sec:
            return

        daily_pnl = self.client.balance - self.client.initial_balance
        if daily_pnl <= -self.client.initial_balance * self.cfg.max_daily_loss_pct:
            logger.warning("Daily loss limit reached! Stopping trading.")
            return

        signals = []
        for symbol in self.cfg.symbols:
            if not self.client.can_trade_symbol(symbol):
                continue
                
            signal = self.analyze_symbol(symbol)
            if signal:
                signals.append(signal)
        
        if signals:
            best_signal = max(signals, key=lambda x: x.get('confidence', 0))
            if best_signal.get('confidence', 0) >= 0.7:
                self.enter_trade(best_signal)

    def enter_trade(self, signal: Dict[str, Any]):
        """Вход в сделку"""
        equity = self.client.get_equity()
        current_price = self.client.get_current_price(signal['symbol'])
        
        notional, quantity, actual_stop = calculate_quantity(
            equity, 
            self.cfg.risk_per_trade_pct,
            self.cfg.initial_stop_pct,
            current_price,
            self.cfg.leverage,
            self.cfg.position_multiplier
        )

        if quantity <= 0 or notional < 10:
            logger.warning(f"❌ Position too small: ${notional:.2f}")
            return

        order = self.client.place_market_order(
            signal['symbol'], 
            signal['side'], 
            quantity
        )

        if order and order.get('status') in ('filled', None):
            self.client.last_trade_time = time.time()
            logger.info(f"🎯 ENTRY {signal['side'].upper()} {signal['symbol']}")
            logger.info(f"   💰 Size: ${notional:.2f} | Risk: ${notional * self.cfg.initial_stop_pct:.2f}")
            logger.info(f"   🎯 Entry: ${current_price:.6f} | Stop: {actual_stop*100:.2f}%")  # 6 знаков у цены
            logger.info(f"   📈 Reason: {signal['reason']}")

    def run(self, runtime_hours: int = 24):
        """Запуск бота"""
        logger.info(f"🚀 Starting SMART SCALPER PRO (Anti-Crowd) | {len(self.cfg.symbols)} symbols")
        logger.info(f"Strategy: {self.entry_strategy.value} | RSI {self.cfg.rsi_period} | Stop: {self.cfg.initial_stop_pct*100}% | Trail: {self.cfg.trailing_start*100}%")
        
        end_time = time.time() + runtime_hours * 3600
        last_status_time = 0
        
        while time.time() < end_time:
            try:
                self.client.update_positions()
                self.execute_trading()

                current_time = time.time()
                if current_time - last_status_time >= 60:  # каждые 60 секунд
                    logger.info(f"📊 Status: {len(self.client.open_positions)} open | Equity: ${self.client.get_equity():.2f}")
                    last_status_time = current_time  # обновляем время
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(10)

        logger.info("⏹️ Runtime finished")
        self.client.close_all_positions()