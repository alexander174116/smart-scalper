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

class SmartAntiCrowdStrategy:
    def __init__(self, client: ScalperProClient, cfg: Config):
        self.client = client
        self.cfg = cfg
        self.entry_strategy = EntryStrategy(cfg.entry_strategy)
        
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
        """Главный анализ символа"""
        data = self.client.fetch_ohlcv(symbol, self.cfg.timeframe, limit=100)
        if len(data) < 50:
            return None

        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        if self.entry_strategy == EntryStrategy.SMART_MOMENTUM:
            signal = self.analyze_smart_momentum(symbol, df)
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