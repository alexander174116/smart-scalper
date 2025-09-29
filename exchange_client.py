import time
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

import ccxt
import pandas as pd

from config import Config
from logger import TradeLogger
from indicators import calculate_quantity

logger = logging.getLogger(__name__)

class ScalperProClient:
    def __init__(self, cfg: Config):
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

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 100):
        """Безопасное получение данных"""
        try:
            ex_cls = getattr(ccxt, self.cfg.exchange_id)
            ex = ex_cls({'enableRateLimit': True})
            data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            return data if data else []
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return []

    def get_current_price(self, symbol: str) -> float:
        """Текущая цена для paper-режима"""
        ohlcv = self.fetch_ohlcv(symbol, self.cfg.timeframe, limit=1)
        return float(ohlcv[-1][4]) if ohlcv else 0

    def can_trade_symbol(self, symbol: str) -> bool:
        """Проверка лимита сделок на символ"""
        symbol_trades = [p for p in self.open_positions if p['symbol'] == symbol]
        return len(symbol_trades) < self.cfg.trades_per_symbol

    def place_market_order(self, symbol: str, side: str, quantity: float):
        """Размещение ордера с проверками и корректировкой количества"""
        current_price = self.get_current_price(symbol)
        
        # Корректируем количество согласно stepSize
        adjusted_quantity = self.adjust_quantity(symbol, quantity)
        if adjusted_quantity <= 0:
            return None
            
        position_value = adjusted_quantity * current_price
        margin_required = position_value / self.cfg.leverage
        
        # Проверка маржи
        if margin_required > self.balance * 0.5:
            logger.warning(f"🚨 Margin too high: ${margin_required:.2f}")
            max_position_value = (self.balance * 0.5) * self.cfg.leverage
            adjusted_quantity = max_position_value / current_price
            # Снова корректируем после изменения
            adjusted_quantity = self.adjust_quantity(symbol, adjusted_quantity)
            logger.info(f"   Adjusted quantity to: {adjusted_quantity:.6f}")
        
        if self.cfg.mode == "paper":
            position = {
                'symbol': symbol,
                'side': side,
                'quantity': adjusted_quantity,
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
                    'quantity': adjusted_quantity,
                    'entry_time': time.time()
                }
            }
            self.open_positions.append(position)
            logger.info(f"[PAPER] {side.upper()} {adjusted_quantity:.6f} {symbol} @ {current_price:.6f}")  
            return {'status': 'filled', 'price': current_price}
        else:
            try:
                return self.exchange.create_order(
                    symbol, 'market', side.lower(), adjusted_quantity
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
            else:
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
                else:
                    new_stop = current_price * (1 + self.cfg.trailing_step)
                    if new_stop < position['current_stop']:
                        position['current_stop'] = new_stop

            # Закрытие позиции по стопу
            if stop_hit:
                self.close_position(position, current_price, "trailing_stop" if position['trailing_activated'] else "initial_stop")

    def close_position(self, position: Dict[str, Any], exit_price: float, reason: str):
        """Закрытие позиции и логирование"""
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

    def get_equity(self) -> float:
        return self.balance

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Автоматическое определение точности символа"""
        try:
            # Для live-режима получаем реальные данные с биржи
            if self.cfg.mode == "live" and self.exchange:
                try:
                    markets = self.exchange.load_markets()
                    if symbol in markets:
                        market = markets[symbol]
                        return {
                            'precision': {
                                'amount': market['precision']['amount'] if market['precision']['amount'] else 0.001,
                                'price': market['precision']['price'] if market['precision']['price'] else 0.0001
                            },
                            'limits': {
                                'amount': market['limits']['amount'],
                                'cost': market['limits']['cost'] if 'cost' in market['limits'] else {'min': 10, 'max': 100000}
                            }
                        }
                except Exception as e:
                    logger.warning(f"Could not load market info for {symbol}: {e}")
            
            # Для paper-режима или при ошибке - АВТООПРЕДЕЛЕНИЕ
            return self._auto_detect_precision(symbol)
            
        except Exception as e:
            logger.error(f"Error in get_symbol_info for {symbol}: {e}")
            return self._get_fallback_precision()

    def _auto_detect_precision(self, symbol: str) -> Dict[str, Any]:
        """Автоматическое определение точности на основе цены"""
        try:
            # Получаем текущую цену для анализа
            current_price = self.get_current_price(symbol)
            
            # Определяем stepSize на основе цены
            if current_price >= 1000:
                amount_step = 0.001
                price_step = 0.1
            elif current_price >= 100:
                amount_step = 0.01
                price_step = 0.01
            elif current_price >= 10:
                amount_step = 0.1
                price_step = 0.001
            elif current_price >= 1:
                amount_step = 1.0
                price_step = 0.0001
            elif current_price >= 0.1:
                amount_step = 10.0
                price_step = 0.00001
            else:
                amount_step = 100.0
                price_step = 0.000001
            
            min_amount = amount_step
            min_cost = 10
            
            logger.info(f"   📊 Current price for {symbol}: ${current_price:.6f}")
            
            return {
                'precision': {
                    'amount': amount_step,
                    'price': price_step
                },
                'limits': {
                    'amount': {'min': min_amount, 'max': 100000},
                    'cost': {'min': min_cost, 'max': 100000}
                }
            }
            
        except Exception as e:
            logger.error(f"Error in auto_detect_precision: {e}")
            return self._get_fallback_precision()

    def _get_fallback_precision(self) -> Dict[str, Any]:
        """Резервные настройки при ошибке"""
        return {
            'precision': {'amount': 0.001, 'price': 0.0001},
            'limits': {
                'amount': {'min': 0.1, 'max': 10000},
                'cost': {'min': 10, 'max': 100000}
            }
        }

    def adjust_quantity(self, symbol: str, quantity: float) -> float:
        """Умная корректировка количества с автоопределением точности"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            step_size = symbol_info['precision']['amount']
            min_qty = symbol_info['limits']['amount']['min']
            min_notional = symbol_info['limits']['cost']['min']
            
            # Округляем до шага количества
            adjusted_qty = round(quantity / step_size) * step_size
            
            # Проверяем минимальное количество
            if adjusted_qty < min_qty:
                logger.warning(f"❌ Quantity too small: {adjusted_qty} < {min_qty}")
                adjusted_qty = min_qty
                logger.info(f"   Increased to min quantity: {adjusted_qty}")
            
            # Проверяем минимальную сумму ордера
            current_price = self.get_current_price(symbol)
            notional = adjusted_qty * current_price
            
            if notional < min_notional:
                logger.warning(f"❌ Notional too small: ${notional:.2f} < ${min_notional}")
                required_qty = min_notional / current_price
                adjusted_qty = round(required_qty / step_size) * step_size
                adjusted_qty = max(adjusted_qty, min_qty)
                logger.info(f"   Increased quantity to meet min notional: {adjusted_qty:.6f}")
                
                new_notional = adjusted_qty * current_price
                if new_notional < min_notional:
                    logger.error(f"🚨 Still below min notional after adjustment: ${new_notional:.2f}")
                    return 0
            
            logger.info(f"   🔧 Quantity: {quantity:.6f} → {adjusted_qty:.6f} (step: {step_size})")
            logger.info(f"   💰 Notional: ${notional:.2f} | Min: ${min_notional}")
            
            return adjusted_qty
            
        except Exception as e:
            logger.error(f"Error adjusting quantity for {symbol}: {e}")
            return quantity

    def close_all_positions(self):
        for position in list(self.open_positions):
            current_price = self.get_current_price(position['symbol'])
            self.close_position(position, current_price, "shutdown")