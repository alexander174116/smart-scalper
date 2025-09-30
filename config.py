from dataclasses import dataclass
from typing import List
import yaml

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
    entry_strategy: str
    multi_timeframe: bool
    tf_primary: str
    tf_confirmation: str
    min_volume_ratio: float
    max_volatility: float
    trend_filter: bool
    position_multiplier: int
    tf_context: str
    use_volume_profile: bool
    use_orderbook: bool
    use_market_depth: bool
    rsi_divergence: bool
    volume_analysis: bool
    price_action: bool
    liquidity_zones: bool
    momentum_filter: bool
    atr_period: int
    compression_period: int

def load_config() -> Config:
    with open("config.yaml", "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)