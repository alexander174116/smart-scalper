import os
import csv
from datetime import datetime
from typing import Dict, Any

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
    
    def log_trade(self, trade_data: Dict[str, Any]):
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