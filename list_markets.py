# list_markets.py
import ccxt, os
ex_id = 'bybit'   # поменяйте, если нужно
ex_cls = getattr(ccxt, ex_id)
ex = ex_cls({'enableRateLimit': True})
markets = ex.load_markets()
for s, m in markets.items():
    if 'MYX' in s or ('symbol' in m and 'MYX' in m['symbol']):
        print(s, m.get('type'), m.get('info', {}) )
