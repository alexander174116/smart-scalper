# SMART SCALPER PRO 🤖

## 🚀 Быстрый старт

Установка зависимостей:
pip install -r requirements.txt

Настройка:
Отредактируй config.yaml
Для live-режима создай .env и добавь API ключи в .env
API_KEY=KEY_HERE
API_SECRET=KEY_HERE

Запуск:
python btc_scalper_bot.py

Настройки
Режимы: paper (тест) / live (реальная торговля)
Риск: 1% на сделку, стоп 0.8%, трейлинг с 1.8% прибыли
Стратегия: Умные входы против толпы (RSI + Volume)

Мониторинг
Логи сделок в trades_log.csv

⚠️Тестируй в paper-режиме перед использованием реальных денег!⚠️