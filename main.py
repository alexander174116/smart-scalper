#!/usr/bin/env python3
"""
SMART SCALPER PRO - Главный файл
"""

import logging
from dotenv import load_dotenv
import sys
from config import load_config
from exchange_client import ScalperProClient
from strategy import SmartAntiCrowdStrategy

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # сразу в stdout
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Загрузка конфигурации
    CFG = load_config()
    load_dotenv()
    
    if CFG.verbose:
        logger.info(f"Config loaded: {CFG}")
    
    # Инициализация клиента и стратегии
    client = ScalperProClient(CFG)
    strategy = SmartAntiCrowdStrategy(client, CFG)
    
    try:
        # Запуск бота
        strategy.run(runtime_hours=24)
    except KeyboardInterrupt:
        logger.info("🛑 Stopped by user")
        client.close_all_positions()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        client.close_all_positions()

if __name__ == '__main__':
    main()