# config.py
import os
class Config:
   SYMBOL = os.getenv("SYMBOL", "BTC")
   DATA_PERIOD = os.getenv("DATA_PERIOD", "1y")
   DATA_INTERVAL = os.getenv("DATA_INTERVAL", "1d")
   SECRET_KEY = os.getenv("SECRET_KEY", "CG-kKmhm8Kman67faEfbLVbJdgz")