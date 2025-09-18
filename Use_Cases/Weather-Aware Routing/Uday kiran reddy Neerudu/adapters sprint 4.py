import time
from functools import lru_cache
def _cache_key(lat, lon, minutes=5):
    bucket = int(time.time() // (60 * minutes))
    return (round(lat, 4), round(lon, 4), bucket)
@lru_cache(maxsize=10000)
def _cached_value(kind, key, default):
    return default
def get_weather_score(lat, lon, mode="FALLBACK", default=0.2):
    key = ("weather",) + _cache_key(lat, lon)
    return _cached_value("weather", key, default)
def get_traffic_proxy(lat, lon, mode="FALLBACK", default=0.2):
    key = ("traffic",) + _cache_key(lat, lon)
    return _cached_value("traffic", key, default)
