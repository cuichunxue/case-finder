"""軽量な計測（外部依存なし）。

検索回数・レイテンシ・キャッシュヒット率・Azure呼び出し/トークン等を集計し、
/api/metrics で確認できるようにする。運用時の健全性把握用。
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

_lock = threading.Lock()
_counters: dict[str, float] = defaultdict(float)
_timers: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])  # name -> [sum, count]
_started = time.time()


def incr(name: str, n: float = 1.0):
    with _lock:
        _counters[name] += n


def observe(name: str, value_ms: float):
    with _lock:
        t = _timers[name]
        t[0] += value_ms
        t[1] += 1


class timed:
    """with metrics.timed('search'): ... でレイテンシを記録するコンテキスト。"""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        observe(self.name, (time.perf_counter() - self.t0) * 1000.0)
        return False


def snapshot() -> dict:
    with _lock:
        counters = dict(_counters)
        timers = {
            k: {"avg_ms": round(v[0] / v[1], 1) if v[1] else 0.0, "count": int(v[1])}
            for k, v in _timers.items()
        }
    return {"uptime_sec": round(time.time() - _started, 1), "counters": counters, "latency": timers}
