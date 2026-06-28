"""スレッドセーフな TTL + LRU キャッシュ（外部依存なし）。

検索結果や Azure 要約のような「同じ入力なら同じ出力」を短時間使い回し、
再計算と API 課金を削減する。インデックス更新時はキー（mtime 込み）が変わるか、
明示 clear() で失効する。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict


class TTLCache:
    def __init__(self, size: int = 256, ttl: float = 300.0):
        self.size = size
        self.ttl = ttl
        self._d: "OrderedDict[object, tuple[float, object]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            item = self._d.get(key)
            if item is None:
                return None
            ts, val = item
            if self.ttl and (time.time() - ts) > self.ttl:
                self._d.pop(key, None)
                return None
            self._d.move_to_end(key)
            return val

    def put(self, key, val):
        if self.size <= 0:
            return
        with self._lock:
            self._d[key] = (time.time(), val)
            self._d.move_to_end(key)
            while len(self._d) > self.size:
                self._d.popitem(last=False)

    def clear(self):
        with self._lock:
            self._d.clear()
