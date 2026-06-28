"""取り込みをバックグラウンドで実行する軽量ジョブ機構（外部依存なし）。

アップロードはファイル保存だけ即座に行い、OCR・埋め込みといった重い処理は
単一のワーカースレッドで順番に処理する。進捗は /api/job で取得できる。

- ジョブ状態はメモリ上（プロセス再起動で消える）。小規模な社内利用を想定。
- ワーカーは1本に固定し、SQLite書き込みとモデル推論の競合を避ける。
"""

from __future__ import annotations

import os
import queue
import threading
import time
import uuid

_jobs: dict[str, dict] = {}
_q: "queue.Queue[tuple[str, list[str], str]]" = queue.Queue()
_lock = threading.Lock()
_worker_started = False


def _set(jid: str, **kw):
    with _lock:
        _jobs.setdefault(jid, {}).update(kw)


def get(jid: str):
    with _lock:
        j = _jobs.get(jid)
        return dict(j) if j else None


def submit(paths: list[str], industry: str = "") -> str:
    """取り込みジョブを投入し、ジョブIDを返す。"""
    _ensure_worker()
    jid = uuid.uuid4().hex[:8]
    _set(
        jid,
        id=jid,
        status="queued",
        total=len(paths),
        done=0,
        current="",
        added=[],
        rejected=[],
        errors=[],
        created=time.time(),
    )
    _q.put((jid, paths, industry))
    return jid


def _process(jid: str, paths: list[str], industry: str):
    # 重い import はワーカー内で（起動を軽く保つ）
    import ingest
    import ocr
    import search

    _set(jid, status="running")
    conn = search.connect()
    search.init_db(conn)
    ocr_ok = ocr.available()
    for path in paths:
        name = os.path.basename(path)
        _set(jid, current=name)
        try:
            if ingest.ingest_file(conn, path, ocr_ok, industry or None):
                with _lock:
                    _jobs[jid]["added"].append(name)
            else:
                with _lock:
                    _jobs[jid]["rejected"].append(name)
        except Exception as e:  # noqa: BLE001
            with _lock:
                _jobs[jid]["errors"].append(f"{name}: {e}")
        with _lock:
            _jobs[jid]["done"] += 1
    conn.close()
    search.invalidate_cache()
    _set(jid, status="done", current="", count=search.stats()["count"],
         industries=search.list_industries())


def _worker():
    while True:
        jid, paths, industry = _q.get()
        try:
            _process(jid, paths, industry)
        except Exception as e:  # noqa: BLE001
            _set(jid, status="error", current="", errors=[str(e)])
        finally:
            _q.task_done()


def _ensure_worker():
    global _worker_started
    with _lock:
        if _worker_started:
            return
        _worker_started = True
    threading.Thread(target=_worker, name="ingest-worker", daemon=True).start()


def warmup():
    """起動時に埋め込みモデルを別スレッドで先読みし、初回検索の待ちを減らす。"""
    def _run():
        try:
            import search

            search.embed(["warmup"], "query")
        except Exception:  # noqa: BLE001
            pass  # モデル未DL等は実検索時に顕在化させる

    threading.Thread(target=_run, name="warmup", daemon=True).start()
