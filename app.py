"""事例ファインダー ローカルサーバー。

    python app.py

起動後、同じネットワークの人は http://<あなたのIP>:5000 でアクセスできます。

任意で簡易Basic認証を有効化できます（社外秘の事例を扱う場合に推奨）:
    CASE_FINDER_PASSWORD=ひみつ python app.py
    （ユーザー名は既定 "user"、CASE_FINDER_USER で変更可）
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time

from flask import (
    Flask,
    Response,
    g,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)

import azure_ai
import cache
import ingest
import jobs
import metrics
import ocr
import search
import topics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("case-finder")

# 任意: ファイル出力＋ローテーション（CASE_FINDER_LOG_FILE 設定時のみ）
_LOG_FILE = os.environ.get("CASE_FINDER_LOG_FILE")
if _LOG_FILE:
    from logging.handlers import RotatingFileHandler

    _fh = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=int(os.environ.get("CASE_FINDER_LOG_MAX_BYTES", str(5 * 1024 * 1024))),
        backupCount=int(os.environ.get("CASE_FINDER_LOG_BACKUPS", "3")),
        encoding="utf-8",
    )
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(_fh)
    log.info("ログをファイルに出力します: %s", _LOG_FILE)

app = Flask(__name__)

# Azure 要約のキャッシュ（同一クエリの再課金を回避）
_ANSWER_CACHE = cache.TTLCache(
    int(os.environ.get("CASE_FINDER_ANSWER_CACHE_SIZE", "128")),
    float(os.environ.get("CASE_FINDER_CACHE_TTL", "300")),
)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("CASE_FINDER_MAX_MB", "64")) * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 任意のBasic認証（読みと書きで分離可能）──
#   CASE_FINDER_PASSWORD       : 設定すると全アクセスに認証が必要
#   CASE_FINDER_WRITE_PASSWORD : 設定すると「読みは自由・書き(upload)のみ要認証」
AUTH_USER = os.environ.get("CASE_FINDER_USER", "user")
AUTH_PASSWORD = os.environ.get("CASE_FINDER_PASSWORD")
WRITE_PASSWORD = os.environ.get("CASE_FINDER_WRITE_PASSWORD")


def _creds_ok(passwords) -> bool:
    a = request.authorization
    if not a or a.username is None or a.password is None:
        return False
    # タイミング攻撃を避けるため定数時間比較を使う
    user_ok = hmac.compare_digest(a.username, AUTH_USER)
    pw_ok = any(hmac.compare_digest(a.password, p) for p in passwords)
    return user_ok and pw_ok


def _unauthorized():
    return Response(
        "認証が必要です", 401, {"WWW-Authenticate": 'Basic realm="case-finder"'}
    )


# 外部送信・課金を伴う読み取り系も「書き込み相当」として保護する
PRIVILEGED_PATHS = {"/api/answer", "/api/answer_stream"}
# 認証を常に通すパス（監視用ヘルスチェック）
PUBLIC_PATHS = {"/healthz"}

# ── Azure呼び出し(要約)の簡易レート制限（クライアントIP毎・分あたり）──
# コストの暴走と誤操作の連打を防ぐ。0 で無効。
ANSWER_RATE = int(os.environ.get("CASE_FINDER_ANSWER_RATE", "10"))
_rate_hits: dict[str, list[float]] = {}
_rate_lock = threading.Lock()


# 追跡するクライアント数の上限（長期稼働でのメモリ膨張を防ぐ）
RATE_MAX_KEYS = int(os.environ.get("CASE_FINDER_RATE_MAX_KEYS", "1000"))


def _rate_limited(key: str) -> bool:
    if ANSWER_RATE <= 0:
        return False
    now = time.time()
    with _rate_lock:
        q = _rate_hits.setdefault(key, [])
        while q and now - q[0] > 60.0:
            q.pop(0)
        # 期限切れのクライアントを掃除し、それでも多い場合は古い順に捨てる
        if len(_rate_hits) > RATE_MAX_KEYS:
            for k in [k for k, v in _rate_hits.items() if not v and k != key]:
                del _rate_hits[k]
            if len(_rate_hits) > RATE_MAX_KEYS:
                for k in sorted(_rate_hits, key=lambda k: _rate_hits[k][-1] if _rate_hits[k] else 0)[
                    : len(_rate_hits) - RATE_MAX_KEYS
                ]:
                    if k != key:
                        del _rate_hits[k]
        if len(q) >= ANSWER_RATE:
            return True
        q.append(now)
        return False


@app.before_request
def _timer_start():
    import time
    g._t0 = time.perf_counter()


@app.after_request
def _access_log(resp):
    try:
        import time
        dt = (time.perf_counter() - getattr(g, "_t0", 0.0)) * 1000.0
        if request.path.startswith("/api/"):
            # クエリ本文はプライバシー配慮でログに残さない（長さのみ）
            qlen = len(request.args.get("q", ""))
            log.info("%s %s %d %.0fms qlen=%d", request.method, request.path,
                     resp.status_code, dt, qlen)
    except Exception:  # noqa: BLE001
        pass
    return resp


@app.before_request
def _require_auth():
    if request.path in PUBLIC_PATHS:
        return None
    is_write = request.method in ("POST", "PUT", "DELETE") or request.path in PRIVILEGED_PATHS
    if is_write:
        # 書き込みは、全体パスワードか書き込み専用パスワードのいずれかで許可
        pws = {p for p in (AUTH_PASSWORD, WRITE_PASSWORD) if p}
        if pws and not _creds_ok(pws):
            return _unauthorized()
    else:
        if AUTH_PASSWORD and not _creds_ok({AUTH_PASSWORD}):
            return _unauthorized()
    # Azure呼び出しはレート制限（コスト保護）
    if request.path in PRIVILEGED_PATHS and _rate_limited(request.remote_addr or "?"):
        metrics.incr("rate_limited")
        return jsonify({"error": "リクエストが多すぎます。1分ほど待って再試行してください。"}), 429
    return None


@app.after_request
def _security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    return resp


@app.errorhandler(Exception)
def _unhandled(e):
    from werkzeug.exceptions import HTTPException

    if isinstance(e, HTTPException):
        return e  # 404 等はそのまま
    log.exception("未処理の例外")  # 詳細はログのみ。クライアントへは漏らさない
    return jsonify({"error": "内部エラーが発生しました。ログを確認してください。"}), 500


def _safe_filename(name: str) -> str:
    """日本語ファイル名は保ちつつ、パス区切りや '..' を排除し、長さも制限する。"""
    name = os.path.basename(name or "").strip()
    if name in ("", ".", "..") or "/" in name or "\\" in name:
        return ""
    if "\x00" in name:
        return ""
    # ファイルシステム上限(255バイト)を超えないよう拡張子を保って切り詰める
    stem, ext = os.path.splitext(name)
    ext = ext[:16]
    max_stem = 200 - len(ext.encode("utf-8"))
    b = stem.encode("utf-8")
    if len(b) > max_stem:
        stem = b[:max_stem].decode("utf-8", errors="ignore")
    return (stem + ext) or ""


def _int_arg(src, key, default, lo, hi):
    """数値パラメータを安全に取得する。不正値は 400 で返せるよう ValueError を投げる。"""
    raw = src.get(key)
    if raw in (None, ""):
        return default
    try:
        v = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{key} は整数で指定してください")
    return max(lo, min(hi, v))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    """監視/自動再起動用の軽量ヘルスチェック（認証不要・モデルロードなし）。"""
    try:
        n = len(search.get_index()["meta"])
        return jsonify({"status": "ok", "cases": n})
    except Exception as e:  # noqa: BLE001
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    industry = request.args.get("industry", "").strip()
    # loose=1 で「関連が弱い候補」も含める（足切りを WEAK_REL まで下げる）
    min_rel = search.WEAK_REL if request.args.get("loose") else None
    try:
        top_k = _int_arg(request.args, "k", 6, 1, 50)
        return jsonify(search.search(q, top_k=top_k, industry=industry, min_rel=min_rel))
    except Exception as e:  # noqa: BLE001  不正パラメータ・次元不一致など
        return jsonify({"error": str(e)}), 400


@app.route("/api/stats")
def api_stats():
    return jsonify(search.stats())


@app.route("/api/map")
def api_map():
    """事例コーパスのトピック地図（クラスタ＋キーワード＋2D配置＋エッジ）。"""
    try:
        k = _int_arg(request.args, "k", None, 1, 50)
        ix = search.get_index()
        return jsonify(topics.build_map(ix, k=k))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400


@app.route("/api/metrics")
def api_metrics():
    m = metrics.snapshot()
    m["cache"] = {"result": search._RESULT_CACHE.stats(), "answer": _ANSWER_CACHE.stats()}
    return jsonify(m)


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    q = (request.form.get("q") or "").strip()[:1000]  # ログ肥大を防ぐ
    vote = request.form.get("vote", "")
    if vote not in ("up", "down"):
        return jsonify({"error": "vote は up/down"}), 400
    try:
        # SQLite の INTEGER 範囲を超える値は None 扱い（OverflowError を防ぐ）
        case_id = _int_arg(request.form, "case_id", None, -(2 ** 63), 2 ** 63 - 1)
    except ValueError:
        case_id = None
    search.record_feedback(q, case_id, vote)
    return jsonify({"ok": True})


@app.route("/api/answer")
def api_answer():
    """検索上位を根拠に Azure 生成AIで要約・示唆を返す（任意機能）。

    Azure 未設定なら answer=null を返し、フロントは通常の検索結果のみ表示する。
    """
    q = request.args.get("q", "").strip()
    industry = request.args.get("industry", "").strip()
    top_k = _int_arg(request.args, "k", 6, 1, 50)  # 検索表示と同じ件数に統一
    min_rel = search.WEAK_REL if request.args.get("loose") else None
    try:
        result = search.search(q, top_k=top_k, industry=industry, min_rel=min_rel)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400
    nodes = result["nodes"]
    # 検索結果(nodes/edges/hidden)も同梱して返し、フロントの二重検索を避ける
    if not azure_ai.available():
        return jsonify({**result, "answer": None, "reason": "Azure未設定"})
    if not nodes:
        return jsonify({**result, "answer": None, "reason": "該当事例なし"})

    # 同一クエリの要約はキャッシュから返す（Azure課金を回避）
    akey = (q, industry, top_k, (min_rel if min_rel is not None else "d"), search.index_version())
    hit = _ANSWER_CACHE.get(akey)
    if hit is not None:
        return jsonify(hit)
    try:
        out = azure_ai.synthesize(q, nodes)
    except Exception as e:  # noqa: BLE001
        return jsonify({**result, "answer": None, "reason": f"生成に失敗: {e}"})
    payload = {**result, "answer": out["answer"],
               "citations": out["citations"], "model": out.get("model")}
    _ANSWER_CACHE.put(akey, payload)
    return jsonify(payload)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """ブラウザからPPT/PDFを受け取り、保存後に取り込みをバックグラウンド実行する。

    ファイル保存だけ即時に行い、重い処理（OCR・埋め込み）はジョブ化して
    job_id を返す。進捗は /api/job/<id> でポーリングできる。
    """
    files = request.files.getlist("files")
    industry = request.form.get("industry", "").strip()
    if not files:
        return jsonify({"error": "ファイルがありません"}), 400

    os.makedirs(search.DATA_DIR, exist_ok=True)
    saved, rejected = [], []
    for f in files:
        name = _safe_filename(f.filename)
        if not name or not name.lower().endswith(ingest.SUPPORTED):
            rejected.append(f.filename or "(名前なし)")
            continue
        path = os.path.join(search.DATA_DIR, name)
        f.save(path)
        saved.append(path)

    if not saved:
        return jsonify({"error": "対応形式のファイルがありません", "rejected": rejected}), 400

    job_id = jobs.submit(saved, industry)
    return jsonify({"job_id": job_id, "queued": len(saved), "rejected": rejected})


@app.route("/api/job/<job_id>")
def api_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "ジョブが見つかりません"}), 404
    return jsonify(job)


def _sse(obj):
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


@app.route("/api/answer_stream")
def api_answer_stream():
    """SSEで「検索結果→要約トークン→出典」を逐次配信（Azure有効時）。"""
    q = request.args.get("q", "").strip()
    industry = request.args.get("industry", "").strip()
    top_k = _int_arg(request.args, "k", 6, 1, 50)
    min_rel = search.WEAK_REL if request.args.get("loose") else None
    try:
        result = search.search(q, top_k=top_k, industry=industry, min_rel=min_rel)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400

    def gen():
        yield _sse({"type": "results", **result})
        nodes = result["nodes"]
        if not azure_ai.available():
            yield _sse({"type": "noai", "reason": "Azure未設定"})
        elif not nodes:
            yield _sse({"type": "noai", "reason": "該当事例なし"})
        else:
            try:
                for kind, val in azure_ai.synthesize_stream(q, nodes):
                    if kind == "token":
                        yield _sse({"type": "token", "text": val})
                    elif kind == "citations":
                        yield _sse({"type": "citations", "citations": val})
                    elif kind == "done":
                        yield _sse({"type": "done", "model": val})
            except Exception as e:  # noqa: BLE001
                yield _sse({"type": "error", "reason": str(e)})
        yield _sse({"type": "end"})

    return Response(stream_with_context(gen()), mimetype="text/event-stream")


@app.route("/data/<path:filename>")
def data_file(filename):
    """検索結果から元の PPT/PDF を開けるようにする。"""
    return send_from_directory(search.DATA_DIR, filename)


def _run():
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("CASE_FINDER_HOST", "0.0.0.0")
    if not AUTH_PASSWORD and not WRITE_PASSWORD:
        print("※ 認証なしで公開します（読み書きとも自由）。社外秘なら CASE_FINDER_PASSWORD"
              " か CASE_FINDER_WRITE_PASSWORD の設定を推奨。")
    elif WRITE_PASSWORD and not AUTH_PASSWORD:
        print("※ 読み取りは無認証です。検索結果や元ファイル(/data)も誰でも閲覧できます。"
              "社外秘の本文を守るには CASE_FINDER_PASSWORD（全体認証）を推奨。")
    if host == "0.0.0.0":
        print("※ ネットワーク全体に公開中。HTTPのBasic認証は平文です。社外秘なら"
              " TLS(リバースプロキシ)か CASE_FINDER_HOST=127.0.0.1＋SSHトンネルを推奨。")
    jobs.warmup()  # 埋め込みモデルを先読みして初回検索を速く
    print(f"\n事例ファインダーを起動します → http://{host}:{port}")
    if host == "0.0.0.0":
        print("同じネットワークの人は http://<このPCのIP>:%d で使えます。\n" % port)
    try:
        from waitress import serve  # 本番向けの安定サーバー

        serve(app, host=host, port=port, threads=int(os.environ.get("THREADS", "8")))
    except ImportError:
        print("（waitress未導入のため開発サーバーで起動。常用は pip install waitress を推奨）")
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    _run()
