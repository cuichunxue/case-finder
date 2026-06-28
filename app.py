"""事例ファインダー ローカルサーバー。

    python app.py

起動後、同じネットワークの人は http://<あなたのIP>:5000 でアクセスできます。

任意で簡易Basic認証を有効化できます（社外秘の事例を扱う場合に推奨）:
    CASE_FINDER_PASSWORD=ひみつ python app.py
    （ユーザー名は既定 "user"、CASE_FINDER_USER で変更可）
"""

from __future__ import annotations

import os

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

import ingest
import ocr
import search

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("CASE_FINDER_MAX_MB", "64")) * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 任意のBasic認証 ──
AUTH_USER = os.environ.get("CASE_FINDER_USER", "user")
AUTH_PASSWORD = os.environ.get("CASE_FINDER_PASSWORD")  # 未設定なら認証なし


@app.before_request
def _require_auth():
    if not AUTH_PASSWORD:
        return None
    a = request.authorization
    if not a or a.username != AUTH_USER or a.password != AUTH_PASSWORD:
        return Response(
            "認証が必要です", 401, {"WWW-Authenticate": 'Basic realm="case-finder"'}
        )
    return None


def _safe_filename(name: str) -> str:
    """日本語ファイル名は保ちつつ、パス区切りや '..' を排除する。"""
    name = os.path.basename(name or "").strip()
    if name in ("", ".", "..") or "/" in name or "\\" in name:
        return ""
    return name


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    industry = request.args.get("industry", "").strip()
    top_k = int(request.args.get("k", 6))
    return jsonify(search.search(q, top_k=top_k, industry=industry))


@app.route("/api/stats")
def api_stats():
    return jsonify(search.stats())


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """ブラウザからPPT/PDFを受け取り、保存→取り込み→キャッシュ更新する。"""
    files = request.files.getlist("files")
    industry = request.form.get("industry", "").strip()
    if not files:
        return jsonify({"error": "ファイルがありません"}), 400

    os.makedirs(search.DATA_DIR, exist_ok=True)
    saved, rejected = [], []
    for f in files:
        name = _safe_filename(f.filename)
        if not name or not name.lower().endswith(ingest.SUPPORTED):
            rejected.append(f.filename)
            continue
        path = os.path.join(search.DATA_DIR, name)
        f.save(path)
        saved.append(path)

    conn = search.connect()
    search.init_db(conn)
    ocr_ok = ocr.available()
    added = []
    for path in saved:
        # industry を指定した場合のみ明示採用、未指定は自動推定
        if ingest.ingest_file(conn, path, ocr_ok, industry or None):
            added.append(os.path.basename(path))
    conn.close()
    search.invalidate_cache()

    return jsonify(
        {
            "added": added,
            "rejected": rejected,
            "count": search.stats()["count"],
            "industries": search.list_industries(),
        }
    )


@app.route("/data/<path:filename>")
def data_file(filename):
    """検索結果から元の PPT/PDF を開けるようにする。"""
    return send_from_directory(os.path.join(BASE_DIR, "data"), filename)


def _run():
    port = int(os.environ.get("PORT", 5000))
    if not AUTH_PASSWORD:
        print("※ 認証なしで公開します。社外秘の事例は CASE_FINDER_PASSWORD の設定を推奨。")
    print(f"\n事例ファインダーを起動します → http://0.0.0.0:{port}")
    print("同じネットワークの人は http://<このPCのIP>:%d で使えます。\n" % port)
    try:
        from waitress import serve  # 本番向けの安定サーバー

        serve(app, host="0.0.0.0", port=port, threads=int(os.environ.get("THREADS", "8")))
    except ImportError:
        print("（waitress未導入のため開発サーバーで起動。常用は pip install waitress を推奨）")
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    _run()
