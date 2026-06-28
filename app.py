"""事例ファインダー ローカルサーバー。

    python app.py

起動後、同じネットワークの人は http://<あなたのIP>:5000 でアクセスできます。
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request, send_from_directory

import search

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    top_k = int(request.args.get("k", 6))
    result = search.search(q, top_k=top_k)
    return jsonify(result)


@app.route("/api/stats")
def api_stats():
    return jsonify(search.stats())


@app.route("/data/<path:filename>")
def data_file(filename):
    """検索結果から元の PPT/PDF を開けるようにする。"""
    return send_from_directory(os.path.join(BASE_DIR, "data"), filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n事例ファインダーを起動します → http://0.0.0.0:{port}")
    print("同じネットワークの人は http://<このPCのIP>:%d で使えます。\n" % port)
    app.run(host="0.0.0.0", port=port, debug=False)
