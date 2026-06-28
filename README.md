# 事例ファインダー（ローカル意味検索サーバー）

手元の事例（PPT / PDF）を、意味の近さで正確に探せるローカルアプリです。
Connected Papers のように「あなたの問題」を中心にしたグラフで関連事例を表示します。
**データは外部に送信しません。** 埋め込みモデルもローカルで動きます（初回のみDL）。

## できること
- `data/` に置いた PPT / PDF を読み取り、意味ベクトル化して登録
- 「困っていること」を文章で入力すると、意味の近い事例をランキング表示
- 事例同士のつながりをグラフで可視化
- 結果から元ファイルをそのまま開ける
- ローカルサーバーなので、同じネットワークの人みんなで使える

## セットアップ
```bash
python -m venv .venv && source .venv/bin/activate   # 任意
pip install -r requirements.txt
```

## 使い方
1. `data/` フォルダに事例の PPT / PDF を入れる
2. 取り込み（初回はモデルを自動ダウンロード）
   ```bash
   python ingest.py
   ```
3. サーバー起動
   ```bash
   python app.py
   ```
4. ブラウザで `http://localhost:5000`
   - 同じネットワークの人は `http://<このPCのIP>:5000` でアクセス

事例を追加・更新したら `python ingest.py` を再実行するだけです
（同じファイル名は上書き、未対応の画像PDFはOCRなしのため抽出されません）。

## 仕組み
- 抽出: `PyMuPDF`（PDF）/ `python-pptx`（PPT）
- 意味ベクトル: `sentence-transformers` の多言語モデル `intfloat/multilingual-e5-small`
- 保存: SQLite（`cases.db`）。再起動しても再計算不要
- 検索: コサイン類似度（ベクトルは正規化済み）

別のモデルを使いたい場合は環境変数で指定できます:
```bash
CASE_FINDER_MODEL=intfloat/multilingual-e5-base python ingest.py
```

## 構成
| ファイル | 役割 |
|---|---|
| `ingest.py` | PPT/PDF を読み取り → ベクトル化 → DB登録 |
| `search.py` | 埋め込み・DB・検索/グラフの中核ロジック |
| `app.py` | Flask サーバー（API + 画面配信） |
| `templates/index.html` | 画面（意味検索＋関係グラフ） |
| `data/` | 事例の PPT/PDF を置く場所 |
| `cases.db` | 事例テキスト＋ベクトルの保存先（自動生成） |

## 今後（v2 案）
- ローカル LLM（Ollama 等）で「課題 / 施策 / 成果」を自動抽出して項目表示
- 画像中心の PDF/PPT 向けに OCR を追加
- 画面からのアップロードで取り込み
