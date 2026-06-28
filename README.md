# 事例ファインダー（ローカル意味検索サーバー）

手元の事例（PPT / PDF）を、意味の近さで正確に探せるローカルアプリです。
Connected Papers のように「あなたの問題」を中心にしたグラフで関連事例を表示します。
**データは外部に送信しません。生成AI（LLM）は使いません。**
意味理解は BERT 系エンコーダ、画像中心の資料は OCR で扱います（モデルは初回のみDL）。

## できること
- `data/` に置いた PPT / PDF を読み取り、意味ベクトル化して登録
- 画像中心のスライド・スキャンPDFは **OCR（Tesseract 日本語）** で文字化
- **BERT 埋め込み**で各事例を「課題 / 施策 / 成果」に自動仕分け（LLM不使用）
- 「困っていること」を文章で入力すると、意味の近い事例をランキング表示
- 事例同士のつながりをグラフで可視化／結果から元ファイルを開ける
- ローカルサーバーなので、同じネットワークの人みんなで使える

## セットアップ
```bash
python -m venv .venv && source .venv/bin/activate   # 任意
pip install -r requirements.txt

# OCR を使う場合は Tesseract 本体も導入（任意・画像中心の資料向け）
#   macOS : brew install tesseract tesseract-lang
#   Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-jpn
```
Tesseract が無くてもテキスト層のある PPT/PDF はそのまま取り込めます
（OCR は自動で有効/無効を判定します）。

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

## 仕組み（生成AI不使用）
- 抽出: `PyMuPDF`（PDF）/ `python-pptx`（PPT）
- OCR: `Tesseract`（`pytesseract`）。テキスト層が乏しいページ/画像スライドを自動でOCR
- 意味ベクトル: `sentence-transformers` の `intfloat/multilingual-e5-small`
  （XLM-RoBERTa ＝ BERT 系エンコーダ。生成LLMではありません）
- 課題/施策/成果の仕分け: 見出し検出 ＋ **BERT埋め込みのゼロショット分類**
  （各文を埋め込み、ラベル代表文との類似度で割り当て）
- 保存: SQLite（`cases.db`）。再起動しても再計算不要
- 検索: コサイン類似度（ベクトルは正規化済み）

別のモデルを使いたい場合は環境変数で指定できます:
```bash
CASE_FINDER_MODEL=intfloat/multilingual-e5-base python ingest.py
```

## 構成
| ファイル | 役割 |
|---|---|
| `ingest.py` | PPT/PDF を読み取り（必要ならOCR）→ 仕分け → ベクトル化 → DB登録 |
| `ocr.py` | 画像中心の資料を Tesseract で文字化 |
| `extract.py` | BERT埋め込みで「課題/施策/成果」に分類 |
| `search.py` | 埋め込み・DB・検索/グラフの中核ロジック |
| `app.py` | Flask サーバー（API + 画面配信） |
| `templates/index.html` | 画面（意味検索＋関係グラフ） |
| `data/` | 事例の PPT/PDF を置く場所 |
| `cases.db` | 事例テキスト＋ベクトルの保存先（自動生成） |

## 今後の案
- OCR エンジンの選択肢追加（EasyOCR / PaddleOCR など、より高精度な日本語OCR）
- 画面からのアップロードで取り込み
- 分類しきい値やラベル代表文の調整UI
