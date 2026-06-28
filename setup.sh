#!/usr/bin/env bash
# ローカル(非Docker)セットアップ。Python 3.10+ が必要。
set -e
cd "$(dirname "$0")"

python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

echo ""
echo "セットアップ完了。次の手順で起動できます:"
echo "  source .venv/bin/activate"
echo "  # data/ に事例(PPT/PDF)を置いてから:"
echo "  python ingest.py        # 初回はモデルを自動DL"
echo "  python app.py           # http://localhost:5000"
echo ""
echo "OCRでTesseractを使う場合のみ別途インストール（既定はEasyOCRでpipのみ）。"
