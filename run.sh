#!/usr/bin/env bash
# サーバー起動の簡易ラッパー。
set -e
cd "$(dirname "$0")"
if [ -d .venv ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi
exec python app.py
