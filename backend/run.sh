#!/usr/bin/env bash
# 백엔드 실행 스크립트
set -e
cd "$(dirname "$0")"
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q -r requirements.txt
uvicorn app.main:app --reload --port 8000
