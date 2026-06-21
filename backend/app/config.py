"""환경설정 로더.

KIS(한국투자증권) OpenAPI 키와 서버 환경을 .env 에서 읽는다.
키는 절대 코드/깃에 커밋하지 않으며 backend/.env 파일로만 관리한다.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# backend/.env 로드
BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")

# KIS 인증 정보
KIS_APP_KEY = os.environ.get("KIS_APP_KEY", "")
KIS_APP_SECRET = os.environ.get("KIS_APP_SECRET", "")

# real(실전) | paper(모의). 시세 조회는 둘 다 가능.
KIS_ENV = os.environ.get("KIS_ENV", "real").strip().lower()

KIS_DOMAIN = (
    "https://openapi.koreainvestment.com:9443"
    if KIS_ENV == "real"
    else "https://openapivts.koreainvestment.com:29443"
)

# 토큰 캐시 파일 (KIS 는 토큰 발급 호출 빈도 제한이 있어 캐싱 필수)
TOKEN_CACHE_PATH = BACKEND_DIR / ".kis_token_cache.json"

# CORS 허용 오리진 (프론트 dev 서버)
CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")


def kis_configured() -> bool:
    return bool(KIS_APP_KEY) and bool(KIS_APP_SECRET)
