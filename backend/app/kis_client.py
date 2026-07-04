"""한국투자증권(KIS) OpenAPI 클라이언트.

기능
  - 접근토큰 발급 및 파일 캐싱 (24h 유효, 발급 빈도 제한 회피)
  - 국내주식 기간별 일봉(OHLCV) 조회 (tr_id: FHKST03010100)
    한 번에 약 100영업일까지만 반환되므로 날짜 구간을 나눠 페이지네이션 후 병합한다.
  - OHLCV 로컬 캐시: 종목별 JSON 파일(backend/.ohlcv_cache/)에 저장하고,
    캐시에 없는 날짜 구간만 증분 조회한다. 반복 실험 시 KIS 재조회를 제거해
    fetching 단계가 수 초 → 즉시로 단축된다. '오늘' 일봉은 장중 미확정이므로
    커버 범위에 포함하지 않아 다음 실행 때 자동 재조회된다.
    (수정주가 이벤트(액면분할 등) 발생 시 캐시 파일을 삭제하면 전체 재조회)

주의: 시세 조회는 실전/모의 도메인 모두 가능하다. 본 클라이언트는 매매 주문을
전혀 수행하지 않으며 오직 과거 시세 조회만 한다.
"""
import json
import time
import datetime as dt
from typing import List, Dict

import requests

from . import config

# keep-alive 연결 재사용 (호출마다 TCP/TLS 핸드셰이크 반복 방지)
_session = requests.Session()


class KISError(RuntimeError):
    pass


def _load_cached_token() -> str | None:
    try:
        with open(config.TOKEN_CACHE_PATH, "r") as f:
            data = json.load(f)
        if data.get("env") != config.KIS_ENV:
            return None
        if time.time() < data.get("expires_at", 0) - 600:  # 10분 여유
            return data.get("access_token")
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return None


def _save_cached_token(token: str, expires_in: int) -> None:
    data = {
        "access_token": token,
        "expires_at": time.time() + expires_in,
        "env": config.KIS_ENV,
    }
    with open(config.TOKEN_CACHE_PATH, "w") as f:
        json.dump(data, f)


def get_access_token() -> str:
    if not config.kis_configured():
        raise KISError(
            "KIS_APP_KEY / KIS_APP_SECRET 가 설정되지 않았습니다. "
            "backend/.env 파일에 키를 넣어주세요 (.env.example 참고)."
        )
    cached = _load_cached_token()
    if cached:
        return cached

    url = f"{config.KIS_DOMAIN}/oauth2/tokenP"
    body = {
        "grant_type": "client_credentials",
        "appkey": config.KIS_APP_KEY,
        "appsecret": config.KIS_APP_SECRET,
    }
    resp = _session.post(url, json=body, timeout=15)
    if resp.status_code != 200:
        raise KISError(f"토큰 발급 실패 ({resp.status_code}): {resp.text}")
    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise KISError(f"토큰 발급 응답에 access_token 없음: {data}")
    _save_cached_token(token, int(data.get("expires_in", 86400)))
    return token


def _headers(tr_id: str) -> Dict[str, str]:
    return {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {get_access_token()}",
        "appkey": config.KIS_APP_KEY,
        "appsecret": config.KIS_APP_SECRET,
        "tr_id": tr_id,
        "custtype": "P",
    }


def _is_rate_limited(status_code: int, body: dict) -> bool:
    """KIS 초당 호출 제한(EGW00201) 여부."""
    if not isinstance(body, dict):
        return False
    return body.get("msg_cd") == "EGW00201" or "초당 거래건수" in str(body.get("msg1", ""))


def _fetch_daily_chunk(code: str, start: str, end: str, max_retries: int = 5) -> List[Dict]:
    """단일 구간(<=100영업일 권장) 일봉 조회. start/end 는 YYYYMMDD.

    KIS 초당 호출 제한(EGW00201)에 걸리면 지수 백오프로 재시도한다.
    """
    url = (
        f"{config.KIS_DOMAIN}"
        "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    )
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",   # 주식
        "FID_INPUT_ISCD": code,
        "FID_INPUT_DATE_1": start,
        "FID_INPUT_DATE_2": end,
        "FID_PERIOD_DIV_CODE": "D",      # 일봉
        "FID_ORG_ADJ_PRC": "0",          # 0: 수정주가 반영
    }
    last_err = None
    for attempt in range(max_retries):
        resp = _session.get(url, headers=_headers("FHKST03010100"), params=params, timeout=15)
        try:
            data = resp.json()
        except ValueError:
            data = {}
        # 초당 호출 제한 -> 백오프 후 재시도 (HTTP 200/500 모두에서 발생 가능)
        if _is_rate_limited(resp.status_code, data):
            wait = 0.6 * (2 ** attempt)  # 0.6, 1.2, 2.4, 4.8, 9.6s
            last_err = data.get("msg1", "초당 거래건수 초과")
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            raise KISError(f"일봉 조회 실패 ({resp.status_code}): {resp.text}")
        if data.get("rt_cd") not in ("0", 0):
            raise KISError(f"일봉 조회 오류: {data.get('msg1')} (code={code})")
        out = data.get("output2") or []
        rows = []
        for r in out:
            if not r or not r.get("stck_bsop_date"):
                continue
            # 종가가 비어있는(미체결/휴장/미래) 행은 건너뛴다.
            if r.get("stck_clpr") in (None, "", "0"):
                continue
            try:
                rows.append({
                    "date": r["stck_bsop_date"],
                    "open": float(r["stck_oprc"]),
                    "high": float(r["stck_hgpr"]),
                    "low": float(r["stck_lwpr"]),
                    "close": float(r["stck_clpr"]),
                    "volume": float(r["acml_vol"]),
                })
            except (KeyError, ValueError, TypeError):
                continue
        return rows
    raise KISError(
        f"일봉 조회 실패: KIS 초당 호출 제한이 계속 발생합니다 ({last_err}). "
        "잠시 후 다시 시도해주세요."
    )


def _date_windows(start: str, end: str, days: int = 100):
    """[start, end] 를 days(달력일) 단위 구간으로 분할."""
    d0 = dt.datetime.strptime(start, "%Y%m%d").date()
    d1 = dt.datetime.strptime(end, "%Y%m%d").date()
    cur = d0
    step = dt.timedelta(days=days)
    while cur <= d1:
        w_end = min(cur + step, d1)
        yield cur.strftime("%Y%m%d"), w_end.strftime("%Y%m%d")
        cur = w_end + dt.timedelta(days=1)


# ----------------------------------------------------------------------
# OHLCV 로컬 캐시
# ----------------------------------------------------------------------
def _cache_path(code: str):
    config.OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config.OHLCV_CACHE_DIR / f"{code}.json"


def _load_ohlcv_cache(code: str) -> Dict:
    try:
        with open(_cache_path(code), "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("rows"), dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {"covered_from": None, "covered_to": None, "rows": {}}


def _save_ohlcv_cache(code: str, cache: Dict) -> None:
    try:
        with open(_cache_path(code), "w") as f:
            json.dump(cache, f)
    except OSError:
        pass  # 캐시 저장 실패는 치명적이지 않음 (다음 실행 시 재조회)


def _shift_day(yyyymmdd: str, days: int) -> str:
    d = dt.datetime.strptime(yyyymmdd, "%Y%m%d").date() + dt.timedelta(days=days)
    return d.strftime("%Y%m%d")


def _fetch_range(code: str, date_from: str, date_to: str) -> Dict[str, Dict]:
    """[date_from, date_to] 구간을 KIS 에서 조회해 {date: row} 로 반환."""
    merged: Dict[str, Dict] = {}
    for w_start, w_end in _date_windows(date_from, date_to, days=140):
        chunk = _fetch_daily_chunk(code, w_start, w_end)
        for row in chunk:
            merged[row["date"]] = row
        time.sleep(0.35)  # 호출 속도 제한 보호 (KIS 초당 제한 회피)
    return merged


def fetch_ohlcv(code: str, date_from: str, date_to: str) -> List[Dict]:
    """국내주식 일봉 OHLCV 를 [date_from, date_to] 전체 구간에 대해 조회.

    로컬 캐시를 우선 사용하고, 커버되지 않은 날짜 구간만 KIS 에서 증분 조회한다.
    반환: date 오름차순 정렬된 dict 리스트. date 는 'YYYYMMDD'.
    """
    code = str(code).zfill(6)
    today = dt.date.today().strftime("%Y%m%d")
    yesterday = _shift_day(today, -1)

    cache = _load_ohlcv_cache(code)
    cov_from, cov_to = cache["covered_from"], cache["covered_to"]

    # 캐시에 없는 구간 계산 (앞쪽/뒤쪽 증분)
    need: List[tuple] = []
    if cov_from is None:
        need.append((date_from, date_to))
    else:
        if date_from < cov_from:
            need.append((date_from, _shift_day(cov_from, -1)))
        if date_to > cov_to:
            need.append((_shift_day(cov_to, 1), date_to))

    if need:
        for f_, t_ in need:
            if f_ > t_:
                continue
            cache["rows"].update(_fetch_range(code, f_, t_))
        # 커버 범위 갱신 — '오늘'은 장중 미확정이라 어제까지만 커버로 기록
        new_from = min(filter(None, [cov_from, date_from]))
        new_to = max(filter(None, [cov_to, min(date_to, yesterday)]))
        cache["covered_from"], cache["covered_to"] = new_from, new_to
        _save_ohlcv_cache(code, cache)

    rows = [r for d, r in cache["rows"].items() if date_from <= d <= date_to]
    rows.sort(key=lambda r: r["date"])
    return rows
