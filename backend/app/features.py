"""피처 엔지니어링.

자료조사 의사결정③(State 설계)을 반영해, OHLCV 만으로 계산 가능한 상태 벡터를
구성한다. quantylab/rltrader 의 v1 비율 피처(이동평균 이격도 등 15종)에
보고서가 강조한 기술적 지표(RSI·MACD·Stochastic·CCI·Bollinger·MFI)를 더한다.

정규화: quantylab 신경망(DNN/LSTM/CNN)의 첫 층이 BatchNorm1d 이므로 입력 정규화는
학습 시 내부적으로 수행된다(학습 구간 통계로만 fit → 룩어헤드 방지). 0~100 유계
지표(RSI/Stochastic/MFI)는 /100 으로, CCI 는 tanh 로 추가 스케일링한다.
"""
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

CHART_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

# quantylab v1 비율 피처 (이동평균 이격도/거래량 이격도 등)
V1_COLUMNS = [
    "open_lastclose_ratio", "high_close_ratio", "low_close_ratio",
    "close_lastclose_ratio", "volume_lastvolume_ratio",
    "close_ma5_ratio", "volume_ma5_ratio",
    "close_ma10_ratio", "volume_ma10_ratio",
    "close_ma20_ratio", "volume_ma20_ratio",
    "close_ma60_ratio", "volume_ma60_ratio",
    "close_ma120_ratio", "volume_ma120_ratio",
]

# 보고서 의사결정③에서 권장한 기술적 지표
TA_COLUMNS = [
    "rsi14", "stoch_k", "stoch_d", "cci14",
    "macd_signal_ratio", "boll_upper_ratio", "boll_lower_ratio", "mfi14",
]

TRAINING_COLUMNS = V1_COLUMNS + TA_COLUMNS

# 프론트 체크박스 UI 용 지표 메타데이터 (id/label/group).
# group 으로 묶어 화면에서 그룹별 표시한다.
FEATURE_META = [
    # --- 가격/거래량 비율 (quantylab v1) ---
    {"id": "open_lastclose_ratio", "label": "시가/전일종가", "group": "가격·거래량 비율"},
    {"id": "high_close_ratio", "label": "고가/종가", "group": "가격·거래량 비율"},
    {"id": "low_close_ratio", "label": "저가/종가", "group": "가격·거래량 비율"},
    {"id": "close_lastclose_ratio", "label": "종가/전일종가", "group": "가격·거래량 비율"},
    {"id": "volume_lastvolume_ratio", "label": "거래량/전일거래량", "group": "가격·거래량 비율"},
    {"id": "close_ma5_ratio", "label": "종가 MA5 이격도", "group": "이동평균 이격도"},
    {"id": "volume_ma5_ratio", "label": "거래량 MA5 이격도", "group": "이동평균 이격도"},
    {"id": "close_ma10_ratio", "label": "종가 MA10 이격도", "group": "이동평균 이격도"},
    {"id": "volume_ma10_ratio", "label": "거래량 MA10 이격도", "group": "이동평균 이격도"},
    {"id": "close_ma20_ratio", "label": "종가 MA20 이격도", "group": "이동평균 이격도"},
    {"id": "volume_ma20_ratio", "label": "거래량 MA20 이격도", "group": "이동평균 이격도"},
    {"id": "close_ma60_ratio", "label": "종가 MA60 이격도", "group": "이동평균 이격도"},
    {"id": "volume_ma60_ratio", "label": "거래량 MA60 이격도", "group": "이동평균 이격도"},
    {"id": "close_ma120_ratio", "label": "종가 MA120 이격도", "group": "이동평균 이격도"},
    {"id": "volume_ma120_ratio", "label": "거래량 MA120 이격도", "group": "이동평균 이격도"},
    # --- 기술적 지표 (보고서 의사결정③) ---
    {"id": "rsi14", "label": "RSI(14)", "group": "기술적 지표"},
    {"id": "stoch_k", "label": "스토캐스틱 %K", "group": "기술적 지표"},
    {"id": "stoch_d", "label": "스토캐스틱 %D", "group": "기술적 지표"},
    {"id": "cci14", "label": "CCI(14)", "group": "기술적 지표"},
    {"id": "macd_signal_ratio", "label": "MACD-시그널", "group": "기술적 지표"},
    {"id": "boll_upper_ratio", "label": "볼린저 상단 이격도", "group": "기술적 지표"},
    {"id": "boll_lower_ratio", "label": "볼린저 하단 이격도", "group": "기술적 지표"},
    {"id": "mfi14", "label": "MFI(14)", "group": "기술적 지표"},
]


def resolve_feature_columns(selected) -> List[str]:
    """선택된 지표 id 리스트를 검증해 TRAINING_COLUMNS 순서대로 정렬해 반환.

    None/빈 값이면 전체(TRAINING_COLUMNS)를 사용한다. 알 수 없는 id 는 무시하고,
    유효한 지표가 하나도 없으면 오류를 낸다.
    """
    if not selected:
        return list(TRAINING_COLUMNS)
    selected_set = set(selected)
    cols = [c for c in TRAINING_COLUMNS if c in selected_set]
    if not cols:
        raise ValueError("선택된 state 지표가 없습니다. 최소 1개 이상 선택해주세요.")
    return cols


def _v1_ratios(df: pd.DataFrame) -> pd.DataFrame:
    windows = [5, 10, 20, 60, 120]
    for w in windows:
        df[f"close_ma{w}"] = df["close"].rolling(w).mean()
        df[f"volume_ma{w}"] = df["volume"].rolling(w).mean()
        df[f"close_ma{w}_ratio"] = (df["close"] - df[f"close_ma{w}"]) / df[f"close_ma{w}"]
        df[f"volume_ma{w}_ratio"] = (df["volume"] - df[f"volume_ma{w}"]) / df[f"volume_ma{w}"]

    df["open_lastclose_ratio"] = 0.0
    df.loc[df.index[1:], "open_lastclose_ratio"] = (
        (df["open"][1:].values - df["close"][:-1].values) / df["close"][:-1].values
    )
    df["high_close_ratio"] = (df["high"] - df["close"]) / df["close"]
    df["low_close_ratio"] = (df["low"] - df["close"]) / df["close"]
    df["close_lastclose_ratio"] = 0.0
    df.loc[df.index[1:], "close_lastclose_ratio"] = (
        (df["close"][1:].values - df["close"][:-1].values) / df["close"][:-1].values
    )
    vol_prev = df["volume"].shift(1).replace(0, np.nan).ffill().bfill()
    df["volume_lastvolume_ratio"] = (df["volume"] - df["volume"].shift(1)) / vol_prev
    df["volume_lastvolume_ratio"] = df["volume_lastvolume_ratio"].fillna(0.0)
    return df


def _technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    # RSI(14) -> [0,1]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = (100 - 100 / (1 + rs)) / 100.0

    # Stochastic %K(14), %D(3) -> [0,1]
    ll = low.rolling(14).min()
    hh = high.rolling(14).max()
    stoch_k = (close - ll) / (hh - ll).replace(0, np.nan) * 100
    df["stoch_k"] = stoch_k / 100.0
    df["stoch_d"] = stoch_k.rolling(3).mean() / 100.0

    # CCI(14) -> tanh 스케일
    tp = (high + low + close) / 3.0
    ma_tp = tp.rolling(14).mean()
    md = (tp - ma_tp).abs().rolling(14).mean()
    cci = (tp - ma_tp) / (0.015 * md.replace(0, np.nan))
    df["cci14"] = np.tanh(cci / 100.0)

    # MACD(12,26,9): (MACD - signal) / close
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_signal_ratio"] = (macd - signal) / close

    # Bollinger Bands(20, 2): 상/하단 대비 이격도
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    df["boll_upper_ratio"] = (close - upper) / close
    df["boll_lower_ratio"] = (close - lower) / close

    # MFI(14) -> [0,1]
    rmf = tp * vol
    pos_mf = rmf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    neg_mf = rmf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    df["mfi14"] = (100 - 100 / (1 + mfr)) / 100.0
    return df


def build_features(rows: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """OHLCV dict 리스트 -> (chart_df, training_df).

    두 DataFrame 모두 'date' 컬럼(문자열 YYYYMMDD)을 0번 인덱스에 포함하도록
    구성한다(quantylab Environment.PRICE_IDX=4 가정과 정합).
    """
    if not rows:
        raise ValueError("입력 OHLCV 데이터가 비어 있습니다.")
    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    df = _v1_ratios(df)
    df = _technical_indicators(df)

    # 워밍업(NaN) 행 제거
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=TRAINING_COLUMNS).reset_index(drop=True)

    chart_df = df[CHART_COLUMNS].copy()
    training_df = df[["date"] + TRAINING_COLUMNS].copy()
    # quantylab 은 chart_data[PRICE_IDX=4]=close, training_data 는 순수 피처만
    # build_sample 에서 training_data.iloc[idx].tolist() 를 그대로 상태로 쓰므로
    # training_df 에서 'date' 는 제외해 넘긴다(엔진에서 처리).
    return chart_df, training_df


def slice_by_date(chart_df, training_df, date_from: str, date_to: str):
    """[date_from, date_to] (YYYYMMDD) 구간으로 잘라 반환."""
    mask = (chart_df["date"] >= date_from) & (chart_df["date"] <= date_to)
    c = chart_df[mask].reset_index(drop=True)
    t = training_df[mask].reset_index(drop=True)
    return c, t
