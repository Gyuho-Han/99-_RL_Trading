"""성과지표 계산 공용 모듈 — quantylab/hanium 두 엔진에서 공용.

기존에 engine.py 와 engine_hanium.py 에 중복돼 있던
  1) 포트폴리오 가치 시계열 → 기본 지표(수익률·Sharpe·MDD)
  2) 보유주식 수 변화 → 체결 내역 복원(+매도별 실현손익)
을 이곳으로 통합하고, PPT 목표 지표였던 확장 지표
  3) 승률 · 손익비 · 월별 양수 수익률 비율 · 월평균 거래횟수
를 추가로 계산한다.
"""
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

TRADING_DAYS = 252


def metrics_from_pv(pv) -> Dict:
    """포트폴리오 가치 시계열 -> 기본 성과지표."""
    pv = np.asarray(pv, dtype=float)
    if len(pv) < 2:
        return dict(cumulative_return=0.0, annual_return=0.0, annual_vol=0.0,
                    sharpe=0.0, mdd=0.0)
    initial = pv[0]
    cum = pv[-1] / initial - 1
    rets = np.diff(pv) / pv[:-1]
    ann_vol = float(np.std(rets) * math.sqrt(TRADING_DAYS)) if len(rets) > 1 else 0.0
    mean_daily = float(np.mean(rets))
    ann_ret = (1 + cum) ** (TRADING_DAYS / len(pv)) - 1
    sharpe = (mean_daily / np.std(rets) * math.sqrt(TRADING_DAYS)) if np.std(rets) > 0 else 0.0
    peak = np.maximum.accumulate(pv)
    dd = (pv - peak) / peak
    mdd = float(dd.min())
    return dict(cumulative_return=float(cum), annual_return=float(ann_ret),
                annual_vol=ann_vol, sharpe=float(sharpe), mdd=mdd)


def restore_trades(dates: Sequence[str], closes: Sequence[float],
                   num_stocks: Sequence[int],
                   charge: float = 0.00015, tax: float = 0.0025):
    """보유주식 수 변화 기반 체결 내역 복원 + 매도별 실현손익.

    반환: (trades_log, trade_summary)
    """
    n = min(len(dates), len(closes), len(num_stocks))
    trades_log: List[Dict] = []
    prev_shares = 0
    avg_cost = 0.0            # 1주당 평균 매입원가(매수 수수료 포함)
    total_traded_amount = 0.0
    total_realized_profit = 0.0
    for i in range(n):
        shares = int(num_stocks[i])
        price = float(closes[i])
        delta = shares - prev_shares
        if delta > 0:  # 매수
            qty = delta
            buy_cost_per = price * (1 + charge)
            avg_cost = (avg_cost * prev_shares + buy_cost_per * qty) / (prev_shares + qty)
            amount = qty * price
            total_traded_amount += amount
            trades_log.append({
                "date": dates[i], "side": "buy", "shares": qty, "price": price,
                "amount": amount, "holding_after": shares,
            })
        elif delta < 0:  # 매도
            qty = -delta
            proceeds_per = price * (1 - charge - tax)
            profit = (proceeds_per - avg_cost) * qty
            ret = (proceeds_per / avg_cost - 1) if avg_cost > 0 else 0.0
            amount = qty * price
            total_traded_amount += amount
            total_realized_profit += profit
            trades_log.append({
                "date": dates[i], "side": "sell", "shares": qty, "price": price,
                "amount": amount, "profit": float(profit), "return": float(ret),
                "avg_cost": float(avg_cost), "holding_after": shares,
            })
        prev_shares = shares
    summary = {
        "total_traded_amount": float(total_traded_amount),
        "total_realized_profit": float(total_realized_profit),
        "num_trades": len(trades_log),
    }
    return trades_log, summary


def _monthly_returns(dates: Sequence[str], pv: Sequence[float]) -> List[float]:
    """월별 수익률 목록. dates 는 YYYYMMDD 문자열, pv 는 같은 길이."""
    n = min(len(dates), len(pv))
    if n < 2:
        return []
    rets = []
    month_start_pv = float(pv[0])
    cur_month = dates[0][:6]
    last_pv = float(pv[0])
    for i in range(1, n):
        m = dates[i][:6]
        if m != cur_month:
            if month_start_pv > 0:
                rets.append(last_pv / month_start_pv - 1)
            cur_month = m
            month_start_pv = last_pv
        last_pv = float(pv[i])
    if month_start_pv > 0:
        rets.append(last_pv / month_start_pv - 1)
    return rets


def extended_metrics(trade_log: List[Dict], dates: Sequence[str],
                     pv: Sequence[float]) -> Dict:
    """확장 지표(PPT 목표 지표): 승률·손익비·월별 양수수익률·월평균 거래횟수.

    - 승률: 매도(청산) 중 실현이익 > 0 비율
    - 손익비: 평균 실현이익 / 평균 실현손실
    - 월별 양수 수익률 비율: 월 수익률 > 0 인 달의 비율
    - 월평균 거래횟수: (매수+매도) / 개월 수
    거래가 없어 계산 불가한 값은 None.
    """
    sells = [t for t in trade_log if t.get("side") == "sell"]
    wins = [t for t in sells if t.get("profit", 0) > 0]
    losses = [t for t in sells if t.get("profit", 0) <= 0]

    win_rate = (len(wins) / len(sells)) if sells else None
    avg_gain = (sum(t["profit"] for t in wins) / len(wins)) if wins else 0.0
    avg_loss = (-sum(t["profit"] for t in losses) / len(losses)) if losses else 0.0
    if wins and losses and avg_loss > 0:
        profit_factor = avg_gain / avg_loss
    elif wins and not losses:
        profit_factor = None   # 손실 매도가 없어 정의 불가(전승)
    else:
        profit_factor = None

    monthly = _monthly_returns(dates, pv)
    monthly_win_rate = (sum(1 for r in monthly if r > 0) / len(monthly)) if monthly else None
    n_months = max(1, len(monthly))
    trades_per_month = len(trade_log) / n_months

    return {
        "win_rate": float(win_rate) if win_rate is not None else None,
        "profit_factor": float(profit_factor) if profit_factor is not None else None,
        "monthly_win_rate": float(monthly_win_rate) if monthly_win_rate is not None else None,
        "trades_per_month": float(trades_per_month),
        "num_months": len(monthly),
    }
