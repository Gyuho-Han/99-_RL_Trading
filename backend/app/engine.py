"""학습 + 백테스트 엔진.

흐름: KIS OHLCV 조회 -> 피처 생성 -> 학습/테스트 구간 분할 ->
선택한 알고리즘(DQN/A2C/PPO)으로 학습 -> 테스트 구간 백테스트 ->
수익률·Sharpe·MDD 등 지표를 단순보유(Buy&Hold)와 함께 계산.
"""
import os
import sys
import math
import tempfile
import datetime as dt
from typing import Callable, Dict, Optional

import numpy as np

# 벤더링한 quantylab/rltrader 패키지 경로 등록
_VENDOR = os.path.join(os.path.dirname(__file__), "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
os.environ.setdefault("RLTRADER_BACKEND", "pytorch")

from quantylab.rltrader.learners import DQNLearner, A2CLearner, PPOLearner  # noqa: E402

from . import kis_client, features  # noqa: E402

TRADING_DAYS = 252

# 알고리즘 메타 (프론트 버튼/설명용)
ALGORITHMS = {
    "dqn": {"label": "DQN", "desc": "가치기반·이산행동(매수/관망/매도). 베이스라인", "net_default": "dnn"},
    "a2c": {"label": "A2C", "desc": "액터-크리틱. 변동성 낮고 하락장 방어에 강함", "net_default": "lstm"},
    "ppo": {"label": "PPO", "desc": "정책기반 클리핑(프레임워크 정합 근사). 추세추종에 강함", "net_default": "lstm"},
}

# 테스트 종목 (자료조사 의사결정① 반영)
STOCKS = [
    {"code": "005930", "name": "삼성전자", "tag": "대형·저변동", "note": "안정적 대형주 (벤치마크용)"},
    {"code": "028300", "name": "에이치엘비", "tag": "바이오·고변동", "note": "논문 [C] 바이오 테마 계열 고변동주"},
    {"code": "086520", "name": "에코프로", "tag": "2차전지·고변동", "note": "고변동 테마주 (추가 선정)"},
]


def _make_learner(algo, code, chart_data, training_data, *, net, num_steps,
                  lr, discount_factor, num_epoches, balance,
                  min_trading_price, max_trading_price, start_epsilon,
                  output_path, reuse_models):
    vpath = os.path.join(output_path, "value.mdl")
    ppath = os.path.join(output_path, "policy.mdl")
    common = dict(
        stock_code=code, chart_data=chart_data, training_data=training_data,
        net=net, num_steps=num_steps, lr=lr, discount_factor=discount_factor,
        num_epoches=num_epoches, balance=balance,
        min_trading_price=min_trading_price, max_trading_price=max_trading_price,
        start_epsilon=start_epsilon, output_path=output_path,
        reuse_models=reuse_models,
    )
    if algo == "dqn":
        return DQNLearner(rl_method="dqn", value_network_path=vpath, **common)
    if algo == "a2c":
        return A2CLearner(rl_method="a2c", value_network_path=vpath,
                          policy_network_path=ppath, **common)
    if algo == "ppo":
        return PPOLearner(rl_method="ppo", value_network_path=vpath,
                          policy_network_path=ppath, **common)
    raise ValueError(f"지원하지 않는 알고리즘: {algo}")


def _metrics_from_pv(pv: np.ndarray) -> Dict:
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
    # 최대 낙폭(MDD)
    peak = np.maximum.accumulate(pv)
    dd = (pv - peak) / peak
    mdd = float(dd.min())
    return dict(cumulative_return=float(cum), annual_return=float(ann_ret),
                annual_vol=ann_vol, sharpe=float(sharpe), mdd=mdd)


def run_experiment(params: Dict, progress_callback: Optional[Callable] = None) -> Dict:
    """단일 실험(학습+백테스트) 실행. params 키는 main.py 의 요청 스키마 참고."""
    code = str(params["stock_code"]).zfill(6)
    algo = params["algorithm"]
    net = params.get("net") or ALGORITHMS[algo]["net_default"]
    num_steps = int(params.get("num_steps") or (1 if net == "dnn" else 5))
    lr = float(params.get("lr", 0.0005))
    discount_factor = float(params.get("discount_factor", 0.9))
    num_epoches = int(params.get("num_epoches", 100))
    balance = int(params.get("balance", 10_000_000))
    min_tp = int(params.get("min_trading_price", 100_000))
    max_tp = int(params.get("max_trading_price", 1_000_000))

    train_start = params["train_start"].replace("-", "")
    train_end = params["train_end"].replace("-", "")
    test_start = params["test_start"].replace("-", "")
    test_end = params["test_end"].replace("-", "")

    # 워밍업 포함해 넉넉히 조회 (ma120 등 지표 계산용으로 train_start 이전 250일)
    fetch_from = (dt.datetime.strptime(train_start, "%Y%m%d").date()
                  - dt.timedelta(days=250)).strftime("%Y%m%d")
    fetch_to = max(train_end, test_end)

    if progress_callback:
        progress_callback("fetching", 0, num_epoches, None, None)
    rows = kis_client.fetch_ohlcv(code, fetch_from, fetch_to)
    if len(rows) < 130:
        raise ValueError(
            f"조회된 일봉이 {len(rows)}건으로 부족합니다(지표 워밍업 130일 필요). "
            "기간을 늘리거나 종목코드를 확인하세요."
        )

    chart_all, train_all = features.build_features(rows)
    chart_tr, feat_tr = features.slice_by_date(chart_all, train_all, train_start, train_end)
    chart_te, feat_te = features.slice_by_date(chart_all, train_all, test_start, test_end)
    if len(chart_tr) < num_steps + 5:
        raise ValueError("학습 구간 데이터가 너무 적습니다. 학습 기간을 늘려주세요.")
    if len(chart_te) < num_steps + 2:
        raise ValueError("테스트 구간 데이터가 너무 적습니다. 테스트 기간을 늘려주세요.")

    # training_data 는 순수 피처만 (date 제외)
    td_tr = feat_tr[features.TRAINING_COLUMNS]
    td_te = feat_te[features.TRAINING_COLUMNS]
    cd_tr = chart_tr[features.CHART_COLUMNS]
    cd_te = chart_te[features.CHART_COLUMNS]

    output_path = tempfile.mkdtemp(prefix=f"rl_{code}_{algo}_")

    # ---- 학습 ----
    def _train_cb(epoch, total, pv, pl):
        if progress_callback:
            progress_callback("training", epoch, total, pv, pl)

    learner = _make_learner(
        algo, code, cd_tr, td_tr, net=net, num_steps=num_steps, lr=lr,
        discount_factor=discount_factor, num_epoches=num_epoches, balance=balance,
        min_trading_price=min_tp, max_trading_price=max_tp, start_epsilon=1.0,
        output_path=output_path, reuse_models=False,
    )
    learner.visualize_enabled = False
    learner.run(learning=True, progress_callback=_train_cb)
    learner.save_models()

    # ---- 백테스트 (테스트 구간, 탐험 0, 단일 에폭) ----
    if progress_callback:
        progress_callback("backtesting", num_epoches, num_epoches, None, None)
    tester = _make_learner(
        algo, code, cd_te, td_te, net=net, num_steps=num_steps, lr=lr,
        discount_factor=discount_factor, num_epoches=1, balance=balance,
        min_trading_price=min_tp, max_trading_price=max_tp, start_epsilon=0.0,
        output_path=output_path, reuse_models=True,
    )
    tester.visualize_enabled = False
    tester.run(learning=False)

    # 백테스트 결과 수집
    model_pv = list(tester.memory_pv)            # 행동 시점별 포트폴리오 가치
    actions = list(tester.memory_action)
    offset = num_steps - 1                        # 앞쪽 num_steps-1 스텝은 행동 없음
    dates = cd_te["date"].tolist()[offset:offset + len(model_pv)]
    closes = cd_te["close"].tolist()[offset:offset + len(model_pv)]

    # 단순보유(Buy&Hold) 곡선: 동일 구간 종가 기준
    if closes:
        base = closes[0]
        bh_pv = [balance * (c / base) for c in closes]
    else:
        bh_pv = []

    # ---- 체결 내역 복원 (보유주식 수 변화 기반) + 매도별 실현손익 ----
    num_stocks = list(tester.memory_num_stocks)[:len(dates)]
    CHARGE = tester.agent.TRADING_CHARGE
    TAX = tester.agent.TRADING_TAX
    trades_log = []
    prev_shares = 0
    avg_cost = 0.0           # 1주당 평균 매입원가(매수 수수료 포함)
    total_traded_amount = 0.0
    total_realized_profit = 0.0
    for i in range(len(dates)):
        shares = int(num_stocks[i]) if i < len(num_stocks) else prev_shares
        price = float(closes[i])
        delta = shares - prev_shares
        if delta > 0:  # 매수
            qty = delta
            buy_cost_per = price * (1 + CHARGE)
            avg_cost = (avg_cost * prev_shares + buy_cost_per * qty) / (prev_shares + qty)
            amount = qty * price
            total_traded_amount += amount
            trades_log.append({
                "date": dates[i], "side": "buy", "shares": qty, "price": price,
                "amount": amount, "holding_after": shares,
            })
        elif delta < 0:  # 매도
            qty = -delta
            proceeds_per = price * (1 - CHARGE - TAX)
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

    model_metrics = _metrics_from_pv(np.array(model_pv))
    bh_metrics = _metrics_from_pv(np.array(bh_pv))

    return {
        "stock_code": code,
        "algorithm": algo,
        "net": net,
        "num_steps": num_steps,
        "num_epoches": num_epoches,
        "train_period": [train_start, train_end],
        "test_period": [test_start, test_end],
        "n_train": int(len(cd_tr)),
        "n_test": int(len(dates)),
        "trades": {"buy": int(tester.agent.num_buy),
                   "sell": int(tester.agent.num_sell),
                   "hold": int(tester.agent.num_hold)},
        "series": {
            "dates": dates,
            "close": closes,
            "model_pv": [float(x) for x in model_pv],
            "buyhold_pv": [float(x) for x in bh_pv],
            "actions": [int(a) for a in actions[:len(dates)]],
            "num_stocks": [int(x) for x in num_stocks],
        },
        "trade_log": trades_log,
        "trade_summary": {
            "total_traded_amount": float(total_traded_amount),
            "total_realized_profit": float(total_realized_profit),
            "num_trades": len(trades_log),
        },
        "metrics": {"model": model_metrics, "buyhold": bh_metrics},
        "initial_balance": balance,
    }
