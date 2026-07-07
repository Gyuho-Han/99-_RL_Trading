"""학습 + 백테스트 엔진.

흐름: KIS OHLCV 조회 -> 피처 생성 -> 학습/테스트 구간 분할 ->
선택한 알고리즘(DQN/A2C/PPO)으로 학습 -> 테스트 구간 백테스트 ->
수익률·Sharpe·MDD 등 지표를 단순보유(Buy&Hold)와 함께 계산.
"""
import os
import sys
import math
import collections
import tempfile
import datetime as dt
from typing import Callable, Dict, Optional

import numpy as np

# 벤더링한 quantylab/rltrader 패키지 경로 등록
_VENDOR = os.path.join(os.path.dirname(__file__), "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
os.environ.setdefault("RLTRADER_BACKEND", "pytorch")

from quantylab.rltrader.learners import (  # noqa: E402
    DQNLearner, DQNPlusLearner, A2CLearner, PPOLearner,
)
from quantylab.rltrader.agent import Agent  # noqa: E402
from quantylab.rltrader.environment import Environment  # noqa: E402

from . import kis_client, features  # noqa: E402
from .metrics import metrics_from_pv, restore_trades, extended_metrics  # noqa: E402

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
                  output_path, reuse_models, dqn_options=None):
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
        # DQN 개선 기법(TD·Replay·Target·Double·Multi-step·PER·Dueling) 체크박스 옵션.
        # 옵션이 모두 꺼져 있으면 기존 quantylab DQN 과 동일하게 동작한다.
        return DQNPlusLearner(rl_method="dqn", value_network_path=vpath,
                              dqn_options=dqn_options, **common)
    if algo == "a2c":
        return A2CLearner(rl_method="a2c", value_network_path=vpath,
                          policy_network_path=ppath, **common)
    if algo == "ppo":
        return PPOLearner(rl_method="ppo", value_network_path=vpath,
                          policy_network_path=ppath, **common)
    raise ValueError(f"지원하지 않는 알고리즘: {algo}")


def _sharpe(pvs):
    """일별 PV 곡선 -> 연환산 Sharpe ratio (rf=0). 변동 없으면 0."""
    a = np.asarray(pvs, dtype=float)
    if len(a) < 3:
        return 0.0
    rets = np.diff(a) / a[:-1]
    sd = float(rets.std())
    if sd < 1e-12:
        return 0.0
    return float(rets.mean() / sd * math.sqrt(TRADING_DAYS))


def _turbulence_series(chart_df, window=252, min_hist=60):
    """단일 종목용 turbulence 지수 {date: value}.

    Yang et al.(ICAIF 2020) 식(3)의 마할라노비스 거리 turbulence 를 단일 자산에
    맞게 1차원으로 축약한 것: 최근 window 일 수익률 분포 대비 당일 수익률의
    표준화 제곱 z^2 = ((r_t-μ)/σ)^2 (1차원 마할라노비스 거리). 과거 데이터만
    사용하므로 룩어헤드가 없다.
    """
    closes = chart_df["close"].astype(float).values
    dates = chart_df["date"].tolist()
    if len(closes) < min_hist + 2:
        return {}
    rets = np.diff(closes) / closes[:-1]
    turb = {}
    for i in range(len(rets)):
        hist = rets[max(0, i - window):i]
        if len(hist) < min_hist:
            continue
        mu, sd = float(hist.mean()), float(hist.std())
        if sd < 1e-12:
            continue
        z = (rets[i] - mu) / sd
        turb[dates[i + 1]] = float(z * z)
    return turb


def _force_liquidate(agent):
    """turbulence 초과 시 전량 매도 + 신규 매수 중단 (Yang et al. 2020 위험회피 규칙)."""
    price = agent.environment.get_price()
    n = agent.num_stocks
    if n > 0:
        agent.balance += price * (1 - (agent.TRADING_TAX + agent.TRADING_CHARGE)) * n
        agent.num_stocks = 0
        agent.avg_buy_price = 0
        agent.num_sell += 1
    else:
        agent.num_hold += 1
    agent.portfolio_value = agent.balance + price * agent.num_stocks
    agent.profitloss = agent.portfolio_value / agent.initial_balance - 1


def _member_decision(tester, sample_seq, agent):
    """멤버 모델의 탐욕(ε=0) 행동 제안. agent 는 액션 마스킹 기준 상태 제공."""
    pred_value = None
    pred_policy = None
    if tester.value_network is not None:
        pred_value = tester.value_network.predict(sample_seq)
    if tester.policy_network is not None:
        pred_policy = tester.policy_network.predict(sample_seq)
    a, conf, _ = agent.decide_action(pred_value, pred_policy, 0.0)
    return int(a), float(conf)


def _sharpe_ensemble_backtest(testers, chart_data, training_data, balance,
                              min_tp, max_tp, leader, window,
                              turb=None, turb_threshold=None):
    """Yang et al.(ICAIF 2020) 앙상블: 최근 Sharpe 최고 멤버가 실제 포트폴리오 운용.

    - 모든 멤버는 자기만의 그림자(shadow) 포트폴리오를 병렬로 운용한다.
    - window 거래일마다 직전 window 구간의 그림자 Sharpe 가 가장 높은 멤버를
      리더로 재선정한다 (논문의 '분기별 검증 Sharpe 로 에이전트 선택'을
      단일 학습·롤링 재선정 형태로 적응).
    - 리더의 행동은 실제 포트폴리오 보유 상태 기준으로 다시 판단해 체결한다.
    - turbulence > threshold 인 날은 전 멤버·실제 포트폴리오 모두 전량 매도
      후 매수 중단, 지수가 내려오면 재개 (논문 식(3), (10)).
    반환: (real_agent, pvs, actions, num_stocks, leader_log)
    """
    m = len(testers)

    def make_pair():
        env = Environment(chart_data)
        agent = Agent(env, balance, min_tp, max_tp)
        env.reset()
        agent.reset()
        return env, agent

    env_r, agent_r = make_pair()
    shadows = [make_pair() for _ in range(m)]
    queues_s = [collections.deque(maxlen=ns) for _, ns in testers]  # 그림자 상태 기준
    queues_r = [collections.deque(maxlen=ns) for _, ns in testers]  # 실제 상태 기준
    max_steps = max(ns for _, ns in testers)
    dates = chart_data["date"].tolist()
    shadow_pvs = [[] for _ in range(m)]
    pvs, acts, stocks, leader_log = [], [], [], []
    since_select = 0

    for i in range(len(training_data)):
        if env_r.observe() is None:
            break
        for env_s, _ in shadows:
            env_s.observe()
        feats = training_data.iloc[i].tolist()
        for k in range(m):
            queues_s[k].append(feats + list(shadows[k][1].get_states()))
            queues_r[k].append(feats + list(agent_r.get_states()))
        if i < max_steps - 1:
            continue

        turbulent = (turb is not None and turb_threshold is not None
                     and turb.get(dates[i], 0.0) > turb_threshold)

        # 그림자 포트폴리오 진행 (멤버별 독립 운용 성과 추적)
        for k in range(m):
            agent_s = shadows[k][1]
            if turbulent:
                _force_liquidate(agent_s)
            else:
                a, c = _member_decision(testers[k][0], list(queues_s[k]), agent_s)
                agent_s.act(a, c)
            shadow_pvs[k].append(agent_s.portfolio_value)

        # 실제 포트폴리오: 리더 모델이 실제 보유 상태 기준으로 행동
        if turbulent:
            _force_liquidate(agent_r)
            act_r = Agent.ACTION_SELL
        else:
            act_r, conf_r = _member_decision(testers[leader][0], list(queues_r[leader]), agent_r)
            agent_r.act(act_r, conf_r)
        pvs.append(agent_r.portfolio_value)
        acts.append(act_r)
        stocks.append(agent_r.num_stocks)
        leader_log.append(leader)

        # window 마다 직전 구간 그림자 Sharpe 로 리더 재선정
        since_select += 1
        if since_select >= window:
            def _score(k):
                hist = shadow_pvs[k][-window:]
                ret = hist[-1] / hist[0] - 1 if hist[0] else 0.0
                return (_sharpe(hist), ret)
            leader = max(range(m), key=_score)
            since_select = 0

    return agent_r, pvs, acts, stocks, leader_log


def _ensemble_backtest(testers, chart_data, training_data, balance, min_tp, max_tp):
    """멤버 모델들의 다수결 앙상블 백테스트 (멤버 1개면 단일 모델 평가와 동일).

    각 스텝에서 멤버별로 액션 마스킹된 탐욕(argmax, ε=0) 행동과 신뢰도를 구하고,
    다수결로 최종 행동을 정한다(동률이면 신뢰도 합이 큰 행동). 체결은 공유 Agent
    하나로 수행하므로 잔고·보유주식 등 포트폴리오 상태는 모든 멤버가 공유한다.
    반환: (agent, pv 리스트, 행동 리스트, 보유주식수 리스트)
    """
    env = Environment(chart_data)
    agent = Agent(env, balance, min_tp, max_tp)
    env.reset()
    agent.reset()
    queues = [collections.deque(maxlen=ns) for _, ns in testers]
    max_steps = max(ns for _, ns in testers)
    pvs, acts, stocks = [], [], []
    for i in range(len(training_data)):
        if env.observe() is None:
            break
        sample = training_data.iloc[i].tolist() + list(agent.get_states())
        for q in queues:
            q.append(sample)
        if i < max_steps - 1:
            continue  # 모든 멤버의 시퀀스가 채워질 때까지 행동 없음
        votes = []
        for (tester, _ns), q in zip(testers, queues):
            pred_value = None
            pred_policy = None
            if tester.value_network is not None:
                pred_value = tester.value_network.predict(list(q))
            if tester.policy_network is not None:
                pred_policy = tester.policy_network.predict(list(q))
            a, conf, _ = agent.decide_action(pred_value, pred_policy, 0.0)
            votes.append((int(a), float(conf)))
        # 다수결 집계: (득표수, 신뢰도 합) 순으로 최종 행동 선택
        tally = {}
        for a, c in votes:
            n_, s_ = tally.get(a, (0, 0.0))
            tally[a] = (n_ + 1, s_ + c)
        action = max(tally.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
        n_win, conf_sum = tally[action]
        agent.act(action, conf_sum / n_win)
        pvs.append(agent.portfolio_value)
        acts.append(action)
        stocks.append(agent.num_stocks)
    return agent, pvs, acts, stocks


def _ensemble_info(method, members, algos, val_sharpes, init_leader,
                   window, turb_threshold, leader_log, dates):
    """결과 JSON 용 앙상블 메타데이터 구성."""
    info = {
        "method": "sharpe_selection" if method == "sharpe" else "majority_vote",
        "members": [{"algorithm": m["algorithm"], "net": m["net"]} for m in members],
    }
    if method == "sharpe":
        info.update({
            "paper": "Yang et al., \"Deep Reinforcement Learning for Automated "
                     "Stock Trading: An Ensemble Strategy\", ICAIF 2020",
            "validation_sharpe": dict(zip(algos, val_sharpes or [])),
            "initial_leader": algos[init_leader] if init_leader is not None else None,
            "reselect_window": window,
            "turbulence_threshold": turb_threshold,
        })
        # 리더 교체 이력: [{date, algorithm}] (변경 시점만 기록)
        segs = []
        if leader_log:
            prev = None
            for d_, k in zip(dates, leader_log):
                if k != prev:
                    segs.append({"date": d_, "algorithm": algos[k]})
                    prev = k
        info["leader_timeline"] = segs
    return info


def run_experiment(params: Dict, progress_callback: Optional[Callable] = None) -> Dict:
    """단일 실험(학습+백테스트) 실행. params 키는 main.py 의 요청 스키마 참고."""
    code = str(params["stock_code"]).zfill(6)

    # 알고리즘 선택: 'algorithms'(복수 선택 → 다수결 앙상블) 우선, 없으면 단일 'algorithm'.
    algos = params.get("algorithms") or [params.get("algorithm")]
    algos = [a for a in dict.fromkeys(algos) if a]
    if not algos:
        raise ValueError("알고리즘을 1개 이상 선택해주세요.")
    unknown = [a for a in algos if a not in ALGORITHMS]
    if unknown:
        raise ValueError(f"지원하지 않는 알고리즘: {unknown}")
    is_ensemble = len(algos) > 1
    algo = algos[0]

    # 재현성: seed 지정 시 numpy/torch 난수를 고정한다. 학습은 무작위 초기화·탐험에
    # 크게 의존하므로 seed 가 없으면 같은 설정이라도 실행마다 결과가 달라진다.
    seed = params.get("seed")
    if seed is not None:
        import random as _random
        _random.seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
        try:
            import torch
            torch.manual_seed(int(seed))
        except ImportError:
            pass
    # 멤버 구성: 단일 실행은 기존처럼 net/num_steps 파라미터를 존중하고,
    # 앙상블은 알고리즘별 기본 네트워크(dqn→dnn, a2c/ppo→lstm)를 사용한다.
    members = []
    for a in algos:
        if is_ensemble:
            net_i = ALGORITHMS[a]["net_default"]
            ns_i = 1 if net_i == "dnn" else 5
        else:
            net_i = params.get("net") or ALGORITHMS[a]["net_default"]
            ns_i = int(params.get("num_steps") or (1 if net_i == "dnn" else 5))
        members.append({"algorithm": a, "net": net_i, "num_steps": ns_i})
    net = "+".join(m["net"] for m in members)
    num_steps = max(m["num_steps"] for m in members)
    lr = float(params.get("lr", 0.0005))
    discount_factor = float(params.get("discount_factor", 0.9))
    num_epoches = int(params.get("num_epoches", 100))
    balance = int(params.get("balance", 10_000_000))
    # 1회 매매 금액: 에이전트는 매 매수/매도 시 신뢰도(confidence)에 비례해
    # min_tp + confidence·(max_tp - min_tp) 원어치만 주문한다(quantylab 설계).
    # max 는 상한을 두지 않고 잔고 전액(balance)으로 고정한다 — 신뢰도가 높을수록
    # 잔고의 더 큰 비중을 매수한다. min 은 UI 고급 설정에서 조절 가능.
    min_tp = int(params.get("min_trading_price", 100_000))
    max_tp = int(params.get("max_trading_price") or balance)
    max_tp = max(max_tp, min_tp)

    # DQN 개선 기법 옵션 (dqn 알고리즘에서만 의미 있음). 의존성 정규화 후 사용.
    dqn_options = None
    if "dqn" in algos:
        dqn_options = DQNPlusLearner.normalize_options(params.get("dqn_options"))

    # state 에 포함할 지표 선택 (없으면 전체). 검증 후 정렬된 컬럼 리스트.
    feature_cols = features.resolve_feature_columns(params.get("features"))

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

    # training_data 는 순수 피처만 (date 제외). 선택된 지표만 state 로 사용.
    td_tr = feat_tr[feature_cols]
    td_te = feat_te[feature_cols]
    cd_tr = chart_tr[features.CHART_COLUMNS]
    cd_te = chart_te[features.CHART_COLUMNS]

    # ---- 앙상블 설정 ----
    # 기본 'sharpe': Yang et al.(ICAIF 2020) — 학습 구간 끝을 검증 구간으로 분리해
    # 검증 Sharpe 최고 멤버로 시작하고, 테스트 중 window 일마다 최근 그림자 Sharpe
    # 최고 멤버로 교체. turbulence 초과 시 전량 매도(위험회피). 'vote' 는 기존 다수결.
    ensemble_method = (str(params.get("ensemble_method") or "sharpe")
                       if is_ensemble else None)
    ensemble_window = int(params.get("ensemble_window") or 21)
    use_turbulence = bool(params.get("use_turbulence", True))
    cd_fit, td_fit = cd_tr, td_tr
    cd_val = td_val = None
    turb_map = None
    turb_threshold = None
    if is_ensemble and ensemble_method == "sharpe":
        val_days = min(63, max(21, len(cd_tr) // 5))
        if len(cd_tr) < val_days + num_steps + 40:
            raise ValueError(
                "학습 구간이 너무 짧아 앙상블 검증 구간(Sharpe 선택용)을 분리할 수 "
                "없습니다. 학습 기간을 늘리거나 다수결(vote) 방식을 사용해주세요.")
        # 검증 구간은 학습에서 제외(out-of-sample 선택, 논문 Step 2)
        cd_fit = cd_tr.iloc[:-val_days].reset_index(drop=True)
        td_fit = td_tr.iloc[:-val_days].reset_index(drop=True)
        tail = val_days + num_steps - 1
        cd_val = cd_tr.iloc[-tail:].reset_index(drop=True)
        td_val = td_tr.iloc[-tail:].reset_index(drop=True)
        if use_turbulence:
            turb_map = _turbulence_series(chart_all)
            _tr_vals = [v for d_, v in turb_map.items() if train_start <= d_ <= train_end]
            if len(_tr_vals) >= 60:
                # 임계값: 학습 구간 turbulence 의 95 분위 (과거 데이터만 사용)
                turb_threshold = float(np.quantile(_tr_vals, 0.95))
            else:
                turb_map = None

    output_path = tempfile.mkdtemp(prefix=f"rl_{code}_{algo}_")

    # ---- 학습 + 백테스트 ----
    total_epochs_all = num_epoches * len(members)

    def _train_and_backtest(attempt):
        total_epochs = total_epochs_all
        testers = []
        for mi, m in enumerate(members):
            mdir = os.path.join(output_path, f"{m['algorithm']}_{attempt}")
            os.makedirs(mdir, exist_ok=True)
            opts_i = dqn_options if m["algorithm"] == "dqn" else None

            def _train_cb(epoch, total, pv, pl, _base=mi * num_epoches):
                if progress_callback:
                    progress_callback("training", _base + epoch, total_epochs, pv, pl)

            learner = _make_learner(
                m["algorithm"], code, cd_fit, td_fit, net=m["net"], num_steps=m["num_steps"],
                lr=lr, discount_factor=discount_factor, num_epoches=num_epoches,
                balance=balance, min_trading_price=min_tp, max_trading_price=max_tp,
                start_epsilon=1.0, output_path=mdir, reuse_models=False, dqn_options=opts_i,
            )
            learner.visualize_enabled = False
            learner.run(learning=True, progress_callback=_train_cb)
            learner.save_models()

            # 학습된 모델을 로드한 평가용 인스턴스 (run 은 호출하지 않고 신경망만 사용)
            tester = _make_learner(
                m["algorithm"], code, cd_te, td_te, net=m["net"], num_steps=m["num_steps"],
                lr=lr, discount_factor=discount_factor, num_epoches=1,
                balance=balance, min_trading_price=min_tp, max_trading_price=max_tp,
                start_epsilon=0.0, output_path=mdir, reuse_models=True, dqn_options=opts_i,
            )
            tester.visualize_enabled = False
            testers.append((tester, m["num_steps"]))

        # ---- 백테스트 (테스트 구간, 탐험 0) ----
        if progress_callback:
            progress_callback("backtesting", total_epochs, total_epochs, None, None)
        val_sharpes = None
        init_leader = None
        leader_log = None
        if is_ensemble and ensemble_method == "sharpe":
            # (1) 검증 구간(학습 제외분)에서 멤버별 Sharpe -> 초기 리더 선정 (논문 Step 2)
            val_sharpes = []
            for k in range(len(testers)):
                _, vpvs, _, _, _ = _sharpe_ensemble_backtest(
                    [testers[k]], cd_val, td_val, balance, min_tp, max_tp,
                    leader=0, window=10 ** 9,
                    turb=turb_map, turb_threshold=turb_threshold)
                val_sharpes.append(round(_sharpe(vpvs), 4))
            init_leader = int(np.argmax(val_sharpes))
            # (2) 테스트: 리더가 운용 + window 일마다 최근 그림자 Sharpe 로 재선정 (Step 3)
            agent, model_pv, actions, num_stocks_all, leader_log = _sharpe_ensemble_backtest(
                testers, cd_te, td_te, balance, min_tp, max_tp,
                leader=init_leader, window=ensemble_window,
                turb=turb_map, turb_threshold=turb_threshold)
        else:
            # 단일 모델(멤버 1개) 또는 다수결(vote) 앙상블
            agent, model_pv, actions, num_stocks_all = _ensemble_backtest(
                testers, cd_te, td_te, balance, min_tp, max_tp)
        return agent, model_pv, actions, num_stocks_all, leader_log, val_sharpes, init_leader

    # 학습이 '전부 관망'으로 붕괴해 백테스트 매수가 0회면, 다른 무작위 초기화로
    # 재학습한다(최대 2회 추가 시도). 장기 학습 구간에서는 MC 회귀 특성상 관망
    # 붕괴 확률이 높아, 사용자가 '매수를 아예 안 한다'고 겪는 문제를 방지한다.
    collapse_retrains = 0
    for attempt in range(3):
        if attempt:
            collapse_retrains = attempt
            if seed is not None:
                _s = (int(seed) + 7919 * attempt) % (2 ** 32)
                np.random.seed(_s)
                try:
                    import torch
                    torch.manual_seed(_s)
                except ImportError:
                    pass
        (agent, model_pv, actions, num_stocks_all,
         leader_log, val_sharpes, init_leader) = _train_and_backtest(attempt)
        if agent.num_buy > 0 or attempt == 2:
            break

    offset = num_steps - 1                        # 앞쪽 max(num_steps)-1 스텝은 행동 없음
    dates = cd_te["date"].tolist()[offset:offset + len(model_pv)]
    closes = cd_te["close"].tolist()[offset:offset + len(model_pv)]

    # 단순보유(Buy&Hold) 곡선: 동일 구간 종가 기준
    if closes:
        base = closes[0]
        bh_pv = [balance * (c / base) for c in closes]
    else:
        bh_pv = []

    # ---- 체결 내역 복원 + 성과지표 (공용 metrics 모듈) ----
    num_stocks = num_stocks_all[:len(dates)]
    trades_log, trade_summary = restore_trades(
        dates, closes, num_stocks,
        charge=agent.TRADING_CHARGE, tax=agent.TRADING_TAX)

    model_metrics = metrics_from_pv(np.array(model_pv))
    model_metrics.update(extended_metrics(trades_log, dates, model_pv))
    bh_metrics = metrics_from_pv(np.array(bh_pv))
    bh_metrics.update(extended_metrics([], dates, bh_pv))

    return {
        "stock_code": code,
        "algorithm": "+".join(algos),
        "net": net,
        "num_steps": num_steps,
        "num_epoches": num_epoches,
        "features": feature_cols,
        "train_period": [train_start, train_end],
        "test_period": [test_start, test_end],
        "n_train": int(len(cd_tr)),
        "n_test": int(len(dates)),
        "trades": {"buy": int(agent.num_buy),
                   "sell": int(agent.num_sell),
                   "hold": int(agent.num_hold)},
        # 앙상블 정보 (단일 실행이면 None)
        "ensemble": (_ensemble_info(
            ensemble_method, members, algos, val_sharpes, init_leader,
            ensemble_window, turb_threshold, leader_log, dates)
            if is_ensemble else None),
        "series": {
            "dates": dates,
            "close": closes,
            "model_pv": [float(x) for x in model_pv],
            "buyhold_pv": [float(x) for x in bh_pv],
            "actions": [int(a) for a in actions[:len(dates)]],
            "num_stocks": [int(x) for x in num_stocks],
        },
        "trade_log": trades_log,
        "trade_summary": trade_summary,
        "metrics": {"model": model_metrics, "buyhold": bh_metrics},
        "initial_balance": balance,
        "collapse_retrains": collapse_retrains,  # 관망 붕괴로 재학습한 횟수
        "dqn_options": dqn_options,   # 적용된 DQN 개선 기법 (dqn 외 알고리즘은 None)
    }
