"""hanium 강화학습 엔진 — 99- 웹앱에 hanium 알고리즘/네트워크를 연결한다.

기존 quantylab 엔진(engine.py)과 **동일한 결과 dict 형태**를 반환하므로,
잡 매니저(jobs.py)와 프론트엔드는 엔진만 바꿔 끼우면 그대로 동작한다.

흐름:
  KIS OHLCV 조회(engine 과 동일) -> features.build_features -> 학습/테스트 분할
  -> hanium TradingEnv(거래세·수수료·액션마스킹 이식) + NetworkRegistry + AgentRegistry
  -> 에피소드 단위 학습(gymnasium step 루프) -> 테스트 구간 백테스트(탐험 0)
  -> 수익률·Sharpe·MDD 를 단순보유(Buy&Hold)와 함께 계산.

quantylab 엔진과의 차이:
  - 학습 단위: 에포크 배치 회귀(quantylab) → step 단위 리플레이/롤아웃(hanium)
  - 알고리즘: DQN/A2C/PPO 3종 → 10종(분포형 IQN·Decision Transformer·앙상블 포함)
  - 네트워크: DNN/LSTM/CNN 3종 → 8종(Mamba·PatchTST·iTransformer·TFT·xLSTM 포함)
"""
import importlib
import json
import re
from typing import Callable, Dict, Optional

import numpy as np

from . import kis_client, features, config
from .metrics import metrics_from_pv, restore_trades, extended_metrics

TRADING_DAYS = 252

# ----------------------------------------------------------------------
# 알고리즘 / 네트워크 메타데이터 (프론트 선택 UI 용)
# ----------------------------------------------------------------------
HANIUM_ALGORITHMS = {
    "dqn":   {"label": "DQN",     "desc": "가치기반 베이스라인. ε-greedy + 리플레이 + 타깃망"},
    "a2c":   {"label": "A2C",     "desc": "Advantage Actor-Critic. 롤아웃 기반 동시 학습"},
    "ppo":   {"label": "PPO",     "desc": "정석 PPO. GAE + 클리핑 + 엔트로피 보너스"},
    "ddpg":  {"label": "DDPG",    "desc": "결정적 정책경사(이산 적용). TD3/SAC 베이스라인"},
    "td3":   {"label": "TD3",     "desc": "쌍둥이 Critic + 지연 업데이트로 과대평가 억제"},
    "sac":   {"label": "SAC",     "desc": "엔트로피 최대화. 하이퍼파라미터에 강건"},
    "rainbow": {"label": "Rainbow", "desc": "Double+PER+Dueling+Multi-step 결합 DQN"},
    "iqn":   {"label": "IQN",     "desc": "분포형 RL. 수익 분포 학습으로 리스크(CVaR) 관리"},
    "decision_transformer": {"label": "Decision Transformer",
                             "desc": "오프라인 RL. (목표수익,상태,행동) 시퀀스를 GPT처럼 예측"},
    "ensemble": {"label": "Ensemble", "desc": "다중 Q-헤드 중 최고 성과 헤드 선택"},
}

HANIUM_NETWORKS = {
    "dnn":  {"label": "DNN",  "sub": "완전연결"},
    "lstm": {"label": "LSTM", "sub": "시계열 순환"},
    "cnn":  {"label": "CNN",  "sub": "1D 합성곱"},
    "mamba": {"label": "Mamba", "sub": "선택적 SSM(선형)"},
    "patchtst": {"label": "PatchTST", "sub": "패치 Transformer"},
    "itransformer": {"label": "iTransformer", "sub": "변수 토큰 Attention"},
    "tft":  {"label": "TFT",  "sub": "변수선택+LSTM+Attention"},
    "xlstm": {"label": "xLSTM", "sub": "지수게이트+행렬메모리"},
}

# 등록명 != 모듈명 인 경우 매핑 (rainbow → rainbow_dqn.py)
_AGENT_MODULES = {
    "dqn": "dqn", "a2c": "a2c", "ppo": "ppo", "ddpg": "ddpg", "td3": "td3",
    "sac": "sac", "rainbow": "rainbow_dqn", "iqn": "iqn",
    "decision_transformer": "decision_transformer", "ensemble": "ensemble",
}
_NETWORK_MODULES = {k: k for k in HANIUM_NETWORKS}


def _ensure_registered(algo: str, net: str):
    """선택된 에이전트/네트워크 모듈을 import 해 레지스트리에 등록한다."""
    if algo not in _AGENT_MODULES:
        raise ValueError(f"지원하지 않는 알고리즘: {algo}")
    if net not in _NETWORK_MODULES:
        raise ValueError(f"지원하지 않는 네트워크: {net}")
    importlib.import_module(f".hanium.agents.{_AGENT_MODULES[algo]}", __package__)
    importlib.import_module(f".hanium.networks.{_NETWORK_MODULES[net]}", __package__)


def _net_params_for(net: str, window_size: int) -> Dict:
    """네트워크별 기본 하이퍼파라미터(시퀀스 길이 등)를 구성한다."""
    p = {"seq_len": window_size}
    if net == "patchtst":
        # 패치 길이가 윈도우보다 크면 안 되므로 안전하게 조정
        patch = 5 if window_size >= 5 else max(1, window_size // 2)
        p.update(patch_len=patch, stride=patch)
    return p


def list_saved_models() -> list:
    """저장된 hanium 모델 목록 (metadata.json 기준, 최신순)."""
    out = []
    root = config.HANIUM_MODELS_DIR
    if not root.is_dir():
        return out
    for d in root.iterdir():
        meta_p = d / "metadata.json"
        if d.is_dir() and meta_p.exists():
            try:
                with open(meta_p, encoding="utf-8") as f:
                    meta = json.load(f)
                meta["model_id"] = d.name
                out.append(meta)
            except (json.JSONDecodeError, OSError):
                continue
    out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return out


def _load_model_meta(model_id: str) -> Dict:
    """저장 모델 metadata 로드. model_id 는 디렉터리명 (경로 조작 방지 검증)."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", model_id or ""):
        raise ValueError(f"잘못된 모델 id: {model_id}")
    meta_p = config.HANIUM_MODELS_DIR / model_id / "metadata.json"
    if not meta_p.exists():
        raise ValueError(f"저장된 모델을 찾을 수 없습니다: {model_id}")
    with open(meta_p, encoding="utf-8") as f:
        return json.load(f)


def run_experiment(params: Dict, progress_callback: Optional[Callable] = None) -> Dict:
    """단일 실험(학습+백테스트) 실행. 결과 dict 는 engine.run_experiment 와 동일 형태.

    saved_model 파라미터가 주어지면 학습을 건너뛰고 저장된 체크포인트를 로드해
    백테스트만 수행한다(알고리즘·네트워크·window·지표는 저장 시점 설정으로 강제).
    """
    import torch  # 지연 임포트(서버 기동 부담 완화)
    from .hanium.agents.registry import AgentRegistry
    from .hanium.networks.registry import NetworkRegistry
    from .hanium.env.trading_env import TradingEnv

    code = str(params["stock_code"]).zfill(6)
    algo = params["algorithm"]
    net = params.get("net") or "lstm"

    # ---- 저장 모델 재사용: 저장 시점 설정으로 덮어쓰기 (입력 차원 일치 필수) ----
    saved_model = (params.get("saved_model") or "").strip() or None
    saved_meta = None
    if saved_model:
        saved_meta = _load_model_meta(saved_model)
        algo = saved_meta["algorithm"]
        net = saved_meta["net"]
        params = dict(params)
        params["features"] = saved_meta.get("features")

    _ensure_registered(algo, net)

    # 프론트에서 빈 값(null)이 오면 .get(key, default) 가 default 대신 None 을 반환하므로
    # `or default` 로 None/0/빈값을 모두 안전하게 기본값 처리한다.
    episodes = int(params.get("episodes") or params.get("num_epoches") or 50)
    window_size = int(params.get("window_size") or 20)
    lr = float(params.get("lr") or 3e-4)
    gamma = float(params.get("discount_factor") or params.get("gamma") or 0.99)
    balance = int(params.get("balance") or 10_000_000)
    trade_ratio = float(params.get("trade_ratio") if params.get("trade_ratio") is not None else 1.0)
    batch_size = int(params.get("batch_size") or 64)
    # 보상 셰이핑 옵션 (실현수익 보너스·손실매도/과매매/MDD 패널티, 기본 0 = 기존과 동일)
    reward_options = dict(params.get("reward_options") or {})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if saved_meta is not None:
        window_size = int(saved_meta["window_size"])
        trade_ratio = float(saved_meta.get("trade_ratio", trade_ratio))

    import datetime as dt
    train_start = params["train_start"].replace("-", "")
    train_end = params["train_end"].replace("-", "")
    test_start = params["test_start"].replace("-", "")
    test_end = params["test_end"].replace("-", "")

    # 지표 워밍업 포함해 넉넉히 조회 (train_start 이전 250일)
    fetch_from = (dt.datetime.strptime(train_start, "%Y%m%d").date()
                  - dt.timedelta(days=250)).strftime("%Y%m%d")
    fetch_to = max(train_end, test_end)

    if progress_callback:
        progress_callback("fetching", 0, episodes, None, None)
    rows = kis_client.fetch_ohlcv(code, fetch_from, fetch_to)
    if len(rows) < 130:
        raise ValueError(
            f"조회된 일봉이 {len(rows)}건으로 부족합니다(지표 워밍업 130일 필요). "
            "기간을 늘리거나 종목코드를 확인하세요."
        )

    chart_all, train_all = features.build_features(rows)
    chart_tr, feat_tr = features.slice_by_date(chart_all, train_all, train_start, train_end)
    chart_te, feat_te = features.slice_by_date(chart_all, train_all, test_start, test_end)

    feat_cols = features.resolve_feature_columns(params.get("features"))
    if len(chart_tr) < window_size + 5:
        raise ValueError("학습 구간 데이터가 너무 적습니다. 학습 기간을 늘리거나 window_size 를 줄여주세요.")
    if len(chart_te) < window_size + 2:
        raise ValueError("테스트 구간 데이터가 너무 적습니다. 테스트 기간을 늘려주세요.")

    df_tr = feat_tr[feat_cols].astype(np.float32).reset_index(drop=True)
    df_te = feat_te[feat_cols].astype(np.float32).reset_index(drop=True)
    price_tr = chart_tr["close"].astype(float).values
    price_te = chart_te["close"].astype(float).values

    # ---- 환경 ----
    # 보상 셰이핑은 학습 환경에만 적용 (테스트는 탐욕 정책이라 보상이 결과에 무관)
    env_tr = TradingEnv(df=df_tr, initial_balance=balance, commission=0.00015,
                        window_size=window_size, trade_ratio=trade_ratio,
                        raw_prices=price_tr, trading_tax=0.0025,
                        reward_options=reward_options)
    env_te = TradingEnv(df=df_te, initial_balance=balance, commission=0.00015,
                        window_size=window_size, trade_ratio=trade_ratio,
                        raw_prices=price_te, trading_tax=0.0025)

    obs_shape = env_tr.observation_space.shape  # (window, n_features+3)
    flat_input_dim = obs_shape[0] * obs_shape[1]

    # ---- 네트워크 + 에이전트 ----
    net_obj = NetworkRegistry.create(net, input_dim=flat_input_dim,
                                     **_net_params_for(net, window_size))
    agent = AgentRegistry.create(
        algo, network=net_obj, device=device, lr=lr, gamma=gamma,
        batch_size=batch_size, num_actions=env_tr.action_space.n,
    )

    # ---- 학습 (저장 모델 재사용 시 건너뜀) ----
    if saved_meta is None:
        for ep in range(1, episodes + 1):
            state, _ = env_tr.reset()
            done = False
            while not done:
                action = agent.select_action(state, explore=True)
                next_state, reward, terminated, truncated, info = env_tr.step(action)
                agent.store_transition(state, action, reward, next_state, terminated)
                agent.train_step()
                state = next_state
                done = terminated or truncated
            agent.on_episode_end(ep)
            if progress_callback:
                progress_callback("training", ep, episodes,
                                  float(info["total_asset"]),
                                  float(info["profit_pct"]))
        # 학습 완료 모델 체크포인트 저장 (metadata 는 백테스트 지표 계산 후 저장)
        model_id = f"{code}_{algo}_{net}_{dt.datetime.now():%Y%m%d_%H%M%S}"
        agent.save(config.HANIUM_MODELS_DIR / model_id / "checkpoint.pt")
    else:
        model_id = saved_model
        agent.load(config.HANIUM_MODELS_DIR / saved_model / "checkpoint.pt")

    # ---- 백테스트 (탐험 0) ----
    if progress_callback:
        progress_callback("backtesting", episodes, episodes, None, None)
    state, _ = env_te.reset()
    done = False
    model_pv, actions, num_stocks = [], [], []
    while not done:
        action = agent.select_action(state, explore=False)
        state, reward, terminated, truncated, info = env_te.step(action)
        model_pv.append(float(info["total_asset"]))
        actions.append(int(action))
        num_stocks.append(int(info["shares"]))
        done = terminated or truncated

    T = len(model_pv)
    offset = window_size
    dates = chart_te["date"].tolist()[offset:offset + T]
    closes = [float(x) for x in price_te[offset:offset + T]]
    # 길이 정합 보정
    n = min(len(dates), len(closes), T)
    dates, closes = dates[:n], closes[:n]
    model_pv, actions, num_stocks = model_pv[:n], actions[:n], num_stocks[:n]

    # ---- 단순보유(Buy&Hold) ----
    if closes:
        base = closes[0]
        bh_pv = [balance * (c / base) for c in closes]
    else:
        bh_pv = []

    # ---- 체결 내역 복원 + 성과지표 (공용 metrics 모듈) ----
    trades_log, trade_summary = restore_trades(dates, closes, num_stocks,
                                               charge=0.00015, tax=0.0025)
    num_buy = sum(1 for t in trades_log if t["side"] == "buy")
    num_sell = sum(1 for t in trades_log if t["side"] == "sell")
    num_hold = n - num_buy - num_sell

    model_metrics = metrics_from_pv(np.array(model_pv))
    model_metrics.update(extended_metrics(trades_log, dates, model_pv))
    bh_metrics = metrics_from_pv(np.array(bh_pv))
    bh_metrics.update(extended_metrics([], dates, bh_pv))

    # ---- 새로 학습한 모델이면 metadata 저장 (저장 모델 목록 API 용) ----
    if saved_meta is None:
        meta = {
            "model_id": model_id, "engine": "hanium", "stock_code": code,
            "algorithm": algo, "net": net, "window_size": window_size,
            "trade_ratio": trade_ratio, "features": feat_cols,
            "episodes": episodes, "lr": lr, "gamma": gamma,
            "train_period": [train_start, train_end],
            "test_period": [test_start, test_end],
            "reward_options": reward_options,
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "metrics": {"model": model_metrics, "buyhold": bh_metrics},
        }
        try:
            with open(config.HANIUM_MODELS_DIR / model_id / "metadata.json",
                      "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except OSError:
            pass  # metadata 저장 실패는 결과 반환을 막지 않음

    return {
        "model_id": model_id,
        "saved_model_used": saved_meta is not None,
        "engine": "hanium",
        "stock_code": code,
        "algorithm": algo,
        "net": net,
        "window_size": window_size,
        "episodes": episodes,
        "num_steps": window_size,           # 프론트 호환(표시용)
        "num_epoches": episodes,            # 프론트 호환(표시용)
        "train_period": [train_start, train_end],
        "test_period": [test_start, test_end],
        "n_train": int(len(chart_tr)),
        "n_test": int(n),
        "trades": {"buy": int(num_buy), "sell": int(num_sell), "hold": int(num_hold)},
        "series": {
            "dates": dates,
            "close": closes,
            "model_pv": [float(x) for x in model_pv],
            "buyhold_pv": [float(x) for x in bh_pv],
            "actions": [int(a) for a in actions],
            "num_stocks": [int(x) for x in num_stocks],
        },
        "trade_log": trades_log,
        "trade_summary": trade_summary,
        "metrics": {"model": model_metrics, "buyhold": bh_metrics},
        "initial_balance": balance,
    }
