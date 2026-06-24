"""hanium 엔진 스모크 테스트 — 모든 (알고리즘 × 네트워크) 조합 검증.

torch 가 설치된 환경에서 합성 데이터로 각 조합이
  네트워크 생성 → 에이전트 생성 → 학습 step → 백테스트(탐험0)
까지 오류 없이 도는지 확인한다.

실행:
    cd backend
    python -m scripts.smoke_hanium            # 전체 조합(80개)
    python -m scripts.smoke_hanium --fast     # 대표 조합만 빠르게

KIS API 없이 동작하도록 합성 OHLCV 를 사용한다.
"""
import argparse
import itertools
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.engine_hanium import (  # noqa: E402
    HANIUM_ALGORITHMS, HANIUM_NETWORKS, _ensure_registered, _net_params_for,
)


def make_env(window=20, n=160, seed=0):
    """합성 피처 + 우상향/노이즈 종가로 TradingEnv 생성."""
    from app.hanium.env.trading_env import TradingEnv
    rng = np.random.default_rng(seed)
    F = 6
    feats = pd.DataFrame(rng.standard_normal((n, F)).astype("float32"),
                         columns=[f"f{i}" for i in range(F)])
    prices = (1000 + np.cumsum(rng.standard_normal(n)) * 5 + np.linspace(0, 200, n))
    prices = np.clip(prices, 100, None)
    return TradingEnv(df=feats, initial_balance=1_000_000, commission=0.00015,
                      window_size=window, raw_prices=prices, trading_tax=0.0025)


def run_combo(algo, net, window=20, train_steps=40):
    import torch  # noqa: F401
    from app.hanium.agents.registry import AgentRegistry
    from app.hanium.networks.registry import NetworkRegistry

    _ensure_registered(algo, net)
    env = make_env(window=window)
    flat = env.observation_space.shape[0] * env.observation_space.shape[1]
    net_obj = NetworkRegistry.create(net, input_dim=flat, **_net_params_for(net, window))
    agent = AgentRegistry.create(algo, network=net_obj, device="cpu", lr=3e-4,
                                 gamma=0.99, batch_size=16,
                                 num_actions=env.action_space.n)

    # 짧은 학습
    state, _ = env.reset()
    for _ in range(train_steps):
        a = agent.select_action(state, explore=True)
        ns, r, term, trunc, _ = env.step(a)
        agent.store_transition(state, a, r, ns, term)
        agent.train_step()
        state = ns
        if term or trunc:
            state, _ = env.reset()
    agent.on_episode_end(1)

    # 백테스트 1 패스 (탐험0)
    state, _ = env.reset()
    done = False
    steps = 0
    while not done:
        a = agent.select_action(state, explore=False)
        state, r, term, trunc, info = env.step(a)
        done = term or trunc
        steps += 1
    return steps, info["total_asset"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="대표 조합만 검증")
    ap.add_argument("--window", type=int, default=20)
    args = ap.parse_args()

    algos = list(HANIUM_ALGORITHMS)
    nets = list(HANIUM_NETWORKS)
    if args.fast:
        algos = ["dqn", "ppo", "sac", "iqn", "rainbow"]
        nets = ["dnn", "lstm", "cnn", "tft", "mamba"]
        combos = list(zip(algos, nets))  # 대각선 5개
    else:
        combos = list(itertools.product(algos, nets))  # 전체 80개

    print(f"검증 조합 수: {len(combos)} (window={args.window})\n")
    ok, fail = 0, []
    for algo, net in combos:
        try:
            steps, asset = run_combo(algo, net, window=args.window)
            print(f"  [OK]   {algo:22s} x {net:13s} → {steps} steps, 자산 {asset:,.0f}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {algo:22s} x {net:13s} → {type(e).__name__}: {e}")
            fail.append((algo, net, traceback.format_exc()))

    print(f"\n결과: {ok} 성공 / {len(fail)} 실패")
    if fail:
        print("\n=== 실패 상세 ===")
        for algo, net, tb in fail:
            print(f"\n--- {algo} x {net} ---\n{tb}")
        sys.exit(1)
    print("ALL COMBOS PASSED ✅")


if __name__ == "__main__":
    main()
