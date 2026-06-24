"""주식 매매 환경 — gymnasium 스타일.

상태: 과거 window_size일의 OHLCV + 기술지표 + 포지션 정보
행동: 0=매수, 1=매도, 2=관망 (이산)
보상: 포트폴리오 가치 변화율

[99-_RL_Trading 이식 사항]
  hanium 원본 환경에 99-(quantylab) 프로토타입의 '거래 현실성'을 이식했다.
    1) 거래세(trading_tax, 기본 0.25%): 매도 시 수수료와 함께 차감
    2) 매매 수수료(commission, 기본 0.015%): 매수/매도 양쪽 적용
    3) 액션 마스킹: 보유 0주에서 매도, 현금 부족 시 매수는 자동으로 관망 처리
       (불가능한 행동으로 인한 정책 붕괴 방지). 마스킹 횟수를 info 로 노출.
    4) 부분 매매: trade_ratio 로 매수 시 잔고 대비 투자 비율을 조절(1.0=전량).
"""

from typing import Tuple

import gymnasium as gym
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    # 행동 상수
    BUY = 0
    SELL = 1
    HOLD = 2

    def __init__(self, df: pd.DataFrame, initial_balance: int = 10_000_000,
                 commission: float = 0.00015, window_size: int = 20,
                 trade_ratio: float = 1.0, raw_prices: np.ndarray = None,
                 trading_tax: float = 0.0025):
        """
        Parameters
        ----------
        df : pd.DataFrame
            전처리 완료된 시세 데이터 (정규화된 특성).
        initial_balance : int
            초기 자본금.
        commission : float
            매매 수수료율(매수·매도 양쪽). 기본 0.015%.
        window_size : int
            상태에 포함할 과거 일수.
        trade_ratio : float
            매수 시 잔고 대비 투자 비율 (1.0 = 전량). 0~1.
        raw_prices : np.ndarray, optional
            정규화 전 원본 종가. None이면 df["close"] 사용.
        trading_tax : float
            매도 시 부과되는 거래세(99- 이식). 기본 0.25%.
        """
        super().__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.commission = commission
        self.trading_tax = trading_tax
        self.window_size = window_size
        self.trade_ratio = float(max(0.0, min(1.0, trade_ratio)))

        # 원본 종가 (매매 가격 계산용) — 정규화되지 않은 값
        self.prices = raw_prices if raw_prices is not None else df["close"].values

        # 상태에 사용할 특성 (모든 수치 컬럼)
        self.features = df.select_dtypes(include=[np.number]).values
        self.n_features = self.features.shape[1]

        # 관측 공간: (window_size, n_features + 3)
        #   +3 = [보유비율, 수익률, 잔고비율]
        obs_shape = (window_size, self.n_features + 3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)

        # 상태 변수 (reset에서 초기화)
        self.current_step = 0
        self.balance = 0
        self.shares = 0
        self.total_asset = 0
        self.trades = []
        self.num_masked = 0  # 액션 마스킹(불가능 행동→관망)된 횟수

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """환경을 초기 상태로 리셋한다."""
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares = 0
        self.total_asset = self.initial_balance
        self.trades = []
        self.num_masked = 0
        self._prev_total_asset = self.initial_balance

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """행동을 실행하고 다음 상태·보상을 반환한다."""
        current_price = self.prices[self.current_step]

        # --- 액션 마스킹(99- 이식): 불가능한 행동은 관망으로 강등 ---
        #   매수: 이미 보유 중이거나 현금이 1주값+수수료에 못 미치면 불가
        #   매도: 보유 주식이 없으면 불가
        if action == self.BUY and (self.shares > 0
                or self.balance < current_price * (1 + self.commission)):
            action = self.HOLD
            self.num_masked += 1
        elif action == self.SELL and self.shares <= 0:
            action = self.HOLD
            self.num_masked += 1

        # --- 행동 실행 ---
        if action == self.BUY and self.shares == 0 and self.balance > 0:
            invest_amount = self.balance * self.trade_ratio
            buy_price = current_price * (1 + self.commission)
            self.shares = int(invest_amount // buy_price)
            if self.shares > 0:
                cost = self.shares * buy_price
                self.balance -= cost
                self.trades.append({
                    "step": self.current_step,
                    "action": "buy",
                    "price": current_price,
                    "qty": self.shares,
                })

        elif action == self.SELL and self.shares > 0:
            # 매도 시 수수료 + 거래세(99- 이식) 차감
            sell_price = current_price * (1 - self.commission - self.trading_tax)
            revenue = self.shares * sell_price
            self.balance += revenue
            self.trades.append({
                "step": self.current_step,
                "action": "sell",
                "price": current_price,
                "qty": self.shares,
            })
            self.shares = 0

        # --- 포트폴리오 가치 계산 ---
        self.total_asset = self.balance + self.shares * current_price

        # --- 보상: 포트폴리오 가치 변화율 ---
        reward = (self.total_asset - self._prev_total_asset) / self._prev_total_asset
        self._prev_total_asset = self.total_asset

        # --- 다음 스텝 ---
        self.current_step += 1
        terminated = self.current_step >= len(self.prices) - 1
        truncated = False

        obs = self._get_observation() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """현재 스텝의 관측값을 구성한다."""
        start = self.current_step - self.window_size
        end = self.current_step

        # 시장 데이터: (window_size, n_features)
        market_data = self.features[start:end]

        # 포지션 정보: (window_size, 3)
        current_price = self.prices[self.current_step]
        stock_value = self.shares * current_price

        if self.total_asset > 0:
            holding_ratio = stock_value / self.total_asset
            balance_ratio = self.balance / self.total_asset
        else:
            holding_ratio = 0.0
            balance_ratio = 0.0

        profit_ratio = (self.total_asset - self.initial_balance) / self.initial_balance

        position_info = np.full(
            (self.window_size, 3),
            [holding_ratio, profit_ratio, balance_ratio],
            dtype=np.float32,
        )

        obs = np.concatenate([market_data, position_info], axis=1).astype(np.float32)
        return obs

    def _get_info(self) -> dict:
        """현재 상태 정보를 반환한다."""
        return {
            "balance": self.balance,
            "shares": self.shares,
            "total_asset": self.total_asset,
            "profit": self.total_asset - self.initial_balance,
            "profit_pct": (self.total_asset - self.initial_balance) / self.initial_balance,
            "n_trades": len(self.trades),
            "n_masked": self.num_masked,
        }

    def render(self, mode="human"):
        info = self._get_info()
        print(
            f"Step {self.current_step} | "
            f"잔고: {info['balance']:,.0f} | "
            f"보유: {self.shares}주 | "
            f"총자산: {info['total_asset']:,.0f} | "
            f"수익률: {info['profit_pct']:.2%}"
        )
