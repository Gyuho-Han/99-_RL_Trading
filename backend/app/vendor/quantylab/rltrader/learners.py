import os
import logging
import abc
import collections
import threading
import time
import json
import numpy as np
from tqdm import tqdm
from quantylab.rltrader.environment import Environment
from quantylab.rltrader.agent import Agent
from quantylab.rltrader.networks import Network, DNN, LSTMNetwork, CNN
from quantylab.rltrader.visualizer import Visualizer
from quantylab.rltrader import utils
from quantylab.rltrader import settings


logger = logging.getLogger(settings.LOGGER_NAME)


class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()

    def __init__(self, rl_method='rl', stock_code=None, 
                chart_data=None, training_data=None,
                min_trading_price=100000, max_trading_price=10000000, 
                net='dnn', num_steps=1, lr=0.0005, 
                discount_factor=0.9, num_epoches=1000,
                balance=100000000, start_epsilon=1,
                min_epsilon=0.1,
                value_network=None, policy_network=None,
                output_path='', reuse_models=True):
        # 인자 확인
        assert min_trading_price > 0
        assert max_trading_price > 0
        assert max_trading_price >= min_trading_price
        assert num_steps > 0
        assert lr > 0
        # 강화학습 설정
        self.rl_method = rl_method
        self.discount_factor = discount_factor
        self.num_epoches = num_epoches
        self.start_epsilon = start_epsilon
        # 탐험 하한(학습 시에만 적용). 기존 quantylab 은 마지막 에포크에서 ε=0 이
        # 되어 학습 배치가 탐욕 행동(대개 관망)으로만 채워졌고, 한 번 Q(관망)이
        # 앞서면 매수/매도 Q 가 갱신되지 못해 '전부 관망' 정책으로 붕괴하는
        # 확률이 높았다. 후반 에포크에도 최소한의 탐험을 유지해 이를 완화한다.
        self.min_epsilon = min_epsilon
        # 환경 설정
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)
        # 에이전트 설정
        self.agent = Agent(self.environment, balance, min_trading_price, max_trading_price)
        # 학습 데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        # 신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        # 가시화 모듈
        self.visualizer = Visualizer()
        # 메모리
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        # 에포크 관련 정보
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
        # 로그 등 출력 경로
        self.output_path = output_path
        # 진행 콜백 및 가시화 토글 (웹 프로토타입용 확장)
        self.progress_callback = None
        self.visualize_enabled = False

    def init_value_network(self, shared_network=None, activation='linear', loss='mse'):
        if self.net == 'dnn':
            self.value_network = DNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.value_network = LSTMNetwork(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.value_network = CNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        if self.reuse_models and os.path.exists(self.value_network_path):
            self.value_network.load_model(model_path=self.value_network_path)

    def init_policy_network(self, shared_network=None, activation='sigmoid', 
                            loss='binary_crossentropy'):
        if self.net == 'dnn':
            self.policy_network = DNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.policy_network = LSTMNetwork(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.policy_network = CNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        if self.reuse_models and os.path.exists(self.policy_network_path):
            self.policy_network.load_model(model_path=self.policy_network_path)

    def reset(self):
        self.sample = None
        self.training_data_idx = -1
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()
        # 가시화 초기화
        self.visualizer.clear([0, len(self.chart_data)])
        # 메모리 초기화
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        # 에포크 관련 정보 초기화
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0

    def build_sample(self):
        self.environment.observe()
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    @abc.abstractmethod
    def get_batch(self):
        pass

    def fit(self):
        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch()
        # 손실 초기화
        self.loss = None
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                # 정책 신경망 갱신
                loss += self.policy_network.train_on_batch(x, y_policy)
            self.loss = loss

    def visualize(self, epoch_str, num_epoches, epsilon):
        self.memory_action = [Agent.ACTION_HOLD] * (self.num_steps - 1) + self.memory_action
        self.memory_num_stocks = [0] * (self.num_steps - 1) + self.memory_num_stocks
        if self.value_network is not None:
            self.memory_value = [np.array([np.nan] * len(Agent.ACTIONS))] \
                                * (self.num_steps - 1) + self.memory_value
        if self.policy_network is not None:
            self.memory_policy = [np.array([np.nan] * len(Agent.ACTIONS))] \
                                * (self.num_steps - 1) + self.memory_policy
        self.memory_pv = [self.agent.initial_balance] * (self.num_steps - 1) + self.memory_pv
        self.visualizer.plot(
            epoch_str=epoch_str, num_epoches=num_epoches, 
            epsilon=epsilon, action_list=Agent.ACTIONS, 
            actions=self.memory_action, 
            num_stocks=self.memory_num_stocks, 
            outvals_value=self.memory_value, 
            outvals_policy=self.memory_policy,
            exps=self.memory_exp_idx, 
            initial_balance=self.agent.initial_balance, 
            pvs=self.memory_pv,
        )
        self.visualizer.save(os.path.join(self.epoch_summary_dir, f'epoch_summary_{epoch_str}.png'))

    def run(self, learning=True, progress_callback=None):
        if progress_callback is not None:
            self.progress_callback = progress_callback
        info = (
            f'[{self.stock_code}] RL:{self.rl_method} NET:{self.net} '
            f'LR:{self.lr} DF:{self.discount_factor} '
        )
        with self.lock:
            logger.debug(info)

        # 시작 시간
        time_start = time.time()

        # 가시화 준비 (웹 프로토타입에서는 기본 비활성)
        if self.visualize_enabled:
            self.visualizer.prepare(self.environment.chart_data, info)

        # 가시화 결과 저장할 폴더 준비 (가시화 활성 시에만)
        if self.visualize_enabled:
            self.epoch_summary_dir = os.path.join(self.output_path, f'epoch_summary_{self.stock_code}')
            if not os.path.isdir(self.epoch_summary_dir):
                os.makedirs(self.epoch_summary_dir)
            else:
                for f in os.listdir(self.epoch_summary_dir):
                    os.remove(os.path.join(self.epoch_summary_dir, f))

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0

        # 에포크 반복
        for epoch in tqdm(range(self.num_epoches)):
            time_start_epoch = time.time()

            # step 샘플을 만들기 위한 큐
            q_sample = collections.deque(maxlen=self.num_steps)
            
            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()

            # 학습을 진행할 수록 탐험 비율 감소 (min_epsilon 하한 유지)
            if learning:
                if self.num_epoches > 1:
                    epsilon = self.start_epsilon * (1 - (epoch / (self.num_epoches - 1)))
                else:
                    epsilon = self.start_epsilon
                epsilon = max(epsilon, self.min_epsilon)
            else:
                epsilon = self.start_epsilon

            for i in tqdm(range(len(self.training_data)), leave=False):
                # 샘플 생성
                next_sample = self.build_sample()
                if next_sample is None:
                    break

                # num_steps만큼 샘플 저장
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue

                # 가치, 정책 신경망 예측
                pred_value = None
                pred_policy = None
                if self.value_network is not None:
                    pred_value = self.value_network.predict(list(q_sample))
                if self.policy_network is not None:
                    pred_policy = self.policy_network.predict(list(q_sample))
                
                # 신경망 또는 탐험에 의한 행동 결정
                action, confidence, exploration = \
                    self.agent.decide_action(pred_value, pred_policy, epsilon)

                # 결정한 행동을 수행하고 보상 획득
                reward = self.agent.act(action, confidence)

                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(reward)
                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)

                # 반복에 대한 정보 갱신
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0

            # 에포크 종료 후 학습
            if learning:
                self.fit()

            # 에포크 관련 정보 로그 기록
            num_epoches_digit = len(str(self.num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            logger.debug(f'[{self.stock_code}][Epoch {epoch_str}/{self.num_epoches}] '
                f'Epsilon:{epsilon:.4f} #Expl.:{self.exploration_cnt}/{self.itr_cnt} '
                f'#Buy:{self.agent.num_buy} #Sell:{self.agent.num_sell} #Hold:{self.agent.num_hold} '
                f'#Stocks:{self.agent.num_stocks} PV:{self.agent.portfolio_value:,.0f} '
                f'Loss:{self.loss:.6f} ET:{elapsed_time_epoch:.4f}')

            # 에포크 관련 정보 가시화 (가시화 활성 시에만)
            if self.visualize_enabled and (self.num_epoches == 1 or (epoch + 1) % max(int(self.num_epoches / 10), 1) == 0):
                self.visualize(epoch_str, self.num_epoches, epsilon)

            # 진행 콜백 호출 (웹 프로토타입 진행률 표시용)
            if self.progress_callback is not None:
                try:
                    self.progress_callback(epoch + 1, self.num_epoches,
                                           float(self.agent.portfolio_value),
                                           float(self.agent.profitloss))
                except Exception:
                    pass

            # 학습 관련 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start

        # 학습 관련 정보 로그 기록
        with self.lock:
            logger.debug(f'[{self.stock_code}] Elapsed Time:{elapsed_time:.4f} '
                f'Max PV:{max_portfolio_value:,.0f} #Win:{epoch_win_cnt}')

    def save_models(self):
        if self.value_network is not None and self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)

    def predict(self):
        # 에이전트 초기화
        self.agent.reset()

        # step 샘플을 만들기 위한 큐
        q_sample = collections.deque(maxlen=self.num_steps)
        
        result = []
        while True:
            # 샘플 생성
            next_sample = self.build_sample()
            if next_sample is None:
                break

            # num_steps만큼 샘플 저장
            q_sample.append(next_sample)
            if len(q_sample) < self.num_steps:
                continue

            # 가치, 정책 신경망 예측
            pred_value = None
            pred_policy = None
            if self.value_network is not None:
                pred_value = self.value_network.predict(list(q_sample))
            if self.policy_network is not None:
                pred_policy = self.policy_network.predict(list(q_sample))
            
            # 신경망에 의한 행동 결정
            action, confidence, _ = self.agent.decide_action(pred_value, pred_policy, 0)
            result.append((self.environment.observation.iloc[0], int(action), float(confidence)))

        with open(os.path.join(self.output_path, f'pred_{self.stock_code}.json'), 'w') as f:
            print(json.dumps(result), file=f)
        return result


class DQNLearner(ReinforcementLearner):
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()

    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        for i, (sample, action, value, reward) in enumerate(memory):
            x[i] = sample
            r = self.memory_reward[-1] - reward
            y_value[i] = value
            y_value[i, action] = r + self.discount_factor * value_max_next
            value_max_next = value.max()
        return x, y_value, None


class DQNPlusLearner(DQNLearner):
    """DQN 한계 극복 기법(고려대 오승상 교수 DRL 강의자료)을 선택 적용하는 확장 DQN.

    dqn_options (dict) — 체크박스 단위로 개별 on/off:
      td:        1-step TD 학습 (강의 p.51-62, 68). 스텝 보상(포트폴리오 가치 변화율)
                 + 부트스트랩 타깃 r + γ·maxQ(s'). False 면 기존 quantylab 방식
                 (에피소드 종료 후 '최종손익-현재손익' MC 회귀) 그대로 동작.
      replay:    Experience Replay (p.70-71). 전이를 버퍼에 쌓고 랜덤 미니배치로
                 학습해 시계열 상관성을 제거. [td 필요]
      target:    Target Network (p.72-74). 타깃 계산 전용 고정 네트워크를 두고
                 주기적으로 하드 업데이트해 moving target 문제 완화. [td 필요]
      double:    Double DQN (p.77). 행동 선택은 온라인망, 평가는 타깃망으로 분리해
                 max 연산의 Q 과대평가 완화. [target 필요]
      multistep: Multi-step Learning (p.76). n-스텝 리턴으로 보상 전파 가속. [td 필요]
      n_step:    multistep 의 n (기본 3).
      per:       Prioritized Experience Replay (p.78-81). |TD 오차| 비례 샘플링
                 + β-어닐링 importance sampling 가중치. [replay 필요]
      dueling:   Dueling DQN (p.82-85). V(s)+A(s,a) 이중 스트림 헤드. [독립 적용 가능]

    기존 quantylab DQN 에 TD 가 없던 이유: memory_reward 에 '누적 손익률'이 저장되고
    get_batch 가 (최종 손익 - 현재 손익)을 보상으로 쓰기 때문에 1-step 보상이 존재하지
    않으며, 학습도 에포크 종료 후 한 번뿐이라 부트스트랩 기반 TD 갱신이 불가능했다.
    본 클래스는 스텝 보상을 직접 계산해 표준 Q-learning 형태로 재구성한다.
    """

    REWARD_SCALE = 100.0   # 손익률(비율) → % 단위로 스케일링 (MSE 그래디언트 안정화)

    # TD 학습 하이퍼파라미터 (프로토타입 데이터 규모에 맞춘 기본값)
    BUFFER_CAP = 100_000
    BATCH_SIZE = 64
    TRAIN_FREQ_REPLAY = 4      # replay 사용 시: 4스텝마다 랜덤 미니배치 1회
    SEQ_BATCH = 32             # replay 미사용 시: 32스텝 순차(상관) 배치로 학습
    WARMUP = 200               # 학습 시작 전 버퍼 최소 크기
    TARGET_UPDATE_FREQ = 200   # 타깃망 하드 업데이트 주기(스텝)
    PER_ALPHA = 0.6
    PER_BETA0 = 0.4
    PER_EPS = 1e-3

    def __init__(self, *args, dqn_options=None, **kwargs):
        self.dqn_options = self.normalize_options(dqn_options)
        self.target_network = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def normalize_options(o):
        """체크박스 값의 의존성 정리: replay/target/multistep ⊂ td, double ⊂ target, per ⊂ replay."""
        o = dict(o or {})
        td = bool(o.get("td"))
        replay = td and bool(o.get("replay"))
        target = td and bool(o.get("target"))
        return {
            "td": td,
            "replay": replay,
            "target": target,
            "double": target and bool(o.get("double")),
            "multistep": td and bool(o.get("multistep")),
            "n_step": max(2, min(int(o.get("n_step") or 3), 10)),
            "per": replay and bool(o.get("per")),
            "dueling": bool(o.get("dueling")),
        }

    def init_value_network(self, shared_network=None, activation='linear', loss='mse'):
        dueling = self.dqn_options.get("dueling", False)

        def build():
            common = dict(input_dim=self.num_features,
                          output_dim=self.agent.NUM_ACTIONS,
                          lr=self.lr, shared_network=shared_network,
                          activation=activation, loss=loss, dueling=dueling)
            if self.net == 'dnn':
                return DNN(**common)
            if self.net == 'lstm':
                return LSTMNetwork(num_steps=self.num_steps, **common)
            if self.net == 'cnn':
                return CNN(num_steps=self.num_steps, **common)
            raise ValueError(f'지원하지 않는 네트워크: {self.net}')

        self.value_network = build()
        if self.reuse_models and os.path.exists(self.value_network_path):
            self.value_network.load_model(model_path=self.value_network_path)
        if self.dqn_options.get("target"):
            self.target_network = build()
            self.target_network.copy_weights_from(self.value_network)

    def run(self, learning=True, progress_callback=None):
        # TD 미적용 시 기존 quantylab 방식(에포크 종료 후 MC 회귀)으로 그대로 학습.
        # (dueling 체크는 네트워크 구조라 이 경로에서도 적용된 상태)
        if not self.dqn_options.get("td"):
            return super().run(learning=learning, progress_callback=progress_callback)
        return self._run_td(learning=learning, progress_callback=progress_callback)

    # ------------------------------------------------------------------
    # TD 기반 학습 루프 (step 단위 Q-learning)
    # ------------------------------------------------------------------
    def _run_td(self, learning=True, progress_callback=None):
        if progress_callback is not None:
            self.progress_callback = progress_callback
        o = self.dqn_options
        gamma = self.discount_factor
        n_step = o["n_step"] if o["multistep"] else 1
        use_replay, use_target = o["replay"], o["target"]
        use_double, use_per = o["double"], o["per"]

        enabled = [k for k in ("td", "replay", "target", "double",
                               "multistep", "per", "dueling") if o.get(k)]
        info = (f'[{self.stock_code}] RL:dqn+({",".join(enabled)}) NET:{self.net} '
                f'LR:{self.lr} DF:{gamma}')
        with self.lock:
            logger.debug(info)

        time_start = time.time()

        # 리플레이 버퍼 (링 버퍼) / PER 우선순위
        buffer, prios = [], []
        ring_idx = 0
        seq_batch = []      # replay 미사용 시 순차 전이 누적
        global_step = 0
        total_steps_est = max(1, self.num_epoches * max(1, len(self.training_data)))

        def store_transition(tr):
            nonlocal ring_idx
            if use_replay:
                p = max(prios) if (use_per and prios) else 1.0
                if len(buffer) < self.BUFFER_CAP:
                    buffer.append(tr)
                    prios.append(p)
                else:
                    buffer[ring_idx] = tr
                    prios[ring_idx] = p
                    ring_idx = (ring_idx + 1) % self.BUFFER_CAP
            else:
                seq_batch.append(tr)

        def train_batch(transitions, idxs=None, weights=None):
            """전이 배치로 Q 갱신. 타깃: G + γ^n · (max 또는 double) Q(s')·(1-done)"""
            b = len(transitions)
            x = np.asarray([t[0] for t in transitions], dtype=np.float32)
            a = np.asarray([t[1] for t in transitions], dtype=np.int64)
            g = np.asarray([t[2] for t in transitions], dtype=np.float32)
            x2 = np.asarray([t[3] for t in transitions], dtype=np.float32)
            d = np.asarray([t[4] for t in transitions], dtype=np.float32)
            disc = np.asarray([t[5] for t in transitions], dtype=np.float32)

            q_now = self.value_network.predict_on_batch(x)
            net_next = self.target_network if use_target else self.value_network
            q_next = net_next.predict_on_batch(x2)
            if use_double:
                # Double DQN: 온라인망으로 a* 선택, 타깃망으로 평가
                a_star = np.argmax(self.value_network.predict_on_batch(x2), axis=1)
                boot = q_next[np.arange(b), a_star]
            else:
                boot = q_next.max(axis=1)
            target = g + disc * boot * (1.0 - d)
            td_err = target - q_now[np.arange(b), a]
            y = q_now.copy()
            y[np.arange(b), a] = target
            loss = self.value_network.train_on_batch(x, y, weights=weights)
            if use_per and idxs is not None:
                for bi, err in zip(idxs, np.abs(td_err)):
                    prios[bi] = float(err) + self.PER_EPS
            return loss

        max_portfolio_value = 0
        epoch_win_cnt = 0

        for epoch in tqdm(range(self.num_epoches)):
            time_start_epoch = time.time()
            q_sample = collections.deque(maxlen=self.num_steps)
            self.reset()

            if learning and self.num_epoches > 1:
                epsilon = self.start_epsilon * (1 - (epoch / (self.num_epoches - 1)))
            else:
                epsilon = self.start_epsilon
            if learning:
                epsilon = max(epsilon, self.min_epsilon)  # 탐험 하한 (정책 붕괴 방지)

            prev_pl = 0.0
            nstep_q = collections.deque()   # (state, action, step_reward)
            losses = []

            for i in tqdm(range(len(self.training_data)), leave=False):
                next_sample = self.build_sample()
                if next_sample is None:
                    break
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue

                state = [list(s) for s in q_sample]

                # 새 상태 s_{t+n} 도착 → 가장 오래된 전이 (s_t, a_t) 완성
                if learning and len(nstep_q) >= n_step:
                    g = sum((gamma ** k) * nstep_q[k][2] for k in range(n_step))
                    s0, a0, _ = nstep_q.popleft()
                    store_transition((s0, a0, g, state, 0.0, gamma ** n_step))

                # 행동 결정 및 실행
                pred_value = self.value_network.predict(list(q_sample))
                action, confidence, exploration = \
                    self.agent.decide_action(pred_value, None, epsilon)
                reward = self.agent.act(action, confidence)

                # 1-step TD 보상: 포트폴리오 가치 변화율(스텝 손익 증분)
                step_reward = (reward - prev_pl) * self.REWARD_SCALE
                prev_pl = reward
                nstep_q.append((state, action, step_reward))

                # 메모리 (백테스트 결과 수집·가시화 호환용 — 기존 형식 유지)
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(reward)
                self.memory_value.append(pred_value)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0
                global_step += 1

                # ---- 학습 ----
                if learning:
                    if use_replay:
                        if (len(buffer) >= max(self.WARMUP, self.BATCH_SIZE)
                                and global_step % self.TRAIN_FREQ_REPLAY == 0):
                            n = len(buffer)
                            if use_per:
                                p = np.asarray(prios, dtype=np.float64) ** self.PER_ALPHA
                                prob = p / p.sum()
                                idxs = np.random.choice(n, self.BATCH_SIZE, p=prob)
                                beta = self.PER_BETA0 + (1.0 - self.PER_BETA0) * min(
                                    1.0, global_step / total_steps_est)
                                w = (n * prob[idxs]) ** (-beta)
                                w = (w / w.max()).astype(np.float32)
                            else:
                                idxs = np.random.randint(0, n, self.BATCH_SIZE)
                                w = None
                            losses.append(train_batch(
                                [buffer[j] for j in idxs], idxs, w))
                    else:
                        # replay 미사용: 순차(상관) 배치 — 강의자료가 지적한
                        # correlated samples 문제를 그대로 보여주는 대조군
                        if len(seq_batch) >= self.SEQ_BATCH:
                            losses.append(train_batch(seq_batch))
                            seq_batch.clear()

                    # 타깃 네트워크 하드 업데이트
                    if use_target and global_step % self.TARGET_UPDATE_FREQ == 0:
                        self.target_network.copy_weights_from(self.value_network)

            # ---- 에피소드 종료: 잔여 n-step 전이 done=True 로 플러시 ----
            if learning:
                last_state = [list(s) for s in q_sample] if len(q_sample) == self.num_steps else None
                while nstep_q and last_state is not None:
                    k_len = len(nstep_q)
                    g = sum((gamma ** k) * nstep_q[k][2] for k in range(k_len))
                    s0, a0, _ = nstep_q.popleft()
                    store_transition((s0, a0, g, last_state, 1.0, gamma ** k_len))
                if not use_replay and len(seq_batch) >= 4:
                    losses.append(train_batch(seq_batch))
                    seq_batch.clear()

            self.loss = float(np.mean(losses)) if losses else 0.0

            # 에포크 로그 (기존 형식 유지)
            num_epoches_digit = len(str(self.num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            elapsed_time_epoch = time.time() - time_start_epoch
            logger.debug(f'[{self.stock_code}][Epoch {epoch_str}/{self.num_epoches}] '
                f'Epsilon:{epsilon:.4f} #Expl.:{self.exploration_cnt}/{self.itr_cnt} '
                f'#Buy:{self.agent.num_buy} #Sell:{self.agent.num_sell} #Hold:{self.agent.num_hold} '
                f'#Stocks:{self.agent.num_stocks} PV:{self.agent.portfolio_value:,.0f} '
                f'Loss:{self.loss:.6f} ET:{elapsed_time_epoch:.4f}')

            if self.progress_callback is not None:
                try:
                    self.progress_callback(epoch + 1, self.num_epoches,
                                           float(self.agent.portfolio_value),
                                           float(self.agent.profitloss))
                except Exception:
                    pass

            max_portfolio_value = max(max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        elapsed_time = time.time() - time_start
        with self.lock:
            logger.debug(f'[{self.stock_code}] Elapsed Time:{elapsed_time:.4f} '
                f'Max PV:{max_portfolio_value:,.0f} #Win:{epoch_win_cnt}')


class PolicyGradientLearner(ReinforcementLearner):
    def __init__(self, *args, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_network_path = policy_network_path
        self.init_policy_network()

    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        for i, (sample, action, policy, reward) in enumerate(memory):
            x[i] = sample
            r = self.memory_reward[-1] - reward
            y_policy[i, :] = policy
            y_policy[i, action] = utils.sigmoid(r)
        return x, None, y_policy


class ActorCriticLearner(ReinforcementLearner):
    def __init__(self, *args, shared_network=None, 
        value_network_path=None, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if shared_network is None:
            self.shared_network = Network.get_shared_network(
                net=self.net, num_steps=self.num_steps, 
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS)
        else:
            self.shared_network = shared_network
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=self.shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=self.shared_network)

    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            r = self.memory_reward[-1] - reward
            y_value[i, :] = value
            y_value[i, action] = r + self.discount_factor * value_max_next
            y_policy[i, :] = policy
            y_policy[i, action] = utils.sigmoid(r)
            value_max_next = value.max()
        return x, y_value, y_policy


class A2CLearner(ActorCriticLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            r = reward_next + self.memory_reward[-1] - reward * 2
            reward_next = reward
            y_value[i, :] = value
            y_value[i, action] = np.tanh(r + self.discount_factor * value_max_next)
            advantage = y_value[i, action] - y_value[i].mean()
            y_policy[i, :] = policy
            y_policy[i, action] = utils.sigmoid(advantage)
            value_max_next = value.max()
        return x, y_value, y_policy


class A3CLearner(ReinforcementLearner):
    def __init__(self, *args, list_stock_code=None, 
        list_chart_data=None, list_training_data=None,
        list_min_trading_price=None, list_max_trading_price=None, 
        value_network_path=None, policy_network_path=None,
        **kwargs):
        assert len(list_training_data) > 0
        super().__init__(*args, **kwargs)
        self.num_features += list_training_data[0].shape[1]

        # 공유 신경망 생성
        self.shared_network = Network.get_shared_network(
            net=self.net, num_steps=self.num_steps, 
            input_dim=self.num_features,
            output_dim=self.agent.NUM_ACTIONS)
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=self.shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=self.shared_network)

        # A2CLearner 생성
        self.learners = []
        for (stock_code, chart_data, training_data, 
            min_trading_price, max_trading_price) in zip(
                list_stock_code, list_chart_data, list_training_data,
                list_min_trading_price, list_max_trading_price
            ):
            learner = A2CLearner(*args, 
                stock_code=stock_code, chart_data=chart_data, 
                training_data=training_data,
                min_trading_price=min_trading_price, 
                max_trading_price=max_trading_price, 
                shared_network=self.shared_network,
                value_network=self.value_network,
                policy_network=self.policy_network, **kwargs)
            self.learners.append(learner)

    def run(self, learning=True):
        threads = []
        for learner in self.learners:
            threads.append(threading.Thread(
                target=learner.run, daemon=True, kwargs={'learning': learning}
            ))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def predict(self):
        threads = []
        for learner in self.learners:
            threads.append(threading.Thread(
                target=learner.predict, daemon=True
            ))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()


class PPOLearner(ActorCriticLearner):
    """PPO 근사 구현 (프레임워크 정합형).

    quantylab/rltrader 의 신경망은 train_on_batch(x, y) 형태로 목표값에 대한
    MSE/BCE 회귀만 수행하므로, 표준 PPO 의 clipped surrogate objective 를
    손실 함수로 직접 표현할 수는 없다. 따라서 본 클래스는 자료조사 의사결정②가
    권장한 PPO 의 '핵심 아이디어'를 이 프레임워크 안에서 표현 가능한 형태로
    근사한다.

      1) Advantage 기반 정책 업데이트 (A2C 와 동일한 골격)
      2) 신구 정책 비율(ratio)의 클리핑: advantage 에 clip(ratio, 1-eps, 1+eps)
         개념을 적용해 한 번에 과도하게 정책이 변하지 않도록 제한
      3) 수집한 배치에 대해 K 에폭 반복 업데이트 (PPO 의 multiple epoch update)

    한계: 완전한 on-policy importance sampling 은 아니며, 자료조사 문서의
    'PPO 는 quantylab 에 없어 같은 스타일로 커스텀 확장' 결정에 따른 근사치다.
    """

    def __init__(self, *args, ppo_clip=0.2, ppo_epochs=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs

    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            r = reward_next + self.memory_reward[-1] - reward * 2
            reward_next = reward
            y_value[i, :] = value
            y_value[i, action] = np.tanh(r + self.discount_factor * value_max_next)
            advantage = y_value[i, action] - y_value[i].mean()
            # PPO 클리핑(트러스트 리전 취지): advantage 자체를 짓누르지 않고,
            # 새 정책 목표가 기존 정책 확률에서 한 번에 ppo_clip 이상 벗어나지 않도록
            # '정책 변화폭'을 제한한다. K-에폭 반복 업데이트로 점진 수렴.
            old_p = policy[action]
            target = utils.sigmoid(advantage)
            y_policy[i, :] = policy
            y_policy[i, action] = old_p + np.clip(target - old_p, -self.ppo_clip, self.ppo_clip)
            value_max_next = value.max()
        return x, y_value, y_policy

    def fit(self):
        # PPO: 동일 배치에 대해 여러 에폭 반복 업데이트
        for _ in range(max(1, self.ppo_epochs)):
            super().fit()
