"""FastAPI 백엔드.

엔드포인트
  GET  /api/health           서버/키 설정 상태
  GET  /api/stocks           테스트 종목 3종
  GET  /api/algorithms       알고리즘 메타(quantylab: DQN/A2C/PPO)
  GET  /api/engines          엔진별(quantylab/hanium) 알고리즘·네트워크 메타
  GET  /api/models           저장된 hanium 학습 모델 목록
  POST /api/train            실험 시작 -> {job_id}
  GET  /api/jobs/{job_id}    진행상황/결과 폴링
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config, engine, engine_hanium, jobs, features

app = FastAPI(title="강화학습 AI 트레이딩 프로토타입", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DQNOptions(BaseModel):
    """DQN 한계 극복 기법 체크박스 (quantylab 엔진 + dqn 알고리즘에서만 사용).

    의존성: replay/target/multistep 은 td 필요, double 은 target 필요, per 는 replay 필요.
    (백엔드에서 normalize_options 로 다시 한 번 정리하므로 잘못 보내도 안전)
    """
    td: bool = False          # 1-step TD 학습 (스텝 보상 + 부트스트랩)
    replay: bool = False      # Experience Replay
    target: bool = False      # Target Network
    double: bool = False      # Double DQN
    multistep: bool = False   # Multi-step Learning
    n_step: int = Field(3, ge=2, le=10)
    per: bool = False         # Prioritized Experience Replay
    dueling: bool = False     # Dueling DQN (V+A 분리 헤드)


class HaniumRewardOptions(BaseModel):
    """hanium TradingEnv 보상 셰이핑 계수 (모두 0이면 기존 순수 PV 변화율 보상).

    PPT 개선 실험 기준값 예시: trade_penalty 0.0005, mdd_penalty 0.3
    """
    sell_profit_bonus: float = Field(0.0, ge=0, le=10)   # 실현수익 보너스 계수
    loss_sell_penalty: float = Field(0.0, ge=0, le=10)   # 손실 매도 패널티 계수
    trade_penalty: float = Field(0.0, ge=0, le=0.1)      # 체결 1회당 고정 패널티
    mdd_penalty: float = Field(0.0, ge=0, le=10)         # 낙폭 갱신 패널티 계수


class TrainRequest(BaseModel):
    stock_code: str
    # 엔진: 'quantylab'(기존 DQN/A2C/PPO) 또는 'hanium'(신규 10종)
    engine: str = Field("quantylab", pattern="^(quantylab|hanium)$")
    # 알고리즘/네트워크는 엔진마다 후보가 달라 자유 문자열로 받고 엔진에서 검증한다.
    algorithm: str
    # quantylab 전용: 복수 알고리즘 앙상블 (1~3개). 지정 시 algorithm 보다 우선.
    algorithms: list[str] | None = None
    # 앙상블 방식: 'sharpe'(Yang et al. ICAIF 2020 — 검증 Sharpe 선택 + 롤링 재선정
    # + turbulence 위험회피, 기본값) 또는 'vote'(단순 다수결)
    ensemble_method: str = Field("sharpe", pattern="^(sharpe|vote)$")
    # sharpe 방식의 리더 재선정 주기 (거래일)
    ensemble_window: int = Field(21, ge=5, le=126)
    # turbulence 위험회피 규칙 사용 여부 (sharpe 방식에서만 의미 있음)
    use_turbulence: bool = True
    net: str | None = None
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    # quantylab: 학습 에포크 수 / hanium: episodes 로 매핑(아래 별도 필드)
    # 기본 40: 삼성전자(학습 2017~2022/테스트 2023~) 에폭 스윕 결과 20/40/80 중
    # 40이 수익률 최고, 80은 시간 2배에 성능 동일 수준이라 40을 기본값으로 채택.
    num_epoches: int = Field(40, ge=1, le=2000)
    episodes: int | None = Field(None, ge=1, le=2000)
    lr: float = Field(0.0005, gt=0)
    discount_factor: float = Field(0.9, gt=0, le=1)
    num_steps: int | None = Field(None, ge=1, le=30)
    window_size: int | None = Field(None, ge=5, le=120)   # hanium 관측 윈도우
    trade_ratio: float | None = Field(None, gt=0, le=1)    # hanium 부분 매매 비율
    balance: int = Field(10_000_000, gt=0)
    # 1회 최소 매매 금액 (quantylab 엔진 전용).
    # 매수/매도 시 '신뢰도(confidence)'에 비례해 min~잔고 사이 금액만큼 주문한다.
    # (금액 = min + confidence·(balance-min), 상한 없음 = 잔고 전액까지)
    min_trading_price: int = Field(100_000, gt=0)
    # (선택) 1회 최대 매매 금액. 미지정 시 잔고 전액.
    max_trading_price: int | None = Field(None, gt=0)
    # 재현용 난수 시드 (None 이면 실행마다 무작위 → 결과가 매번 달라짐)
    seed: int | None = None
    # state 에 포함할 지표 id 목록. None/빈 값이면 전체 지표 사용.
    features: list[str] | None = None
    # DQN 개선 기법 체크박스 (quantylab + dqn 전용, None 이면 전부 미적용=기존 방식)
    dqn_options: DQNOptions | None = None
    # hanium 보상 셰이핑 계수 (None 이면 전부 0 = 기존 보상)
    reward_options: HaniumRewardOptions | None = None
    # hanium 저장 모델 재사용: 모델 id(디렉터리명). 지정 시 학습 생략, 백테스트만 수행.
    saved_model: str | None = None

    def validate_engine_choice(self):
        """엔진별 알고리즘/네트워크 후보 검증."""
        if self.max_trading_price is not None and self.max_trading_price < self.min_trading_price:
            raise HTTPException(400, "max_trading_price 는 min_trading_price 이상이어야 합니다.")
        if self.min_trading_price > self.balance:
            raise HTTPException(400, "min_trading_price 가 초기자본(balance)보다 큽니다.")
        if self.engine == "quantylab":
            algos, nets = {"dqn", "a2c", "ppo"}, {"dnn", "lstm", "cnn"}
            if self.algorithms:
                sel = [a for a in dict.fromkeys(self.algorithms) if a]
                bad = [a for a in sel if a not in algos]
                if not sel:
                    raise HTTPException(400, "algorithms 는 1개 이상 선택해야 합니다.")
                if bad:
                    raise HTTPException(400, f"'quantylab' 엔진에서 지원하지 않는 알고리즘: {bad}")
                if len(sel) > 3:
                    raise HTTPException(400, "앙상블은 최대 3개(dqn/a2c/ppo)까지 선택 가능합니다.")
                return  # 앙상블은 알고리즘별 기본 네트워크를 사용하므로 net 검증 생략
        else:
            algos = set(engine_hanium.HANIUM_ALGORITHMS)
            nets = set(engine_hanium.HANIUM_NETWORKS)
            if self.saved_model:
                return  # 저장 모델 재사용 시 알고리즘/네트워크는 metadata 가 결정
        if self.algorithm not in algos:
            raise HTTPException(400, f"'{self.engine}' 엔진에서 지원하지 않는 알고리즘: {self.algorithm}")
        if self.net is not None and self.net not in nets:
            raise HTTPException(400, f"'{self.engine}' 엔진에서 지원하지 않는 네트워크: {self.net}")


@app.get("/api/health")
def health():
    return {"status": "ok", "kis_configured": config.kis_configured(),
            "kis_env": config.KIS_ENV}


@app.get("/api/stocks")
def stocks():
    return engine.STOCKS


@app.get("/api/algorithms")
def algorithms():
    return [{"id": k, **v} for k, v in engine.ALGORITHMS.items()]


@app.get("/api/engines")
def engines():
    """엔진별 알고리즘·네트워크 메타데이터(프론트 선택 UI 용)."""
    return {
        "quantylab": {
            "label": "quantylab (기존)",
            "desc": "에포크 단위 회귀 학습. DQN·A2C·PPO 3종(거래세·액션마스킹 내장).",
            "algorithms": [{"id": k, **v} for k, v in engine.ALGORITHMS.items()],
            "networks": [
                {"id": "dnn", "label": "DNN", "sub": "완전연결"},
                {"id": "lstm", "label": "LSTM", "sub": "시계열"},
                {"id": "cnn", "label": "CNN", "sub": "합성곱"},
            ],
            "param_kind": "epoches",
        },
        "hanium": {
            "label": "hanium (신규)",
            "desc": "gymnasium step 학습. 10종 알고리즘 × 8종 네트워크 자유 조합(99- 거래 현실성 이식).",
            "algorithms": [{"id": k, **v} for k, v in engine_hanium.HANIUM_ALGORITHMS.items()],
            "networks": [{"id": k, **v} for k, v in engine_hanium.HANIUM_NETWORKS.items()],
            "param_kind": "episodes",
        },
    }


@app.get("/api/features")
def feature_list():
    """state 에 넣을 수 있는 지표 목록(체크박스 UI 용)."""
    return features.FEATURE_META


@app.get("/api/models")
def saved_models():
    """저장된 hanium 학습 모델 목록 (최신순, 재사용 드롭다운용)."""
    return engine_hanium.list_saved_models()


@app.post("/api/train")
def train(req: TrainRequest):
    if not config.kis_configured():
        raise HTTPException(
            status_code=400,
            detail="KIS API 키가 설정되지 않았습니다. backend/.env 에 KIS_APP_KEY/KIS_APP_SECRET 를 넣어주세요.",
        )
    req.validate_engine_choice()
    job_id = jobs.start_job(req.model_dump())
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    # params 는 응답에서 생략(요청 그대로라 불필요), traceback 은 디버깅용으로만
    out = {k: v for k, v in job.items() if k != "params"}
    return out
