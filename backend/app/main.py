"""FastAPI 백엔드.

엔드포인트
  GET  /api/health           서버/키 설정 상태
  GET  /api/stocks           테스트 종목 3종
  GET  /api/algorithms       알고리즘 메타(quantylab: DQN/A2C/PPO)
  GET  /api/engines          엔진별(quantylab/hanium) 알고리즘·네트워크 메타
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


class TrainRequest(BaseModel):
    stock_code: str
    # 엔진: 'quantylab'(기존 DQN/A2C/PPO) 또는 'hanium'(신규 10종)
    engine: str = Field("quantylab", pattern="^(quantylab|hanium)$")
    # 알고리즘/네트워크는 엔진마다 후보가 달라 자유 문자열로 받고 엔진에서 검증한다.
    algorithm: str
    net: str | None = None
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    # quantylab: 학습 에포크 수 / hanium: episodes 로 매핑(아래 별도 필드)
    num_epoches: int = Field(100, ge=1, le=2000)
    episodes: int | None = Field(None, ge=1, le=2000)
    lr: float = Field(0.0005, gt=0)
    discount_factor: float = Field(0.9, gt=0, le=1)
    num_steps: int | None = Field(None, ge=1, le=30)
    window_size: int | None = Field(None, ge=5, le=120)   # hanium 관측 윈도우
    trade_ratio: float | None = Field(None, gt=0, le=1)    # hanium 부분 매매 비율
    balance: int = Field(10_000_000, gt=0)
    # state 에 포함할 지표 id 목록. None/빈 값이면 전체 지표 사용.
    features: list[str] | None = None

    def validate_engine_choice(self):
        """엔진별 알고리즘/네트워크 후보 검증."""
        if self.engine == "quantylab":
            algos, nets = {"dqn", "a2c", "ppo"}, {"dnn", "lstm", "cnn"}
        else:
            algos = set(engine_hanium.HANIUM_ALGORITHMS)
            nets = set(engine_hanium.HANIUM_NETWORKS)
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
