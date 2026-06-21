"""FastAPI 백엔드.

엔드포인트
  GET  /api/health           서버/키 설정 상태
  GET  /api/stocks           테스트 종목 3종
  GET  /api/algorithms       알고리즘 메타(DQN/A2C/PPO)
  POST /api/train            실험 시작 -> {job_id}
  GET  /api/jobs/{job_id}    진행상황/결과 폴링
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config, engine, jobs

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
    algorithm: str = Field(..., pattern="^(dqn|a2c|ppo)$")
    net: str | None = Field(None, pattern="^(dnn|lstm|cnn)$")
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    num_epoches: int = Field(100, ge=1, le=2000)
    lr: float = Field(0.0005, gt=0)
    discount_factor: float = Field(0.9, gt=0, le=1)
    num_steps: int | None = Field(None, ge=1, le=30)
    balance: int = Field(10_000_000, gt=0)


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


@app.post("/api/train")
def train(req: TrainRequest):
    if not config.kis_configured():
        raise HTTPException(
            status_code=400,
            detail="KIS API 키가 설정되지 않았습니다. backend/.env 에 KIS_APP_KEY/KIS_APP_SECRET 를 넣어주세요.",
        )
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
