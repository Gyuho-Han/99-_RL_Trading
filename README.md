# 강화학습 AI 트레이딩 프로토타입

자료조사 워드 2종(`강화학습_트레이딩_조사보고서`, `AI트레이딩_의사결정`)의 의사결정을
그대로 반영한 웹 프로토타입입니다. 국내 개별주를 대상으로 **DQN / A2C / PPO** 강화학습
모델을 직접 학습·백테스트하고, 동일 구간 **단순보유(Buy&Hold)** 와 성과를 비교합니다.

```
프로토타입/
├── backend/        FastAPI + quantylab/rltrader 기반 학습·백테스트 엔진
└── frontend/       React + Vite UI (종목·알고리즘·기간 선택, 결과 차트)
```

## 자료조사 반영 내용

| 의사결정 | 반영 |
|---|---|
| ① 타겟 시장 | 국내 단일 우량주. 삼성전자(대형·저변동), 에이치엘비(바이오·고변동, 논문 [C] 계열), 에코프로(2차전지·고변동) |
| ② 알고리즘 | DQN(베이스라인) · A2C · PPO 버튼 전환. *PPO는 quantylab에 없어 동일 프레임워크 스타일로 근사 구현(아래 참고)* |
| ③ State 설계 | OHLCV 비율 피처 15종 + 기술적 지표(RSI·MACD·Stochastic·CCI·Bollinger·MFI) + 에이전트 상태(보유비중·손익·평단 대비). 신경망 첫 층 BatchNorm으로 정규화(학습구간 통계만 사용 → 룩어헤드 방지) |
| ④ 뉴스/LLM | **요청대로 이번 단계 제외**. 추후 FinBERT/KR-FinBert-SC 감성 스칼라를 state에 추가하는 자리만 설계상 남겨둠 |
| 평가 | 누적수익률·연환산수익률·변동성·Sharpe·MDD를 모두 단순보유와 병기. 테스트는 ε=0(탐험 없음) |

> **PPO 근사 안내**: quantylab/rltrader의 신경망은 `train_on_batch(x, y)`로 목표값 회귀만
> 수행하므로 표준 PPO의 clipped surrogate objective를 손실함수로 직접 표현할 수 없습니다.
> 본 구현은 A2C 골격에 PPO 핵심 아이디어(advantage 클리핑 + K-에폭 반복 업데이트)를
> 적용한 프레임워크 정합형 근사입니다. 엄밀한 PPO가 필요하면 Stable-Baselines3로 별도
> 통합하는 방향을 권장합니다.

## 사전 준비 — KIS API 키

1. [한국투자증권 API 포털](https://apiportal.koreainvestment.com)에서 앱 등록 후
   **APP_KEY / APP_SECRET** 발급.
2. `backend/.env.example`을 `backend/.env`로 복사하고 키를 채웁니다.

```bash
cp backend/.env.example backend/.env
# backend/.env 편집:
# KIS_APP_KEY=...
# KIS_APP_SECRET=...
# KIS_ENV=real     # 또는 paper (시세 조회는 둘 다 가능)
```

> `.env`와 토큰 캐시(`.kis_token_cache.json`)는 `.gitignore`에 포함되어 커밋되지 않습니다.

## 실행

### 백엔드 (포트 8000)
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# 또는: ./run.sh
```

### 프론트엔드 (포트 5173)
```bash
cd frontend
npm install
npm run dev
```
브라우저에서 http://localhost:5173 접속.

## 사용법

1. **종목** 선택 (삼성전자 / 에이치엘비 / 에코프로)
2. **알고리즘** 선택 (DQN / A2C / PPO) + 신경망(DNN / LSTM / CNN)
3. **학습 기간**과 **테스트 기간**을 각각 지정 (학습 구간 이전 ~250일은 지표 워밍업에 자동 사용)
4. *고급 설정*에서 에폭 수·학습률 조정 가능
5. **학습 + 백테스트 실행** → 진행률 표시 후, 성과 지표표와 포트폴리오 가치 추이 차트 확인

## API 요약

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/health` | 서버/키 설정 상태 |
| GET | `/api/stocks` | 테스트 종목 3종 |
| GET | `/api/algorithms` | 알고리즘 메타 |
| POST | `/api/train` | 실험 시작 → `{job_id}` |
| GET | `/api/jobs/{job_id}` | 진행상황/결과 폴링 |

## 주의

- 본 프로토타입은 **시세 조회 + 백테스트 전용**이며 실제 매매 주문은 전혀 수행하지 않습니다.
- 백테스트 결과는 과거 데이터 기반이며 미래 수익을 보장하지 않습니다. 투자 판단의 근거로
  사용하지 마세요.
- 학습 작업 상태는 프로세스 메모리에만 저장됩니다(서버 재시작 시 초기화).

## 기술 스택

- **백엔드**: FastAPI, PyTorch, pandas, [quantylab/rltrader](https://github.com/quantylab/rltrader)(벤더링), 한국투자증권 OpenAPI
- **프론트엔드**: React 18, Vite, Recharts
