# hanium 알고리즘 통합 가이드

99-_RL_Trading 웹앱(FastAPI + React)에 **hanium-rl-trading의 강화학습 프레임워크**를
이식했습니다. 기존 quantylab 엔진은 그대로 두고, UI에서 **엔진을 토글**해 둘 다
실험할 수 있습니다.

## 무엇이 추가됐나

| 구분 | quantylab (기존) | hanium (신규) |
|---|---|---|
| 학습 방식 | 에포크 단위 회귀(`train_on_batch`) | gymnasium step 단위(리플레이/롤아웃) |
| 알고리즘 | DQN · A2C · PPO (3종) | DQN · A2C · PPO · DDPG · TD3 · SAC · Rainbow · IQN · Decision Transformer · Ensemble (**10종**) |
| 네트워크 | DNN · LSTM · CNN (3종) | + Mamba · PatchTST · iTransformer · TFT · xLSTM (**8종**) |
| 조합 | 9 | **80 (10 × 8 자유 조합)** |

두 엔진 모두 **거래세 0.25% · 수수료 0.015% · 액션 마스킹**을 적용합니다(hanium
환경에 99-의 거래 현실성을 이식). hanium 환경은 `trade_ratio`로 부분 매매도 지원합니다.

## 추가/변경된 파일

추가
- `backend/app/hanium/` — hanium의 agents · networks · env 패키지(벤더링)
  - `env/trading_env.py` — 99- 거래 현실성(거래세·마스킹·부분매매) 이식
- `backend/app/engine_hanium.py` — KIS 데이터 → hanium 학습/백테스트, 기존과 동일한 결과 형태 반환
- `backend/scripts/smoke_hanium.py` — 전 조합(80개) 검증 스크립트

변경
- `backend/app/main.py` — `engine` 필드, `/api/engines` 엔드포인트, 엔진별 검증
- `backend/app/jobs.py` — `engine` 값에 따라 엔진 라우팅
- `backend/requirements.txt` — `gymnasium>=0.29` 추가
- `frontend/src/App.jsx`, `frontend/src/api.js` — 엔진/알고리즘/네트워크 선택 UI

## 실행

```bash
# 백엔드
cd backend
pip install -r requirements.txt        # torch, gymnasium 포함
uvicorn app.main:app --reload --port 8000

# 프론트엔드
cd frontend && npm install && npm run dev
```

UI에서 **② 엔진**을 `hanium (신규)`로 바꾸면 10종 알고리즘 × 8종 네트워크를
자유롭게 선택할 수 있습니다. 고급 설정에서 에피소드 수와 관측 윈도우를 조절합니다.

## 검증

```bash
cd backend
python -m scripts.smoke_hanium --fast   # 대표 조합 빠른 검증
python -m scripts.smoke_hanium          # 전체 80개 조합(합성 데이터, KIS 불필요)
```

> 참고: 이 통합은 샌드박스에서 환경 로직(거래세·마스킹·관측 셰이프)과 전 조합의
> 셰이프 배선·메타데이터 일관성을 검증했습니다. torch 실연산이 포함된 80개 조합의
> 최종 실행 확인은 위 스모크 스크립트를 **torch가 설치된 로컬 환경**에서 1회 돌려
> 주세요(권장).
```
