import React, { useEffect, useState, useRef } from 'react'
import { api } from './api'
import EquityChart from './components/EquityChart'
import MetricsTable from './components/MetricsTable'
import PriceTradeChart from './components/PriceTradeChart'
import TradeLog from './components/TradeLog'

// 엔진 메타가 로드되기 전 사용할 기본 네트워크 목록(quantylab)
const NETS = [
  { id: 'dnn', label: 'DNN', sub: '완전연결' },
  { id: 'lstm', label: 'LSTM', sub: '시계열' },
  { id: 'cnn', label: 'CNN', sub: '합성곱' },
]

const PHASE_LABEL = {
  queued: '대기 중', starting: '준비 중', fetching: 'KIS 시세 조회 중',
  training: '학습 중', backtesting: '백테스트 중', done: '완료', error: '오류',
}

// DQN 한계 극복 기법 (고려대 오승상 교수 DRL 강의자료 기준)
// requires: 해당 기법의 전제조건 체크박스 id
const DQN_IMPROVEMENTS = [
  { id: 'td', label: 'TD 학습', sub: '1-step TD 타깃 r+γ·maxQ(s′). 기존 MC 방식 대체 · 아래 기법들의 기반', requires: null },
  { id: 'replay', label: 'Experience Replay', sub: '리플레이 버퍼 + 랜덤 미니배치로 샘플 상관성 제거', requires: 'td' },
  { id: 'target', label: 'Target Network', sub: '타깃 전용 고정 네트워크 (주기적 동기화)로 학습 안정화', requires: 'td' },
  { id: 'double', label: 'Double DQN', sub: '선택은 온라인망·평가는 타깃망 → Q 과대평가 완화', requires: 'target' },
  { id: 'multistep', label: 'Multi-step (n=3)', sub: 'n-스텝 리턴으로 보상 전파 가속', requires: 'td' },
  { id: 'per', label: 'Prioritized Replay', sub: '|TD 오차| 비례 우선 샘플링 + IS 가중 보정', requires: 'replay' },
  { id: 'dueling', label: 'Dueling DQN', sub: 'V(s)+A(s,a) 이중 스트림 헤드 (독립 적용 가능)', requires: null },
]

export default function App() {
  const [stocks, setStocks] = useState([])
  const [algos, setAlgos] = useState([])
  const [health, setHealth] = useState(null)

  const [engines, setEngines] = useState({})
  const [engine, setEngine] = useState('quantylab')

  const [stock, setStock] = useState('005930')
  const [algo, setAlgo] = useState('dqn')
  // quantylab 전용: 복수 선택(1~3개) → 앙상블
  const [qlAlgos, setQlAlgos] = useState(['dqn'])
  // 앙상블 방식: 'sharpe'(논문: 검증 Sharpe 선택+롤링 재선정+turbulence) / 'vote'(다수결)
  const [ensMethod, setEnsMethod] = useState('sharpe')
  const [ensWindow, setEnsWindow] = useState(21)
  const [net, setNet] = useState('dnn')
  const [windowSize, setWindowSize] = useState(20)
  const [trainStart, setTrainStart] = useState('2017-01-01')
  const [trainEnd, setTrainEnd] = useState('2022-12-31')
  const [testStart, setTestStart] = useState('2023-01-01')
  const [testEnd, setTestEnd] = useState(new Date().toISOString().slice(0, 10))
  const [epochs, setEpochs] = useState(40)
  const [lr, setLr] = useState(0.0005)
  // 초기자본 (두 엔진 공통)
  const [balance, setBalance] = useState(10000000)
  // 1회 최소 매매 금액 (quantylab): 신뢰도에 비례해 min~잔고 전액 사이 금액 주문
  const [minTp, setMinTp] = useState(100000)
  const [showAdv, setShowAdv] = useState(false)

  const [featureMeta, setFeatureMeta] = useState([])
  const [selectedFeatures, setSelectedFeatures] = useState([])

  // DQN 개선 기법 체크박스 (기본: 2015 Nature DQN 구성 = TD + Replay + Target)
  const [dqnOpts, setDqnOpts] = useState({
    td: true, replay: true, target: true,
    double: false, multistep: false, per: false, dueling: false,
  })

  // hanium: 매수 비율 · 보상 셰이핑 · 저장 모델 재사용
  const [tradeRatio, setTradeRatio] = useState(1.0)
  const [rewardOpts, setRewardOpts] = useState({
    sell_profit_bonus: 0, loss_sell_penalty: 0, trade_penalty: 0, mdd_penalty: 0,
  })
  const [savedModels, setSavedModels] = useState([])
  const [savedModel, setSavedModel] = useState('')   // '' = 새로 학습

  const [job, setJob] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const pollRef = useRef(null)

  useEffect(() => {
    api.stocks().then(setStocks).catch(() => {})
    api.algorithms().then(setAlgos).catch(() => {})
    api.engines().then(setEngines).catch(() => {})
    api.health().then(setHealth).catch(() => setHealth({ status: 'down' }))
    api.features().then((fs) => {
      setFeatureMeta(fs)
      setSelectedFeatures(fs.map((f) => f.id)) // 기본값: 전체 선택
    }).catch(() => {})
    api.models().then(setSavedModels).catch(() => {})
  }, [])

  // 현재 엔진의 알고리즘/네트워크 목록 (엔진 메타 로드 전엔 기본값)
  const engMeta = engines[engine]
  const algoList = engMeta ? engMeta.algorithms : algos
  const netList = engMeta ? engMeta.networks : NETS
  const paramKind = engMeta ? engMeta.param_kind : 'epoches'

  // 엔진을 바꾸면 해당 엔진의 첫 알고리즘/네트워크로 초기화
  function switchEngine(next) {
    if (next === engine) return
    setEngine(next)
    const m = engines[next]
    if (m) {
      if (m.algorithms?.length) setAlgo(m.algorithms[0].id)
      if (m.networks?.length) setNet((m.networks[1] || m.networks[0]).id)
    }
  }

  function toggleFeature(id) {
    setSelectedFeatures((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id])
  }

  // DQN 개선 기법 토글 + 의존성 자동 정리 (td 해제 → 하위 기법 전부 해제 등)
  function toggleDqnOpt(id) {
    setDqnOpts((prev) => {
      const next = { ...prev, [id]: !prev[id] }
      if (!next.td) {
        next.replay = false; next.target = false; next.double = false
        next.multistep = false; next.per = false
      }
      if (!next.replay) next.per = false
      if (!next.target) next.double = false
      return next
    })
  }
  // quantylab 알고리즘 복수 선택 토글 (최소 1개 유지, 1개만 남으면 기본 네트워크 적용)
  function toggleQlAlgo(id) {
    setQlAlgos((prev) => {
      const next = prev.includes(id)
        ? (prev.length > 1 ? prev.filter((x) => x !== id) : prev)
        : [...prev, id]
      if (next.length === 1) {
        const meta = algoList.find((x) => x.id === next[0])
        if (meta?.net_default) setNet(meta.net_default)
      }
      return next
    })
  }
  const qlEnsemble = engine === 'quantylab' && qlAlgos.length > 1
  const showDqnOpts = engine === 'quantylab' && qlAlgos.includes('dqn')

  // group -> [feature, ...] 로 묶기 (메타 순서 유지)
  const featureGroups = featureMeta.reduce((acc, f) => {
    (acc[f.group] = acc[f.group] || []).push(f)
    return acc
  }, {})

  useEffect(() => () => clearInterval(pollRef.current), [])

  const running = job && (job.status === 'queued' || job.status === 'running')

  async function start() {
    setError(null)
    setResult(null)
    clearInterval(pollRef.current)   // 이전 폴링 잔여분 정리
    try {
      const body = {
        stock_code: stock, engine, algorithm: algo, net,
        train_start: trainStart, train_end: trainEnd,
        test_start: testStart, test_end: testEnd,
        lr: Number(lr), features: selectedFeatures,
        balance: Number(balance),
      }
      if (engine === 'hanium') {
        body.episodes = Number(epochs)
        body.window_size = Number(windowSize)
        body.trade_ratio = Number(tradeRatio)
        body.reward_options = {
          sell_profit_bonus: Number(rewardOpts.sell_profit_bonus) || 0,
          loss_sell_penalty: Number(rewardOpts.loss_sell_penalty) || 0,
          trade_penalty: Number(rewardOpts.trade_penalty) || 0,
          mdd_penalty: Number(rewardOpts.mdd_penalty) || 0,
        }
        if (savedModel) body.saved_model = savedModel
      } else {
        body.algorithm = qlAlgos[0]
        body.algorithms = qlAlgos          // 2개 이상이면 앙상블
        body.ensemble_method = ensMethod
        body.ensemble_window = Number(ensWindow)
        body.num_epoches = Number(epochs)
        body.min_trading_price = Number(minTp)
        if (qlAlgos.includes('dqn')) body.dqn_options = dqnOpts
      }
      const { job_id } = await api.train(body)
      setJob({ status: 'queued', phase: 'queued', progress: 0 })
      pollRef.current = setInterval(async () => {
        try {
          const j = await api.job(job_id)
          setJob(j)
          if (j.status === 'done') {
            clearInterval(pollRef.current)
            setResult(j.result)
            // 새 모델이 저장됐을 수 있으므로 목록 갱신
            if (engine === 'hanium') api.models().then(setSavedModels).catch(() => {})
          } else if (j.status === 'error') {
            clearInterval(pollRef.current)
            setError(j.error || '학습 중 오류가 발생했습니다.')
          }
        } catch (e) {
          clearInterval(pollRef.current)
          setError(e.message)
        }
      }, 1000)
    } catch (e) {
      setError(e.message)
    }
  }

  const progressPct = Math.round((job?.progress || 0) * 100)

  return (
    <div className="app">
      <div className="header">
        <h1>강화학습 AI 트레이딩 프로토타입</h1>
        <p>국내 개별주 · 2개 엔진(quantylab/hanium) · 학습/테스트 기간을 설정해 단순보유 대비 성과를 비교합니다.</p>
      </div>

      {health && !health.kis_configured && (
        <div className="banner">
          ⚠ KIS API 키가 설정되지 않았습니다. <code>backend/.env</code> 에
          KIS_APP_KEY / KIS_APP_SECRET 를 넣고 백엔드를 재시작하세요.
        </div>
      )}

      <div className="panel">
        <h2>① 종목 선택</h2>
        <div className="row">
          {stocks.map((s) => (
            <button key={s.code}
              className={`btn stock ${stock === s.code ? 'active' : ''}`}
              onClick={() => setStock(s.code)}>
              {s.name} <span className="sub">{s.tag} · {s.code}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="panel">
        <h2>② 엔진</h2>
        <div className="row">
          {Object.entries(engines).map(([id, m]) => (
            <button key={id}
              className={`btn algo ${engine === id ? 'active' : ''}`}
              onClick={() => switchEngine(id)}>
              {m.label} <span className="sub">{m.desc}</span>
            </button>
          ))}
          {Object.keys(engines).length === 0 && (
            <span className="note">엔진 목록을 불러오는 중…</span>
          )}
        </div>
      </div>

      <div className="panel">
        <h2>③ 알고리즘 · 네트워크</h2>
        <div className="feat-group-title">
          알고리즘 ({algoList.length}종)
          {engine === 'quantylab' && ' — 복수 선택 시 다수결 앙상블'}
        </div>
        <div className="row">
          {algoList.map((a) => (
            <button key={a.id}
              className={`btn algo ${(engine === 'quantylab'
                ? qlAlgos.includes(a.id) : algo === a.id) ? 'active' : ''}`}
              onClick={() => {
                if (engine === 'quantylab') toggleQlAlgo(a.id)
                else { setAlgo(a.id); if (a.net_default) setNet(a.net_default) }
              }}>
              {a.label} <span className="sub">{a.desc}</span>
            </button>
          ))}
        </div>
        {qlEnsemble && (
          <>
            <div className="feat-group-title" style={{ marginTop: 14 }}>앙상블 방식</div>
            <div className="row">
              <button className={`btn small ${ensMethod === 'sharpe' ? 'active' : ''}`}
                onClick={() => setEnsMethod('sharpe')}>
                Sharpe 선택 <span className="sub">Yang et al. (ICAIF 2020) — 권장</span>
              </button>
              <button className={`btn small ${ensMethod === 'vote' ? 'active' : ''}`}
                onClick={() => setEnsMethod('vote')}>
                다수결 투표 <span className="sub">매 스텝 과반 행동 (단순 베이스라인)</span>
              </button>
            </div>
            <p className="note" style={{ marginTop: 8 }}>
              {ensMethod === 'sharpe'
                ? `앙상블(논문 방식): ${qlAlgos.map((x) => x.toUpperCase()).join(' + ')} 를 각각 학습한 뒤,
                   학습 구간 끝을 검증 구간으로 분리해 Sharpe 최고 모델로 시작하고,
                   테스트 중 ${ensWindow}거래일마다 최근 성과(그림자 포트폴리오 Sharpe) 최고 모델로 교체합니다.
                   변동성(turbulence) 급등 시 전량 매도 후 매수를 중단하는 위험회피 규칙이 적용됩니다.`
                : `앙상블(다수결): 매 스텝 ${qlAlgos.map((x) => x.toUpperCase()).join(' + ')} 의
                   행동을 투표해 과반(동률이면 신뢰도 합) 행동을 실행합니다.`}
              {' '}네트워크는 알고리즘별 기본값(DQN→DNN, A2C/PPO→LSTM)을 사용합니다.
            </p>
          </>
        )}
        {!qlEnsemble && (
          <>
            <div className="feat-group-title" style={{ marginTop: 14 }}>
              네트워크 ({netList.length}종) — 특성 추출기
            </div>
            <div className="row">
              {netList.map((nw) => (
                <button key={nw.id}
                  className={`btn small ${net === nw.id ? 'active' : ''}`}
                  onClick={() => setNet(nw.id)}>
                  {nw.label} <span className="sub">{nw.sub}</span>
                </button>
              ))}
            </div>
          </>
        )}
        {engine === 'hanium' && (
          <>
            <p className="note" style={{ marginTop: 10 }}>
              hanium 엔진은 알고리즘 10종 × 네트워크 8종을 자유롭게 조합해 실험할 수 있습니다.
              (거래세 0.25% · 수수료 · 액션 마스킹이 환경에 이식되어 있습니다.)
            </p>
            <div className="feat-group-title" style={{ marginTop: 14 }}>
              저장된 학습 모델 재사용 ({savedModels.length}개 저장됨)
            </div>
            <select value={savedModel}
              onChange={(e) => setSavedModel(e.target.value)}
              style={{ width: '100%', padding: '8px 10px', marginTop: 6,
                       background: '#1e222b', color: '#e6e9ef',
                       border: '1px solid #2a2f3a', borderRadius: 8 }}>
              <option value="">새로 학습 (학습 완료 후 자동 저장)</option>
              {savedModels.map((m) => (
                <option key={m.model_id} value={m.model_id}>
                  {m.model_id}
                  {m.metrics?.model != null
                    ? ` — 수익률 ${(m.metrics.model.cumulative_return * 100).toFixed(1)}% · MDD ${(m.metrics.model.mdd * 100).toFixed(1)}%`
                    : ''}
                </option>
              ))}
            </select>
            {savedModel && (
              <p className="note" style={{ marginTop: 8 }}>
                저장 모델 재사용: 학습을 건너뛰고 백테스트만 수행합니다.
                알고리즘·네트워크·State 지표·윈도우는 저장 시점 설정으로 자동 적용됩니다.
              </p>
            )}
          </>
        )}
        {showDqnOpts && (
          <>
            <div className="feat-group-title" style={{ marginTop: 14 }}>
              DQN 개선 기법 ({Object.values(dqnOpts).filter(Boolean).length}개 적용) — 강의자료 기반
            </div>
            <div className="feat-list">
              {DQN_IMPROVEMENTS.map((imp) => {
                const blocked = imp.requires && !dqnOpts[imp.requires]
                return (
                  <label key={imp.id}
                    className={`feat-chip ${dqnOpts[imp.id] ? 'on' : ''}`}
                    style={blocked ? { opacity: 0.4, cursor: 'not-allowed' } : undefined}
                    title={imp.sub + (imp.requires ? ` (요구: ${
                      DQN_IMPROVEMENTS.find((x) => x.id === imp.requires)?.label})` : '')}>
                    <input type="checkbox"
                      checked={dqnOpts[imp.id]}
                      disabled={blocked}
                      onChange={() => toggleDqnOpt(imp.id)} />
                    {imp.label}
                  </label>
                )
              })}
            </div>
            <p className="note" style={{ marginTop: 8 }}>
              {dqnOpts.td
                ? '체크된 기법이 적용된 step 단위 Q-learning 으로 학습합니다. Double은 Target, Prioritized는 Replay가 먼저 필요합니다.'
                : 'TD 학습을 끄면 기존 방식(에포크 종료 후 누적수익 회귀)으로 학습하며 Replay·Target 등은 적용되지 않습니다. (Dueling은 구조 옵션이라 단독 적용 가능)'}
            </p>
          </>
        )}
      </div>

      <div className="panel">
        <div className="feat-head">
          <h2>④ State 지표 선택</h2>
          <div className="feat-actions">
            <span className="feat-count">{selectedFeatures.length} / {featureMeta.length} 선택</span>
            <button className="btn small"
              onClick={() => setSelectedFeatures(featureMeta.map((f) => f.id))}>전체 선택</button>
            <button className="btn small"
              onClick={() => setSelectedFeatures([])}>전체 해제</button>
          </div>
        </div>
        <p className="note" style={{ marginTop: 0 }}>
          체크된 지표만 학습 state(관측값)로 사용됩니다. 최소 1개 이상 선택하세요.
        </p>
        {Object.entries(featureGroups).map(([group, items]) => (
          <div key={group} className="feat-group">
            <div className="feat-group-title">{group}</div>
            <div className="feat-list">
              {items.map((f) => (
                <label key={f.id}
                  className={`feat-chip ${selectedFeatures.includes(f.id) ? 'on' : ''}`}>
                  <input type="checkbox"
                    checked={selectedFeatures.includes(f.id)}
                    onChange={() => toggleFeature(f.id)} />
                  {f.label}
                </label>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="panel">
        <h2>⑤ 기간 설정</h2>
        <div className="grid-dates">
          <div>
            <div className="period-tag">학습 기간 (Train)</div>
            <div className="sub-grid">
              <div className="field"><label>시작</label>
                <input type="date" value={trainStart} onChange={(e) => setTrainStart(e.target.value)} /></div>
              <div className="field"><label>종료</label>
                <input type="date" value={trainEnd} onChange={(e) => setTrainEnd(e.target.value)} /></div>
            </div>
          </div>
          <div>
            <div className="period-tag">테스트 기간 (Test)</div>
            <div className="sub-grid">
              <div className="field"><label>시작</label>
                <input type="date" value={testStart} onChange={(e) => setTestStart(e.target.value)} /></div>
              <div className="field"><label>종료</label>
                <input type="date" value={testEnd} onChange={(e) => setTestEnd(e.target.value)} /></div>
            </div>
          </div>
        </div>
        <button className="toggle-adv" onClick={() => setShowAdv(!showAdv)}>
          {showAdv ? '− 고급 설정 숨기기'
            : `+ 고급 설정 (${paramKind === 'episodes' ? '에피소드/윈도우/초기자본/' : '에폭/초기자본/매매금액/'}학습률)`}
        </button>
        {showAdv && (
          <div className="sub-grid" style={{ marginTop: 10 }}>
            <div className="field">
              <label>{paramKind === 'episodes' ? '학습 에피소드 수 (episodes)' : '학습 에폭 수 (num_epoches)'}</label>
              <input type="number" min="1" max="2000" value={epochs}
                onChange={(e) => setEpochs(e.target.value)} /></div>
            <div className="field"><label>학습률 (lr)</label>
              <input type="number" step="0.0001" value={lr}
                onChange={(e) => setLr(e.target.value)} /></div>
            <div className="field"><label>초기자본 (원)</label>
              <input type="number" step="1000000" min="100000" value={balance}
                onChange={(e) => setBalance(e.target.value)} /></div>
            {qlEnsemble && ensMethod === 'sharpe' && (
              <div className="field"><label>앙상블 리더 재선정 주기 (거래일)</label>
                <input type="number" min="5" max="126" value={ensWindow}
                  onChange={(e) => setEnsWindow(e.target.value)} /></div>
            )}
            {engine === 'quantylab' && (
              <>
                <div className="field"><label>1회 최소 매매 금액 (원) — 최대는 잔고 전액</label>
                  <input type="number" step="100000" min="10000" value={minTp}
                    onChange={(e) => setMinTp(e.target.value)} /></div>
              </>
            )}
            {engine === 'hanium' && (
              <>
                <div className="field"><label>관측 윈도우 (window_size, 일)</label>
                  <input type="number" min="5" max="120" value={windowSize}
                    onChange={(e) => setWindowSize(e.target.value)} /></div>
                <div className="field"><label>매수 비율 (trade_ratio, 0~1)</label>
                  <input type="number" step="0.1" min="0.1" max="1" value={tradeRatio}
                    onChange={(e) => setTradeRatio(e.target.value)} /></div>
                <div className="field"><label>실현수익 보너스 계수</label>
                  <input type="number" step="0.1" min="0" value={rewardOpts.sell_profit_bonus}
                    onChange={(e) => setRewardOpts({ ...rewardOpts, sell_profit_bonus: e.target.value })} /></div>
                <div className="field"><label>손실매도 패널티 계수</label>
                  <input type="number" step="0.1" min="0" value={rewardOpts.loss_sell_penalty}
                    onChange={(e) => setRewardOpts({ ...rewardOpts, loss_sell_penalty: e.target.value })} /></div>
                <div className="field"><label>거래 패널티 (체결당, 예: 0.0005)</label>
                  <input type="number" step="0.0001" min="0" value={rewardOpts.trade_penalty}
                    onChange={(e) => setRewardOpts({ ...rewardOpts, trade_penalty: e.target.value })} /></div>
                <div className="field"><label>MDD 패널티 계수 (예: 0.3)</label>
                  <input type="number" step="0.1" min="0" value={rewardOpts.mdd_penalty}
                    onChange={(e) => setRewardOpts({ ...rewardOpts, mdd_penalty: e.target.value })} /></div>
              </>
            )}
          </div>
        )}
        <div className="note">
          학습 구간 이전 약 250일은 지표(이동평균·RSI 등) 워밍업에 자동 사용됩니다.
          테스트는 탐험 없이(ε=0) 평가하며, 동일 구간 단순보유와 비교합니다.
        </div>
      </div>

      <div className="panel">
        <button className="run-btn" onClick={start}
          disabled={running || (health && !health.kis_configured) || selectedFeatures.length === 0}>
          {running ? `${PHASE_LABEL[job.phase] || '진행 중'}…`
            : selectedFeatures.length === 0 ? '▶ State 지표를 1개 이상 선택하세요'
            : '▶ 학습 + 백테스트 실행'}
        </button>
        {job && running && (
          <div style={{ marginTop: 14 }}>
            <div className="progress-wrap">
              <div className="progress-bar" style={{ width: `${progressPct}%` }} />
            </div>
            <div className="progress-meta">
              <span>{PHASE_LABEL[job.phase] || job.phase}
                {job.phase === 'training' && job.total ? ` (${job.step}/${job.total} 회차)` : ''}</span>
              <span>{progressPct}%</span>
            </div>
          </div>
        )}
        {error && <div className="error" style={{ marginTop: 14 }}>{error}</div>}
      </div>

      {result && (
        <>
          <div className="panel">
            <h2>성과 지표 (테스트 구간)</h2>
            <MetricsTable metrics={result.metrics} trades={result.trades} />
            <div className="note">
              {result.stock_code} · {result.algorithm.toUpperCase()} ({result.net.toUpperCase()})
              {result.ensemble && (result.ensemble.method === 'sharpe_selection'
                ? ` · 앙상블(Sharpe 선택, 시작 ${result.ensemble.initial_leader?.toUpperCase()})`
                : ' · 앙상블(다수결)')}
              · 학습 {result.n_train}일 → 테스트 {result.n_test}일
              · 초기자본 {result.initial_balance.toLocaleString()}원
              {result.dqn_options && (
                <> · DQN 기법: {
                  DQN_IMPROVEMENTS.filter((i) => result.dqn_options[i.id])
                    .map((i) => i.label).join(', ') || '미적용(기존 방식)'
                }</>
              )}
              {result.model_id && (
                <> · 모델: {result.model_id}{result.saved_model_used ? ' (재사용)' : ' (신규 저장)'}</>
              )}
            </div>
          </div>
          <div className="panel">
            <h2>포트폴리오 가치 추이</h2>
            <EquityChart series={result.series} initial={result.initial_balance} />
          </div>
          <div className="panel">
            <h2>매수 / 매도 시점 (종가 기준)</h2>
            <PriceTradeChart series={result.series} tradeLog={result.trade_log} />
          </div>
          <div className="panel">
            <h2>거래 내역</h2>
            <TradeLog tradeLog={result.trade_log} summary={result.trade_summary}
              initialBalance={result.initial_balance} />
          </div>
        </>
      )}
    </div>
  )
}
