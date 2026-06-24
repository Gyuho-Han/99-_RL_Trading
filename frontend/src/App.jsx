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

export default function App() {
  const [stocks, setStocks] = useState([])
  const [algos, setAlgos] = useState([])
  const [health, setHealth] = useState(null)

  const [engines, setEngines] = useState({})
  const [engine, setEngine] = useState('quantylab')

  const [stock, setStock] = useState('005930')
  const [algo, setAlgo] = useState('dqn')
  const [net, setNet] = useState('dnn')
  const [windowSize, setWindowSize] = useState(20)
  const [trainStart, setTrainStart] = useState('2022-01-01')
  const [trainEnd, setTrainEnd] = useState('2024-06-30')
  const [testStart, setTestStart] = useState('2024-07-01')
  const [testEnd, setTestEnd] = useState('2025-06-30')
  const [epochs, setEpochs] = useState(80)
  const [lr, setLr] = useState(0.0005)
  const [showAdv, setShowAdv] = useState(false)

  const [featureMeta, setFeatureMeta] = useState([])
  const [selectedFeatures, setSelectedFeatures] = useState([])

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
    try {
      const body = {
        stock_code: stock, engine, algorithm: algo, net,
        train_start: trainStart, train_end: trainEnd,
        test_start: testStart, test_end: testEnd,
        lr: Number(lr), features: selectedFeatures,
      }
      if (engine === 'hanium') {
        body.episodes = Number(epochs)
        body.window_size = Number(windowSize)
      } else {
        body.num_epoches = Number(epochs)
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
        <div className="feat-group-title">알고리즘 ({algoList.length}종)</div>
        <div className="row">
          {algoList.map((a) => (
            <button key={a.id}
              className={`btn algo ${algo === a.id ? 'active' : ''}`}
              onClick={() => { setAlgo(a.id); if (a.net_default) setNet(a.net_default) }}>
              {a.label} <span className="sub">{a.desc}</span>
            </button>
          ))}
        </div>
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
        {engine === 'hanium' && (
          <p className="note" style={{ marginTop: 10 }}>
            hanium 엔진은 알고리즘 10종 × 네트워크 8종을 자유롭게 조합해 실험할 수 있습니다.
            (거래세 0.25% · 수수료 · 액션 마스킹이 환경에 이식되어 있습니다.)
          </p>
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
            : `+ 고급 설정 (${paramKind === 'episodes' ? '에피소드/윈도우/' : '에폭/'}학습률)`}
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
            {engine === 'hanium' && (
              <div className="field"><label>관측 윈도우 (window_size, 일)</label>
                <input type="number" min="5" max="120" value={windowSize}
                  onChange={(e) => setWindowSize(e.target.value)} /></div>
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
              · 학습 {result.n_train}일 → 테스트 {result.n_test}일
              · 초기자본 {result.initial_balance.toLocaleString()}원
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
