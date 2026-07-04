import React from 'react'

const pct = (v) => `${(v * 100).toFixed(2)}%`
const cls = (v) => (v >= 0 ? 'pos' : 'neg')

const orDash = (v, fmt) => (v == null ? '-' : fmt(v))

export default function MetricsTable({ metrics, trades }) {
  const m = metrics.model
  const b = metrics.buyhold
  const rows = [
    ['누적수익률', pct(m.cumulative_return), pct(b.cumulative_return), true],
    ['연환산 수익률', pct(m.annual_return), pct(b.annual_return), true],
    ['연환산 변동성', pct(m.annual_vol), pct(b.annual_vol), false],
    ['Sharpe', m.sharpe.toFixed(2), b.sharpe.toFixed(2), true],
    ['최대낙폭(MDD)', pct(m.mdd), pct(b.mdd), true],
    // 확장 지표 (거래 품질 — B&H 는 거래가 없어 해당 없음)
    ['승률 (익절 매도 비율)', orDash(m.win_rate, pct), '-', false],
    ['손익비 (평균이익/평균손실)', orDash(m.profit_factor, (v) => v.toFixed(2)), '-', false],
    ['월별 양수 수익률 비율', orDash(m.monthly_win_rate, pct),
      orDash(b?.monthly_win_rate, pct), false],
    ['월평균 거래횟수', orDash(m.trades_per_month, (v) => `${v.toFixed(1)}회`), '-', false],
  ]
  return (
    <>
      <table className="metrics-table">
        <thead>
          <tr><th>지표</th><th>강화학습 모델</th><th>단순보유(B&H)</th></tr>
        </thead>
        <tbody>
          {rows.map(([label, mv, bv, color]) => (
            <tr key={label}>
              <td>{label}</td>
              <td className={color ? cls(parseFloat(mv)) : ''}>{mv}</td>
              <td className={color ? cls(parseFloat(bv)) : ''}>{bv}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="chips" style={{ marginTop: 14 }}>
        <span className="chip">매수 {trades.buy}회</span>
        <span className="chip">매도 {trades.sell}회</span>
        <span className="chip">관망 {trades.hold}회</span>
      </div>
    </>
  )
}
