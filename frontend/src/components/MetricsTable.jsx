import React from 'react'

const pct = (v) => `${(v * 100).toFixed(2)}%`
const cls = (v) => (v >= 0 ? 'pos' : 'neg')

export default function MetricsTable({ metrics, trades }) {
  const m = metrics.model
  const b = metrics.buyhold
  const rows = [
    ['누적수익률', pct(m.cumulative_return), pct(b.cumulative_return), true],
    ['연환산 수익률', pct(m.annual_return), pct(b.annual_return), true],
    ['연환산 변동성', pct(m.annual_vol), pct(b.annual_vol), false],
    ['Sharpe', m.sharpe.toFixed(2), b.sharpe.toFixed(2), true],
    ['최대낙폭(MDD)', pct(m.mdd), pct(b.mdd), true],
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
