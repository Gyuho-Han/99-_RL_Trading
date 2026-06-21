import React from 'react'

const fmtDate = (d) => (d ? `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}` : '')
const won = (v) => `${Math.round(v).toLocaleString()}원`
const pct = (v) => `${(v * 100).toFixed(2)}%`

export default function TradeLog({ tradeLog, summary, initialBalance }) {
  if (!tradeLog || tradeLog.length === 0) {
    return <div className="note">이 구간에서 체결된 거래가 없습니다.</div>
  }
  const realizedRet = summary.total_realized_profit / initialBalance
  return (
    <>
      <div className="chips" style={{ marginBottom: 14 }}>
        <span className="chip">총 거래 {summary.num_trades}건</span>
        <span className="chip">총 거래금액 {won(summary.total_traded_amount)}</span>
        <span className="chip">
          실현손익{' '}
          <b className={summary.total_realized_profit >= 0 ? 'pos' : 'neg'}>
            {summary.total_realized_profit >= 0 ? '+' : ''}{won(summary.total_realized_profit)}
            {' '}({pct(realizedRet)})
          </b>
        </span>
      </div>
      <div style={{ maxHeight: 360, overflowY: 'auto' }}>
        <table className="metrics-table trade-table">
          <thead>
            <tr>
              <th>날짜</th><th>구분</th><th>주수</th><th>체결가</th>
              <th>거래금액</th><th>수익금</th><th>수익률</th>
            </tr>
          </thead>
          <tbody>
            {tradeLog.map((t, i) => (
              <tr key={i}>
                <td>{fmtDate(t.date)}</td>
                <td>
                  <span className={t.side === 'buy' ? 'tag-buy' : 'tag-sell'}>
                    {t.side === 'buy' ? '매수' : '매도'}
                  </span>
                </td>
                <td>{t.shares.toLocaleString()}</td>
                <td>{won(t.price)}</td>
                <td>{won(t.amount)}</td>
                <td className={t.side === 'sell' ? (t.profit >= 0 ? 'pos' : 'neg') : ''}>
                  {t.side === 'sell' ? `${t.profit >= 0 ? '+' : ''}${won(t.profit)}` : '-'}
                </td>
                <td className={t.side === 'sell' ? (t.return >= 0 ? 'pos' : 'neg') : ''}>
                  {t.side === 'sell' ? `${t.return >= 0 ? '+' : ''}${pct(t.return)}` : '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="note">
        수익금/수익률은 평균매입원가(매수 수수료 포함) 대비 매도 체결가(매도 수수료·거래세 차감) 기준
        실현손익입니다. 매수 행의 수익은 아직 미실현이라 '-'로 표시됩니다.
      </div>
    </>
  )
}
