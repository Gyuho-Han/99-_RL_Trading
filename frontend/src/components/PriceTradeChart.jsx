import React from 'react'
import {
  ResponsiveContainer, ComposedChart, Line, Scatter, XAxis, YAxis,
  Tooltip, CartesianGrid, Legend,
} from 'recharts'

const fmtDate = (d) => (d ? `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}` : '')

export default function PriceTradeChart({ series, tradeLog }) {
  const buyByDate = {}
  const sellByDate = {}
  for (const t of tradeLog) {
    if (t.side === 'buy') buyByDate[t.date] = t.price
    else sellByDate[t.date] = t.price
  }
  const data = series.dates.map((d, i) => ({
    date: fmtDate(d),
    close: Math.round(series.close[i]),
    buy: buyByDate[d] != null ? Math.round(buyByDate[d]) : null,
    sell: sellByDate[d] != null ? Math.round(sellByDate[d]) : null,
  }))
  const n = data.length
  const tickStep = Math.max(1, Math.floor(n / 6))
  return (
    <div style={{ width: '100%', height: 320 }}>
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
          <CartesianGrid stroke="#262b35" />
          <XAxis dataKey="date" stroke="#9aa3b2" fontSize={11}
            ticks={data.filter((_, i) => i % tickStep === 0).map((d) => d.date)} />
          <YAxis stroke="#9aa3b2" fontSize={11} domain={['auto', 'auto']}
            tickFormatter={(v) => `${(v / 1000).toFixed(0)}천`} width={50} />
          <Tooltip contentStyle={{ background: '#1e222b', border: '1px solid #2a2f3a', color: '#e6e9ef' }}
            formatter={(v, name) => [v != null ? `${v.toLocaleString()}원` : '-', name]} />
          <Legend />
          <Line type="monotone" dataKey="close" name="종가" stroke="#8a93a6"
            dot={false} strokeWidth={1.6} />
          <Scatter dataKey="buy" name="매수" fill="#2ecc71" shape="triangle" />
          <Scatter dataKey="sell" name="매도" fill="#ff5d5d" shape="diamond" />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
