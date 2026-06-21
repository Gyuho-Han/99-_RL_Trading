import React from 'react'
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip,
  CartesianGrid, Legend, ScatterChart, Scatter, ZAxis,
} from 'recharts'

const fmtDate = (d) => (d ? `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}` : '')

export default function EquityChart({ series, initial }) {
  const data = series.dates.map((d, i) => ({
    date: fmtDate(d),
    model: Math.round(series.model_pv[i]),
    buyhold: Math.round(series.buyhold_pv[i]),
  }))
  const n = data.length
  const tickStep = Math.max(1, Math.floor(n / 6))
  return (
    <div style={{ width: '100%', height: 320 }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
          <CartesianGrid stroke="#262b35" />
          <XAxis dataKey="date" stroke="#9aa3b2" fontSize={11}
            ticks={data.filter((_, i) => i % tickStep === 0).map((d) => d.date)} />
          <YAxis stroke="#9aa3b2" fontSize={11}
            tickFormatter={(v) => `${(v / 10000).toFixed(0)}만`} width={50} />
          <Tooltip contentStyle={{ background: '#1e222b', border: '1px solid #2a2f3a', color: '#e6e9ef' }}
            formatter={(v) => `${v.toLocaleString()}원`} />
          <Legend />
          <Line type="monotone" dataKey="model" name="강화학습 모델" stroke="#4f8cff"
            dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="buyhold" name="단순보유(B&H)" stroke="#00c2a8"
            dot={false} strokeWidth={2} strokeDasharray="5 4" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
