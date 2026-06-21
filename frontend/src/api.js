const BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

async function j(path, opts) {
  const res = await fetch(`${BASE}${path}`, opts)
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `요청 실패 (${res.status})`)
  return data
}

export const api = {
  health: () => j('/api/health'),
  stocks: () => j('/api/stocks'),
  algorithms: () => j('/api/algorithms'),
  train: (body) =>
    j('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  job: (id) => j(`/api/jobs/${id}`),
}
