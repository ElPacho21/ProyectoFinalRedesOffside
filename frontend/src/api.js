const BASE = '/api'

export async function uploadMedia(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getFrame(mediaId, index) {
  const res = await fetch(`${BASE}/frame/${mediaId}?index=${index}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function runDetection(mediaId, frameIndex, confidence) {
  const res = await fetch(`${BASE}/detect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ media_id: mediaId, frame_index: frameIndex, confidence }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function computeVP(points) {
  const res = await fetch(`${BASE}/vanishing-point`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ points }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function analyzeOffside(payload) {
  const res = await fetch(`${BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}
