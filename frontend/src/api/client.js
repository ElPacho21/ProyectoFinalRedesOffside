import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 120000,
})

api.interceptors.request.use((config) => {
  config._startTime = performance.now()
  console.log(`→ ${config.method?.toUpperCase()} ${config.url}`)
  return config
})

api.interceptors.response.use(
  (response) => {
    const ms = (performance.now() - response.config._startTime).toFixed(1)
    console.log(`← ${response.config.method?.toUpperCase()} ${response.config.url} ${response.status} (${ms}ms)`)
    return response
  },
  (error) => {
    const cfg = error.config || {}
    const ms = cfg._startTime ? (performance.now() - cfg._startTime).toFixed(1) : '?'
    const status = error.response?.status ?? 'ERR'
    console.error(`← ${cfg.method?.toUpperCase()} ${cfg.url} ${status} (${ms}ms)`, error.message)
    return Promise.reject(error)
  },
)

export function b64toBlob(b64Data, contentType = 'image/jpeg') {
  const byteChars = atob(b64Data)
  const byteArrays = []
  for (let offset = 0; offset < byteChars.length; offset += 512) {
    const slice = byteChars.slice(offset, offset + 512)
    const byteNums = new Array(slice.length)
    for (let i = 0; i < slice.length; i++) {
      byteNums[i] = slice.charCodeAt(i)
    }
    byteArrays.push(new Uint8Array(byteNums))
  }
  return new Blob(byteArrays, { type: contentType })
}

export default api
