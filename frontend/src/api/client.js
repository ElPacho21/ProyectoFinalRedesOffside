import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 120000,
})

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
