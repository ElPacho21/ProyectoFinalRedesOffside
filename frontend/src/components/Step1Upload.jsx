import { useRef, useState, useCallback } from 'react'
import {
  IconCloudUpload, IconMovie, IconX, IconSearch,
  IconChevronLeft, IconChevronRight, IconPhoto, IconLoader2,
} from '@tabler/icons-react'
import { b64toBlob } from '../api/client'

const IMAGE_EXT = ['image/jpeg', 'image/png', 'image/webp']
const VIDEO_EXT = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo']
const ACCEPTED  = '.jpg,.jpeg,.png,.webp,.mp4,.avi,.mov,.mkv,.m4v'

function formatTime(frame, fps) {
  const secs = Math.floor(frame / (fps || 30))
  return `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, '0')}`
}

export default function Step1Upload({ loading, onDetect, onExtractFrame }) {
  const inputRef    = useRef(null)
  const debounceRef = useRef(null)

  const [file,         setFile]         = useState(null)
  const [isVideo,      setIsVideo]      = useState(false)
  const [preview,      setPreview]      = useState(null)
  const [frameNumber,  setFrameNumber]  = useState(0)
  const [totalFrames,  setTotalFrames]  = useState(0)
  const [fps,          setFps]          = useState(30)
  const [frameLoading, setFrameLoading] = useState(false)
  const [dragging,     setDragging]     = useState(false)

  const handleFile = useCallback(async (f) => {
    if (!f) return
    const video = VIDEO_EXT.includes(f.type) || f.type.startsWith('video/')
    const image = IMAGE_EXT.includes(f.type) || f.type.startsWith('image/')
    if (!video && !image) return

    setFile(f)
    setIsVideo(video)
    setFrameNumber(0)

    if (image) {
      setPreview(URL.createObjectURL(f))
    } else {
      setFrameLoading(true)
      try {
        const data = await onExtractFrame(f, 0)
        setTotalFrames(data.total_frames)
        setFps(data.fps)
        setPreview(`data:image/jpeg;base64,${data.image_b64}`)
      } finally {
        setFrameLoading(false)
      }
    }
  }, [onExtractFrame])

  const handleFrameSeek = (n) => {
    setFrameNumber(n)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(async () => {
      setFrameLoading(true)
      try {
        const data = await onExtractFrame(file, n)
        setPreview(`data:image/jpeg;base64,${data.image_b64}`)
      } finally {
        setFrameLoading(false)
      }
    }, 200)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files?.[0])
  }

  const handleAnalyze = async () => {
    let imageFile = file
    if (isVideo && preview) {
      const b64  = preview.replace('data:image/jpeg;base64,', '')
      const blob = b64toBlob(b64)
      imageFile  = new File([blob], 'frame.jpg', { type: 'image/jpeg' })
    }
    await onDetect(imageFile)
  }

  const removeFile = () => { setFile(null); setPreview(null); setIsVideo(false) }

  const progressPct = totalFrames > 1 ? (frameNumber / (totalFrames - 1)) * 100 : 0
  const trackBg     = (pct) => `linear-gradient(to right, #22c55e ${pct}%, #1a2540 ${pct}%)`

  return (
    <div className="space-y-4">

      {/* ── Drop zone ─────────────────────────────────────────── */}
      {!file ? (
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className="cursor-pointer rounded-2xl border-2 border-dashed p-20 flex flex-col items-center gap-5 transition-all duration-200"
          style={{
            borderColor: dragging ? '#22c55e' : '#1a2540',
            background:  dragging ? 'rgba(34,197,94,0.05)' : 'rgba(11,17,32,0.5)',
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED}
            className="hidden"
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
          <div
            className="w-20 h-20 rounded-2xl flex items-center justify-center"
            style={{
              background: 'rgba(34,197,94,0.08)',
              border: '1px solid rgba(34,197,94,0.2)',
              boxShadow: dragging ? '0 0 32px rgba(34,197,94,0.25)' : 'none',
            }}
          >
            <IconCloudUpload size={36} stroke={1.5} className="text-accent" />
          </div>
          <div className="text-center">
            <p className="font-semibold text-tx-primary text-[17px]">
              Arrastrá o <span className="text-accent">hacé click</span> para subir
            </p>
            <p className="text-tx-muted text-sm mt-2">
              Imágenes: JPG, PNG · Videos: MP4, AVI, MOV, MKV
            </p>
          </div>
        </div>
      ) : (
        /* ── Media card ─────────────────────────────────────────── */
        <div className="rounded-2xl overflow-hidden border border-border-subtle">
          <div className="relative bg-black">
            <img
              src={preview}
              alt="preview"
              className="w-full block object-contain"
              style={{ maxHeight: '72vh' }}
            />

            {frameLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/60">
                <IconLoader2 size={40} stroke={2} className="text-accent animate-spin" />
              </div>
            )}

            {isVideo && (
              <div
                className="absolute top-4 left-4 flex items-center gap-1.5 text-tx-secondary text-xs px-2.5 py-1 rounded-full"
                style={{ background: 'rgba(0,0,0,0.7)', border: '1px solid #1a2540' }}
              >
                <IconMovie size={12} stroke={2} />
                Video
              </div>
            )}

            <button
              onClick={removeFile}
              className="absolute top-4 right-4 w-8 h-8 rounded-full flex items-center justify-center transition-all"
              style={{ background: 'rgba(0,0,0,0.7)', border: '1px solid #1a2540', color: '#8896b0' }}
              onMouseEnter={(e) => { e.currentTarget.style.color = '#e2e8f5'; e.currentTarget.style.borderColor = '#f43f5e' }}
              onMouseLeave={(e) => { e.currentTarget.style.color = '#8896b0'; e.currentTarget.style.borderColor = '#1a2540' }}
            >
              <IconX size={14} stroke={2.5} />
            </button>

            {/* YouTube-style controls */}
            {isVideo && totalFrames > 0 && (
              <div
                className="absolute inset-x-0 bottom-0 px-5 pb-4 pt-12"
                style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.96) 0%, transparent 100%)' }}
              >
                <input
                  type="range"
                  min={0}
                  max={totalFrames - 1}
                  value={frameNumber}
                  onChange={(e) => handleFrameSeek(Number(e.target.value))}
                  className="video-range w-full mb-3"
                  style={{ background: trackBg(progressPct) }}
                />

                <div className="flex items-center gap-3">
                  <button
                    onClick={() => handleFrameSeek(Math.max(0, frameNumber - 1))}
                    className="w-8 h-8 rounded-full flex items-center justify-center text-white transition-colors"
                    style={{ background: 'rgba(255,255,255,0.12)' }}
                    onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.22)'}
                    onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.12)'}
                  >
                    <IconChevronLeft size={16} stroke={2.5} />
                  </button>
                  <button
                    onClick={() => handleFrameSeek(Math.min(totalFrames - 1, frameNumber + 1))}
                    className="w-8 h-8 rounded-full flex items-center justify-center text-white transition-colors"
                    style={{ background: 'rgba(255,255,255,0.12)' }}
                    onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.22)'}
                    onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.12)'}
                  >
                    <IconChevronRight size={16} stroke={2.5} />
                  </button>

                  <span className="text-white text-sm font-mono">
                    {formatTime(frameNumber, fps)}
                    <span className="opacity-40"> / {formatTime(totalFrames - 1, fps)}</span>
                  </span>

                  <span className="ml-auto text-white/40 text-xs font-mono tabular-nums">
                    Frame {frameNumber.toLocaleString()} / {(totalFrames - 1).toLocaleString()}
                  </span>
                </div>
              </div>
            )}
          </div>

          <div className="px-4 py-2.5 bg-bg-elevated border-t border-border-subtle flex items-center gap-2">
            <IconPhoto size={14} stroke={2} className="text-tx-muted shrink-0" />
            <span className="text-xs text-tx-muted truncate">{file.name}</span>
          </div>
        </div>
      )}

      {/* ── CTA ───────────────────────────────────────────────── */}
      <button
        onClick={handleAnalyze}
        disabled={!file || loading || frameLoading}
        className="btn-primary w-full py-4 text-[15px] flex items-center justify-center gap-2.5"
      >
        {loading ? (
          <>
            <IconLoader2 size={16} stroke={2} className="animate-spin" />
            Analizando imagen…
          </>
        ) : (
          <>
            <IconSearch size={16} stroke={2.5} />
            Analizar imagen
          </>
        )}
      </button>
    </div>
  )
}
