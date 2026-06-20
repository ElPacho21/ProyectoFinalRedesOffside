import { useState, useRef, useCallback } from 'react'
import { uploadMedia, getFrame } from '../api.js'

export default function Step1Upload({ onDone }) {
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [mediaInfo, setMediaInfo] = useState(null)
  const [frameIndex, setFrameIndex] = useState(0)
  const [frameData, setFrameData] = useState(null)
  const [confidence, setConfidence] = useState(0.25)
  const fileRef = useRef()

  const handleFile = useCallback(async (file) => {
    setError(null)
    setLoading(true)
    try {
      const info = await uploadMedia(file)
      setMediaInfo(info)
      setFrameIndex(0)
      setFrameData(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  const fetchFrame = useCallback(async (idx) => {
    if (!mediaInfo?.is_video) return
    setLoading(true)
    try {
      const data = await getFrame(mediaInfo.media_id, idx)
      setFrameData(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [mediaInfo])

  const changeFrame = (newIdx) => {
    const clamped = Math.max(0, Math.min(newIdx, (mediaInfo?.total_frames ?? 1) - 1))
    setFrameIndex(clamped)
    fetchFrame(clamped)
  }

  const currentImage = mediaInfo?.is_video
    ? (frameData?.frame_base64 ?? mediaInfo?.preview_base64)
    : mediaInfo?.preview_base64

  const fps = mediaInfo?.fps ?? 25
  const totalFrames = mediaInfo?.total_frames ?? 1
  const ts = frameIndex / fps
  const tsStr = `${String(Math.floor(ts / 60)).padStart(2, '0')}:${(ts % 60).toFixed(2).padStart(5, '0')}s`

  const canContinue = !!mediaInfo

  return (
    <div>
      <div className="card">
        <div className="card-title">Upload image or video</div>

        {!mediaInfo ? (
          <div
            className={`upload-zone${dragging ? ' dragover' : ''}`}
            onClick={() => fileRef.current.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <div className="upload-icon">⚽</div>
            <p style={{ fontWeight: 600, marginBottom: 4 }}>Drag & drop or <strong>click to browse</strong></p>
            <p className="upload-text">Supported formats: JPG, PNG, MP4, AVI, MOV, MKV</p>
          </div>
        ) : (
          <div>
            <div className="img-wrap">
              {currentImage
                ? <img src={`data:image/jpeg;base64,${currentImage}`} alt="preview" />
                : <div className="spinner-wrap"><div className="spinner" /> Loading frame...</div>
              }
            </div>

            {mediaInfo.is_video && (
              <>
                <div className="video-controls">
                  <button className="btn btn-secondary" onClick={() => changeFrame(frameIndex - 1)}>◀</button>
                  <input
                    type="range"
                    min={0}
                    max={totalFrames - 1}
                    value={frameIndex}
                    onChange={(e) => changeFrame(Number(e.target.value))}
                  />
                  <button className="btn btn-secondary" onClick={() => changeFrame(frameIndex + 1)}>▶</button>
                </div>
                <div className="frame-info">
                  {totalFrames} frames · {fps.toFixed(1)} FPS · Frame {frameIndex}/{totalFrames - 1} · {tsStr}
                </div>
              </>
            )}

            <div style={{ marginTop: 16 }}>
              <label className="label">Confidence threshold</label>
              <div className="slider-wrap">
                <input
                  type="range"
                  min={0.05}
                  max={0.90}
                  step={0.05}
                  value={confidence}
                  onChange={(e) => setConfidence(Number(e.target.value))}
                />
                <span className="slider-val">{(confidence * 100).toFixed(0)}%</span>
              </div>
            </div>

            <button
              className="btn btn-secondary"
              style={{ marginTop: 12 }}
              onClick={() => { setMediaInfo(null); setFrameData(null) }}
            >
              Change file
            </button>
          </div>
        )}

        <input
          ref={fileRef}
          type="file"
          accept=".jpg,.jpeg,.png,.mp4,.avi,.mov,.mkv,.m4v"
          style={{ display: 'none' }}
          onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])}
        />
      </div>

      {loading && (
        <div className="spinner-wrap"><div className="spinner" /> Processing...</div>
      )}
      {error && <div className="alert alert-error">⚠ {error}</div>}

      <div className="nav-row">
        <span />
        <button
          className="btn btn-primary btn-lg"
          disabled={!canContinue || loading}
          onClick={() => onDone({ mediaInfo, frameIndex, confidence })}
        >
          Detect objects →
        </button>
      </div>
    </div>
  )
}
