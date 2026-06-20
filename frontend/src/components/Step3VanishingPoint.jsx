import { useState } from 'react'
import { computeVP } from '../api.js'
import CanvasVP from './CanvasVP.jsx'

export default function Step3VanishingPoint({ mediaInfo, currentFrameBase64, onDone, onBack }) {
  const [points, setPoints] = useState([])
  const [vp, setVp] = useState(null)
  const [vpError, setVpError] = useState(null)
  const [loading, setLoading] = useState(false)

  const handlePointsChange = (pts) => {
    setPoints(pts)
    setVp(null)
    setVpError(null)
  }

  const handleCompute = async () => {
    if (points.length !== 4) return
    setLoading(true)
    setVpError(null)
    try {
      const data = await computeVP(points)
      if (data.vp) {
        setVp(data.vp)
      } else {
        setVpError(data.error ?? 'Lines are parallel — choose different points')
      }
    } catch (e) {
      setVpError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const imageBase64 = currentFrameBase64
  const { width, height } = mediaInfo

  return (
    <div>
      <div className="card">
        <div className="card-title">Vanishing point selection</div>

        <div className="alert alert-info" style={{ marginBottom: 16 }}>
          ℹ The vanishing point (VP) corrects for camera perspective. Select 2 parallel field lines
          using 4 clicks: points 1-2 on line A, points 3-4 on line B.
        </div>

        <CanvasVP
          imageBase64={imageBase64}
          imageWidth={width}
          imageHeight={height}
          onPointsChange={handlePointsChange}
        />

        {points.length === 4 && !vp && (
          <button
            className="btn btn-primary"
            style={{ marginTop: 12 }}
            disabled={loading}
            onClick={handleCompute}
          >
            {loading ? 'Computing...' : 'Compute vanishing point'}
          </button>
        )}

        {vpError && <div className="alert alert-error" style={{ marginTop: 10 }}>⚠ {vpError}</div>}

        {vp && (
          <div className="alert alert-success" style={{ marginTop: 10 }}>
            ✓ Vanishing point: ({vp[0].toFixed(0)}, {vp[1].toFixed(0)})
            {(vp[0] < 0 || vp[0] > width || vp[1] < 0 || vp[1] > height) && (
              <span style={{ marginLeft: 6, fontSize: 11, opacity: 0.8 }}>
                (outside image — this is normal for shallow angles)
              </span>
            )}
          </div>
        )}
      </div>

      <div className="nav-row">
        <button className="btn btn-secondary" onClick={onBack}>← Back</button>
        <button
          className="btn btn-primary btn-lg"
          disabled={!vp}
          onClick={() => onDone({ vp })}
        >
          Select attacking team →
        </button>
      </div>
    </div>
  )
}
