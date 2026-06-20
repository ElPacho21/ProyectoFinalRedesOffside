import { useState, useEffect } from 'react'
import { runDetection } from '../api.js'

const CLASS_COLORS = {
  'Ball': '#ffffff',
  'GoalKeeper': '#ffd700',
  'Referee': '#ff8c00',
  'TEAM 1': '#ff4444',
  'TEAM 2': '#dddd00',
  'Corner': '#888888',
  'Goal_Net': '#aaaaaa',
}

export default function Step2Detection({ mediaInfo, frameIndex, confidence, onDone, onBack }) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    runDetection(mediaInfo.media_id, frameIndex, confidence)
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [mediaInfo.media_id, frameIndex, confidence])

  if (loading) {
    return (
      <div className="card">
        <div className="spinner-wrap">
          <div className="spinner" />
          Running YOLOv8 detection...
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div>
        <div className="alert alert-error">⚠ {error}</div>
        <button className="btn btn-secondary" onClick={onBack}>← Back</button>
      </div>
    )
  }

  const { detections, image_base64, counts, warnings, suggested_team } = result
  const sortedCounts = Object.entries(counts).sort((a, b) => b[1] - a[1])

  return (
    <div>
      <div className="card">
        <div className="card-title">Detection results</div>
        <div className="grid-2">
          <div className="img-wrap">
            <img src={`data:image/jpeg;base64,${image_base64}`} alt="detections" />
          </div>
          <div>
            <div style={{ fontWeight: 600, marginBottom: 12, fontSize: 13 }}>Detected objects</div>
            {sortedCounts.length === 0 ? (
              <div className="alert alert-error">No objects detected</div>
            ) : (
              <table className="det-table">
                <thead>
                  <tr>
                    <th>Class</th>
                    <th>Count</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedCounts.map(([name, n]) => (
                    <tr key={name}>
                      <td style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span style={{
                          width: 10, height: 10, borderRadius: 2,
                          background: CLASS_COLORS[name] ?? '#888',
                          display: 'inline-block', flexShrink: 0
                        }} />
                        {name}
                      </td>
                      <td>
                        <span className="badge badge-blue">{n}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            {warnings.map((w, i) => (
              <div className="alert alert-warn" key={i}>⚠ {w}</div>
            ))}

            {suggested_team && (
              <div className="alert alert-info" style={{ marginTop: 8 }}>
                💡 TEAM {suggested_team === 5 ? '1' : '2'} appears closest to the ball
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="nav-row">
        <button className="btn btn-secondary" onClick={onBack}>← Back</button>
        <button
          className="btn btn-primary btn-lg"
          disabled={detections.length === 0}
          onClick={() => onDone({ detections, suggested_team })}
        >
          Set vanishing point →
        </button>
      </div>
    </div>
  )
}
