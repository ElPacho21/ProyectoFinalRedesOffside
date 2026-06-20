import { useState, useEffect } from 'react'
import { analyzeOffside } from '../api.js'

export default function Step5Result({ mediaInfo, frameIndex, detections, vp, attackingTeamId, direction, refDefenderIdx, footPoint, onBack, onRestart }) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    analyzeOffside({
      media_id: mediaInfo.media_id,
      frame_index: frameIndex,
      detections,
      vp,
      attacking_team_id: attackingTeamId,
      direction: direction ?? null,
      ref_defender_idx: refDefenderIdx ?? null,
      foot_point: footPoint,
    })
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="card">
        <div className="spinner-wrap">
          <div className="spinner" />
          Computing offside geometry...
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

  const { hay_offside, advertencias, atacantes_resultado, annotated_image_base64 } = result
  const teamName = `TEAM ${attackingTeamId === 5 ? '1' : '2'}`

  return (
    <div>
      <div className="card">
        <div className="card-title">Offside analysis result</div>

        {advertencias?.length > 0 && advertencias.map((w, i) => (
          <div className="alert alert-warn" key={i}>⚠ {w}</div>
        ))}

        <div className="grid-2">
          <div className="img-wrap">
            <img src={`data:image/jpeg;base64,${annotated_image_base64}`} alt="result" />
          </div>

          <div>
            <div className={`verdict ${hay_offside ? 'verdict-offside' : 'verdict-onside'}`}>
              <div className="verdict-icon">{hay_offside ? '🚩' : '✅'}</div>
              <div className="verdict-text">{hay_offside ? 'OFFSIDE' : 'ONSIDE'}</div>
              <div style={{ fontSize: 12, marginTop: 4, opacity: 0.7 }}>{teamName}</div>
            </div>

            <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 10 }}>Player breakdown</div>
            {atacantes_resultado?.length === 0 ? (
              <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>No attacking players to evaluate</p>
            ) : (
              atacantes_resultado?.map((item, i) => {
                const det = item.detection
                return (
                  <div className="player-row" key={i}>
                    <span>{det.class_name} <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>({(det.conf * 100).toFixed(0)}%)</span></span>
                    <span className={item.offside ? 'tag-offside' : 'tag-onside'}>
                      {item.offside ? '🔴 OFFSIDE' : '🟢 ONSIDE'}
                    </span>
                  </div>
                )
              })
            )}
          </div>
        </div>
      </div>

      <div className="nav-row">
        <button className="btn btn-secondary" onClick={onBack}>← Adjust settings</button>
        <button className="btn btn-primary" onClick={onRestart}>Analyze new image</button>
      </div>
    </div>
  )
}
