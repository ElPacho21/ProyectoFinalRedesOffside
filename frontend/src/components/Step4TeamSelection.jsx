import { useState } from 'react'

export default function Step4TeamSelection({ suggestedTeam, onDone, onBack }) {
  const [attackingTeam, setAttackingTeam] = useState(suggestedTeam ?? 5)
  const [direction, setDirection] = useState('auto')     // 'auto' | 'right' | 'left'
  const [refDefender, setRefDefender] = useState('auto') // 'auto' | '0' | '1' | '2'
  const [footPoint, setFootPoint] = useState('medio')    // 'medio' | 'izquierdo' | 'derecho'

  const handleNext = () => {
    const dirMap = { auto: null, right: true, left: false }
    const refMap = { auto: null, '0': 0, '1': 1, '2': 2 }

    onDone({
      attackingTeamId: attackingTeam,
      direction: dirMap[direction],
      refDefenderIdx: refMap[refDefender],
      footPoint,
    })
  }

  return (
    <div>
      <div className="card">
        <div className="card-title">Attacking team</div>

        {suggestedTeam && (
          <div className="alert alert-info" style={{ marginBottom: 14 }}>
            💡 TEAM {suggestedTeam === 5 ? '1' : '2'} appears closest to the ball
          </div>
        )}

        <label className="label">Which team is attacking?</label>
        <div className="radio-group" style={{ marginBottom: 20 }}>
          {[{ id: 5, label: 'TEAM 1' }, { id: 6, label: 'TEAM 2' }].map(({ id, label }) => (
            <button
              key={id}
              className={`radio-btn${attackingTeam === id ? ' selected' : ''}`}
              onClick={() => setAttackingTeam(id)}
            >
              {label}
            </button>
          ))}
        </div>

        <details>
          <summary>Advanced offside settings</summary>
          <div className="details-body">
            <div className="grid-3" style={{ gap: 20 }}>
              <div>
                <label className="label" style={{ marginBottom: 8 }}>Attack direction</label>
                <div className="radio-group" style={{ flexDirection: 'column' }}>
                  {[
                    { val: 'auto', label: '🔄 Auto-detect' },
                    { val: 'right', label: '➡️ Attack right' },
                    { val: 'left', label: '⬅️ Attack left' },
                  ].map(({ val, label }) => (
                    <button
                      key={val}
                      className={`radio-btn${direction === val ? ' selected' : ''}`}
                      onClick={() => setDirection(val)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="label" style={{ marginBottom: 8 }}>Reference defender</label>
                <div className="radio-group" style={{ flexDirection: 'column' }}>
                  {[
                    { val: 'auto', label: '🔄 Auto (penultimate)' },
                    { val: '0', label: '1st closest to goal' },
                    { val: '1', label: '2nd closest (penultimate)' },
                    { val: '2', label: '3rd closest' },
                  ].map(({ val, label }) => (
                    <button
                      key={val}
                      className={`radio-btn${refDefender === val ? ' selected' : ''}`}
                      onClick={() => setRefDefender(val)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="label" style={{ marginBottom: 8 }}>Player reference point</label>
                <div className="radio-group" style={{ flexDirection: 'column' }}>
                  {[
                    { val: 'medio', label: '⏺️ Bottom center' },
                    { val: 'izquierdo', label: '◀️ Bottom left' },
                    { val: 'derecho', label: '▶️ Bottom right' },
                  ].map(({ val, label }) => (
                    <button
                      key={val}
                      className={`radio-btn${footPoint === val ? ' selected' : ''}`}
                      onClick={() => setFootPoint(val)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </details>
      </div>

      <div className="nav-row">
        <button className="btn btn-secondary" onClick={onBack}>← Back</button>
        <button className="btn btn-primary btn-lg" onClick={handleNext}>
          Analyze offside →
        </button>
      </div>
    </div>
  )
}
