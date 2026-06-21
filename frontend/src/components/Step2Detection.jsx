const CLASS_BADGES = {
  Ball:       { color: '#94a3b8', bg: 'rgba(148,163,184,0.1)' },
  GoalKeeper: { color: '#fbbf24', bg: 'rgba(251,191,36,0.1)'  },
  Referee:    { color: '#f97316', bg: 'rgba(249,115,22,0.1)'  },
  'TEAM 1':   { color: '#f472b6', bg: 'rgba(244,114,182,0.1)' },
  'TEAM 2':   { color: '#fcd34d', bg: 'rgba(252,211,77,0.1)'  },
  Corner:     { color: '#818cf8', bg: 'rgba(129,140,248,0.1)' },
  Goal_Net:   { color: '#6b7280', bg: 'rgba(107,114,128,0.1)' },
}

export default function Step2Detection({ detectedImageB64, classCount, onNext, onBack }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* ── Image ──────────────────────────────────────────────── */}
        <div className="lg:col-span-2 glass rounded-2xl overflow-hidden">
          <div className="px-4 py-3 border-b border-border-subtle">
            <span className="text-sm font-600 text-tx-primary">Detecciones YOLO</span>
          </div>
          <img
            src={`data:image/jpeg;base64,${detectedImageB64}`}
            alt="Detected"
            className="w-full block"
          />
        </div>

        {/* ── Class counts ───────────────────────────────────────── */}
        <div className="glass rounded-2xl p-4">
          <p className="text-[11px] font-700 uppercase tracking-widest text-tx-muted mb-3">Detecciones</p>
          {Object.keys(classCount).length === 0 ? (
            <p className="text-sm text-tx-muted">Sin detecciones</p>
          ) : (
            <div className="space-y-2">
              {Object.entries(classCount).map(([name, count]) => {
                const st = CLASS_BADGES[name] || { color: '#8896b0', bg: 'rgba(136,150,176,0.1)' }
                return (
                  <div key={name} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-5 h-5 rounded-md flex items-center justify-center"
                        style={{ background: st.bg, border: `1px solid ${st.color}33` }}>
                        <div className="w-2 h-2 rounded-full" style={{ background: st.color }} />
                      </div>
                      <span className="text-sm text-tx-primary">{name}</span>
                    </div>
                    <span className="text-sm font-700 text-tx-primary tabular-nums">{count}</span>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* Nav */}
      <div className="flex gap-3 justify-between pt-1">
        <button onClick={onBack} className="btn-ghost px-6 py-2.5 text-sm font-600">
          Volver
        </button>
        <button onClick={onNext} className="btn-primary px-8 py-2.5 text-sm">
          Siguiente
        </button>
      </div>
    </div>
  )
}
