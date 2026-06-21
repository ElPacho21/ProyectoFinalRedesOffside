import { IconArrowLeft, IconArrowRight, IconUser, IconCheck, IconTarget } from '@tabler/icons-react'

const DIRECTIONS = [
  { value: false, Icon: IconArrowLeft,  label: 'Izquierda' },
  { value: true,  Icon: IconArrowRight, label: 'Derecha'   },
]

const REF_POINTS = [
  { v: 'medio',     label: 'Centro'    },
  { v: 'izquierdo', label: 'Izquierdo' },
  { v: 'derecho',   label: 'Derecho'   },
]

function TeamCard({ label, sampleB64, selected, onClick }) {
  return (
    <button
      onClick={onClick}
      className="flex-1 rounded-2xl p-5 flex flex-col items-center gap-3 transition-all duration-200"
      style={{
        background: selected ? 'rgba(34,197,94,0.07)' : 'rgba(11,17,32,0.7)',
        border: `1.5px solid ${selected ? '#22c55e' : '#1a2540'}`,
        boxShadow: selected
          ? '0 0 28px rgba(34,197,94,0.14), 0 4px 20px rgba(0,0,0,0.3)'
          : '0 4px 16px rgba(0,0,0,0.2)',
        transform: selected ? 'translateY(-2px)' : 'none',
      }}
    >
      <div
        className="relative w-24 h-32 rounded-xl overflow-hidden"
        style={{
          border: `1.5px solid ${selected ? '#22c55e' : '#1a2540'}`,
          boxShadow: selected ? '0 0 16px rgba(34,197,94,0.25)' : 'none',
        }}
      >
        {sampleB64 ? (
          <img src={`data:image/jpeg;base64,${sampleB64}`} alt={label} className="w-full h-full object-cover" />
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-bg-elevated">
            <IconUser size={32} stroke={1.5} className="text-tx-muted" />
          </div>
        )}
        {selected && (
          <div
            className="absolute inset-0 flex items-end justify-center pb-2.5"
            style={{ background: 'linear-gradient(to top, rgba(34,197,94,0.55) 0%, transparent 55%)' }}
          >
            <div className="w-6 h-6 rounded-full bg-accent flex items-center justify-center">
              <IconCheck size={14} stroke={3} color="white" />
            </div>
          </div>
        )}
      </div>
      <div className="text-center">
        <p className="font-display font-700 text-[15px]"
          style={{ color: selected ? '#4ade80' : '#e2e8f5' }}>
          {label}
        </p>
        {selected && (
          <p className="text-[11px] text-accent font-600 mt-0.5 tracking-wider uppercase">Atacante</p>
        )}
      </div>
    </button>
  )
}

export default function Step2TeamConfig({
  teamSamples, attackingTeam, goalOnRight, goalOnRightAutoDetected,
  referencePoint, onChange, onNext, onBack,
}) {
  const canProceed = attackingTeam !== null && goalOnRight !== null && goalOnRight !== undefined

  return (
    <div className="max-w-xl mx-auto space-y-4">

      {/* ── Team selection ─────────────────────────────────────── */}
      <div className="glass rounded-2xl p-5">
        <h2 className="font-display font-700 text-[17px] text-tx-primary mb-1">¿Quién está atacando?</h2>
        <p className="text-sm text-tx-muted mb-5">Seleccioná el equipo que ataca hacia el arco rival</p>
        <div className="flex gap-3">
          <TeamCard label="TEAM 1" sampleB64={teamSamples?.team1}
            selected={attackingTeam === 5} onClick={() => onChange({ attackingTeam: 5 })} />
          <TeamCard label="TEAM 2" sampleB64={teamSamples?.team2}
            selected={attackingTeam === 6} onClick={() => onChange({ attackingTeam: 6 })} />
        </div>
      </div>

      {/* ── Direction ─────────────────────────────────────────── */}
      <div className="glass rounded-2xl p-5">
        <p className="font-600 text-tx-primary mb-1">Dirección de ataque</p>
        <p className="text-xs text-tx-muted mb-4">Hacia qué lado del campo ataca el equipo seleccionado</p>
        <div className="flex gap-2">
          {DIRECTIONS.map(({ value, Icon, label }) => {
            const active = goalOnRight === value
            return (
              <button
                key={String(value)}
                onClick={() => onChange({ goalOnRight: value })}
                className="flex-1 py-3 rounded-xl flex flex-col items-center gap-1.5 transition-all duration-150"
                style={{
                  background: active ? 'rgba(34,197,94,0.1)' : 'rgba(26,37,64,0.25)',
                  border: `1.5px solid ${active ? '#22c55e' : '#1a2540'}`,
                  color: active ? '#4ade80' : '#8896b0',
                  boxShadow: active ? '0 0 14px rgba(34,197,94,0.1)' : 'none',
                }}
              >
                <Icon size={20} stroke={2} />
                <span className="text-[11px] font-600">{label}</span>
              </button>
            )
          })}
        </div>
        {goalOnRightAutoDetected && goalOnRight !== null && (
          <p className="text-[11px] text-tx-muted mt-3 flex items-center gap-1.5">
            <span style={{ color: '#22c55e' }}>●</span>
            Auto-detectado desde la posición de la red del arco — podés cambiarlo si es incorrecto
          </p>
        )}
        {!goalOnRightAutoDetected && (
          <p className="text-[11px] text-tx-muted mt-3">
            No se detectó la red del arco — seleccioná manualmente hacia qué lado ataca el equipo
          </p>
        )}
      </div>

      {/* ── Reference point ───────────────────────────────────── */}
      <div className="glass rounded-2xl p-5">
        <div className="flex items-center gap-2 mb-1">
          <IconTarget size={16} stroke={2} className="text-tx-muted" />
          <p className="font-600 text-tx-primary">Punto de referencia del pie</p>
        </div>
        <p className="text-xs text-tx-muted mb-4">Qué esquina del bounding box representa el pie del jugador</p>
        <div className="flex gap-2">
          {REF_POINTS.map(({ v, label }) => {
            const active = referencePoint === v
            return (
              <button
                key={v}
                onClick={() => onChange({ referencePoint: v })}
                className="flex-1 py-2.5 rounded-xl text-xs font-600 transition-all"
                style={{
                  background: active ? 'rgba(34,197,94,0.1)' : 'rgba(26,37,64,0.25)',
                  border: `1.5px solid ${active ? '#22c55e' : '#1a2540'}`,
                  color: active ? '#4ade80' : '#8896b0',
                }}
              >
                {label}
              </button>
            )
          })}
        </div>
      </div>

      {/* ── Nav ───────────────────────────────────────────────── */}
      <div className="flex gap-3">
        <button onClick={onBack} className="btn-ghost px-6 py-3 text-sm font-600">
          Volver
        </button>
        <button
          onClick={onNext}
          disabled={!canProceed}
          className="btn-primary flex-1 py-3 text-[15px]"
        >
          Siguiente
        </button>
      </div>
    </div>
  )
}
