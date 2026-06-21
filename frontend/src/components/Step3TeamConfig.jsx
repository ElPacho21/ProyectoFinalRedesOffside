import { useRef, useState, useEffect } from 'react'
import {
  IconArrowLeft, IconArrowRight, IconArrowsHorizontal,
  IconUser, IconCheck, IconLoader2, IconTarget,
  IconCircleCheck, IconAlertTriangle, IconRefresh, IconCrosshair,
} from '@tabler/icons-react'

const DIRECTIONS = [
  { value: false, Icon: IconArrowLeft,        label: 'Izquierda' },
  { value: null,  Icon: IconArrowsHorizontal, label: 'Auto' },
  { value: true,  Icon: IconArrowRight,        label: 'Derecha' },
]

const REF_POINTS = [
  { v: 'medio',     label: 'Centro'    },
  { v: 'izquierdo', label: 'Izquierdo' },
  { v: 'derecho',   label: 'Derecho'   },
]

function calcIntersection(p1, p2, p3, p4) {
  const dx1 = p2.ox - p1.ox, dy1 = p2.oy - p1.oy
  const dx2 = p4.ox - p3.ox, dy2 = p4.oy - p3.oy
  const denom = dx1 * dy2 - dy1 * dx2
  if (Math.abs(denom) < 1e-6) return null
  const t = ((p3.ox - p1.ox) * dy2 - (p3.oy - p1.oy) * dx2) / denom
  return { x: p1.ox + t * dx1, y: p1.oy + t * dy1 }
}

function TeamCard({ label, sampleB64, selected, onClick }) {
  return (
    <button
      onClick={onClick}
      className="flex-1 rounded-2xl p-5 flex flex-col items-center gap-3 transition-all duration-200"
      style={{
        background: selected ? 'rgba(34,197,94,0.07)' : 'rgba(11,17,32,0.7)',
        border: `1.5px solid ${selected ? '#22c55e' : '#1a2540'}`,
        boxShadow: selected ? '0 0 28px rgba(34,197,94,0.14), 0 4px 20px rgba(0,0,0,0.3)' : '0 4px 16px rgba(0,0,0,0.2)',
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
        <p className="font-display font-700 text-[15px]" style={{ color: selected ? '#4ade80' : '#e2e8f5' }}>
          {label}
        </p>
        {selected && (
          <p className="text-[11px] text-accent font-600 mt-0.5 tracking-wider uppercase">Atacante</p>
        )}
      </div>
    </button>
  )
}

function VPSection({ detectedImageB64, imageSize, vpAuto, onVPChange }) {
  const imgRef = useRef(null)
  const [manualMode, setManualMode] = useState(!vpAuto)
  const [points,     setPoints]     = useState([])
  const [manualVP,   setManualVP]   = useState(null)
  const [imgReady,   setImgReady]   = useState(false)

  const activeVP = manualMode ? manualVP : vpAuto

  useEffect(() => { onVPChange?.(activeVP) }, [activeVP])

  const resetManual = () => { setPoints([]); setManualVP(null) }

  const handleClick = (e) => {
    if (!manualMode || points.length >= 4 || !imageSize) return
    const rect = imgRef.current.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width)  * 100
    const py = ((e.clientY - rect.top)  / rect.height) * 100
    const ox = ((e.clientX - rect.left) / rect.width)  * imageSize.width
    const oy = ((e.clientY - rect.top)  / rect.height) * imageSize.height
    const newPts = [...points, { px, py, ox, oy }]
    setPoints(newPts)
    if (newPts.length === 4) {
      const vp = calcIntersection(newPts[0], newPts[1], newPts[2], newPts[3])
      if (vp) setManualVP(vp)
    }
  }

  const vpDisplay = activeVP && imageSize
    ? { px: (activeVP.x / imageSize.width) * 100, py: (activeVP.y / imageSize.height) * 100 }
    : null
  const vpInBounds = vpDisplay
    && vpDisplay.px >= 0 && vpDisplay.px <= 100
    && vpDisplay.py >= 0 && vpDisplay.py <= 100

  if (!detectedImageB64) return null

  return (
    <div className="glass rounded-2xl overflow-hidden">
      {/* Header */}
      <div className="px-5 pt-5 pb-4 border-b border-border-subtle flex items-start justify-between gap-4">
        <div>
          <p className="font-600 text-tx-primary">Punto de Fuga</p>
          <p className="text-xs text-tx-muted mt-0.5">
            {activeVP
              ? (manualMode ? 'VP calculado manualmente' : 'Auto-detectado desde Hough Lines')
              : 'Seleccioná 4 puntos: 2 en la línea A, 2 en la línea B'}
          </p>
        </div>
        {activeVP ? (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full shrink-0"
            style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)' }}>
            <IconCircleCheck size={13} stroke={2} className="text-accent" />
            <span className="text-[11px] font-600 text-accent font-mono">
              ({activeVP.x.toFixed(0)}, {activeVP.y.toFixed(0)})
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full shrink-0"
            style={{ background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)' }}>
            <IconAlertTriangle size={13} stroke={2} className="text-warn" />
            <span className="text-[11px] font-600 text-warn">Sin VP</span>
          </div>
        )}
      </div>

      {/* Image */}
      <div
        className="relative"
        style={{ cursor: manualMode && points.length < 4 ? 'crosshair' : 'default' }}
        onClick={handleClick}
      >
        {manualMode && points.length < 4 && (
          <div className="absolute top-3 left-3 z-10 flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full pointer-events-none"
            style={{ background: 'rgba(0,0,0,0.75)', border: '1px solid rgba(245,158,11,0.35)', color: '#f59e0b' }}>
            <IconCrosshair size={12} stroke={2.5} />
            {4 - points.length} clic{4 - points.length !== 1 ? 's' : ''} restante{4 - points.length !== 1 ? 's' : ''}
          </div>
        )}
        <img
          ref={imgRef}
          src={`data:image/jpeg;base64,${detectedImageB64}`}
          alt="Campo"
          className="w-full block"
          style={{ maxHeight: '280px', objectFit: 'contain', background: '#000' }}
          onLoad={() => setImgReady(true)}
        />
        {imgReady && (
          <svg className="absolute inset-0 w-full h-full pointer-events-none">
            {points.length >= 2 && (
              <line x1={`${points[0].px}%`} y1={`${points[0].py}%`}
                x2={`${points[1].px}%`} y2={`${points[1].py}%`}
                stroke="#22c55e" strokeWidth="1.5" strokeDasharray="6 3" opacity="0.9" />
            )}
            {points.length >= 4 && (
              <line x1={`${points[2].px}%`} y1={`${points[2].py}%`}
                x2={`${points[3].px}%`} y2={`${points[3].py}%`}
                stroke="#fbbf24" strokeWidth="1.5" strokeDasharray="6 3" opacity="0.9" />
            )}
            {points.map((p, i) => (
              <g key={i}>
                <circle cx={`${p.px}%`} cy={`${p.py}%`} r="8"
                  fill={i < 2 ? '#22c55e' : '#fbbf24'} stroke="#060910" strokeWidth="2" />
                <text x={`${p.px}%`} y={`${p.py}%`} dy="4.5"
                  textAnchor="middle" fill="#060910" fontSize="9" fontWeight="800"
                  fontFamily="Plus Jakarta Sans, sans-serif">{i + 1}</text>
              </g>
            ))}
            {vpInBounds && (
              <g>
                <circle cx={`${vpDisplay.px}%`} cy={`${vpDisplay.py}%`} r="16"
                  fill="rgba(34,197,94,0.12)" stroke="#22c55e" strokeWidth="1.5" />
                <circle cx={`${vpDisplay.px}%`} cy={`${vpDisplay.py}%`} r="4" fill="#22c55e" />
                <text x={`${vpDisplay.px}%`} y={`${vpDisplay.py}%`} dy="-20"
                  textAnchor="middle" fill="#22c55e" fontSize="10" fontWeight="700"
                  fontFamily="Plus Jakarta Sans, sans-serif">VP</text>
              </g>
            )}
          </svg>
        )}
      </div>

      {/* Actions */}
      <div className="px-5 py-3 flex gap-2 border-t border-border-subtle">
        {vpAuto && !manualMode && (
          <button onClick={() => { setManualMode(true); resetManual() }}
            className="btn-ghost flex-1 py-2 text-xs font-600">
            Seleccionar manualmente
          </button>
        )}
        {!vpAuto && !manualMode && (
          <button onClick={() => setManualMode(true)}
            className="flex-1 py-2 text-xs font-600 rounded-xl transition-all flex items-center justify-center gap-1.5"
            style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.25)', color: '#22c55e' }}>
            <IconCrosshair size={13} stroke={2.5} />
            Seleccionar 4 puntos
          </button>
        )}
        {manualMode && (
          <>
            {vpAuto && (
              <button onClick={() => { setManualMode(false); resetManual() }}
                className="btn-ghost flex-1 py-2 text-xs font-600">
                Usar auto
              </button>
            )}
            <button onClick={resetManual}
              className="btn-ghost flex-1 py-2 text-xs font-600 flex items-center justify-center gap-1.5">
              <IconRefresh size={12} stroke={2.5} />
              Limpiar
            </button>
          </>
        )}
      </div>
    </div>
  )
}

export default function Step3TeamConfig({
  detectedImageB64, imageSize, vpAuto, vpFinal, onVPChange,
  teamSamples, attackingTeam, goalOnRight, referencePoint,
  loading, onChange, onCalculate, onBack,
}) {
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
        <p className="text-xs text-tx-muted mb-4">Hacia qué lado del campo ataca el equipo</p>
        <div className="flex gap-2">
          {DIRECTIONS.map(({ value, Icon, label }) => {
            const active = goalOnRight === value
            return (
              <button key={String(value)} onClick={() => onChange({ goalOnRight: value })}
                className="flex-1 py-3 rounded-xl flex flex-col items-center gap-1.5 transition-all duration-150"
                style={{
                  background: active ? 'rgba(34,197,94,0.1)' : 'rgba(26,37,64,0.25)',
                  border: `1.5px solid ${active ? '#22c55e' : '#1a2540'}`,
                  color: active ? '#4ade80' : '#8896b0',
                  boxShadow: active ? '0 0 14px rgba(34,197,94,0.1)' : 'none',
                }}>
                <Icon size={20} stroke={2} />
                <span className="text-[11px] font-600">{label}</span>
              </button>
            )
          })}
        </div>
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
              <button key={v} onClick={() => onChange({ referencePoint: v })}
                className="flex-1 py-2.5 rounded-xl text-xs font-600 transition-all"
                style={{
                  background: active ? 'rgba(34,197,94,0.1)' : 'rgba(26,37,64,0.25)',
                  border: `1.5px solid ${active ? '#22c55e' : '#1a2540'}`,
                  color: active ? '#4ade80' : '#8896b0',
                }}>
                {label}
              </button>
            )
          })}
        </div>
      </div>

      {/* ── Vanishing Point ───────────────────────────────────── */}
      <VPSection
        detectedImageB64={detectedImageB64}
        imageSize={imageSize}
        vpAuto={vpAuto}
        onVPChange={onVPChange}
      />

      {/* ── Nav ───────────────────────────────────────────────── */}
      <div className="flex gap-3">
        <button onClick={onBack} className="btn-ghost px-6 py-3 text-sm font-600">
          Volver
        </button>
        <button
          onClick={onCalculate}
          disabled={attackingTeam === null || !vpFinal || loading}
          className="btn-primary flex-1 py-3 text-[15px] flex items-center justify-center gap-2.5"
        >
          {loading
            ? <><IconLoader2 size={16} stroke={2} className="animate-spin" /> Calculando…</>
            : 'Calcular offside'
          }
        </button>
      </div>
    </div>
  )
}
