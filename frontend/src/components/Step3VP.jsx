import { useRef, useState, useEffect } from 'react'
import { IconCircleCheck, IconAlertTriangle, IconRefresh, IconCrosshair, IconLoader2 } from '@tabler/icons-react'
import api from '../api/client'

function calcIntersection(p1, p2, p3, p4) {
  const dx1 = p2.ox - p1.ox, dy1 = p2.oy - p1.oy
  const dx2 = p4.ox - p3.ox, dy2 = p4.oy - p3.oy
  const denom = dx1 * dy2 - dy1 * dx2
  if (Math.abs(denom) < 1e-6) return null
  const t = ((p3.ox - p1.ox) * dy2 - (p3.oy - p1.oy) * dx2) / denom
  return { x: p1.ox + t * dx1, y: p1.oy + t * dy1 }
}

export default function Step3VP({
  detectedImageB64, imageSize, vpAuto, vpLines, onVPChange,
  loading, onCalculate, onBack,
}) {
  const imgRef = useRef(null)
  const [manualMode, setManualMode] = useState(!vpAuto)
  const [points,     setPoints]     = useState([])
  const [manualVP,   setManualVP]   = useState(null)
  const [imgReady,   setImgReady]   = useState(false)
  const [lineSelA,   setLineSelA]   = useState([])
  const [lineSelB,   setLineSelB]   = useState([])
  const [localVP,    setLocalVP]    = useState(null) // VP from custom line selection

  // Reset line selection when vpLines changes
  useEffect(() => {
    if (vpLines) {
      setLineSelA(vpLines.a.map(() => true))
      setLineSelB(vpLines.b.map(() => true))
      setLocalVP(null)
    }
  }, [vpLines])

  const activeVP = manualMode ? manualVP : (localVP ?? vpAuto)

  useEffect(() => { onVPChange?.(activeVP) }, [activeVP])

  const resetManual = () => { setPoints([]); setManualVP(null) }

  const toggleLine = async (group, idx) => {
    if (!vpLines || !imageSize) return
    const nextA = group === 'a' ? lineSelA.map((v, i) => i === idx ? !v : v) : lineSelA
    const nextB = group === 'b' ? lineSelB.map((v, i) => i === idx ? !v : v) : lineSelB
    if (group === 'a') setLineSelA(nextA)
    else setLineSelB(nextB)
    const activeA = vpLines.a.filter((_, i) => nextA[i])
    const activeB = vpLines.b.filter((_, i) => nextB[i])
    if (!activeA.length || !activeB.length) { setLocalVP(null); return }
    try {
      const { data } = await api.post('/api/vp-from-lines', {
        lines_a: activeA,
        lines_b: activeB,
        image_width: imageSize.width,
        image_height: imageSize.height,
      })
      setLocalVP(data.vp)
    } catch { setLocalVP(null) }
  }

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

  const hasCustomSel = lineSelA.some(v => !v) || lineSelB.some(v => !v)
  const activeLineCount = lineSelA.filter(Boolean).length + lineSelB.filter(Boolean).length
  const MIN_LINES = 2
  const tooFewLines = !manualMode && vpLines && activeLineCount < MIN_LINES

  return (
    <div className="space-y-4">

      {/* ── Status bar ─────────────────────────────────────────── */}
      <div className="glass rounded-2xl px-5 py-4 flex items-center justify-between gap-4">
        <div>
          <p className="font-600 text-tx-primary">Punto de Fuga (VP)</p>
          <p className="text-xs text-tx-muted mt-0.5">
            {activeVP
              ? manualMode
                ? 'Calculado manualmente desde 4 puntos'
                : hasCustomSel
                  ? `Auto-detectado · ${activeLineCount} líneas seleccionadas`
                  : 'Auto-detectado · clic en líneas para ajustar'
              : manualMode
                ? 'Hacé clic en 2 puntos sobre la línea A y 2 sobre la línea B'
                : 'No se detectó automáticamente — seleccioná manualmente'}
          </p>
        </div>
        {activeVP ? (
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full shrink-0"
            style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)' }}>
            <IconCircleCheck size={14} stroke={2} className="text-accent" />
            <span className="text-xs font-600 text-accent font-mono">
              ({activeVP.x.toFixed(0)}, {activeVP.y.toFixed(0)})
            </span>
          </div>
        ) : (
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full shrink-0"
            style={{ background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)' }}>
            <IconAlertTriangle size={14} stroke={2} className="text-warn" />
            <span className="text-xs font-600 text-warn">Sin punto de fuga</span>
          </div>
        )}
      </div>

      {/* ── Image ──────────────────────────────────────────────── */}
      <div className="glass rounded-2xl overflow-hidden">
        <div className="px-4 py-2.5 border-b border-border-subtle flex items-center justify-between">
          <span className="text-xs font-600 text-tx-muted">
            {manualMode && points.length < 4
              ? <span className="flex items-center gap-1.5" style={{ color: '#f59e0b' }}>
                  <IconCrosshair size={12} stroke={2.5} />
                  {4 - points.length} clic{4 - points.length !== 1 ? 's' : ''} restante{4 - points.length !== 1 ? 's' : ''}
                </span>
              : !manualMode && vpLines
                ? <span className="text-tx-muted">
                    <span style={{ color: '#22c55e' }}>{activeLineCount}</span>{' / '}{vpLines.a.length + vpLines.b.length} líneas detectadas{' · '}
                    <span style={{ color: '#f43f5e' }}>clic para desactivar</span>
                  </span>
                : 'Imagen detectada'
            }
          </span>
          <div className="flex gap-2">
            {vpAuto && !manualMode && (
              <>
                {hasCustomSel && (
                  <button
                    onClick={() => {
                      setLineSelA(vpLines.a.map(() => true))
                      setLineSelB(vpLines.b.map(() => true))
                      setLocalVP(null)
                    }}
                    className="btn-ghost py-1 px-2.5 text-[11px] font-600 flex items-center gap-1">
                    <IconRefresh size={11} stroke={2.5} />
                    Resetear
                  </button>
                )}
                <button onClick={() => { setManualMode(true); resetManual() }}
                  className="btn-ghost py-1 px-2.5 text-[11px] font-600">
                  4 puntos manual
                </button>
              </>
            )}
            {!vpAuto && !manualMode && (
              <button onClick={() => setManualMode(true)}
                className="py-1 px-2.5 text-[11px] font-600 rounded-lg flex items-center gap-1"
                style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.25)', color: '#22c55e' }}>
                <IconCrosshair size={11} stroke={2.5} />
                Seleccionar 4 puntos
              </button>
            )}
            {manualMode && (
              <div className="flex gap-1.5">
                {vpAuto && (
                  <button onClick={() => { setManualMode(false); resetManual() }}
                    className="btn-ghost py-1 px-2.5 text-[11px] font-600">
                    Usar auto
                  </button>
                )}
                <button onClick={resetManual}
                  className="btn-ghost py-1 px-2.5 text-[11px] font-600 flex items-center gap-1">
                  <IconRefresh size={11} stroke={2.5} />
                  Limpiar
                </button>
              </div>
            )}
          </div>
        </div>

        <div
          className="relative"
          style={{ cursor: manualMode && points.length < 4 ? 'crosshair' : 'default' }}
          onClick={handleClick}
        >
          {detectedImageB64 && (
            <img
              ref={imgRef}
              src={`data:image/jpeg;base64,${detectedImageB64}`}
              alt="Campo"
              className="block"
              style={{ display: 'block', width: '100%', height: 'auto' }}
              onLoad={() => setImgReady(true)}
            />
          )}
          {imgReady && (
            <svg className="absolute inset-0 w-full h-full" style={{ pointerEvents: 'none' }}>
              {/* Auto-detected field lines — clickable to toggle */}
              {!manualMode && vpLines && imageSize && (
                <>
                  {vpLines.a.map(([x1, y1, x2, y2], i) => {
                    const active = lineSelA[i] ?? true
                    const x1p = `${(x1 / imageSize.width)  * 100}%`
                    const y1p = `${(y1 / imageSize.height) * 100}%`
                    const x2p = `${(x2 / imageSize.width)  * 100}%`
                    const y2p = `${(y2 / imageSize.height) * 100}%`
                    return (
                      <g key={`a${i}`} style={{ pointerEvents: 'stroke', cursor: 'pointer' }}
                        onClick={(e) => { e.stopPropagation(); toggleLine('a', i) }}>
                        {/* Dark outline for contrast */}
                        <line x1={x1p} y1={y1p} x2={x2p} y2={y2p}
                          stroke="#000" strokeWidth="6" opacity={active ? 0.5 : 0.2}
                          strokeLinecap="round" />
                        <line x1={x1p} y1={y1p} x2={x2p} y2={y2p}
                          stroke={active ? '#22c55e' : '#6b7280'}
                          strokeWidth={active ? 4 : 2}
                          strokeDasharray={active ? 'none' : '8 5'}
                          strokeLinecap="round"
                          opacity={active ? 1 : 0.5}
                          style={{ filter: active ? 'drop-shadow(0 0 5px #22c55e) drop-shadow(0 0 10px #22c55e)' : 'none' }} />
                      </g>
                    )
                  })}
                  {vpLines.b.map(([x1, y1, x2, y2], i) => {
                    const active = lineSelB[i] ?? true
                    const x1p = `${(x1 / imageSize.width)  * 100}%`
                    const y1p = `${(y1 / imageSize.height) * 100}%`
                    const x2p = `${(x2 / imageSize.width)  * 100}%`
                    const y2p = `${(y2 / imageSize.height) * 100}%`
                    return (
                      <g key={`b${i}`} style={{ pointerEvents: 'stroke', cursor: 'pointer' }}
                        onClick={(e) => { e.stopPropagation(); toggleLine('b', i) }}>
                        <line x1={x1p} y1={y1p} x2={x2p} y2={y2p}
                          stroke="#000" strokeWidth="6" opacity={active ? 0.5 : 0.2}
                          strokeLinecap="round" />
                        <line x1={x1p} y1={y1p} x2={x2p} y2={y2p}
                          stroke={active ? '#22c55e' : '#6b7280'}
                          strokeWidth={active ? 4 : 2}
                          strokeDasharray={active ? 'none' : '8 5'}
                          strokeLinecap="round"
                          opacity={active ? 1 : 0.5}
                          style={{ filter: active ? 'drop-shadow(0 0 5px #22c55e) drop-shadow(0 0 10px #22c55e)' : 'none' }} />
                      </g>
                    )
                  })}
                </>
              )}
              {/* Manual mode lines and points */}
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
                  <circle cx={`${p.px}%`} cy={`${p.py}%`} r="9"
                    fill={i < 2 ? '#22c55e' : '#fbbf24'} stroke="#060910" strokeWidth="2" />
                  <text x={`${p.px}%`} y={`${p.py}%`} dy="4.5"
                    textAnchor="middle" fill="#060910" fontSize="9" fontWeight="800"
                    fontFamily="Plus Jakarta Sans, sans-serif">{i + 1}</text>
                </g>
              ))}
              {vpInBounds && (
                <g>
                  <circle cx={`${vpDisplay.px}%`} cy={`${vpDisplay.py}%`} r="18"
                    fill="rgba(34,197,94,0.1)" stroke="#22c55e" strokeWidth="1.5" />
                  <circle cx={`${vpDisplay.px}%`} cy={`${vpDisplay.py}%`} r="5" fill="#22c55e" />
                  <text x={`${vpDisplay.px}%`} y={`${vpDisplay.py}%`} dy="-24"
                    textAnchor="middle" fill="#22c55e" fontSize="11" fontWeight="700"
                    fontFamily="Plus Jakarta Sans, sans-serif">VP</text>
                </g>
              )}
            </svg>
          )}
        </div>
      </div>

      {/* ── Nav ────────────────────────────────────────────────── */}
      <div className="flex gap-3">
        <button onClick={onBack} className="btn-ghost px-6 py-3 text-sm font-600">
          Volver
        </button>
        {tooFewLines && (
          <div className="flex-1 flex items-center justify-center px-4 py-3 rounded-2xl text-sm font-600"
            style={{ background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.25)', color: '#f59e0b' }}>
            Mínimo {MIN_LINES} líneas para calcular el punto de fuga — activá más líneas o usá 4 puntos manual
          </div>
        )}
        {!tooFewLines && (
          <button
            onClick={() => onCalculate(activeVP)}
            disabled={!activeVP || loading}
            className="btn-primary flex-1 py-3 text-[15px] flex items-center justify-center gap-2.5"
          >
            {loading
              ? <><IconLoader2 size={16} stroke={2} className="animate-spin" /> Calculando…</>
              : 'Calcular offside'
            }
          </button>
        )}
      </div>
    </div>
  )
}
