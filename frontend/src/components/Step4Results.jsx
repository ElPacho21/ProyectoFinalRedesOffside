import { useRef, useState } from 'react'
import { IconZoomIn, IconZoomOut, IconRefresh, IconAlertTriangle, IconFlag, IconShieldCheck } from '@tabler/icons-react'

export default function Step4Results({ resultImageB64, resultado, imageSize, onReset }) {
  const [zoom,         setZoom]         = useState(1)
  const [hoveredIndex, setHoveredIndex] = useState(null)
  const [tooltipPos,   setTooltipPos]   = useState({ x: 0, y: 0 })

  // Refs to overlay divs — used to anchor tooltip above player bbox
  const overlayRefs = useRef([])
  // Debounce ref to prevent flicker when moving between card and image overlay
  const clearRef    = useRef(null)

  if (!resultado || !resultImageB64) return null

  const { hay_offside, atacantes_resultado = [], advertencias = [] } = resultado
  const offsideCount = atacantes_resultado.filter((x) => x.offside).length
  const onsideCount  = atacantes_resultado.filter((x) => !x.offside).length

  const onEnter = (i) => {
    if (clearRef.current) clearTimeout(clearRef.current)
    const rect = overlayRefs.current[i]?.getBoundingClientRect()
    if (rect) setTooltipPos({ x: rect.left + rect.width / 2, y: rect.top })
    setHoveredIndex(i)
  }

  const onLeave = () => {
    clearRef.current = setTimeout(() => setHoveredIndex(null), 200)
  }

  const handleWheel = (e) => {
    if (!e.ctrlKey) return
    e.preventDefault()
    setZoom((z) => Math.max(0.4, Math.min(5, z + (e.deltaY > 0 ? -0.15 : 0.15))))
  }

  const hoveredItem = hoveredIndex !== null ? atacantes_resultado[hoveredIndex] : null

  return (
    <div className="space-y-4 animate-fade-in">

      {/* ── Compact verdict bar ─────────────────────────────────── */}
      <div
        className="rounded-2xl px-5 py-3.5 flex items-center gap-4 relative overflow-hidden"
        style={{
          background: hay_offside
            ? 'linear-gradient(135deg, rgba(244,63,94,0.1) 0%, rgba(5,9,17,0.85) 100%)'
            : 'linear-gradient(135deg, rgba(34,197,94,0.09) 0%, rgba(5,9,17,0.85) 100%)',
          border: `1.5px solid ${hay_offside ? 'rgba(244,63,94,0.3)' : 'rgba(34,197,94,0.3)'}`,
          boxShadow: hay_offside ? '0 0 32px rgba(244,63,94,0.07)' : '0 0 32px rgba(34,197,94,0.07)',
        }}
      >
        {hay_offside
          ? <IconFlag size={22} stroke={1.5} className="text-danger shrink-0" />
          : <IconShieldCheck size={22} stroke={1.5} className="text-accent shrink-0" />
        }

        <span className="font-display font-800 text-2xl tracking-tight"
          style={{ color: hay_offside ? '#f43f5e' : '#22c55e' }}>
          {hay_offside ? 'OFFSIDE' : 'ONSIDE'}
        </span>

        <div className="flex items-center gap-3 ml-1">
          {offsideCount > 0 && (
            <div className="flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-danger" />
              <span className="text-xs font-600 text-danger">{offsideCount} fuera de juego</span>
            </div>
          )}
          {onsideCount > 0 && (
            <div className="flex items-center gap-1.5">
              <div className="w-1.5 h-1.5 rounded-full bg-accent" />
              <span className="text-xs font-600 text-accent">{onsideCount} en juego</span>
            </div>
          )}
        </div>

        <button onClick={onReset}
          className="btn-ghost ml-auto py-1.5 px-4 text-sm font-600 flex items-center gap-2 shrink-0">
          <IconRefresh size={13} stroke={2.5} />
          Nueva imagen
        </button>
      </div>

      {/* ── Result image — full width ───────────────────────────── */}
      <div className="glass rounded-2xl overflow-hidden">
        <div className="px-4 py-2.5 border-b border-border-subtle flex items-center justify-between">
          <span className="text-xs font-600 text-tx-muted">
            Resultado
            <span className="ml-2 font-400 opacity-40">Ctrl+Scroll para zoom · hover para info</span>
          </span>
          <div className="flex items-center gap-1">
            <button onClick={() => setZoom(z => Math.max(0.4, z - 0.2))}
              className="btn-ghost w-7 h-7 flex items-center justify-center rounded-lg">
              <IconZoomOut size={14} stroke={2.5} />
            </button>
            <span className="text-xs font-mono text-tx-muted w-12 text-center tabular-nums">
              {(zoom * 100).toFixed(0)}%
            </span>
            <button onClick={() => setZoom(z => Math.min(5, z + 0.2))}
              className="btn-ghost w-7 h-7 flex items-center justify-center rounded-lg">
              <IconZoomIn size={14} stroke={2.5} />
            </button>
            <button onClick={() => setZoom(1)}
              className="btn-ghost px-2.5 py-1 text-[11px] font-600 rounded-lg ml-1">
              1:1
            </button>
          </div>
        </div>

        <div className="overflow-auto bg-black" onWheel={handleWheel}>
          <div style={{ width: `${zoom * 100}%`, minWidth: '100%', position: 'relative', display: 'inline-block' }}>
            <img
              src={`data:image/jpeg;base64,${resultImageB64}`}
              alt="Resultado"
              className="w-full block"
              draggable={false}
            />
            {imageSize && atacantes_resultado.map((item, i) => {
              const d = item.detection
              const isHovered = hoveredIndex === i
              return (
                <div
                  key={i}
                  ref={(el) => overlayRefs.current[i] = el}
                  style={{
                    position: 'absolute',
                    left:   `${(d.x1 / imageSize.width)  * 100}%`,
                    top:    `${(d.y1 / imageSize.height) * 100}%`,
                    width:  `${((d.x2 - d.x1) / imageSize.width)  * 100}%`,
                    height: `${((d.y2 - d.y1) / imageSize.height) * 100}%`,
                    cursor: 'pointer',
                    borderRadius: '3px',
                    transition: 'background 0.15s, outline 0.15s',
                    background: isHovered
                      ? (item.offside ? 'rgba(244,63,94,0.22)' : 'rgba(34,197,94,0.22)')
                      : 'transparent',
                    outline: isHovered
                      ? `2px solid ${item.offside ? '#f43f5e' : '#22c55e'}`
                      : 'none',
                  }}
                  onMouseEnter={() => onEnter(i)}
                  onMouseLeave={onLeave}
                />
              )
            })}
          </div>
        </div>
      </div>

      {/* ── Player chips — horizontal scrollable strip ───────────── */}
      <div className="glass rounded-2xl p-3">
        <div className="flex items-center gap-2 overflow-x-auto pb-0.5" style={{ scrollbarWidth: 'none' }}>
          <span className="text-[10px] font-700 uppercase tracking-widest text-tx-muted shrink-0 pr-2 border-r border-border-subtle">
            Atacantes
          </span>
          {atacantes_resultado.length === 0 ? (
            <span className="text-xs text-tx-muted">Sin jugadores detectados</span>
          ) : atacantes_resultado.map((item, i) => {
            const isHovered = hoveredIndex === i
            return (
              <button
                key={i}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl shrink-0 transition-all duration-150"
                style={{
                  background: isHovered
                    ? (item.offside ? 'rgba(244,63,94,0.18)' : 'rgba(34,197,94,0.16)')
                    : (item.offside ? 'rgba(244,63,94,0.07)' : 'rgba(34,197,94,0.05)'),
                  border: `1px solid ${isHovered
                    ? (item.offside ? 'rgba(244,63,94,0.5)' : 'rgba(34,197,94,0.5)')
                    : (item.offside ? 'rgba(244,63,94,0.2)' : 'rgba(34,197,94,0.18)')}`,
                  transform: isHovered ? 'translateY(-2px)' : 'none',
                }}
                onMouseEnter={() => onEnter(i)}
                onMouseLeave={onLeave}
              >
                <span className="text-[11px] font-700 text-tx-primary">#{i + 1}</span>
                <span
                  className="text-[10px] font-800 tracking-wide"
                  style={{ color: item.offside ? '#f43f5e' : '#22c55e' }}
                >
                  {item.offside ? 'OFFSIDE' : 'OK'}
                </span>
                <span className="text-[10px] text-tx-muted font-mono">
                  {(item.detection.conf * 100).toFixed(0)}%
                </span>
              </button>
            )
          })}
          {advertencias.length > 0 && (
            <div className="flex items-center gap-1.5 ml-auto shrink-0 pl-2 border-l border-border-subtle">
              <IconAlertTriangle size={12} stroke={2} className="text-warn" />
              <span className="text-[10px] text-warn font-600">{advertencias.length} advertencia{advertencias.length > 1 ? 's' : ''}</span>
            </div>
          )}
        </div>
      </div>

      {/* Tooltip — position only updates while hovering */}
      {hoveredItem && (
        <div
          className="tooltip-box pointer-events-none"
          style={{ position: 'fixed', left: tooltipPos.x, top: tooltipPos.y - 10, transform: 'translate(-50%, -100%)', zIndex: 9999 }}
        >
          <div className="flex items-center gap-2 mb-2.5">
            <div className="w-2.5 h-2.5 rounded-full shrink-0"
              style={{ background: hoveredItem.offside ? '#f43f5e' : '#22c55e' }} />
            <span className="font-700 text-tx-primary text-[13px]">{hoveredItem.detection.class_name}</span>
          </div>
          <div className="space-y-1.5 text-xs">
            <div className="flex justify-between gap-6">
              <span className="text-tx-muted">Estado</span>
              <span className="font-700" style={{ color: hoveredItem.offside ? '#f43f5e' : '#22c55e' }}>
                {hoveredItem.offside ? 'OFFSIDE' : 'ONSIDE'}
              </span>
            </div>
            <div className="flex justify-between gap-6">
              <span className="text-tx-muted">Confianza</span>
              <span className="text-tx-primary font-600">{(hoveredItem.detection.conf * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between gap-6">
              <span className="text-tx-muted">Clase ID</span>
              <span className="text-tx-primary font-mono">{hoveredItem.detection.class_id}</span>
            </div>
            <div className="mt-2 pt-2 border-t border-border-subtle">
              <p className="text-tx-muted text-[10px] mb-1 uppercase tracking-wider">BBox (px)</p>
              <p className="font-mono text-[11px] text-tx-secondary">
                ({Math.round(hoveredItem.detection.x1)}, {Math.round(hoveredItem.detection.y1)})
                {' → '}
                ({Math.round(hoveredItem.detection.x2)}, {Math.round(hoveredItem.detection.y2)})
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
