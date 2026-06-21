import { useRef, useState, useEffect } from 'react'
import { IconZoomIn, IconZoomOut, IconRefresh, IconAlertTriangle, IconFlag, IconShieldCheck } from '@tabler/icons-react'

export default function Step4Results({ resultImageB64, resultado, imageSize, onReset }) {
  const [zoom,         setZoom]         = useState(1)
  const [naturalSize,  setNaturalSize]  = useState({ w: 0, h: 0 })
  const [hoveredIndex, setHoveredIndex] = useState(null)
  const [tooltipPos,   setTooltipPos]   = useState({ x: 0, y: 0 })

  const overlayRefs    = useRef([])
  const defOverlayRef  = useRef(null)
  const clearRef       = useRef(null)
  const scrollRef     = useRef(null)
  const contentDivRef = useRef(null)
  const spacerDivRef  = useRef(null)
  const zoomRef       = useRef(1)
  const naturalRef    = useRef({ w: 0, h: 0 })
  const isDragging    = useRef(false)
  const lastPos       = useRef({ x: 0, y: 0 })

  useEffect(() => { zoomRef.current = zoom }, [zoom])

  // Ctrl+Scroll — direct DOM manipulation: transform + scroll in same synchronous call, no flicker
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const handler = (e) => {
      if (!e.ctrlKey) return
      e.preventDefault()
      const rect     = el.getBoundingClientRect()
      const viewX    = e.clientX - rect.left
      const viewY    = e.clientY - rect.top
      const contentX = el.scrollLeft + viewX
      const contentY = el.scrollTop  + viewY
      const oldZoom  = zoomRef.current
      const newZoom  = Math.max(1, Math.min(5, oldZoom + (e.deltaY > 0 ? -0.15 : 0.15)))
      zoomRef.current = newZoom

      if (contentDivRef.current)
        contentDivRef.current.style.transform = `scale(${newZoom})`
      const ns = naturalRef.current
      if (spacerDivRef.current && ns.w > 0) {
        spacerDivRef.current.style.width  = `${ns.w * newZoom}px`
        spacerDivRef.current.style.height = `${ns.h * newZoom}px`
      }
      el.scrollLeft = contentX * (newZoom / oldZoom) - viewX
      el.scrollTop  = contentY * (newZoom / oldZoom) - viewY

      setZoom(newZoom) // only for toolbar % display
    }
    el.addEventListener('wheel', handler, { passive: false })
    return () => el.removeEventListener('wheel', handler)
  }, [])

  // Drag-to-pan
  useEffect(() => {
    const onMove = (e) => {
      if (!isDragging.current) return
      const dx = e.clientX - lastPos.current.x
      const dy = e.clientY - lastPos.current.y
      lastPos.current = { x: e.clientX, y: e.clientY }
      if (scrollRef.current) {
        scrollRef.current.scrollLeft -= dx
        scrollRef.current.scrollTop  -= dy
      }
    }
    const onUp = () => {
      if (!isDragging.current) return
      isDragging.current = false
      document.body.style.cursor     = ''
      document.body.style.userSelect = ''
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup',   onUp)
    return () => {
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup',   onUp)
    }
  }, [])

  if (!resultado || !resultImageB64) return null

  const { hay_offside, atacantes_resultado = [], advertencias = [], penultimo_defensor, pelota } = resultado
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

  const hoveredItem  = hoveredIndex !== null ? atacantes_resultado[hoveredIndex] : null
  const showBallRef  = !!(hoveredItem?.adelantado_al_defensor && !hoveredItem?.adelantado_a_pelota)
  const showDefRef   = hoveredIndex !== null && !showBallRef

  const spacerW = naturalSize.w > 0 ? naturalSize.w * zoom : '100%'
  const spacerH = naturalSize.h > 0 ? naturalSize.h * zoom : 'auto'

  return (
    <div className="space-y-4 animate-fade-in">

      {/* ── Verdict bar ─────────────────────────────────────────── */}
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

      {/* ── Image ───────────────────────────────────────────────── */}
      <div className="glass rounded-2xl overflow-hidden">
        <div className="px-4 py-2.5 border-b border-border-subtle flex items-center justify-between">
          <span className="text-xs font-600 text-tx-muted">
            Resultado
            <span className="ml-2 font-400 opacity-40">Ctrl+Scroll para zoom · arrastrar para mover</span>
          </span>
          <div className="flex items-center gap-1">
            <button onClick={() => setZoom(z => Math.max(1, z - 0.2))}
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

        <div
          ref={scrollRef}
          className="overflow-auto bg-black"
          style={{ cursor: 'grab', maxHeight: '100vh' }}
          onMouseDown={(e) => {
            if (e.button !== 0) return
            isDragging.current = true
            lastPos.current = { x: e.clientX, y: e.clientY }
            document.body.style.cursor     = 'grabbing'
            document.body.style.userSelect = 'none'
          }}
        >
          <div
            ref={spacerDivRef}
            style={{ position: 'relative', width: spacerW, height: spacerH, minWidth: '100%' }}
          >
            <div
              ref={contentDivRef}
              style={{
                position: 'absolute', top: 0, left: 0,
                transformOrigin: 'top left',
                transform: `scale(${zoom})`,
                width: naturalSize.w > 0 ? naturalSize.w : '100%',
                willChange: 'transform',
              }}
            >
              <img
                src={`data:image/jpeg;base64,${resultImageB64}`}
                alt="Resultado"
                style={{ display: 'block', width: '100%' }}
                draggable={false}
                onLoad={(e) => {
                  const w = e.target.offsetWidth
                  const h = e.target.offsetHeight
                  naturalRef.current = { w, h }
                  setNaturalSize({ w, h })
                }}
              />
              {/* Defender overlay — visible when any attacker is hovered */}
              {/* Defender overlay */}
              {imageSize && penultimo_defensor && (() => {
                const d = penultimo_defensor
                return (
                  <div
                    ref={defOverlayRef}
                    style={{
                      position: 'absolute',
                      left:   `${(d.x1 / imageSize.width)  * 100}%`,
                      top:    `${(d.y1 / imageSize.height) * 100}%`,
                      width:  `${((d.x2 - d.x1) / imageSize.width)  * 100}%`,
                      height: `${((d.y2 - d.y1) / imageSize.height) * 100}%`,
                      borderRadius: '3px',
                      pointerEvents: 'none',
                      transition: 'background 0.15s, outline 0.15s, box-shadow 0.15s',
                      background: showDefRef ? 'rgba(251,191,36,0.3)' : 'transparent',
                      outline: showDefRef ? '3px solid #fbbf24' : 'none',
                      boxShadow: showDefRef
                        ? '0 0 0 4px rgba(251,191,36,0.2), inset 0 0 20px rgba(251,191,36,0.15)'
                        : 'none',
                    }}
                  />
                )
              })()}

              {/* Ball overlay */}
              {imageSize && pelota && (() => {
                const d = pelota
                return (
                  <div
                    style={{
                      position: 'absolute',
                      left:   `${(d.x1 / imageSize.width)  * 100}%`,
                      top:    `${(d.y1 / imageSize.height) * 100}%`,
                      width:  `${((d.x2 - d.x1) / imageSize.width)  * 100}%`,
                      height: `${((d.y2 - d.y1) / imageSize.height) * 100}%`,
                      borderRadius: '50%',
                      pointerEvents: 'none',
                      transition: 'background 0.15s, outline 0.15s, box-shadow 0.15s',
                      background: showBallRef ? 'rgba(251,191,36,0.3)' : 'transparent',
                      outline: showBallRef ? '3px solid #fbbf24' : 'none',
                      boxShadow: showBallRef
                        ? '0 0 0 4px rgba(251,191,36,0.2), inset 0 0 20px rgba(251,191,36,0.15)'
                        : 'none',
                    }}
                  />
                )
              })()}

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
                        ? (item.offside ? 'rgba(244,63,94,0.35)' : 'rgba(34,197,94,0.35)')
                        : 'transparent',
                      outline: isHovered
                        ? `3px solid ${item.offside ? '#f43f5e' : '#22c55e'}`
                        : 'none',
                      boxShadow: isHovered
                        ? `0 0 0 4px ${item.offside ? 'rgba(244,63,94,0.2)' : 'rgba(34,197,94,0.2)'}, inset 0 0 20px ${item.offside ? 'rgba(244,63,94,0.15)' : 'rgba(34,197,94,0.15)'}`
                        : 'none',
                    }}
                    onMouseEnter={() => { if (!isDragging.current) onEnter(i) }}
                    onMouseLeave={onLeave}
                  />
                )
              })}
            </div>
          </div>
        </div>
      </div>

      {/* ── Player chips ────────────────────────────────────────── */}
      <div className="glass rounded-2xl overflow-x-auto px-3" style={{ scrollbarWidth: 'none', paddingTop: '10px', paddingBottom: '10px' }}>
        <div className="flex items-center gap-2">
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
                  transform: isHovered ? 'translateY(-3px)' : 'none',
                  boxShadow: isHovered
                    ? (item.offside ? '0 0 12px rgba(244,63,94,0.4)' : '0 0 12px rgba(34,197,94,0.4)')
                    : 'none',
                }}
                onMouseEnter={() => onEnter(i)}
                onMouseLeave={onLeave}
              >
                <span className="text-[11px] font-700 text-tx-primary">#{i + 1}</span>
                <span className="text-[10px] font-800 tracking-wide"
                  style={{ color: item.offside ? '#f43f5e' : '#22c55e' }}>
                  {item.offside ? 'OFFSIDE' : 'OK'}
                </span>
                <span className="text-[10px] text-tx-muted font-mono">
                  {(item.detection.conf * 100).toFixed(0)}%
                </span>
              </button>
            )
          })}
        </div>
      </div>

      {/* ── Advertencias ────────────────────────────────────────── */}
      {advertencias.length > 0 && (
        <div className="rounded-2xl px-5 py-3.5 flex flex-col gap-2"
          style={{ background: 'rgba(245,158,11,0.05)', border: '1px solid rgba(245,158,11,0.2)' }}>
          <div className="flex items-center gap-2">
            <IconAlertTriangle size={14} stroke={2} className="text-warn shrink-0" />
            <span className="text-[11px] font-700 uppercase tracking-widest text-warn">
              Advertencia{advertencias.length > 1 ? 's' : ''}
            </span>
          </div>
          <ul className="space-y-1">
            {advertencias.map((w, i) => (
              <li key={i} className="text-sm text-tx-secondary leading-relaxed">{w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* ── Tooltip anchored above player bbox ──────────────────── */}
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
            {showBallRef && pelota ? (
              <div className="mt-2 pt-2 border-t border-border-subtle flex justify-between gap-6">
                <span style={{ color: '#fbbf24' }}>vs. pelota</span>
                <span className="text-tx-primary font-600">{pelota.class_name}</span>
              </div>
            ) : penultimo_defensor ? (
              <div className="mt-2 pt-2 border-t border-border-subtle flex justify-between gap-6">
                <span style={{ color: '#fbbf24' }}>vs. defensor</span>
                <span className="text-tx-primary font-600">{penultimo_defensor.class_name}</span>
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  )
}
