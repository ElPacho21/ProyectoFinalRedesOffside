import { useRef, useEffect, useState, useCallback } from 'react'

export default function CanvasVP({ imageBase64, imageWidth, imageHeight, onPointsChange }) {
  const canvasRef = useRef()
  const [points, setPoints] = useState([])
  const imgRef = useRef(null)
  const [imgLoaded, setImgLoaded] = useState(false)

  // Load image once
  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      imgRef.current = img
      setImgLoaded(true)
    }
    img.src = `data:image/jpeg;base64,${imageBase64}`
  }, [imageBase64])

  // Draw on canvas whenever points or image change
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !imgLoaded) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width
    const H = canvas.height

    ctx.clearRect(0, 0, W, H)
    ctx.drawImage(imgRef.current, 0, 0, W, H)

    // Draw guide lines between pairs
    if (points.length >= 2) {
      ctx.strokeStyle = 'rgba(255,255,100,0.6)'
      ctx.lineWidth = 1.5
      ctx.setLineDash([6, 3])
      ctx.beginPath()
      ctx.moveTo(points[0].cx, points[0].cy)
      ctx.lineTo(points[1].cx, points[1].cy)
      ctx.stroke()
    }
    if (points.length >= 4) {
      ctx.beginPath()
      ctx.moveTo(points[2].cx, points[2].cy)
      ctx.lineTo(points[3].cx, points[3].cy)
      ctx.stroke()
    }
    ctx.setLineDash([])

    // Draw X markers
    points.forEach((p, i) => {
      const R = 8
      const COLORS = ['#ff4444', '#ff4444', '#44aaff', '#44aaff']
      const c = COLORS[i] ?? '#ff4444'

      // White outline X
      ctx.lineWidth = 4
      ctx.strokeStyle = 'rgba(255,255,255,0.8)'
      ctx.beginPath()
      ctx.moveTo(p.cx - R, p.cy - R); ctx.lineTo(p.cx + R, p.cy + R)
      ctx.moveTo(p.cx + R, p.cy - R); ctx.lineTo(p.cx - R, p.cy + R)
      ctx.stroke()

      // Colored X
      ctx.lineWidth = 2.5
      ctx.strokeStyle = c
      ctx.beginPath()
      ctx.moveTo(p.cx - R, p.cy - R); ctx.lineTo(p.cx + R, p.cy + R)
      ctx.moveTo(p.cx + R, p.cy - R); ctx.lineTo(p.cx - R, p.cy + R)
      ctx.stroke()

      // Number label
      ctx.fillStyle = c
      ctx.font = 'bold 12px sans-serif'
      ctx.fillText(i + 1, p.cx + R + 2, p.cy - R)
    })
  }, [points, imgLoaded])

  const handleClick = useCallback((e) => {
    if (points.length >= 4) return
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    const cx = (e.clientX - rect.left) * scaleX
    const cy = (e.clientY - rect.top) * scaleY

    // Convert canvas coords → original image coords
    const ox = (cx / canvas.width) * imageWidth
    const oy = (cy / canvas.height) * imageHeight

    const newPoints = [...points, { cx, cy, ox, oy }]
    setPoints(newPoints)
    onPointsChange(newPoints.map(p => [p.ox, p.oy]))
  }, [points, imageWidth, imageHeight, onPointsChange])

  const reset = () => {
    setPoints([])
    onPointsChange([])
  }

  // Canvas dimensions: fit within 900px wide
  const displayW = Math.min(900, imageWidth)
  const displayH = Math.round(displayW * imageHeight / imageWidth)

  return (
    <div>
      <p className="canvas-instructions">
        Click <strong>4 points</strong> on the image: points <strong style={{color:'#ff6666'}}>1–2</strong> on one field line,
        points <strong style={{color:'#66aaff'}}>3–4</strong> on a parallel line (e.g. sideline + penalty area line).
        The vanishing point is computed as their intersection.
      </p>

      <div className="canvas-container">
        <canvas
          ref={canvasRef}
          width={displayW}
          height={displayH}
          onClick={handleClick}
          style={{ borderRadius: 8, maxWidth: '100%', cursor: points.length < 4 ? 'crosshair' : 'default' }}
        />
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 10 }}>
        <div className="points-counter">
          {[0,1,2,3].map(i => (
            <div key={i} className={`point-badge ${points[i] ? 'filled' : 'empty'}`}>
              {i + 1}
            </div>
          ))}
          <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 4, alignSelf: 'center' }}>
            {points.length}/4 points placed
          </span>
        </div>
        {points.length > 0 && (
          <button className="btn btn-danger" style={{ padding: '4px 10px', fontSize: 12 }} onClick={reset}>
            Reset
          </button>
        )}
      </div>
    </div>
  )
}
