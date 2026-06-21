import { IconCheck } from '@tabler/icons-react'

const STEPS = [
  { n: 1, label: 'Subir' },
  { n: 2, label: 'Equipo' },
  { n: 3, label: 'Punto de Fuga' },
  { n: 4, label: 'Resultado' },
]

export default function StepIndicator({ currentStep, onStepClick }) {
  return (
    <div className="flex items-center justify-center">
      {STEPS.map((step, i) => {
        const done     = currentStep > step.n
        const active   = currentStep === step.n
        const clickable = done

        return (
          <div key={step.n} className="flex items-center">
            <div className="flex flex-col items-center gap-1.5">
              <div
                onClick={() => clickable && onStepClick?.(step.n)}
                className="w-9 h-9 rounded-full flex items-center justify-center transition-all duration-300"
                style={{
                  background: done ? 'linear-gradient(135deg, #22c55e, #16a34a)' : 'transparent',
                  border: done ? 'none' : active ? '2px solid #22c55e' : '2px solid #1a2540',
                  boxShadow: active ? '0 0 16px rgba(34,197,94,0.35)' : 'none',
                  animation: active ? 'glowPulse 2s ease-in-out infinite' : 'none',
                  cursor: clickable ? 'pointer' : 'default',
                }}
                onMouseEnter={(e) => { if (clickable) e.currentTarget.style.filter = 'brightness(1.15)' }}
                onMouseLeave={(e) => { if (clickable) e.currentTarget.style.filter = 'none' }}
              >
                {done ? (
                  <IconCheck size={16} stroke={2.5} color="white" />
                ) : (
                  <span className="font-display text-sm font-700"
                    style={{ color: active ? '#22c55e' : '#4a5878' }}>
                    {step.n}
                  </span>
                )}
              </div>
              <span
                className="text-[11px] font-semibold tracking-wide whitespace-nowrap transition-colors"
                style={{ color: done ? '#22c55e' : active ? '#e2e8f5' : '#4a5878', cursor: clickable ? 'pointer' : 'default' }}
                onClick={() => clickable && onStepClick?.(step.n)}
              >
                {step.label}
              </span>
            </div>

            {i < STEPS.length - 1 && (
              <div
                className="mx-2 mb-5 transition-all duration-500"
                style={{
                  width: '64px', height: '2px',
                  background: currentStep > step.n ? 'linear-gradient(90deg, #22c55e, #16a34a)' : '#1a2540',
                  borderRadius: '1px',
                }}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}
