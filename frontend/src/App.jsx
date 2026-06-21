import { useState } from 'react'
import { IconAlertCircle, IconBallFootball } from '@tabler/icons-react'
import api, { b64toBlob } from './api/client'
import StepIndicator from './components/StepIndicator'
import Step1Upload from './components/Step1Upload'
import Step2TeamConfig from './components/Step2TeamConfig'
import Step3VP from './components/Step3VP'
import Step4Results from './components/Step4Results'

const IOU = 0.7

const INITIAL_STATE = {
  step: 1,
  originalImageB64: null,
  detectedImageB64: null,
  detections: [],
  vpAuto: null,
  vpLines: null,
  vpFinal: null,
  classCount: {},
  teamSamples: { team1: null, team2: null },
  imageSize: null,
  attackingTeam: null,
  goalOnRight: null,
  referencePoint: 'medio',
  resultImageB64: null,
  resultado: null,
  loading: false,
  error: null,
}

export default function App() {
  const [s, setS] = useState(INITIAL_STATE)
  const set = (patch) => setS((prev) => ({ ...prev, ...patch }))

  const handleDetect = async (imageFile) => {
    set({ loading: true, error: null })
    try {
      const form = new FormData()
      form.append('image', imageFile, 'image.jpg')
      form.append('iou', IOU)
      const { data } = await api.post('/api/detect', form)
      set({
        originalImageB64: data.image_b64,
        detectedImageB64: data.detected_image_b64,
        detections: data.detections,
        vpAuto: data.vp,
        vpLines: data.vp_lines,
        vpFinal: data.vp,
        classCount: data.class_counts,
        teamSamples: data.team_samples,
        imageSize: data.image_size,
        step: 2,
        loading: false,
      })
    } catch (err) {
      set({ loading: false, error: err?.response?.data?.detail || 'Error al detectar. Verificá que el backend esté corriendo en :8000.' })
    }
  }

  const handleExtractFrame = async (videoFile, frameNumber) => {
    const form = new FormData()
    form.append('video', videoFile)
    form.append('frame', frameNumber)
    const { data } = await api.post('/api/extract-frame', form)
    return data
  }

  const handleCalculateOffside = async (vp) => {
    set({ loading: true, error: null })
    try {
      const { data } = await api.post('/api/calculate-offside', {
        image_b64: s.originalImageB64,
        detections: s.detections,
        vp: vp ?? s.vpFinal,
        attacking_team_id: s.attackingTeam,
        goal_on_right: s.goalOnRight,
        reference_point: s.referencePoint,
      })
      set({
        resultImageB64: data.result_image_b64,
        resultado: data.resultado,
        imageSize: data.image_size,
        step: 4,
        loading: false,
      })
    } catch (err) {
      set({ loading: false, error: err?.response?.data?.detail || 'Error al calcular offside.' })
    }
  }

  const handleReset = () => setS(INITIAL_STATE)

  const handleStepClick = (n) => {
    if (n < s.step) set({ step: n })
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="sticky top-0 z-20 glass border-b border-border-subtle">
        <div className="max-w-5xl mx-auto px-5 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-accent-muted border border-accent/20 flex items-center justify-center">
              <IconBallFootball size={18} stroke={1.6} className="text-accent" />
            </div>
            <div className="flex items-baseline gap-2">
              <span className="font-display font-700 text-sm tracking-widest text-gradient">OFFSIDE</span>
              <span className="font-display text-sm tracking-widest text-tx-muted font-400">DETECTION</span>
            </div>
          </div>
          <span className="text-[11px] text-tx-muted font-mono">YOLOv26m</span>
        </div>
      </header>

      <main className="flex-1 max-w-5xl mx-auto w-full px-4 py-8">
        <StepIndicator currentStep={s.step} onStepClick={handleStepClick} />

        {s.error && (
          <div className="mt-5 px-4 py-3 rounded-xl glass flex items-center gap-2.5 animate-fade-in"
            style={{ border: '1px solid rgba(244,63,94,0.3)', color: '#f87171' }}>
            <IconAlertCircle size={16} stroke={2} className="shrink-0" />
            <span className="text-sm">{s.error}</span>
          </div>
        )}

        <div className="mt-7 animate-slide-up" key={s.step}>
          {s.step === 1 && (
            <Step1Upload
              loading={s.loading}
              onDetect={handleDetect}
              onExtractFrame={handleExtractFrame}
            />
          )}
          {s.step === 2 && (
            <Step2TeamConfig
              teamSamples={s.teamSamples}
              attackingTeam={s.attackingTeam}
              goalOnRight={s.goalOnRight}
              referencePoint={s.referencePoint}
              onChange={(patch) => set(patch)}
              onNext={() => set({ step: 3 })}
              onBack={() => set({ step: 1 })}
            />
          )}
          {s.step === 3 && (
            <Step3VP
              detectedImageB64={s.detectedImageB64}
              imageSize={s.imageSize}
              vpAuto={s.vpAuto}
              vpLines={s.vpLines}
              onVPChange={(vp) => set({ vpFinal: vp })}
              loading={s.loading}
              onCalculate={handleCalculateOffside}
              onBack={() => set({ step: 2 })}
            />
          )}
          {s.step === 4 && (
            <Step4Results
              resultImageB64={s.resultImageB64}
              resultado={s.resultado}
              imageSize={s.imageSize}
              onReset={handleReset}
            />
          )}
        </div>
      </main>

      <footer className="border-t border-border-subtle glass mt-auto">
        <div className="max-w-5xl mx-auto px-5 py-3.5 flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <IconBallFootball size={14} stroke={1.5} className="text-accent opacity-50" />
            <span className="text-[11px] font-600 text-tx-muted">Offside Detection</span>
            <span className="text-[10px] text-tx-muted opacity-30 mx-1">·</span>
            <span className="text-[10px]" style={{ color: '#4a5878' }}>UTN FRM · Redes Neuronales</span>
          </div>
          <p className="text-[10px] text-right" style={{ color: '#4a5878' }}>
            F. Pacci · N. Ocampo · V. Isgro · B. Lucero · J.P. Costa
          </p>
        </div>
      </footer>
    </div>
  )
}
