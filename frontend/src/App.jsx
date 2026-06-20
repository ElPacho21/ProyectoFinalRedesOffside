import { useState } from 'react'
import Step1Upload from './components/Step1Upload.jsx'
import Step2Detection from './components/Step2Detection.jsx'
import Step3VanishingPoint from './components/Step3VanishingPoint.jsx'
import Step4TeamSelection from './components/Step4TeamSelection.jsx'
import Step5Result from './components/Step5Result.jsx'

const STEPS = [
  'Upload',
  'Detect',
  'Vanishing Point',
  'Team',
  'Result',
]

export default function App() {
  const [step, setStep] = useState(1)

  // Step 1 output
  const [mediaInfo, setMediaInfo] = useState(null)
  const [frameIndex, setFrameIndex] = useState(0)
  const [confidence, setConfidence] = useState(0.25)

  // Current frame base64 (for canvas in step 3)
  const [currentFrameBase64, setCurrentFrameBase64] = useState(null)

  // Step 2 output
  const [detections, setDetections] = useState(null)
  const [suggestedTeam, setSuggestedTeam] = useState(null)

  // Step 3 output
  const [vp, setVp] = useState(null)

  // Step 4 output
  const [attackingTeamId, setAttackingTeamId] = useState(5)
  const [direction, setDirection] = useState(null)
  const [refDefenderIdx, setRefDefenderIdx] = useState(null)
  const [footPoint, setFootPoint] = useState('medio')

  const reset = () => {
    setStep(1)
    setMediaInfo(null)
    setFrameIndex(0)
    setConfidence(0.25)
    setCurrentFrameBase64(null)
    setDetections(null)
    setSuggestedTeam(null)
    setVp(null)
    setAttackingTeamId(5)
    setDirection(null)
    setRefDefenderIdx(null)
    setFootPoint('medio')
  }

  const handleStep1Done = ({ mediaInfo: mi, frameIndex: fi, confidence: co }) => {
    setMediaInfo(mi)
    setFrameIndex(fi)
    setConfidence(co)
    setCurrentFrameBase64(mi.preview_base64)
    setStep(2)
  }

  const handleStep2Done = ({ detections: dets, suggested_team }) => {
    setDetections(dets)
    setSuggestedTeam(suggested_team)
    // Use current preview image for canvas
    setCurrentFrameBase64(mediaInfo.preview_base64)
    setStep(3)
  }

  const handleStep3Done = ({ vp: v }) => {
    setVp(v)
    setStep(4)
  }

  const handleStep4Done = ({ attackingTeamId: atk, direction: dir, refDefenderIdx: rdi, footPoint: fp }) => {
    setAttackingTeamId(atk)
    setDirection(dir)
    setRefDefenderIdx(rdi)
    setFootPoint(fp)
    setStep(5)
  }

  return (
    <div className="app">
      <div className="header">
        <h1>⚽ Offside Detection</h1>
        <p>Football offside analysis using YOLOv8 + geometric projection</p>
      </div>

      <div className="stepper">
        {STEPS.map((label, i) => {
          const n = i + 1
          const isDone = step > n
          const isActive = step === n
          return (
            <div className="step-item" key={n}>
              {i > 0 && <div className={`step-connector${isDone ? ' done' : ''}`} />}
              <div className={`step-circle${isActive ? ' active' : isDone ? ' done' : ''}`}>
                {isDone ? '✓' : n}
              </div>
              <span className={`step-label${isActive ? ' active' : isDone ? ' done' : ''}`}>
                {label}
              </span>
            </div>
          )
        })}
      </div>

      {step === 1 && (
        <Step1Upload onDone={handleStep1Done} />
      )}

      {step === 2 && mediaInfo && (
        <Step2Detection
          mediaInfo={mediaInfo}
          frameIndex={frameIndex}
          confidence={confidence}
          onDone={handleStep2Done}
          onBack={() => setStep(1)}
        />
      )}

      {step === 3 && mediaInfo && currentFrameBase64 && (
        <Step3VanishingPoint
          mediaInfo={mediaInfo}
          currentFrameBase64={currentFrameBase64}
          onDone={handleStep3Done}
          onBack={() => setStep(2)}
        />
      )}

      {step === 4 && (
        <Step4TeamSelection
          suggestedTeam={suggestedTeam}
          onDone={handleStep4Done}
          onBack={() => setStep(3)}
        />
      )}

      {step === 5 && mediaInfo && detections && vp && (
        <Step5Result
          mediaInfo={mediaInfo}
          frameIndex={frameIndex}
          detections={detections}
          vp={vp}
          attackingTeamId={attackingTeamId}
          direction={direction}
          refDefenderIdx={refDefenderIdx}
          footPoint={footPoint}
          onBack={() => setStep(4)}
          onRestart={reset}
        />
      )}
    </div>
  )
}
