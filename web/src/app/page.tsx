'use client'

import dynamic from 'next/dynamic'
import { useCallback, useEffect, useState } from 'react'
import { detect, LEVEL_COLOR, reports as fetchReports, type Report, type Result } from '@/lib/api'
import PhotoResult from '@/components/PhotoResult'

// Leaflet touches `window`, so the map only renders in the browser
const DamageMap = dynamic(() => import('@/components/DamageMap'), { ssr: false })

export default function Home() {
  const [reports, setReports] = useState<Report[]>([])
  const [picked, setPicked] = useState<[number, number] | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<Result | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const load = useCallback(() => {
    fetchReports().then(setReports).catch(() => setError('API not reachable'))
  }, [])
  useEffect(load, [load])

  async function onFile(file: File) {
    setError('')
    setResult(null)
    setPreview(URL.createObjectURL(file))
    let where = picked
    try {
      // a phone photo usually carries its GPS position
      const { gps } = await import('exifr')
      const pos = await gps(file)
      if (pos?.latitude && pos?.longitude) where = [pos.latitude, pos.longitude]
    } catch {}
    if (where) setPicked(where)
    setBusy(true)
    try {
      const r = await detect(file, where?.[0], where?.[1])
      setResult(r)
      if (r.id) {
        setSelected(r.id)
        load()
      }
    } catch (e) {
      setError(String((e as Error).message))
    } finally {
      setBusy(false)
    }
  }

  const counts = (['urgent', 'high', 'medium', 'low'] as const).map((l) => [l, reports.filter((r) => r.level === l).length] as const)
  const sel = reports.find((r) => r.id === selected)

  return (
    <div className="shell">
      <header className="top">
        <div className="brand">
          <span className="logo" />
          Road damage · repair priorities
        </div>
        <div className="counts">
          {counts.map(([l, n]) => (
            <span key={l}>
              <i style={{ background: LEVEL_COLOR[l] }} />
              {n} {l}
            </span>
          ))}
        </div>
        <span className="model">RT-DETR fine-tuned on RDD2022 · TensorRT FP16</span>
      </header>

      <main className="body">
        <aside className="panel">
          <section>
            <h2>Report damage</h2>
            <p className="hint">
              Click the map where the photo was taken (or use a phone photo with GPS), then upload a dashcam-style photo of
              the road.
            </p>
            <p className="where">
              {picked ? `Location: ${picked[0].toFixed(5)}, ${picked[1].toFixed(5)}` : 'No location yet: the photo is analysed but not saved'}
            </p>
            <label className="upload">
              <input type="file" accept="image/*" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
              {busy ? 'Analysing…' : 'Upload a road photo'}
            </label>
            {error && <p className="error">{error}</p>}
          </section>

          {preview && result && <PhotoResult src={preview} result={result} />}

          <section>
            <h2>Work queue</h2>
            <ol className="queue">
              {reports.map((r) => (
                <li key={r.id} className={r.id === selected ? 'sel' : ''} onClick={() => setSelected(r.id)}>
                  <i style={{ background: LEVEL_COLOR[r.level] }} />
                  <div>
                    <b>#{r.id} · {r.level}</b>
                    <span>{Object.entries(r.treatments).map(([t, v]) => `${v.count} ${t}`).join(' · ') || 'no damage'}</span>
                  </div>
                  <em>{r.score}</em>
                </li>
              ))}
              {!reports.length && <li className="empty">No reports yet.</li>}
            </ol>
            {sel && (
              <p className="hint">
                #{sel.id}: {sel.detections.map((d) => d.name).join(', ') || 'no damage'}
              </p>
            )}
          </section>
        </aside>
        <DamageMap reports={reports} picked={picked} onPick={(a, b) => setPicked([a, b])} selected={selected} onSelect={setSelected} />
      </main>
    </div>
  )
}
