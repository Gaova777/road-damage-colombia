'use client'

import { CLASS_COLOR, LEVEL_COLOR, type Result } from '@/lib/api'

export default function PhotoResult({ src, result }: { src: string; result: Result }) {
  return (
    <div className="result">
      <div className="photo">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt="road photo" />
        <svg viewBox="0 0 1 1" preserveAspectRatio="none">
          {result.detections.map((d, i) => {
            const [x1, y1, x2, y2] = d.box
            return (
              <rect key={i} x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="none"
                stroke={CLASS_COLOR[d.cls]} strokeWidth={3} vectorEffect="non-scaling-stroke" />
            )
          })}
        </svg>
      </div>
      <div className="verdict" style={{ background: LEVEL_COLOR[result.level] }}>
        {result.level === 'none' ? 'No damage found' : `${result.level} priority`} <span>score {result.score}</span>
      </div>
      <table className="dets">
        <thead>
          <tr>
            <th>Damage</th>
            <th>Treatment</th>
            <th>Conf.</th>
            <th>Severity</th>
          </tr>
        </thead>
        <tbody>
          {result.detections
            .slice()
            .sort((a, b) => b.severity - a.severity)
            .map((d, i) => (
              <tr key={i}>
                <td>
                  <i className="swatch" style={{ background: CLASS_COLOR[d.cls] }} />
                  {d.name} {d.in_lane && <em>in lane</em>}
                </td>
                <td>{d.treatment}</td>
                <td>{Math.round(d.confidence * 100)}%</td>
                <td>{d.severity.toFixed(2)}</td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  )
}
