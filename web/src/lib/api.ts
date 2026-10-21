export const API = process.env.NEXT_PUBLIC_API ?? 'http://localhost:8000'

export type Level = 'none' | 'low' | 'medium' | 'high' | 'urgent'

export interface Detection {
  cls: 'D00' | 'D10' | 'D20' | 'D40'
  name: string
  confidence: number
  box: [number, number, number, number]
  treatment: string
  severity: number
  in_lane: boolean
}

export interface Result {
  id: number | null
  score: number
  level: Level
  treatments: Record<string, { count: number; image_share: number }>
  detections: Detection[]
}

export interface Report extends Omit<Result, 'id'> {
  id: number
  created: number
  lat: number
  lng: number
}

export async function detect(file: File, lat?: number, lng?: number): Promise<Result> {
  const form = new FormData()
  form.append('photo', file)
  if (lat !== undefined && lng !== undefined) {
    form.append('lat', String(lat))
    form.append('lng', String(lng))
  }
  const res = await fetch(`${API}/detect`, { method: 'POST', body: form })
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? res.statusText)
  return res.json()
}

export async function reports(): Promise<Report[]> {
  const res = await fetch(`${API}/reports`)
  return res.ok ? res.json() : []
}

export const LEVEL_COLOR: Record<Level, string> = {
  none: '#6b7280',
  low: '#2f9e44',
  medium: '#e8a317',
  high: '#e8590c',
  urgent: '#c92a2a',
}

export const CLASS_COLOR: Record<Detection['cls'], string> = {
  D00: '#1c7ed6',
  D10: '#7048e8',
  D20: '#e8590c',
  D40: '#c92a2a',
}
