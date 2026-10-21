'use client'

import { CircleMarker, MapContainer, Popup, TileLayer, useMapEvents } from 'react-leaflet'
import { LEVEL_COLOR, type Report } from '@/lib/api'

// Pereira, where this project would be used first
const CENTER: [number, number] = [4.8133, -75.6961]

function PickLocation({ onPick }: { onPick: (lat: number, lng: number) => void }) {
  useMapEvents({ click: (e) => onPick(e.latlng.lat, e.latlng.lng) })
  return null
}

export default function DamageMap({
  reports,
  picked,
  onPick,
  selected,
  onSelect,
}: {
  reports: Report[]
  picked: [number, number] | null
  onPick: (lat: number, lng: number) => void
  selected: number | null
  onSelect: (id: number) => void
}) {
  return (
    <MapContainer center={CENTER} zoom={14} className="map" scrollWheelZoom>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <PickLocation onPick={onPick} />
      {reports.map((r) => (
        <CircleMarker
          key={r.id}
          center={[r.lat, r.lng]}
          radius={r.id === selected ? 12 : 8}
          pathOptions={{ color: '#fff', weight: 2, fillColor: LEVEL_COLOR[r.level], fillOpacity: 0.95 }}
          eventHandlers={{ click: () => onSelect(r.id) }}
        >
          <Popup>
            <b>Report #{r.id}</b> · {r.level} priority
            <br />
            {r.detections.length} damage{r.detections.length === 1 ? '' : 's'}: {Object.keys(r.treatments).join(', ') || '—'}
          </Popup>
        </CircleMarker>
      ))}
      {picked && (
        <CircleMarker center={picked} radius={7} pathOptions={{ color: '#1c7ed6', weight: 3, fillOpacity: 0 }} />
      )}
    </MapContainer>
  )
}
