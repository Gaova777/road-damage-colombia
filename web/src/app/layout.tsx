import type { Metadata } from 'next'
import { Public_Sans, IBM_Plex_Mono } from 'next/font/google'
import 'leaflet/dist/leaflet.css'
import './globals.css'

const sans = Public_Sans({ subsets: ['latin'], variable: '--sans' })
const mono = IBM_Plex_Mono({ subsets: ['latin'], weight: ['400', '500'], variable: '--mono' })

export const metadata: Metadata = {
  title: 'Road damage · repair priorities',
  description: 'Road damage detection with RT-DETR, turned into repair priorities on a map.',
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  )
}
