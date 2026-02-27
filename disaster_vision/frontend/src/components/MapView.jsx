/**
 * MapView — Leaflet map displaying color-coded tile markers.
 *
 * Each tile is rendered as a CircleMarker colored by damage severity.
 * Top-priority tiles get a pulsing ring.
 * Clicking a marker opens a detailed popup.
 */

import React, { useEffect, useRef } from 'react'
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import { severityColor, CLASS_LABELS } from '../utils/colorMap.js'

const LEGEND_ITEMS = [
    { cls: 'no-damage', label: 'No Damage' },
    { cls: 'minor', label: 'Minor' },
    { cls: 'major', label: 'Major' },
    { cls: 'destroyed', label: 'Destroyed' },
]

/** Auto-fit map bounds when tiles are provided. */
function BoundsAdjuster({ tiles }) {
    const map = useMap()
    useEffect(() => {
        if (!tiles || tiles.length === 0) return
        const valid = tiles.filter(t => t.lat != null && t.lon != null)
        if (valid.length === 0) return
        const lats = valid.map(t => t.lat)
        const lons = valid.map(t => t.lon)
        const sw = [Math.min(...lats) - 0.01, Math.min(...lons) - 0.01]
        const ne = [Math.max(...lats) + 0.01, Math.max(...lons) + 0.01]
        map.fitBounds([sw, ne], { padding: [30, 30] })
    }, [tiles, map])
    return null
}

function TilePopup({ tile, topPriorityIds }) {
    const isTop = topPriorityIds.has(tile.tile_id)
    const probs = tile.probability_distribution || {}

    return (
        <div className="popup-inner">
            <div className="popup-title">
                {isTop ? '⭐ ' : ''}
                {tile.tile_id}
            </div>
            <div className="popup-row">
                <span className="popup-key">Damage Class</span>
                <span className="popup-val" style={{ color: severityColor(tile.damage_class) }}>
                    {CLASS_LABELS[tile.damage_class] || tile.damage_class}
                </span>
            </div>
            <div className="popup-row">
                <span className="popup-key">Confidence</span>
                <span className="popup-val">{(tile.confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="popup-row">
                <span className="popup-key">Priority Score</span>
                <span className="popup-val">{tile.priority_score.toFixed(3)}</span>
            </div>
            <div className="popup-row">
                <span className="popup-key">Pre-image</span>
                <span className="popup-val" style={{ fontSize: 10, maxWidth: 120, wordBreak: 'break-all' }}>
                    {tile.filename_pre}
                </span>
            </div>
            {Object.keys(probs).length > 0 && (
                <div className="popup-prob-bar">
                    {Object.entries(probs).map(([cls, prob]) => (
                        <div key={cls} className="prob-row">
                            <span className="prob-label">{CLASS_LABELS[cls] || cls}</span>
                            <div className="prob-track">
                                <div
                                    className="prob-fill"
                                    style={{ width: `${(prob * 100).toFixed(1)}%`, background: severityColor(cls) }}
                                />
                            </div>
                            <span className="prob-pct">{(prob * 100).toFixed(1)}%</span>
                        </div>
                    ))}
                </div>
            )}
            {tile.status === 'error' && (
                <div style={{ marginTop: 8, color: '#f85149', fontSize: 11 }}>
                    ⚠ Error: {tile.error_message}
                </div>
            )}
        </div>
    )
}

export default function MapView({ tiles, topPriorityTiles }) {
    const topIds = new Set((topPriorityTiles || []).map(t => t.tile_id))

    const validTile = tiles?.find(t => t.lat != null && t.lon != null)
    const center = validTile
        ? [validTile.lat, validTile.lon]
        : [20, 78]

    return (
        <div className="map-container">
            <MapContainer
                center={center}
                zoom={tiles && tiles.length > 0 ? 14 : 5}
                className="map-wrap"
                zoomControl
            >
                <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                    maxZoom={19}
                />
                {tiles && tiles.map(tile => {
                    if (tile.lat == null || tile.lon == null) return null
                    const color = severityColor(tile.damage_class)
                    const isTop = topIds.has(tile.tile_id)
                    const radius = isTop ? 14 : 10

                    return (
                        <CircleMarker
                            key={tile.tile_id}
                            center={[tile.lat, tile.lon]}
                            radius={radius}
                            pathOptions={{
                                color: isTop ? '#ffffff' : color,
                                weight: isTop ? 3 : 1.5,
                                fillColor: color,
                                fillOpacity: 0.85,
                            }}
                        >
                            <Popup className="tile-popup" maxWidth={280}>
                                <TilePopup tile={tile} topPriorityIds={topIds} />
                            </Popup>
                        </CircleMarker>
                    )
                })}
                <BoundsAdjuster tiles={tiles} />
            </MapContainer>

            {/* Legend */}
            <div className="map-legend">
                <div className="legend-title">Damage Severity</div>
                {LEGEND_ITEMS.map(({ cls, label }) => (
                    <div key={cls} className="legend-row">
                        <div className="legend-dot" style={{ background: severityColor(cls) }} />
                        {label}
                    </div>
                ))}
                <div className="legend-row" style={{ marginTop: 4, borderTop: '1px solid #30363d', paddingTop: 4 }}>
                    <div className="legend-dot" style={{ background: '#58a6ff', border: '2px solid #fff' }} />
                    Top Priority
                </div>
            </div>

            {(!tiles || tiles.length === 0) && (
                <div className="empty-state" style={{ position: 'absolute', inset: 0, zIndex: 999, background: 'rgba(13,17,23,0.7)' }}>
                    <div className="empty-icon">🗺</div>
                    <div>Upload and analyze images to see results on the map.</div>
                </div>
            )}
        </div>
    )
}
