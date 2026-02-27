/**
 * SummaryPanel — Region aggregate statistics and top-priority tile list.
 */

import React from 'react'
import { severityColor, CLASS_LABELS, priorityStyle } from '../utils/colorMap.js'

const CLASS_ORDER = ['no-damage', 'minor', 'major', 'destroyed']

export default function SummaryPanel({ stats, topPriority }) {
    if (!stats || stats.ok_tiles === 0) {
        return (
            <div className="empty-state">
                <div className="empty-icon">📊</div>
                <div>Analysis summary will appear here after processing.</div>
            </div>
        )
    }

    const { ok_tiles, failed_tiles, avg_confidence, avg_severity_score, class_counts, class_percentages } = stats

    function severityLabel(s) {
        if (s >= 0.75) return { label: 'CRITICAL', color: '#f85149' }
        if (s >= 0.5) return { label: 'SEVERE', color: '#db6d28' }
        if (s >= 0.25) return { label: 'MODERATE', color: '#d29922' }
        if (s > 0.0) return { label: 'MINOR', color: '#3fb950' }
        return { label: 'NEGLIGIBLE', color: '#8b949e' }
    }

    const { label: sevLabel, color: sevColor } = severityLabel(avg_severity_score)

    return (
        <div className="summary-section">

            {/* Overall assessment */}
            <div className="stat-card">
                <div className="stat-card-label">Overall Assessment</div>
                <div className="stat-card-value" style={{ color: sevColor }}>{sevLabel}</div>
                <div style={{ marginTop: 6, fontSize: 12, color: 'var(--text-secondary)' }}>
                    Avg severity: {avg_severity_score.toFixed(2)} · Confidence: {(avg_confidence * 100).toFixed(0)}%
                </div>
            </div>

            {/* Quick stats */}
            <div className="stat-grid">
                <div className="stat-card">
                    <div className="stat-card-value">{ok_tiles}</div>
                    <div className="stat-card-label">Tiles Scored</div>
                </div>
                <div className="stat-card">
                    <div className="stat-card-value" style={{ color: failed_tiles > 0 ? '#f85149' : '#3fb950' }}>
                        {failed_tiles}
                    </div>
                    <div className="stat-card-label">Failed Tiles</div>
                </div>
                <div className="stat-card">
                    <div className="stat-card-value">{(avg_confidence * 100).toFixed(0)}%</div>
                    <div className="stat-card-label">Avg Confidence</div>
                </div>
                <div className="stat-card">
                    <div className="stat-card-value">
                        {(class_counts?.destroyed ?? 0) + (class_counts?.major ?? 0)}
                    </div>
                    <div className="stat-card-label">High-Damage Tiles</div>
                </div>
            </div>

            {/* Damage breakdown */}
            <div>
                <div className="section-title">Damage Breakdown</div>
                <div className="damage-breakdown">
                    {CLASS_ORDER.map(cls => {
                        const count = class_counts?.[cls] ?? 0
                        const pct = class_percentages?.[cls] ?? 0
                        const color = severityColor(cls)
                        return (
                            <div key={cls} className="breakdown-row">
                                <div className="breakdown-color-dot" style={{ background: color }} />
                                <span className="breakdown-label">{CLASS_LABELS[cls]}</span>
                                <div className="breakdown-bar-track">
                                    <div className="breakdown-bar-fill" style={{ width: `${pct}%`, background: color }} />
                                </div>
                                <span className="breakdown-count">{count}</span>
                                <span className="breakdown-pct">{pct.toFixed(1)}%</span>
                            </div>
                        )
                    })}
                </div>
            </div>

            {/* Top priority tiles */}
            {topPriority && topPriority.length > 0 && (
                <div>
                    <div className="section-title">⭐ Top Priority Zones</div>
                    <div className="priority-list">
                        {topPriority.map((tile, i) => {
                            const style = priorityStyle(tile.priority_score)
                            const color = severityColor(tile.damage_class)
                            return (
                                <div key={tile.tile_id} className="priority-tile-card">
                                    <div className="priority-rank">#{i + 1}</div>
                                    <div className="priority-tile-info">
                                        <div className="priority-tile-id">{tile.tile_id}</div>
                                        <div className="priority-tile-sub" style={{ color }}>
                                            {CLASS_LABELS[tile.damage_class]} · {(tile.confidence * 100).toFixed(0)}% confidence
                                        </div>
                                    </div>

                                    {/* Grad-CAM Heatmap Preview */}
                                    {tile.gradcam && (
                                        <div className="gradcam-container">
                                            <img
                                                src={`data:image/jpeg;base64,${tile.gradcam}`}
                                                alt="AI Attention Heatmap"
                                                className="gradcam-img"
                                                title="AI Attention Heatmap (Grad-CAM)"
                                            />
                                            <div className="gradcam-label">AI FOCUS</div>
                                        </div>
                                    )}

                                    <div className="priority-score-badge" style={style}>
                                        {tile.priority_score.toFixed(2)}
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </div>
            )}
        </div>
    )
}
