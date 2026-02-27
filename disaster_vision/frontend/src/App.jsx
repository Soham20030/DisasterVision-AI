/**
 * App — Root component for DisasterVision.
 *
 * Manages global state: upload pairs, session lifecycle, polling, results.
 * Renders the top bar, sidebar (tabs: Upload / Summary / Report), and map.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react'
import MapView from './components/MapView.jsx'
import UploadPanel from './components/UploadPanel.jsx'
import SummaryPanel from './components/SummaryPanel.jsx'
import SituationReport from './components/SituationReport.jsx'
import { startAnalysis, getStatus, getResults } from './utils/api.js'
import { IconUpload, IconSummary, IconReport } from './components/Icons.jsx'

const POLL_INTERVAL_MS = 1500

export default function App() {
    const [activeTab, setActiveTab] = useState('upload')   // upload | summary | report
    const [isProcessing, setIsProcessing] = useState(false)
    const [progress, setProgress] = useState({ done: 0, total: 0 })
    const [sessionId, setSessionId] = useState(null)
    const [tiles, setTiles] = useState([])
    const [stats, setStats] = useState(null)
    const [topPriority, setTopPriority] = useState([])
    const [report, setReport] = useState(null)
    const [error, setError] = useState(null)
    const [mockMode, setMockMode] = useState(false)

    const pollRef = useRef(null)

    // ── Check backend health on mount ──────────────────────────────────────────
    useEffect(() => {
        fetch('/healthz')
            .then(r => r.json())
            .then(data => setMockMode(data.inference_mode === 'mock'))
            .catch(() => { })
    }, [])

    // ── Polling ─────────────────────────────────────────────────────────────────
    const startPolling = useCallback((sid) => {
        if (pollRef.current) clearInterval(pollRef.current)

        pollRef.current = setInterval(async () => {
            try {
                const status = await getStatus(sid)
                setProgress({ done: status.completed + status.failed, total: status.total_tiles })

                if (status.status === 'done') {
                    clearInterval(pollRef.current)
                    const results = await getResults(sid)
                    setTiles(results.tiles || [])
                    setStats(results.stats)
                    setTopPriority(results.top_priority || [])
                    setReport(results.situation_report)
                    setIsProcessing(false)
                    setActiveTab('summary')   // auto-switch to summary when done
                }

                if (status.status === 'error') {
                    clearInterval(pollRef.current)
                    setError('Server-side processing error. Please retry.')
                    setIsProcessing(false)
                }
            } catch (e) {
                clearInterval(pollRef.current)
                setError(`Polling error: ${e.message}`)
                setIsProcessing(false)
            }
        }, POLL_INTERVAL_MS)
    }, [])

    // ── Analysis trigger ────────────────────────────────────────────────────────
    async function handleAnalyze(pairs, centerLat, centerLon, perTileCoords, mode) {
        setError(null)
        setTiles([])
        setStats(null)
        setTopPriority([])
        setReport(null)
        setIsProcessing(true)
        setProgress({ done: 0, total: pairs.length })

        try {
            const res = await startAnalysis(pairs, centerLat, centerLon, perTileCoords, mode)
            setSessionId(res.session_id)
            startPolling(res.session_id)
        } catch (e) {
            setError(`Failed to start analysis: ${e.message}`)
            setIsProcessing(false)
        }
    }

    // ── Progress % ──────────────────────────────────────────────────────────────
    const pct = progress.total > 0 ? Math.round((progress.done / progress.total) * 100) : 0

    return (
        <div className="app">
            {/* ── Top bar ─────────────────────────────────────────────────────── */}
            <header className="topbar">
                <div className="topbar-logo">
                    <span className="dot" />
                    DisasterVision
                </div>
                <span className="topbar-badge">Decision Support</span>
                <div className="topbar-spacer" />
                {mockMode && (
                    <span style={{ fontSize: 11, color: '#d29922', fontFamily: 'var(--mono)', marginRight: 12 }}>
                        ⚠ Mock inference active
                    </span>
                )}
                {isProcessing && (
                    <span className="topbar-status">
                        Processing: {progress.done} / {progress.total} tiles ({pct}%)
                    </span>
                )}
                {!isProcessing && tiles.length > 0 && (
                    <span className="topbar-status">
                        ✓ {tiles.length} tile{tiles.length !== 1 ? 's' : ''} analyzed
                    </span>
                )}
            </header>

            {/* ── Main layout ─────────────────────────────────────────────────── */}
            <div className="main-layout">

                {/* ── Sidebar ───────────────────────────────────────────────────── */}
                <aside className="sidebar">
                    <div className="sidebar-tabs">
                        {[
                            { id: 'upload', label: 'Upload', icon: <IconUpload /> },
                            { id: 'summary', label: 'Summary', icon: <IconSummary /> },
                            { id: 'report', label: 'Report', icon: <IconReport /> },
                        ].map(tab => (
                            <button
                                key={tab.id}
                                className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
                                onClick={() => setActiveTab(tab.id)}
                            >
                                <span className="tab-icon">{tab.icon}</span>
                                {tab.label}
                            </button>
                        ))}
                    </div>

                    <div className="tab-content">
                        {/* Progress bar (visible while processing) */}
                        {isProcessing && (
                            <div className="progress-container" style={{ marginBottom: 16 }}>
                                <div className="progress-label">
                                    <span>Analyzing tiles…</span>
                                    <span>{pct}%</span>
                                </div>
                                <div className="progress-bar-track">
                                    <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
                                </div>
                                <div className="progress-label">
                                    <span>{progress.done} complete · {progress.total - progress.done} remaining</span>
                                </div>
                            </div>
                        )}

                        {error && <div className="error-banner" style={{ marginBottom: 14 }}>⚠ {error}</div>}

                        {activeTab === 'upload' && (
                            <UploadPanel onAnalyze={handleAnalyze} isProcessing={isProcessing} />
                        )}
                        {activeTab === 'summary' && (
                            <SummaryPanel stats={stats} topPriority={topPriority} />
                        )}
                        {activeTab === 'report' && (
                            <SituationReport report={report} sessionId={sessionId} />
                        )}
                    </div>
                </aside>

                {/* ── Map ───────────────────────────────────────────────────────── */}
                <MapView tiles={tiles} topPriorityTiles={topPriority} />
            </div>
        </div>
    )
}
