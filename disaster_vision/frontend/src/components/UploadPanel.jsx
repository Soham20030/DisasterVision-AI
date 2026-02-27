/**
 * UploadPanel — drag-and-drop interface for pre/post image pair uploads.
 *
 * The model requires paired images (pre-disaster + post-disaster).
 * Users can upload pre and post images separately; this component
 * matches them by order into pairs.
 */

import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { IconFolder, IconImage, IconMapPin } from './Icons.jsx'

const ACCEPTED = { 'image/jpeg': [], 'image/png': [], 'image/tiff': [], 'image/webp': [], 'text/csv': [] }
const CSV_ACCEPTED = { 'text/csv': [] }

function Dropzone({ label, files, onAdd, onRemove, acceptTypes = ACCEPTED }) {
    const onDrop = useCallback(accepted => onAdd(accepted), [onAdd])
    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop, accept: acceptTypes, multiple: true,
    })

    return (
        <div>
            <div className="upload-pair-title">{label}</div>
            <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
                <input {...getInputProps()} />
                <div className="dropzone-icon">
                    <IconFolder />
                </div>
                <div className="dropzone-label">
                    <strong>Click or drag</strong> images here
                </div>
            </div>
            {files.length > 0 && (
                <div className="file-list">
                    {files.map((f, i) => (
                        <div key={i} className="file-chip">
                            <IconImage className="file-icon" />
                            {f.name}
                            <button className="remove-btn" onClick={() => onRemove(i)} title="Remove">✕</button>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}

export default function UploadPanel({ onAnalyze, isProcessing }) {
    const [preFiles, setPreFiles] = useState([])
    const [postFiles, setPostFiles] = useState([])
    const [mode, setMode] = useState('dual') // 'dual' | 'single'
    const [csvFile, setCsvFile] = useState([])
    const [latitude, setLatitude] = useState('')
    const [longitude, setLongitude] = useState('')
    const [error, setError] = useState(null)

    const addPre = fs => setPreFiles(prev => [...prev, ...fs])
    const addPost = fs => setPostFiles(prev => [...prev, ...fs])
    const addCsv = fs => setCsvFile(fs.slice(0, 1)) // Only allow one CSV file
    const removePre = i => setPreFiles(prev => prev.filter((_, idx) => idx !== i))
    const removePost = i => setPostFiles(prev => prev.filter((_, idx) => idx !== i))
    const removeCsv = () => setCsvFile([])

    // Pair count = min of pre and post counts
    const isDual = mode === 'dual'
    const pairCount = isDual ? Math.min(preFiles.length, postFiles.length) : postFiles.length
    const pairs = Array.from({ length: pairCount }, (_, i) => ({
        pre: isDual ? preFiles[i] : null,
        post: postFiles[i],
    }))

    async function handleAnalyze() {
        setError(null)
        if (pairCount === 0) {
            setError(isDual ? 'Upload at least one pre-disaster and one post-disaster image.' : 'Upload at least one post-disaster image.')
            return
        }

        let cLat = latitude.trim() !== '' ? parseFloat(latitude) : null
        let cLon = longitude.trim() !== '' ? parseFloat(longitude) : null

        if (latitude.trim() !== '' && isNaN(cLat)) {
            setError('Invalid latitude format.')
            return
        }
        if (longitude.trim() !== '' && isNaN(cLon)) {
            setError('Invalid longitude format.')
            return
        }

        let coords = null

        if (csvFile.length > 0) {
            try {
                const text = await csvFile[0].text()
                const lines = text.trim().split('\n').filter(l => l.trim())
                coords = []
                for (const line of lines) {
                    const parts = line.split(',')
                    if (parts.length < 2) throw new Error('Invalid row format')
                    const lat = parseFloat(parts[0])
                    const lon = parseFloat(parts[1])
                    if (isNaN(lat) || isNaN(lon)) throw new Error('Valid numbers required')
                    coords.push([lat, lon])
                }
                if (coords.length !== pairCount) {
                    throw new Error(`CSV has ${coords.length} rows, but you have ${pairCount} pairs.`)
                }
            } catch (e) {
                setError(`CSV Error: ${e.message}. Use format: lat,lon (one per line).`)
                return
            }
        }

        onAnalyze(pairs, cLat, cLon, coords, mode)
    }

    return (
        <div className="upload-section">
            <div className="mode-toggle" style={{ marginBottom: 20, display: 'flex', gap: 10 }}>
                <button
                    className={`tab-btn ${isDual ? 'active' : ''}`}
                    onClick={() => setMode('dual')}
                    style={{ flex: 1 }}
                >
                    Dual-Image (Pre & Post)
                </button>
                <button
                    className={`tab-btn ${!isDual ? 'active' : ''}`}
                    onClick={() => setMode('single')}
                    style={{ flex: 1 }}
                >
                    Single-Image (Post-only)
                </button>
            </div>

            {isDual && <Dropzone label="Pre-Disaster Images" files={preFiles} onAdd={addPre} onRemove={removePre} />}
            <Dropzone label="Post-Disaster Images" files={postFiles} onAdd={addPost} onRemove={removePost} />

            <div className="coordinate-input-section">
                <div className="section-title">
                    <IconMapPin style={{ marginRight: 8 }} />
                    Coordinate Input (Optional)
                </div>
                <div className="coordinate-group">
                    <div className="coordinate-row">
                        <label htmlFor="latitude">Region Center Latitude:</label>
                        <input
                            id="latitude"
                            type="text"
                            value={latitude}
                            onChange={e => setLatitude(e.target.value)}
                            placeholder="e.g., 34.0522"
                            className="coordinate-input"
                        />
                    </div>
                    <div className="coordinate-row">
                        <label htmlFor="longitude">Region Center Longitude:</label>
                        <input
                            id="longitude"
                            type="text"
                            value={longitude}
                            onChange={e => setLongitude(e.target.value)}
                            placeholder="e.g., -118.2437"
                            className="coordinate-input"
                        />
                    </div>
                </div>
                <Dropzone
                    label="Per-Tile Coordinates (Optional CSV)"
                    files={csvFile}
                    onAdd={addCsv}
                    onRemove={removeCsv}
                    acceptTypes={CSV_ACCEPTED}
                />
            </div>

            <div>
                <div className="section-title">Analysis Queue ({pairCount} {isDual ? 'pairs' : 'tiles'})</div>
                <div className="pair-group">
                    {pairs.map((p, i) => (
                        <div key={i} style={{ marginBottom: i < pairs.length - 1 ? 12 : 0, paddingBottom: i < pairs.length - 1 ? 12 : 0, borderBottom: i < pairs.length - 1 ? '1px solid rgba(255,255,255,0.05)' : 'none' }}>
                            {isDual && (
                                <div className="pair-row">
                                    <span className="pair-label pre-label">PRE</span>
                                    <span className="pair-filename">{p.pre.name}</span>
                                </div>
                            )}
                            <div className="pair-row">
                                <span className="pair-label post-label">POST</span>
                                <span className="pair-filename">{p.post.name}</span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {isDual && preFiles.length !== postFiles.length && (
                <div className="error-banner">
                    ⚠ Unequal count — {preFiles.length} pre / {postFiles.length} post.
                    Only the first {pairCount} pairs will be analyzed.
                </div>
            )}

            {error && <div className="error-banner">⚠ {error}</div>}

            <button
                className="analyze-btn"
                onClick={handleAnalyze}
                disabled={isProcessing || pairCount === 0}
            >
                {isProcessing ? '⏳ Analyzing…' : `▶ Analyze Region (${pairCount} ${isDual ? 'pair' + (pairCount !== 1 ? 's' : '') : 'tile' + (pairCount !== 1 ? 's' : '')})`}
            </button>
        </div>
    )
}
