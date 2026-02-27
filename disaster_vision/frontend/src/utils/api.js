/** API utilities for communicating with the DisasterVision FastAPI backend. */

const BASE = ''  // proxied via vite dev server

/**
 * Upload pre/post image pairs and start batch analysis.
 * @param {Array<{pre: File, post: File}>} pairs
 * @param {number} centerLat - latitude of region center (used for grid layout)
 * @param {number} centerLon - longitude of region center (used for grid layout)
 * @param {Array<[number,number]>|null} perTileCoords - optional exact [lat,lon] per tile
 * @returns {Promise<{session_id: string, total_tiles: number, status: string}>}
 */
export async function startAnalysis(pairs, centerLat = 0.0, centerLon = 0.0, perTileCoords = null, mode = 'dual') {
    const form = new FormData()
    pairs.forEach(({ pre, post }) => {
        if (pre) form.append('pre_images', pre)
        form.append('post_images', post)
    })
    form.append('mode', mode)
    if (centerLat !== null) form.append('center_lat', String(centerLat))
    if (centerLon !== null) form.append('center_lon', String(centerLon))

    // If per-tile coordinates were provided, send them as comma-separated lists
    if (perTileCoords && perTileCoords.length === pairs.length) {
        form.append('lats', perTileCoords.map(c => c[0]).join(','))
        form.append('lons', perTileCoords.map(c => c[1]).join(','))
    }

    const res = await fetch(`${BASE}/api/analyze`, { method: 'POST', body: form })
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || `Server error ${res.status}`)
    }
    return res.json()
}

/**
 * Poll the processing status of a session.
 */
export async function getStatus(sessionId) {
    const res = await fetch(`${BASE}/api/status/${sessionId}`)
    if (!res.ok) throw new Error(`Status error ${res.status}`)
    return res.json()
}

/**
 * Get full results (tiles, stats, report) for a completed session.
 */
export async function getResults(sessionId) {
    const res = await fetch(`${BASE}/api/results/${sessionId}`)
    if (!res.ok) throw new Error(`Results error ${res.status}`)
    return res.json()
}

/**
 * Download the situation report as a .txt file.
 */
export function downloadReport(sessionId) {
    window.open(`${BASE}/api/report/${sessionId}`, '_blank')
}
