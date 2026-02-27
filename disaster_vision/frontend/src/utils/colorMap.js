/** Maps damage class names to UI colors. */

export const CLASS_COLORS = {
    'no-damage': '#3fb950',
    'minor': '#d29922',
    'major': '#db6d28',
    'destroyed': '#f85149',
    'unknown': '#6e7681',
}

export const CLASS_LABELS = {
    'no-damage': 'No Damage',
    'minor': 'Minor',
    'major': 'Major',
    'destroyed': 'Destroyed',
    'unknown': 'Unknown',
}

/** Returns a hex color for a given damage class name. */
export function severityColor(cls) {
    return CLASS_COLORS[cls] || CLASS_COLORS['unknown']
}

/**
 * Returns a CSS rgba string with optional opacity for map markers.
 */
export function severityColorAlpha(cls, alpha = 0.85) {
    const hex = severityColor(cls)
    const r = parseInt(hex.slice(1, 3), 16)
    const g = parseInt(hex.slice(3, 5), 16)
    const b = parseInt(hex.slice(5, 7), 16)
    return `rgba(${r},${g},${b},${alpha})`
}

/** Priority → badge style */
export function priorityStyle(score) {
    if (score >= 0.7) return { background: '#6e011a33', color: '#f85149', border: '1px solid #f85149' }
    if (score >= 0.4) return { background: '#5a1e0233', color: '#db6d28', border: '1px solid #db6d28' }
    if (score >= 0.1) return { background: '#3b280033', color: '#d29922', border: '1px solid #d29922' }
    return { background: '#0d1d0e33', color: '#3fb950', border: '1px solid #3fb950' }
}
