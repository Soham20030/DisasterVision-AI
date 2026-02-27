/**
 * SituationReport — Displays and allows copy/download of the generated text report.
 */

import React, { useState } from 'react'
import { downloadReport } from '../utils/api.js'

export default function SituationReport({ report, sessionId }) {
    const [copied, setCopied] = useState(false)

    if (!report) {
        return (
            <div className="empty-state">
                <div className="empty-icon">📋</div>
                <div>The situation report will appear here after analysis completes.</div>
            </div>
        )
    }

    function handleCopy() {
        navigator.clipboard.writeText(report).then(() => {
            setCopied(true)
            setTimeout(() => setCopied(false), 2000)
        })
    }

    return (
        <div className="report-section">
            <div className="section-title">AI-Generated Situation Report</div>
            <textarea
                className="report-textarea"
                value={report}
                readOnly
                spellCheck={false}
            />
            <div className="report-actions">
                <button className="btn-secondary" onClick={handleCopy}>
                    {copied ? '✓ Copied!' : '📋 Copy to Clipboard'}
                </button>
                {sessionId && (
                    <button className="btn-secondary" onClick={() => downloadReport(sessionId)}>
                        ⬇ Download .txt
                    </button>
                )}
            </div>
        </div>
    )
}
