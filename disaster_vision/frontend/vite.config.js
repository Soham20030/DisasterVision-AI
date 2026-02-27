import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
    plugins: [react()],
    server: {
        port: 5175,
        proxy: {
            // Forward /api/* and /healthz to FastAPI
            '/api': { target: 'http://localhost:8000', changeOrigin: true },
            '/healthz': { target: 'http://localhost:8000', changeOrigin: true },
        },
    },
})
