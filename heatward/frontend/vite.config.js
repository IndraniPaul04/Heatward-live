import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// In dev, /api/* is proxied to the FastAPI backend so the browser sees one origin
// (no CORS, no hard-coded host). For production either serve both behind one
// reverse proxy, or build with VITE_API_BASE=https://your-api.example.com
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': { target: process.env.HEATWARD_API || 'http://localhost:8000', changeOrigin: true } } },
  preview: { proxy: { '/api': { target: process.env.HEATWARD_API || 'http://localhost:8000', changeOrigin: true } } }
});
