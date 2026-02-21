import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/monitor': {
        target: 'http://127.0.0.1:8090',
        changeOrigin: true,
        ws: true,
        rewrite: (value) => value.replace(/^\/monitor/, ''),
      },
    },
  },
  preview: {
    proxy: {
      '/monitor': {
        target: 'http://127.0.0.1:8090',
        changeOrigin: true,
        ws: true,
        rewrite: (value) => value.replace(/^\/monitor/, ''),
      },
    },
  },
})
