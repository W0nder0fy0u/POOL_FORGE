// Vite config. `npm run dev` serves the dashboard on :5173 and forwards
// /api and /ws to the coordinator on :8000. `npm run build` writes dist/,
// which the coordinator serves in production.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
  build: { chunkSizeWarningLimit: 1000 },
});
