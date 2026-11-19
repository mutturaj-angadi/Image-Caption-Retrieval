/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// In development the browser talks to Vite (port 5173); Vite forwards API calls to FastAPI (port 8000).
// This avoids CORS entirely. Set VITE_API_BASE to call a different backend directly.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    globals: true,
    css: false,
  },
});
