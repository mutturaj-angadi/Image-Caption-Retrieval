/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
      },
      colors: {
        brand: { 50: "#eff9f8", 100: "#d7efed", 500: "#1b8989", 600: "#147779", 700: "#105f65", 900: "#153f4d" },
      },
    },
  },
  plugins: [],
};
