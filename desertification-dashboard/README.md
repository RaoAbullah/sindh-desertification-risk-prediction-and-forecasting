# Desertification Dashboard (Frontend)

[![Next.js 16](https://img.shields.io/badge/Next.js-16.2.4-black.svg)](https://nextjs.org/)
[![React 19](https://img.shields.io/badge/React-19.2.4-61DAFB.svg)](https://react.dev/)
[![Leaflet](https://img.shields.io/badge/Leaflet-1.9.4-199900.svg)](https://leafletjs.com/)
[![Tailwind CSS 4](https://img.shields.io/badge/Tailwind-v4.0-38B2AC.svg)](https://tailwindcss.com/)

A modern, high-performance geospatial web dashboard built with **Next.js 16** (App Router) and **React 19**. It provides interactive visualization, spatial query tools, deep learning diagnostic telemetry, time-series forecasting players, streaming generative AI copilot interactions, and automated policy report generation for desertification risk across Sindh, Pakistan.

---

## 🌟 Key Capabilities

### 1. Interactive Geospatial Workspace (`MapWidget.tsx`)
- **Geographic Boundary Rendering:** Integrates administrative boundaries of all Sindh districts from GeoJSON layers with dynamic hovering, selection highlights, and auto-zoom centering.
- **Flexible Area Selection Modes:**
  - **Administrative District Selection:** One-click selection of precomputed districts (e.g., Hyderabad, Tharparkar, Badin, Sukkur).
  - **Interactive Bounding Box (Square):** Click and drag rectangular bounding boxes on the map canvas.
  - **Freeform Polygon Drawing (`leaflet-draw`):** Draw arbitrary spatial polygons to assess custom zones such as agricultural belts, riverine reserves, or desert margins.
- **Coordinate & Surface Telemetry:** Real-time computation of bounding latitude/longitude boundaries and calculated surface area ($\text{km}^2$).

### 2. Dual Analysis Workflow (`page.tsx`)
- **Current Baseline Analysis (`/analyze/current`):**
  - Instantaneous multi-sensor assessment using **Model A (Spatio-Temporal UNet)**.
  - Generates comprehensive 4-panel diagnostic dashboards, 33-band satellite input grids, global feature importance rankings, and bivariate hexbin joint distributions.
- **Multi-Year Forecast Analysis (`/analyze/forecast`):**
  - Autoregressive sequence-to-sequence rollouts using **Model B (ConvGRU-UNet)**.
  - Visualizes quarterly multi-year trajectories from 2026 to 2030.
  - Interactive **Timelapse GIF Player** with frame-by-frame scrubbing and replay capabilities.
  - Thematic anomaly maps, stacked percentage risk bar charts, seasonal variance curves, and spatial polarization metrics.

### 3. Generative AI Copilot & Automated Reporting
- **Streaming Multimodal RAG Copilot:** Full-screen conversational AI assistant powered by Google Gemini 1.5/2.0. Streams responses token-by-token with formatted Markdown and math notation (`react-markdown`, `remark-gfm`). Retains conversational memory and context from current visual analyses.
- **One-Click Comprehensive Policy Reports:** Generates structured executive environmental assessment reports grounded in multi-sensor telemetry. Includes instant client-side PDF export via `html2pdf.js`.

### 4. Historical Analysis Management
- Side-drawer interface displaying past analyses stored in **Supabase** and **MongoDB**.
- Filter by analysis type (`current` or `forecast`), view timestamps, reload historical maps, or permanently delete records along with remote storage assets.

---

## 🏗️ Architecture & Component Hierarchy

```text
desertification-dashboard/
├── app/
│   ├── favicon.ico
│   ├── globals.css              # Tailwind CSS imports & custom map styling
│   ├── layout.tsx               # Root layout, metadata, and font declarations
│   └── page.tsx                 # Core interactive application state, sidebar & modals
├── components/
│   ├── MapWidget.tsx            # Leaflet map container, tile layers & drawing controls
│   └── SearchBar.tsx            # Debounced district autocomplete search component
├── public/
│   ├── sindh_districts.geojson  # High-precision Sindh administrative boundary vectors
│   └── ...                      # Static icons and assets
├── package.json                 # Project dependencies and script declarations
├── tsconfig.json                # TypeScript compiler configuration
├── next.config.ts               # Next.js runtime configuration
└── eslint.config.mjs            # Code quality and linting rules
```

---

## 🛠️ Technology Stack

| Category | Technologies |
| :--- | :--- |
| **Framework** | Next.js 16 (App Router, Turbopack, Server-Side / Client Split) |
| **Language** | TypeScript 5 & React 19 |
| **Mapping Engine** | Leaflet 1.9.4, React-Leaflet 5.0, Leaflet-Draw 1.0.4 |
| **Styling** | Tailwind CSS v4 (PostCSS 4) |
| **Document Generation** | `html2pdf.js` (Canvas to PDF rendering) |
| **Markdown / RAG UI** | `react-markdown` 10, `remark-gfm` 4 |
| **Icons** | Lucide React |

---

## 🚀 Getting Started

### 1. Prerequisites
- **Node.js**: `v18.17.0` or higher (Node 20+ recommended)
- **Package Manager**: `npm`, `pnpm`, or `yarn`

### 2. Installation
```bash
# Clone the repository (if not already done)
git clone https://github.com/ahmedfarazsyk/sindh-desertification-risk-prediction-and-forecasting.git
cd sindh-desertification-risk-prediction-and-forecasting/desertification-dashboard

# Install dependencies
npm install
```

### 3. Environment Configuration
Create a `.env.local` file in the root of `desertification-dashboard`:
```bash
cp .env.example .env.local
```

Configure the backend API URL:
```env
# URL pointing to the running desertification_api backend
NEXT_PUBLIC_API_URL=http://localhost:9000/api/v1
```

### 4. Development Server
Start the development server with Hot Module Replacement:
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 📦 Production Build & Deployment

To generate an optimized production build:
```bash
# Build the production bundle
npm run build

# Start the production server
npm run start
```

### Linting & Formatting
```bash
npm run lint
```

---

## 🧩 Key Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `NEXT_PUBLIC_API_URL` | Base URL of the FastAPI backend service | `http://localhost:9000/api/v1` |
