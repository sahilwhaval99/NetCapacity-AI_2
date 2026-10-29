# NetCapacity AI — AI-Based Intelligent Network Capacity Planning

**B.Tech Third-Year Artificial Intelligence & Data Science (AI & DS) Full-Stack ML Project**

NetCapacity AI is an end-to-end machine learning and network analytics web application that ingests historical network traffic telemetry, engineers leak-free autoregressive lag and seasonal features, forecasts future bandwidth demand across **24-hour**, **7-day**, and **30-day** horizons using a **Random Forest Regressor**, detects **SLA capacity bottlenecks**, and recommends **carrier-grade optical infrastructure upgrades**.

---

## 1. Key Features

1. **Responsive Dark-Navy Network Analytics Dashboard**:
   - Executive KPI cards for Historical Peak Load, Growth-Adjusted Forecast Peak, Effective SLA Capacity Gap/Margin, and Random Forest Out-of-Sample $R^2$ & RMSE skill score.
   - Unified Recharts `ComposedChart` displaying observed historical traffic, multi-step forecasts, 90% empirical prediction intervals, physical link capacity, and SLA threshold ceilings.
   - Diurnal (24-hour) and Weekly seasonality profiles alongside M/M/1 queuing delay (latency in ms) and active user correlation charts.
2. **Dataset Manager & Validated CSV Ingestion Pipeline**:
   - Ships with **3 reproducible synthetic network telemetry datasets** (`campus_core_hourly_60d.csv`, `isp_edge_hourly_90d.csv`, and `datacenter_wan_daily_180d.csv`), clearly badged as synthetic (`seed=42`, `seed=77`, `seed=108`).
   - Supports uploading custom `.csv` datasets with automatic column alias detection, timestamp parsing, chronological sorting, duplicate timestamp deduplication, regular time-grid reindexing, and selectable missing-value imputation (`Linear Interpolation`, `Forward Fill`, or `7-Step Rolling Median`).
3. **Genuine, Leak-Free Machine Learning Pipeline**:
   - **Strictly Chronological Train/Test Split** (default 80% earliest observations for training, 20% most recent unseen observations for testing — zero random shuffling, zero future data leakage).
   - **Causal Feature Engineering**: Autoregressive lags ($y_{t-1}, y_{t-2}, y_{t-3}, y_{t-6}, y_{t-12}, y_{t-24}, y_{t-48}, y_{t-168}$), strictly shifted rolling window statistics (`shift(1)` before `.rolling()`), velocity differences, and cyclical Fourier calendar encodings ($\sin/\cos$ of hour-of-day and day-of-week).
   - **Baseline Benchmarking**: Evaluates **Random Forest Regressor** directly against both a **Persistence Baseline** ($\hat{y}_t = y_{t-1}$) and a **Seasonal-Naive Baseline** ($\hat{y}_t = y_{t-24}$ for hourly or $y_{t-7}$ for daily) using **MAE**, **RMSE**, **$R^2$**, and **MAPE (%)**.
4. **Interactive Multi-Horizon Forecast Explorer (`24h`, `7d`, `30d`)**:
   - Recursive multi-step autoregressive forecasting with live user-growth multiplier slider (`-20%` to `+100%`), 90% prediction interval bands, and filterable breach tables.
5. **Interactive Capacity Planner & Upgrade Recommendation Engine**:
   - Editable **Allocated Link Bandwidth (Mbps)**, **SLA Utilization Threshold (%)**, **Expected User Growth (%)**, and **Safety Headroom Buffer (%)**.
   - Computes Effective SLA Ceiling ($C_{\text{eff}}$), SLA Capacity Gap ($\Delta_{\text{SLA}}$), Physical Link Deficit, Breach Probability, and recommends standard IEEE/Telco optical carrier tiers (`500M`, `1GBASE-LX`, `2.5G LACP`, `5G CIR`, `10GBASE-LR`, `25GBASE-LR`, `40GBASE-LR4`, `100GBASE-LR4`) with hardware specifications and estimated monthly circuit OpEx.
6. **Persistent SQLite Storage, Alert Center & Downloadable CSV Reports**:
   - Persists configuration settings, registered dataset metadata, model training runs, and acknowledgeable capacity/QoS alerts in SQLite (`backend/data/netcapacity.db`).
   - One-click export of **Forecast Schedule CSV** and **Executive Capacity Audit CSV** reports.

---

## 2. Project Directory Structure

```text
NetCapacity-AI/
├── README.md                          # Project overview, setup, benchmarks & GitHub guide
├── .gitignore                         # Excludes node_modules, venv, caches & runtime DB
├── requirements.txt                   # Root Python dependencies for instant GitHub detection
├── start_app.sh                       # 1-click starter script for macOS / Linux
├── backend/                           # Flat Python FastAPI + ML + SQLite Backend
│   ├── main.py                        # FastAPI server & REST endpoints (run: python3 main.py)
│   ├── ml_engine.py                   # Leak-free lag features, Random Forest & baselines
│   ├── capacity_planner.py            # SLA ceiling math, bottleneck & upgrade engine
│   ├── preprocessing.py               # CSV validation, timestamp checks & NaN imputation
│   ├── database.py                    # SQLite persistence layer
│   ├── test_api_and_ml.py             # Automated test suite (run: python3 test_api_and_ml.py -v)
│   └── requirements.txt               # Backend Python packages
├── frontend/                          # Flat React + Vite + Tailwind CSS + Recharts Frontend
│   ├── package.json
│   ├── vite.config.js                 # Vite config with /api proxy to 127.0.0.1:8000
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── index.html
│   └── src/                           # All React components & API client in one flat folder
│       ├── main.jsx
│       ├── App.jsx
│       ├── index.css
│       ├── api.js
│       ├── Sidebar.jsx
│       ├── DashboardPage.jsx
│       ├── ForecastExplorerPage.jsx
│       ├── CapacityPlannerPage.jsx
│       ├── ModelEvaluationPage.jsx
│       ├── DatasetManagerPage.jsx
│       └── AlertsAndSettingsPage.jsx
├── datasets/                          # Synthetic network telemetry CSVs & generator script
│   ├── campus_core_hourly_60d.csv     # 1,440 hourly rows (1 Gbps Campus Core, seed=42)
│   ├── isp_edge_hourly_90d.csv        # 2,160 hourly rows (2.5 Gbps ISP Edge, seed=77)
│   ├── datacenter_wan_daily_180d.csv  # 180 daily rows (5 Gbps Data Center WAN, seed=108)
│   └── generate_synthetic_data.py     # Reproducible synthetic dataset generator
└── docs/
    └── ACADEMIC_DOCUMENTATION.md      # Architecture, math formulas, limitations & 25 Viva Q&As
```

---

## 3. Verified Model Evaluation Benchmark (Campus Core 60-Day Hourly Dataset)

Executed on the held-out chronological test split (final 20% unseen hours, `255` test samples following `1,017` training samples after `168`-hour lag warm-up):

| Model | Role | Test MAE (Mbps) | Test RMSE (Mbps) | Test $R^2$ Score | Test MAPE (%) | Skill vs. Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | Primary ML | **27.389** | **37.721** | **0.9395** | **3.93%** | **Primary Model** |
| **Persistence Baseline** ($\hat{y}_t = y_{t-1}$) | Baseline 1 | 40.231 | 54.198 | 0.8751 | 5.79% | RF reduces RMSE by **30.40%** |
| **Seasonal-Naive Baseline** ($\hat{y}_t = y_{t-24}$) | Baseline 2 | 103.551 | 140.693 | 0.1583 | 15.65% | RF reduces RMSE by **73.19%** |

---

## 4. Local Setup & Run Instructions

### Prerequisites
- **Python**: `3.10+` (`3.11` recommended)
- **Node.js**: `18.0+` and `npm`

### Step 1: Start the Python FastAPI Backend
Open a terminal in the `NetCapacity-AI/backend` directory:

```bash
cd NetCapacity-AI/backend

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# On Windows:
# .venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server on http://127.0.0.1:8000
python3 main.py   # or: uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
- Interactive Swagger API documentation is automatically available at: `http://127.0.0.1:8000/docs`

### Step 2: Start the React + Vite Frontend
Open a second terminal in the `NetCapacity-AI/frontend` directory:

```bash
cd NetCapacity-AI/frontend

# Install Node packages
npm install

# Start the Vite development server on http://localhost:5173
npm run dev
```
Open **`http://localhost:5173`** in your browser. Vite automatically proxies `/api/*` requests to `http://127.0.0.1:8000`.

### Step 3: Run the Automated Test Suite
From `NetCapacity-AI/backend`:

```bash
cd NetCapacity-AI/backend

# Run via Python's built-in unittest runner:
python3 test_api_and_ml.py -v

# Or run via pytest:
pytest -v
```

---

## 5. How to Upload to GitHub

1. **Extract `NetCapacity-AI.zip`** on your computer so you have the `NetCapacity-AI/` folder.
2. **Create a new empty repository on GitHub** (e.g., `NetCapacity-AI`) at [https://github.com/new](https://github.com/new) **without** initializing a README or `.gitignore` (since they are already included).
3. **Push via Git CLI** from inside the extracted `NetCapacity-AI/` directory:

```bash
cd NetCapacity-AI
git init
git add .
git commit -m "Initial commit: NetCapacity AI - B.Tech AI & DS Network Capacity Planning Project"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/NetCapacity-AI.git
git push -u origin main
```
*(Alternatively, you can click **"uploading an existing file"** on your new GitHub repository page and drag-and-drop the contents of the extracted `NetCapacity-AI/` folder directly into the browser.)*
