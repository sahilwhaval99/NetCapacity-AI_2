# Academic Project Documentation: AI-Based Intelligent Network Capacity Planning (NetCapacity AI)

**Course / Degree**: Third-Year B.Tech in Artificial Intelligence & Data Science (AI & DS)  
**Project Title**: AI-Based Intelligent Network Capacity Planning (*NetCapacity AI*)  
**Domain**: Machine Learning for Time-Series Telemetry, Network Traffic Engineering & Infrastructure Provisioning  

---

## 1. Abstract & Problem Statement

Modern enterprise campuses, cloud data centers, and regional Internet Service Providers (ISPs) experience non-stationary, bursty bandwidth demand driven by cloud synchronization, video conferencing, and diurnal user patterns. Traditional network provisioning relies on static rules of thumb—either **over-provisioning** expensive optical circuits (wasting CapEx/OpEx) or **reactive upgrading** only after link saturation causes severe M/M/1 queuing delay and packet loss.

**NetCapacity AI** solves this problem by combining:
1. Automated time-series preprocessing, timestamp continuity verification, and missing-value imputation.
2. Leak-free supervised feature engineering (autoregressive lags, strictly causal rolling-window statistics, and cyclical Fourier calendar encodings).
3. Out-of-sample chronological **Random Forest Regression** benchmarked against **Persistence** and **Seasonal-Naive** baselines.
4. Multi-step recursive forecasting across **24-hour**, **7-day**, and **30-day** horizons with 90% empirical prediction intervals.
5. An interactive **Capacity Planning & Carrier Upgrade Engine** that translates forecasted peak bandwidth demand, user growth projections, SLA utilization ceilings, and safety headroom buffers into concrete optical hardware upgrade recommendations.

---

## 2. System Architecture

NetCapacity AI follows a decoupled **Three-Tier Full-Stack Architecture**:

1. **Presentation Layer (React 18 + Vite + Tailwind CSS + Recharts)**:
   - Responsive dark-navy single-page application (`DashboardPage`, `ForecastExplorerPage`, `CapacityPlannerPage`, `ModelEvaluationPage`, `DatasetManagerPage`, `AlertsAndSettingsPage`).
   - Communicates asynchronously with the FastAPI backend via REST JSON endpoints and streaming CSV report downloads.
2. **Application & ML Engine Layer (Python + FastAPI + Pandas + NumPy + Scikit-learn)**:
   - **Ingestion & Preprocessing (`preprocessing.py`)**: Schema validation, column alias mapping, timestamp sorting, duplicate timestamp aggregation, regular grid reindexing, and missing-value imputation.
   - **Forecasting Engine (`ml_engine.py`)**: Supervised lag matrix builder, chronological split engine, Random Forest Regressor (`scikit-learn` with vectorized NumPy CART ensemble fallback), baseline evaluators, and recursive multi-step forecaster.
   - **Capacity Planner (`capacity_planner.py`)**: Computes SLA ceiling, capacity gap, breach probability, and matches required target bandwidth against standard IEEE/Telco optical carrier tiers.
3. **Persistence Layer (SQLite + Joblib / Pickle)**:
   - **SQLite (`netcapacity.db`)**: Stores global configuration settings, registered dataset metadata, model training run logs, and acknowledgeable capacity alerts.
   - **Model Artifact Store (`backend/models/*.joblib`)**: Caches trained Random Forest ensembles and evaluation diagnostics per dataset.

---

## 3. Mathematical Formulations & Algorithms

### 3.1 Leak-Free Time-Series Feature Engineering
Given a univariate bandwidth time series $\{y_1, y_2, \dots, y_T\}$ sampled at regular interval $\Delta t$ (hourly or daily), predicting $y_t$ requires constructing feature vector $\mathbf{x}_t$ using **strictly past observations** $\{y_{t-1}, y_{t-2}, \dots\}$:

1. **Autoregressive Lag Features**:
   $$L_{t,k} = y_{t-k}, \quad k \in \{1, 2, 3, 6, 12, 24, 48, 168\} \text{ (hourly)} \quad \text{or} \quad k \in \{1, 2, 3, 7, 14, 30\} \text{ (daily)}$$

2. **Strictly Causal Rolling Window Statistics** (computed on $y_{t-1}$ via `y.shift(1).rolling(w)` so $y_t$ is never leaked):
   $$\mu_{t,w} = \frac{1}{w} \sum_{j=1}^{w} y_{t-j}, \qquad \sigma_{t,w} = \sqrt{\frac{1}{w-1} \sum_{j=1}^{w} (y_{t-j} - \mu_{t,w})^2}, \qquad M_{t,w} = \max_{1 \le j \le w} y_{t-j}$$

3. **First-Order & Seasonal Velocity Differences**:
   $$\Delta_1 y_t = y_{t-1} - y_{t-2}, \qquad \Delta_s y_t = y_{t-1} - y_{t-s}$$

4. **Cyclical Fourier Calendar Encodings** (preserving continuity between 23:00 and 00:00):
   $$h_{\sin}(t) = \sin\left(\frac{2\pi \cdot \text{hour}(t)}{24}\right), \quad h_{\cos}(t) = \cos\left(\frac{2\pi \cdot \text{hour}(t)}{24}\right)$$
   $$d_{\sin}(t) = \sin\left(\frac{2\pi \cdot \text{dow}(t)}{7}\right), \quad d_{\cos}(t) = \cos\left(\frac{2\pi \cdot \text{dow}(t)}{7}\right)$$

---

### 3.2 Chronological Train/Test Partitioning
Unlike standard tabular classification where rows are independent and identically distributed (i.i.d.), random K-fold cross-validation on time series causes **temporal data leakage** because future values $y_{t+1}$ contain target $y_t$ inside their lag features $L_{t+1,1} = y_t$.

NetCapacity AI enforces a strict **Chronological Split** at index $T_{\text{split}} = \lfloor (1 - r_{\text{test}}) \cdot N \rfloor$:
- **Training Set**: $\mathcal{D}_{\text{train}} = \{(\mathbf{x}_t, y_t)\}_{t=1}^{T_{\text{split}}}$
- **Testing Set**: $\mathcal{D}_{\text{test}} = \{(\mathbf{x}_t, y_t)\}_{t=T_{\text{split}}+1}^{N}$

All timestamps in $\mathcal{D}_{\text{train}}$ strictly precede all timestamps in $\mathcal{D}_{\text{test}}$.

---

### 3.3 Random Forest Regression & Baselines
A **Random Forest Regressor** (Breiman, 2001) is an ensemble of $B$ regression trees $\{T_b(\mathbf{x})\}_{b=1}^{B}$, each trained on an independent bootstrap sample of $\mathcal{D}_{\text{train}}$ with random feature subspace selection ($m_{\text{try}} \approx \lfloor\sqrt{p}\rfloor$) at each split node. Each node chooses feature $j^*$ and threshold $s^*$ that maximize **Variance Reduction (SSE Gain)**:

$$\Delta \text{SSE}(j, s) = \sum_{i \in \text{Node}} (y_i - \bar{y}_{\text{Node}})^2 - \left[ \sum_{i \in \text{Left}(j,s)} (y_i - \bar{y}_{L})^2 + \sum_{i \in \text{Right}(j,s)} (y_i - \bar{y}_{R})^2 \right]$$

The ensemble point prediction is the unweighted mean across all $B$ trees:
$$\hat{y}_t^{\text{RF}} = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x}_t)$$

**Baseline Models**:
1. **Persistence Baseline (Naive Walk)**: $\hat{y}_t^{\text{pers}} = y_{t-1}$
2. **Seasonal-Naive Baseline**: $\hat{y}_t^{\text{seas}} = y_{t-s}$, where $s = 24$ (hourly) or $s = 7$ (daily).

---

### 3.4 Evaluation Metrics
Evaluated on $\mathcal{D}_{\text{test}}$ ($n = |\mathcal{D}_{\text{test}}|$):

$$\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat{y}_t|, \qquad \text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2}$$

$$R^2 = 1 - \frac{\sum_{t=1}^{n} (y_t - \hat{y}_t)^2}{\sum_{t=1}^{n} (y_t - \bar{y}_{\text{test}})^2}, \qquad \text{MAPE}(\%) = \frac{100}{n} \sum_{t=1}^{n} \left|\frac{y_t - \hat{y}_t}{\max(|y_t|, 1)}\right|$$

---

### 3.5 Capacity Planning & Infrastructure Upgrade Equations
Given:
- Allocated physical link bandwidth $C_{\text{alloc}}$ (Mbps)
- SLA maximum safe utilization threshold $\tau_{\text{SLA}} \in (0, 1]$ (e.g., $0.80$ for $80\%$)
- Expected user/traffic growth rate $g \in [-0.5, 3.0]$ (e.g., $0.12$ for $+12\%$)
- Safety burst headroom buffer $h_{\text{safety}} \in [0, 1]$ (e.g., $0.15$ for $+15\%$)
- Base ML forecasted peak bandwidth over horizon $H$: $\hat{y}_{\text{peak}} = \max_{1 \le k \le H} \hat{y}_{T+k}$

1. **Effective SLA Safe Ceiling (Mbps)**:
   $$C_{\text{eff}} = C_{\text{alloc}} \times \tau_{\text{SLA}}$$

2. **Growth-Adjusted Peak Demand (Mbps)**:
   $$D_{\text{peak}} = \hat{y}_{\text{peak}} \times (1 + g)$$

3. **SLA Capacity Gap (Deficit in Mbps)**:
   $$\Delta_{\text{SLA}} = \max\left(0, \, D_{\text{peak}} - C_{\text{eff}}\right)$$

4. **Physical Link Deficit (Mbps)**:
   $$\Delta_{\text{phys}} = \max\left(0, \, D_{\text{peak}} - C_{\text{alloc}}\right)$$

5. **Required Target Link Capacity with Safety Headroom (Mbps)**:
   $$C_{\text{req}} = \frac{D_{\text{peak}} \times (1 + h_{\text{safety}})}{\tau_{\text{SLA}}}$$

6. **Carrier Tier Selection**:
   Select the smallest standard carrier optical tier $C_{\text{tier}} \in \{500, 1000, 2500, 5000, 10000, 25000, 40000, 100000\} \text{ Mbps}$ such that $C_{\text{tier}} \ge C_{\text{req}}$.

---

## 4. Limitations & Engineering Assumptions

1. **Error Accumulation in Multi-Step Recursive Forecasting**:
   - Because 7-day (`168` steps) and 30-day (`720` steps) forecasts feed step $k$'s prediction $\hat{y}_{T+k}$ back as lag features for step $k+1$, prediction variance widens over longer horizons. NetCapacity AI explicitly models this via widening 90% confidence intervals $\propto \sqrt{1 + 0.18(k/s)}$.
2. **Tree Extrapolation Ceiling**:
   - Decision tree ensembles cannot extrapolate target values strictly outside the minimum/maximum range observed in $\mathcal{D}_{\text{train}}$ without explicit trend scaling. NetCapacity AI addresses secular user growth via the explicit $(1 + g)$ user growth multiplier in the Capacity Planner.
3. **Synthetic Telemetry Characteristics**:
   - The 3 built-in datasets are deterministic synthetic datasets modeling realistic diurnal/weekly cycles, M/M/1 queuing delay, and micro-bursts. Real production SNMP interface counters (`ifHCInOctets`) may contain hardware counter wraps or routing flap outages that require SNMP-specific pre-filters.

---

## 5. Future Scope

1. **Live SNMP / NetFlow / Prometheus Streaming Ingestion**: Connect directly to router SNMPv3 polling agents or Prometheus Pushgateway for real-time 5-minute telemetry ingestion.
2. **Hybrid Residual Boosting (SARIMAX + XGBoost / Temporal Fusion Transformer)**: Combine linear trend decomposition with gradient-boosted trees or deep sequence models to extrapolate unbounded secular trends natively.
3. **Multi-Link BGP / SD-WAN Traffic Engineering**: Extend the single-link planner to a multi-node network topology graph using linear programming (Min-Cost Multi-Commodity Flow) to balance traffic across redundant ISP uplinks before purchasing hardware upgrades.

---

## 6. B.Tech Viva-Voce Questions & Answers (25 Questions)

### Q1. Why did you choose a Random Forest Regressor for network bandwidth forecasting?
**Answer**: Network traffic exhibits non-linear interactions between autoregressive lags (e.g., traffic 1 hour ago vs. 24 hours ago vs. 168 hours ago) and calendar states (business hours vs. weekends). Random Forest captures these non-linear interactions without requiring strict stationarity transformations (unlike ARIMA), is robust against isolated burst outliers, provides built-in feature importance rankings, and enables empirical uncertainty estimation across trees.

### Q2. How do you convert a time-series problem into a supervised regression problem?
**Answer**: Using the **sliding window (time-delay embedding)** method. At each timestamp $t$, we construct an input vector $\mathbf{x}_t$ from past lagged values ($y_{t-1}, y_{t-2}, \dots, y_{t-168}$), causal rolling window statistics ($\mu_{t,6}, \mu_{t,24}$), and deterministic calendar encodings, with the current observation $y_t$ as the target label.

### Q3. What is data leakage in time-series forecasting, and how did you prevent it?
**Answer**: Data leakage occurs when information from the future ($t$ or $t+k$) accidentally enters the training features used to predict $t$. We prevented leakage in three ways:
1. Using a **strict chronological train/test split** (earliest 80% for training, final 20% for testing) rather than random shuffling.
2. Shifting the target series by 1 step (`y.shift(1)`) **before** computing rolling means, standard deviations, and maximums so $y_t$ is never included in its own rolling window.
3. Normalizing the trend index strictly by the training set length.

### Q4. Why is random K-Fold Cross-Validation invalid for autoregressive time series?
**Answer**: In an autoregressive dataset, row $t+1$ contains $y_t$ as its `lag_1` feature. If row $t+1$ is placed in the training fold and row $t$ in the test fold, the model already sees the test target inside the training features, resulting in artificially inflated accuracy. Time-series evaluation must use chronological holdout or expanding-window walk-forward validation.

### Q5. What are the Persistence and Seasonal-Naive baselines, and why are they necessary?
**Answer**:
- **Persistence Baseline**: Predicts that bandwidth at step $t$ equals bandwidth at step $t-1$ ($\hat{y}_t = y_{t-1}$).
- **Seasonal-Naive Baseline**: Predicts that bandwidth at step $t$ equals bandwidth at the same hour yesterday ($\hat{y}_t = y_{t-24}$) or same day last week ($\hat{y}_t = y_{t-7}$).
In machine learning, a complex model is only justified if it achieves lower MAE and RMSE than trivial baselines. On our Campus Core dataset, Random Forest reduces RMSE by **30.40%** over Persistence and **73.19%** over Seasonal-Naive.

### Q6. Why do we encode hour-of-day and day-of-week using Sine and Cosine transforms?
**Answer**: Raw integer hours ($0$ to $23$) treat `23:00` and `00:00` as being 23 units apart, even though they are adjacent in time (1 hour apart). Mapping hour $h$ onto the unit circle via $(\sin(2\pi h/24), \cos(2\pi h/24))$ preserves cyclical continuity across midnight and week boundaries.

### Q7. How does multi-step recursive forecasting work for 24-hour, 7-day, and 30-day horizons?
**Answer**: To forecast $H$ steps into the future ($T+1, \dots, T+H$), the model first predicts $\hat{y}_{T+1}$ using actual historical lags up to $T$. Then $\hat{y}_{T+1}$ is appended to the history buffer and used as `lag_1` when predicting $\hat{y}_{T+2}$, repeating recursively for $24$, $168$, or $720$ steps.

### Q8. How are the 90% prediction intervals computed in your forecast charts?
**Answer**: We combine the out-of-sample test residual standard deviation ($\sigma_{\text{res}}$) with the inter-tree ensemble standard deviation ($\sigma_{\text{trees}}$) at step $k$, scaled by a horizon growth factor $\sqrt{1 + 0.18(k/s)}$:
$$\text{CI}_{90\%}(T+k) = \hat{y}_{T+k} \pm 1.645 \sqrt{\sigma_{\text{res}}^2 + (0.65 \sigma_{\text{trees}})^2} \cdot \sqrt{1 + 0.18\frac{k}{s}}$$

### Q9. What is the difference between MAE, RMSE, and $R^2$?
**Answer**:
- **MAE (Mean Absolute Error)**: Average absolute difference in Mbps; weights all errors equally.
- **RMSE (Root Mean Squared Error)**: Penalizes large peak-traffic prediction errors quadratically, which is critical in capacity planning where missing a traffic spike causes packet loss.
- **$R^2$ (Coefficient of Determination)**: Proportion of variance in observed traffic explained by the model ($1.0$ is perfect, $0.0$ is equivalent to predicting the mean).

### Q10. Why is RMSE always greater than or equal to MAE?
**Answer**: By the Cauchy-Schwarz / Jensen inequality for convex function $f(x) = x^2$, $\frac{1}{n}\sum |e_i| \le \sqrt{\frac{1}{n}\sum e_i^2}$. Equality holds if and only if all error magnitudes $|e_i|$ are identical.

### Q11. What is the Effective SLA Ceiling ($C_{\text{eff}}$), and why don't network engineers run links at 100% utilization?
**Answer**: According to **M/M/1 queuing theory**, average queuing delay is proportional to $\frac{\rho}{1 - \rho}$, where $\rho$ is link utilization. Above $75\%\text{–}80\%$ utilization, queuing latency and tail-drop packet loss increase exponentially during micro-bursts. Therefore, organizations enforce an SLA utilization threshold (typically $\tau_{\text{SLA}} = 80\%$), making $C_{\text{eff}} = C_{\text{alloc}} \times 0.80$.

### Q12. Walk through an example of your Required Link Capacity formula.
**Answer**: Suppose current link capacity $C_{\text{alloc}} = 1000\text{ Mbps}$, SLA threshold $\tau_{\text{SLA}} = 80\%$ ($C_{\text{eff}} = 800\text{ Mbps}$), base forecasted peak $\hat{y}_{\text{peak}} = 780\text{ Mbps}$, expected user growth $g = 15\%$, and safety headroom $h_{\text{safety}} = 15\%$:
1. Growth-adjusted peak: $D_{\text{peak}} = 780 \times 1.15 = 897\text{ Mbps}$.
2. SLA Capacity Gap: $\Delta_{\text{SLA}} = 897 - 800 = 97\text{ Mbps}$ deficit.
3. Required Target Capacity: $C_{\text{req}} = \frac{897 \times 1.15}{0.80} = 1289.44\text{ Mbps}$.
4. Recommended Carrier Tier: Next standard tier above $1289.44\text{ Mbps}$ is **2.5 Gbps (`2500 Mbps`)**.

### Q13. How does your preprocessing pipeline handle missing values and duplicate timestamps in uploaded CSVs?
**Answer**:
1. Parses timestamps and drops unparseable rows.
2. Sorts observations chronologically and merges duplicate timestamps by averaging numeric telemetry columns.
3. Infers sampling frequency (`hourly` vs. `daily`) from median time delta and reindexes onto a continuous timestamp grid to expose hidden gaps.
4. Imputes missing (`NaN`) or negative corrupted bandwidth values using linear interpolation (or user-selected forward fill / rolling median).

### Q14. Why is the 24-hour forecast button disabled when a daily dataset is active?
**Answer**: A daily dataset has a sampling resolution of $\Delta t = 24\text{ hours}$ (1 data point per day). Therefore, a 24-hour window is only a single step, whereas 7-day (`7` steps) and 30-day (`30` steps) horizons provide meaningful multi-step trajectories.

### Q15. Which features have the highest importance in your trained Random Forest model?
**Answer**: On hourly network datasets, `lag_1` (immediate traffic inertia), `lag_24` (same hour yesterday), `lag_168` (same hour last week), `rolling_mean_6`, and `hour_cos` / `hour_sin` consistently rank at the top, accounting for over 80% of total impurity reduction.

### Q16. What is the role of Joblib in the backend?
**Answer**: `joblib` serializes the trained Random Forest model object, feature configuration, and evaluation metrics to disk (`backend/models/rf_model_dataset_<id>.joblib`) so the API can serve sub-second forecasts without retraining from scratch on every HTTP request.

### Q17. Why did you use SQLite in this architecture?
**Answer**: SQLite provides zero-configuration, ACID-compliant relational persistence for system settings, registered dataset metadata, model run audit logs, and alert acknowledgements in a single portable file (`netcapacity.db`), ideal for academic reproducibility and local deployment.

### Q18. How does FastAPI communicate with the React + Vite frontend?
**Answer**: FastAPI serves RESTful JSON endpoints under `/api/*` with `CORSMiddleware` enabled. During development, Vite's dev server proxies `/api` requests to `http://127.0.0.1:8000`, avoiding cross-origin issues.

### Q19. What happens when a user clicks "Acknowledge" on an alert?
**Answer**: The React frontend sends `POST /api/alerts/{id}/acknowledge` to FastAPI, which executes `UPDATE alerts SET acknowledged = 1 WHERE id = ?` in SQLite and returns the updated alert list and unacknowledged badge count.

### Q20. How are downloadable CSV reports generated?
**Answer**: The endpoints `/api/reports/forecast-csv` and `/api/reports/capacity-csv` dynamically run the forecast and capacity planner using the active dataset and settings, serialize the resulting Pandas DataFrame to an in-memory `io.StringIO` buffer, and return a FastAPI `Response` with `media_type="text/csv"` and `Content-Disposition: attachment`.

### Q21. Can a Random Forest model predict bandwidth values higher than the maximum seen in the training data?
**Answer**: Pure CART trees output the average of training target values in their leaf nodes, so their raw base prediction $\hat{y}_{\text{base}}$ is bounded by $[\min(y_{\text{train}}), \max(y_{\text{train}})]$. However, when combined with the user growth factor $(1 + g)$ and the 90% upper confidence bound in the Capacity Planner, the system accurately projects capacity breaches well above historical peaks.

### Q22. How did you model latency and packet loss in the synthetic datasets?
**Answer**: Using queuing physics: latency remains near a baseline propagation delay ($8.5\text{ ms}$) at low utilization, but exhibits an M/M/1 non-linear hockey-stick curve once link utilization exceeds $72\%$.Similarly, packet loss rises sharply when utilization exceeds $80\%$.

### Q23. What hyperparameters control overfitting in a Random Forest Regressor?
**Answer**:
- `max_depth`: Limits tree depth so trees do not memorize single noisy timestamps.
- `min_samples_split` and `min_samples_leaf`: Require a minimum number of observations in each leaf node, smoothing out predictions.
- `max_features` (`sqrt`): Decorrelates individual trees so no single lag feature dominates every root split.
- `n_estimators`: Increasing the number of trees reduces variance without increasing bias.

### Q24. How is your project tested automatically?
**Answer**: `backend/tests/test_api_and_ml.py` uses FastAPI's `TestClient` and Python's `unittest`/`pytest` framework to execute 7 end-to-end automated tests covering messy CSV imputation, invalid CSV rejection, leak-free chronological split verification, 24h/7d/30d forecast generation, capacity gap math, custom CSV upload, and CSV report downloads.

### Q25. How can a network administrator act on NetCapacity AI's recommendations before a physical fiber upgrade arrives?
**Answer**: When NetCapacity AI flags an `SLA_THRESHOLD_BREACH`, it recommends immediate traffic-engineering mitigations—specifically enabling **Hierarchical QoS (HQoS)** with DSCP Expedited Forwarding (EF) for real-time voice/video, applying WRED/rate-limiting on bulk transfers, and shifting scheduled cloud backups to the `01:00–05:00` night trough window.
