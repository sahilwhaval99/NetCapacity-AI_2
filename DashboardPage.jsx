import React from 'react';
import {
  Activity,
  AlertOctagon,
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  Cpu,
  Download,
  Gauge,
  Info,
  Layers,
  RefreshCw,
  Server,
  TrendingUp,
  Wifi,
} from 'lucide-react';
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { getCapacityCsvUrl, getForecastCsvUrl } from './api';

const STATUS_BADGE_STYLES = {
  CRITICAL_SHORTAGE: 'bg-rose-500/15 text-rose-300 border-rose-500/40',
  SLA_THRESHOLD_BREACH: 'bg-amber-500/15 text-amber-300 border-amber-500/40',
  NEAR_CAPACITY_WATCH: 'bg-yellow-500/15 text-yellow-300 border-yellow-500/40',
  OVERPROVISIONED: 'bg-sky-500/15 text-sky-300 border-sky-500/40',
  HEALTHY_OPTIMAL: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40',
};

export default function DashboardPage({
  dashboardData,
  horizon,
  onHorizonChange,
  onNavigate,
  onRefresh,
  loading,
}) {
  if (!dashboardData) return null;

  const {
    organization_name,
    active_dataset,
    capacity_plan,
    forecast,
    model_metrics,
    engine_used,
    telemetry_series,
    seasonality_profile,
    alerts,
  } = dashboardData;

  const params = capacity_plan.parameters;
  const histMetrics = capacity_plan.historical_metrics;
  const forecastMetrics = capacity_plan.forecast_metrics;
  const gap = capacity_plan.capacity_gap_analysis;
  const recTier = capacity_plan.recommended_upgrade_tier;
  const rfTest = model_metrics.random_forest_test;
  const supportedHorizons = forecast.supported_horizons || ['24h', '7d', '30d'];

  // Combine recent historical context and forecasted points for the unified timeline chart
  const unifiedSeries = [
    ...forecast.historical_context.map((pt) => ({
      timestamp: pt.timestamp,
      historical_mbps: pt.actual_mbps,
      predicted_mbps: null,
      lower_bound_mbps: null,
      upper_bound_mbps: null,
    })),
    ...forecast.forecast_series.map((pt) => ({
      timestamp: pt.timestamp,
      historical_mbps: null,
      predicted_mbps: pt.predicted_mbps,
      base_forecast_mbps: pt.base_forecast_mbps,
      lower_bound_mbps: pt.lower_bound_mbps,
      upper_bound_mbps: pt.upper_bound_mbps,
    })),
  ];

  const unackAlerts = (alerts || []).filter((a) => !a.acknowledged);

  return (
    <div className="space-y-6">
      {/* Top Header & Synthetic Data Disclosure Banner */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90">
        <div>
          <div className="flex flex-wrap items-center gap-2.5 mb-1">
            <span className="text-xs font-mono uppercase tracking-wider px-2.5 py-0.5 rounded-full bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
              {organization_name}
            </span>
            <span
              className={`text-xs font-mono px-2.5 py-0.5 rounded-full border ${
                active_dataset.is_synthetic
                  ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
                  : 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
              }`}
            >
              {active_dataset.is_synthetic
                ? 'SYNTHETIC REPRODUCIBLE TELEMETRY (SEED=42)'
                : 'USER UPLOADED TELEMETRY CSV'}
            </span>
            <span className="text-xs font-mono px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
              {engine_used}
            </span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Network Capacity & Bandwidth Forecasting Center
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Dataset: <strong className="text-slate-200">{active_dataset.name}</strong> ({active_dataset.row_count}{' '}
            {active_dataset.frequency} observations • {active_dataset.start_timestamp} to{' '}
            {active_dataset.end_timestamp})
          </p>
        </div>

        {/* Horizon Controls & Actions */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center bg-[#111C35] p-1 rounded-xl border border-slate-800">
            {['24h', '7d', '30d'].map((h) => {
              const supported = supportedHorizons.includes(h);
              const active = horizon === h;
              return (
                <button
                  key={h}
                  disabled={!supported}
                  onClick={() => supported && onHorizonChange(h)}
                  title={
                    supported
                      ? `Switch forecast horizon to ${h.toUpperCase()}`
                      : `24h horizon requires hourly dataset (current dataset is ${active_dataset.frequency})`
                  }
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition ${
                    active
                      ? 'bg-cyan-500 text-slate-950 shadow-sm'
                      : supported
                      ? 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                      : 'text-slate-600 cursor-not-allowed'
                  }`}
                >
                  {h.toUpperCase()}
                </button>
              );
            })}
          </div>

          <button
            onClick={onRefresh}
            disabled={loading}
            className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-[#111C35] hover:bg-slate-800 text-xs font-medium text-slate-200 border border-slate-700/80 transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-cyan-400' : 'text-slate-400'}`} />
            <span>Refresh</span>
          </button>

          <a
            href={getForecastCsvUrl(horizon)}
            download
            className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-cyan-500/15 hover:bg-cyan-500/25 text-xs font-medium text-cyan-300 border border-cyan-500/30 transition"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export CSV</span>
          </a>
        </div>
      </div>

      {/* 4 Primary KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        {/* KPI 1: Historical / Current Utilization */}
        <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Historical Peak Load
            </span>
            <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
              <Wifi className="w-4 h-4" />
            </div>
          </div>
          <div className="my-3">
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold font-mono text-white">{histMetrics.peak_mbps}</span>
              <span className="text-xs font-mono text-slate-400">Mbps</span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Avg: <strong className="text-slate-200 font-mono">{histMetrics.avg_mbps} Mbps</strong> • P95:{' '}
              <strong className="text-slate-200 font-mono">{histMetrics.p95_mbps} Mbps</strong>
            </p>
          </div>
          <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-xs">
            <span className="text-slate-400">Allocated Link:</span>
            <span className="font-mono font-semibold text-blue-300">
              {params.allocated_bandwidth_mbps} Mbps ({histMetrics.peak_utilization_pct}% Peak)
            </span>
          </div>
        </div>

        {/* KPI 2: Forecasted Peak Demand */}
        <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Projected Peak ({forecast.horizon.toUpperCase()})
            </span>
            <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              <TrendingUp className="w-4 h-4" />
            </div>
          </div>
          <div className="my-3">
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold font-mono text-cyan-300">
                {forecastMetrics.growth_adjusted_peak_mbps}
              </span>
              <span className="text-xs font-mono text-slate-400">Mbps</span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Base ML: <strong className="text-slate-200 font-mono">{forecastMetrics.base_peak_mbps} Mbps</strong> (
              +{params.expected_user_growth_pct}% growth)
            </p>
          </div>
          <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-xs">
            <span className="text-slate-400">90% Upper Burst:</span>
            <span className="font-mono font-semibold text-amber-300">
              {forecastMetrics.upper_90_peak_mbps} Mbps ({forecastMetrics.peak_utilization_pct}% Util)
            </span>
          </div>
        </div>

        {/* KPI 3: Capacity Gap & SLA Threshold Status */}
        <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              SLA Capacity Gap ({params.utilization_threshold_pct}% Limit)
            </span>
            <div
              className={`p-2 rounded-lg border ${
                gap.sla_capacity_gap_mbps > 0
                  ? 'bg-rose-500/10 text-rose-400 border-rose-500/20'
                  : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
              }`}
            >
              <Gauge className="w-4 h-4" />
            </div>
          </div>
          <div className="my-3">
            <div className="flex items-baseline gap-2">
              <span
                className={`text-2xl font-bold font-mono ${
                  gap.sla_capacity_gap_mbps > 0 ? 'text-rose-400' : 'text-emerald-400'
                }`}
              >
                {gap.sla_capacity_gap_mbps > 0
                  ? `+${gap.sla_capacity_gap_mbps} Mbps Deficit`
                  : `${Math.abs(gap.available_sla_margin_mbps)} Mbps Margin`}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              SLA Ceiling: <strong className="text-slate-200 font-mono">{params.effective_sla_ceiling_mbps} Mbps</strong>{' '}
              • Breach Steps: <strong className="text-slate-200 font-mono">{gap.sla_breach_intervals_count}</strong>
            </p>
          </div>
          <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-xs">
            <span
              className={`px-2 py-0.5 rounded font-mono text-[11px] border ${
                STATUS_BADGE_STYLES[gap.status] || STATUS_BADGE_STYLES.HEALTHY_OPTIMAL
              }`}
            >
              {gap.status_label}
            </span>
            <button
              onClick={() => onNavigate('capacity')}
              className="text-cyan-400 hover:underline flex items-center gap-0.5 font-medium"
            >
              <span>Plan</span>
              <ArrowUpRight className="w-3 h-3" />
            </button>
          </div>
        </div>

        {/* KPI 4: Random Forest Model Accuracy */}
        <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Random Forest Accuracy
            </span>
            <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <Cpu className="w-4 h-4" />
            </div>
          </div>
          <div className="my-3">
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold font-mono text-emerald-400">
                R² = {rfTest.r2.toFixed(4)}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              RMSE: <strong className="text-slate-200 font-mono">{rfTest.rmse} Mbps</strong> • MAE:{' '}
              <strong className="text-slate-200 font-mono">{rfTest.mae} Mbps</strong>
            </p>
          </div>
          <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-xs">
            <span className="text-slate-400">vs. Persistence:</span>
            <span className="font-mono font-semibold text-emerald-300">
              +{model_metrics.rmse_improvement_over_persistence_pct}% Lower Error
            </span>
          </div>
        </div>
      </div>

      {/* Main Historical Context + Multi-Horizon Forecast Chart */}
      <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span>Historical Telemetry vs. ML Forecasted Bandwidth Demand ({forecast.horizon.toUpperCase()})</span>
            </h2>
            <p className="text-xs text-slate-400">
              Includes recent actual traffic, growth-adjusted Random Forest forecast (+{params.expected_user_growth_pct}
              %), 90% prediction interval, and SLA capacity thresholds
            </p>
          </div>
          <div className="flex items-center gap-3 text-xs font-mono">
            <span className="flex items-center gap-1.5 text-slate-300">
              <span className="w-3 h-0.5 bg-blue-400 inline-block" /> Historical
            </span>
            <span className="flex items-center gap-1.5 text-cyan-300">
              <span className="w-3 h-0.5 bg-cyan-400 inline-block" /> RF Forecast
            </span>
            <span className="flex items-center gap-1.5 text-amber-300">
              <span className="w-3 h-0.5 bg-amber-400 inline-block border-b border-dashed" /> SLA Limit (
              {params.effective_sla_ceiling_mbps}M)
            </span>
            <span className="flex items-center gap-1.5 text-rose-400">
              <span className="w-3 h-0.5 bg-rose-500 inline-block" /> Physical Link ({params.allocated_bandwidth_mbps}M)
            </span>
          </div>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={unifiedSeries} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <defs>
                <linearGradient id="ciBandGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#06B6D4" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#06B6D4" stopOpacity={0.03} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis
                dataKey="timestamp"
                stroke="#64748B"
                tick={{ fontSize: 11, fill: '#94A3B8' }}
                minTickGap={36}
              />
              <YAxis
                stroke="#64748B"
                tick={{ fontSize: 11, fill: '#94A3B8' }}
                unit=" M"
                domain={[0, (dataMax) => Math.max(dataMax * 1.1, params.allocated_bandwidth_mbps * 1.08)]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0F172A',
                  borderColor: '#334155',
                  borderRadius: '12px',
                  fontSize: '12px',
                  color: '#F1F5F9',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '12px' }} />

              <ReferenceLine
                y={params.allocated_bandwidth_mbps}
                stroke="#F43F5E"
                strokeWidth={1.5}
                strokeDasharray="6 4"
                label={{
                  value: `Link Capacity (${params.allocated_bandwidth_mbps} Mbps)`,
                  fill: '#FB7185',
                  fontSize: 11,
                  position: 'insideTopRight',
                }}
              />
              <ReferenceLine
                y={params.effective_sla_ceiling_mbps}
                stroke="#F59E0B"
                strokeWidth={1.5}
                strokeDasharray="4 4"
                label={{
                  value: `SLA Threshold (${params.effective_sla_ceiling_mbps} Mbps)`,
                  fill: '#FBBF24',
                  fontSize: 11,
                  position: 'insideBottomRight',
                }}
              />

              <Area
                type="monotone"
                dataKey="upper_bound_mbps"
                name="90% Upper Burst Envelope"
                stroke="none"
                fill="url(#ciBandGrad)"
              />
              <Line
                type="monotone"
                dataKey="historical_mbps"
                name="Observed Historical Traffic (Mbps)"
                stroke="#3B82F6"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="predicted_mbps"
                name="Growth-Adjusted Forecast (Mbps)"
                stroke="#06B6D4"
                strokeWidth={2.5}
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Middle Row: Seasonality Profile + Telemetry QoS (Latency & Loss) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Diurnal / Weekly Traffic Profile */}
        <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90">
          <div className="mb-4">
            <h3 className="text-sm font-bold text-white">
              {active_dataset.frequency === 'hourly'
                ? 'Diurnal 24-Hour Bandwidth Seasonality Profile'
                : 'Weekly Day-of-Week Bandwidth Seasonality Profile'}
            </h3>
            <p className="text-xs text-slate-400">
              Historical average vs. peak bandwidth demand per cycle slot
            </p>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={seasonality_profile} margin={{ top: 5, right: 15, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis dataKey="hour" stroke="#64748B" tick={{ fontSize: 11, fill: '#94A3B8' }} />
                <YAxis stroke="#64748B" tick={{ fontSize: 11, fill: '#94A3B8' }} unit=" M" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0F172A',
                    borderColor: '#334155',
                    borderRadius: '10px',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Bar dataKey="avg_mbps" name="Mean Traffic (Mbps)" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="peak_mbps" name="Peak Traffic (Mbps)" fill="#06B6D4" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Queuing Latency & Packet Loss Correlation */}
        <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90">
          <div className="mb-4">
            <h3 className="text-sm font-bold text-white">
              Recent Network QoS Telemetry: Queuing Latency (ms) & Active Users
            </h3>
            <p className="text-xs text-slate-400">
              Monitors M/M/1 queuing delay spikes as bandwidth utilization approaches link saturation
            </p>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={telemetry_series.slice(-48)} margin={{ top: 5, right: 15, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis
                  dataKey="timestamp"
                  stroke="#64748B"
                  tick={{ fontSize: 11, fill: '#94A3B8' }}
                  minTickGap={28}
                />
                <YAxis yAxisId="left" stroke="#F59E0B" tick={{ fontSize: 11, fill: '#FBBF24' }} unit=" ms" />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  stroke="#10B981"
                  tick={{ fontSize: 11, fill: '#34D399' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0F172A',
                    borderColor: '#334155',
                    borderRadius: '10px',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="latency_ms"
                  name="Round-Trip Latency (ms)"
                  stroke="#F59E0B"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="active_users"
                  name="Concurrent Active Users"
                  stroke="#10B981"
                  strokeWidth={1.8}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Bottom Row: Infrastructure Upgrade Recommendation + Active Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Upgrade Recommendation Box */}
        <div className="lg:col-span-7 bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Server className="w-4 h-4 text-cyan-400" />
                <span>AI Infrastructure Upgrade & Provisioning Recommendation</span>
              </h3>
              <span className="text-xs font-mono text-cyan-300 bg-cyan-500/10 px-2.5 py-0.5 rounded-full border border-cyan-500/30">
                Required Target: {gap.required_link_capacity_mbps} Mbps
              </span>
            </div>

            <div className="p-4 rounded-xl bg-[#111C35] border border-slate-800 mb-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <span className="text-[11px] font-mono uppercase text-slate-400">
                    Recommended Carrier Optical Tier
                  </span>
                  <h4 className="text-base font-bold text-cyan-300">{recTier.label}</h4>
                  <p className="text-xs text-slate-300 mt-0.5 font-mono">{recTier.interface_spec}</p>
                </div>
                <div className="text-left sm:text-right">
                  <span className="text-[11px] font-mono uppercase text-slate-400">Est. Carrier OpEx</span>
                  <p className="text-lg font-bold font-mono text-emerald-400">
                    ${recTier.typical_monthly_usd.toLocaleString()}/mo
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-2.5">
              {capacity_plan.recommendations.map((rec, idx) => (
                <div
                  key={idx}
                  className="p-3.5 rounded-xl bg-slate-900/70 border border-slate-800/90 flex items-start gap-3"
                >
                  <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
                  <div>
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-xs font-bold text-white">{rec.title}</span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-cyan-300">
                        {rec.priority}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 mt-1 leading-relaxed">{rec.detail}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between">
            <span className="text-xs text-slate-400">
              Want to simulate custom bandwidth tiers, SLA thresholds, or user growth?
            </span>
            <button
              onClick={() => onNavigate('capacity')}
              className="px-3.5 py-1.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs transition"
            >
              Open Capacity Simulator →
            </button>
          </div>
        </div>

        {/* Active Capacity & Anomaly Alerts */}
        <div className="lg:col-span-5 bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                <span>Active Capacity & Telemetry Alerts</span>
              </h3>
              <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-slate-800 text-slate-300">
                {unackAlerts.length} Unacknowledged
              </span>
            </div>

            {alerts && alerts.length > 0 ? (
              <div className="space-y-2.5 max-h-72 overflow-y-auto pr-1">
                {alerts.slice(0, 5).map((alert) => {
                  const isCrit = alert.severity === 'CRITICAL';
                  const isWarn = alert.severity === 'WARNING';
                  return (
                    <div
                      key={alert.id}
                      className={`p-3.5 rounded-xl border ${
                        isCrit
                          ? 'bg-rose-500/10 border-rose-500/30'
                          : isWarn
                          ? 'bg-amber-500/10 border-amber-500/30'
                          : 'bg-sky-500/10 border-sky-500/30'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span
                          className={`text-[10px] font-mono font-bold uppercase px-1.5 py-0.5 rounded ${
                            isCrit
                              ? 'bg-rose-500/20 text-rose-300'
                              : isWarn
                              ? 'bg-amber-500/20 text-amber-300'
                              : 'bg-sky-500/20 text-sky-300'
                          }`}
                        >
                          {alert.severity} • {alert.category}
                        </span>
                        <span className="text-[11px] font-mono text-slate-400">{alert.event_timestamp}</span>
                      </div>
                      <h4 className="text-xs font-bold text-white mt-1.5">{alert.title}</h4>
                      <p className="text-xs text-slate-300 mt-1 leading-relaxed">{alert.message}</p>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="p-8 text-center text-xs text-slate-400">
                No active capacity or QoS alerts for this dataset.
              </div>
            )}
          </div>

          <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between">
            <span className="text-xs text-slate-400">Manage alert thresholds & acknowledgements</span>
            <button
              onClick={() => onNavigate('settings')}
              className="text-xs font-semibold text-cyan-400 hover:underline"
            >
              View All Alerts →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
