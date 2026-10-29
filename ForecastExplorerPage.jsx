import React, { useEffect, useState } from 'react';
import {
  Calendar,
  Download,
  Filter,
  Sliders,
  TrendingUp,
  AlertCircle,
} from 'lucide-react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { fetchForecast, getForecastCsvUrl } from './api';

export default function ForecastExplorerPage({
  initialHorizon = '24h',
  initialGrowth = 12.0,
  activeDataset,
}) {
  const [horizon, setHorizon] = useState(initialHorizon);
  const [userGrowthPct, setUserGrowthPct] = useState(initialGrowth);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [filterBreachOnly, setFilterBreachOnly] = useState(false);

  const supportedHorizons =
    activeDataset?.frequency === 'daily' ? ['7d', '30d'] : ['24h', '7d', '30d'];

  useEffect(() => {
    if (activeDataset?.frequency === 'daily' && horizon === '24h') {
      setHorizon('7d');
    }
  }, [activeDataset, horizon]);

  const loadForecast = async (targetHorizon, targetGrowth) => {
    setLoading(true);
    try {
      const res = await fetchForecast(targetHorizon, targetGrowth);
      setData(res);
    } catch (err) {
      console.error('Failed to load forecast:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadForecast(horizon, userGrowthPct);
  }, [horizon, activeDataset?.id]);

  if (!data) {
    return (
      <div className="p-8 bg-[#0B1325] rounded-2xl border border-slate-800 text-sm text-slate-400">
        Loading multi-step bandwidth forecast...
      </div>
    );
  }

  const { forecast, capacity_plan } = data;
  const params = capacity_plan.parameters;
  const summary = forecast.summary;
  const series = forecast.forecast_series || [];

  const filteredTableRows = filterBreachOnly
    ? series.filter((pt) => pt.predicted_mbps > params.effective_sla_ceiling_mbps)
    : series;

  return (
    <div className="space-y-6">
      {/* Control Header */}
      <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-cyan-400" />
            <span>Multi-Horizon Autoregressive Bandwidth Forecast Explorer</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Recursive multi-step Random Forest forecasting with 90% empirical prediction intervals and seasonal-naive
            comparison
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Horizon Selector */}
          <div className="flex items-center bg-[#111C35] p-1 rounded-xl border border-slate-800">
            {[
              { id: '24h', label: '24 Hours (24 Steps)' },
              { id: '7d', label: '7 Days' },
              { id: '30d', label: '30 Days' },
            ].map((item) => {
              const supported = supportedHorizons.includes(item.id);
              const active = forecast.horizon === item.id;
              return (
                <button
                  key={item.id}
                  disabled={!supported}
                  onClick={() => supported && setHorizon(item.id)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition ${
                    active
                      ? 'bg-cyan-500 text-slate-950'
                      : supported
                      ? 'text-slate-300 hover:text-white'
                      : 'text-slate-600 cursor-not-allowed'
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </div>

          <a
            href={getForecastCsvUrl(forecast.horizon)}
            download
            className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-cyan-500/15 hover:bg-cyan-500/25 text-xs font-medium text-cyan-300 border border-cyan-500/30 transition"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download {forecast.horizon.toUpperCase()} CSV</span>
          </a>
        </div>
      </div>

      {/* Interactive Growth Rate Bar + Summary Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        <div className="lg:col-span-5 bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-cyan-400" />
              <span>Expected User / Traffic Growth Multiplier</span>
            </span>
            <span className="text-sm font-mono font-bold text-cyan-300">
              {userGrowthPct >= 0 ? `+${userGrowthPct}%` : `${userGrowthPct}%`}
            </span>
          </div>

          <div className="my-4">
            <input
              type="range"
              min="-20"
              max="100"
              step="1"
              value={userGrowthPct}
              onChange={(e) => setUserGrowthPct(Number(e.target.value))}
              onMouseUp={() => loadForecast(horizon, userGrowthPct)}
              onTouchEnd={() => loadForecast(horizon, userGrowthPct)}
              className="w-full accent-cyan-400 cursor-pointer"
            />
            <div className="flex justify-between text-[11px] font-mono text-slate-500 mt-1">
              <span>-20% (Contraction)</span>
              <span>0% (Base ML)</span>
              <span>+50%</span>
              <span>+100% (2x Surge)</span>
            </div>
          </div>

          <div className="flex items-center justify-between pt-2 border-t border-slate-800">
            <span className="text-xs text-slate-400">
              Drag slider or click Apply to recompute multi-step forecast:
            </span>
            <button
              onClick={() => loadForecast(horizon, userGrowthPct)}
              disabled={loading}
              className="px-3 py-1 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs transition"
            >
              {loading ? 'Updating...' : 'Apply Growth'}
            </button>
          </div>
        </div>

        {/* Summary Forecast Cards */}
        <div className="lg:col-span-7 grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-[#0B1325] p-4 rounded-2xl border border-slate-800/90">
            <span className="text-[11px] font-mono uppercase text-slate-400">Base ML Peak</span>
            <p className="text-xl font-bold font-mono text-white mt-1">{summary.base_peak_mbps} M</p>
            <p className="text-[11px] text-slate-400 mt-1">Before user growth</p>
          </div>
          <div className="bg-[#0B1325] p-4 rounded-2xl border border-slate-800/90">
            <span className="text-[11px] font-mono uppercase text-slate-400">Growth-Adj Peak</span>
            <p className="text-xl font-bold font-mono text-cyan-300 mt-1">{summary.predicted_peak_mbps} M</p>
            <p className="text-[11px] text-slate-400 mt-1">At {summary.peak_timestamp}</p>
          </div>
          <div className="bg-[#0B1325] p-4 rounded-2xl border border-slate-800/90">
            <span className="text-[11px] font-mono uppercase text-slate-400">Projected P95</span>
            <p className="text-xl font-bold font-mono text-blue-300 mt-1">{summary.predicted_p95_mbps} M</p>
            <p className="text-[11px] text-slate-400 mt-1">Mean: {summary.predicted_avg_mbps} Mbps</p>
          </div>
          <div className="bg-[#0B1325] p-4 rounded-2xl border border-slate-800/90">
            <span className="text-[11px] font-mono uppercase text-slate-400">90% Upper CI Peak</span>
            <p className="text-xl font-bold font-mono text-amber-300 mt-1">{summary.upper_90_peak_mbps} M</p>
            <p className="text-[11px] text-slate-400 mt-1">Worst-case burst</p>
          </div>
        </div>
      </div>

      {/* Main Forecast Comparison Chart */}
      <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90">
        <div className="mb-4 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-sm font-bold text-white">
              Random Forest Multi-Step Forecast vs. Seasonal-Naive Baseline ({forecast.steps}{' '}
              {forecast.frequency === 'hourly' ? 'Hours' : 'Days'} Ahead)
            </h2>
            <p className="text-xs text-slate-400">
              Shaded band shows the 90% empirical prediction interval [Lower 90% CI, Upper 90% CI]
            </p>
          </div>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={series} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis
                dataKey="timestamp"
                stroke="#64748B"
                tick={{ fontSize: 11, fill: '#94A3B8' }}
                minTickGap={35}
              />
              <YAxis
                stroke="#64748B"
                tick={{ fontSize: 11, fill: '#94A3B8' }}
                unit=" M"
                domain={[0, (dataMax) => Math.max(dataMax * 1.08, params.allocated_bandwidth_mbps * 1.05)]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0F172A',
                  borderColor: '#334155',
                  borderRadius: '12px',
                  fontSize: '12px',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '12px' }} />

              <ReferenceLine
                y={params.allocated_bandwidth_mbps}
                stroke="#F43F5E"
                strokeDasharray="6 4"
                label={{
                  value: `Allocated Capacity (${params.allocated_bandwidth_mbps} Mbps)`,
                  fill: '#FB7185',
                  fontSize: 11,
                  position: 'insideTopRight',
                }}
              />
              <ReferenceLine
                y={params.effective_sla_ceiling_mbps}
                stroke="#F59E0B"
                strokeDasharray="4 4"
                label={{
                  value: `SLA Ceiling (${params.effective_sla_ceiling_mbps} Mbps)`,
                  fill: '#FBBF24',
                  fontSize: 11,
                  position: 'insideBottomRight',
                }}
              />

              <Area
                type="monotone"
                dataKey="upper_bound_mbps"
                name="90% Upper Confidence Bound"
                stroke="none"
                fill="#06B6D4"
                fillOpacity={0.14}
              />
              <Line
                type="monotone"
                dataKey="predicted_mbps"
                name="Growth-Adjusted RF Forecast (Mbps)"
                stroke="#06B6D4"
                strokeWidth={2.5}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="base_forecast_mbps"
                name="Base ML Forecast (0% Growth)"
                stroke="#3B82F6"
                strokeWidth={1.8}
                strokeDasharray="4 4"
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="seasonal_baseline_mbps"
                name="Seasonal-Naive Baseline"
                stroke="#94A3B8"
                strokeWidth={1.4}
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Forecast Data Table */}
      <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
          <div>
            <h3 className="text-sm font-bold text-white">
              Step-by-Step Forecast Schedule ({filteredTableRows.length} rows shown)
            </h3>
            <p className="text-xs text-slate-400">
              Exact timestamped bandwidth predictions, confidence intervals, and SLA breach flags
            </p>
          </div>
          <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={filterBreachOnly}
              onChange={(e) => setFilterBreachOnly(e.target.checked)}
              className="rounded accent-cyan-500"
            />
            <span>Show only SLA Threshold Breaches (&gt; {params.effective_sla_ceiling_mbps} Mbps)</span>
          </label>
        </div>

        <div className="overflow-x-auto max-h-96">
          <table className="w-full text-left border-collapse text-xs">
            <thead className="sticky top-0 bg-[#111C35] text-slate-300 border-b border-slate-700 font-mono uppercase text-[11px]">
              <tr>
                <th className="py-2.5 px-3">Step</th>
                <th className="py-2.5 px-3">Timestamp</th>
                <th className="py-2.5 px-3 text-right">Base ML (Mbps)</th>
                <th className="py-2.5 px-3 text-right">Growth Forecast (Mbps)</th>
                <th className="py-2.5 px-3 text-right">90% CI Range (Mbps)</th>
                <th className="py-2.5 px-3 text-right">Link Util (%)</th>
                <th className="py-2.5 px-3 text-center">SLA Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70 font-mono">
              {filteredTableRows.slice(0, 120).map((row) => {
                const utilPct = ((row.predicted_mbps / params.allocated_bandwidth_mbps) * 100).toFixed(1);
                const isSat = row.predicted_mbps >= params.allocated_bandwidth_mbps;
                const isBreach = row.predicted_mbps > params.effective_sla_ceiling_mbps;

                return (
                  <tr key={row.step} className="hover:bg-slate-800/40">
                    <td className="py-2 px-3 text-slate-400">#{row.step}</td>
                    <td className="py-2 px-3 text-slate-200">{row.timestamp}</td>
                    <td className="py-2 px-3 text-right text-slate-300">{row.base_forecast_mbps}</td>
                    <td className="py-2 px-3 text-right font-bold text-cyan-300">{row.predicted_mbps}</td>
                    <td className="py-2 px-3 text-right text-slate-400">
                      [{row.lower_bound_mbps} – {row.upper_bound_mbps}]
                    </td>
                    <td className="py-2 px-3 text-right text-slate-200">{utilPct}%</td>
                    <td className="py-2 px-3 text-center">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                          isSat
                            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                            : isBreach
                            ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                            : 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30'
                        }`}
                      >
                        {isSat ? 'SATURATION' : isBreach ? 'SLA BREACH' : 'WITHIN SLA'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
