import React, { useEffect, useState } from 'react';
import {
  CheckCircle2,
  Cpu,
  Download,
  Gauge,
  Layers,
  Play,
  Save,
  Server,
  ShieldAlert,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { getCapacityCsvUrl, runCapacityPlan } from './api';

const QUICK_TIERS = [500, 1000, 2500, 5000, 10000];

export default function CapacityPlannerPage({
  initialPlan,
  activeDataset,
  onPlanSaved,
}) {
  const params = initialPlan?.parameters || {
    allocated_bandwidth_mbps: 1000,
    utilization_threshold_pct: 80,
    expected_user_growth_pct: 12,
    safety_headroom_pct: 15,
    horizon: '24h',
  };

  const [allocatedMbps, setAllocatedMbps] = useState(params.allocated_bandwidth_mbps);
  const [thresholdPct, setThresholdPct] = useState(params.utilization_threshold_pct);
  const [userGrowthPct, setUserGrowthPct] = useState(params.expected_user_growth_pct);
  const [headroomPct, setHeadroomPct] = useState(params.safety_headroom_pct);
  const [horizon, setHorizon] = useState(params.horizon || '24h');
  const [planResult, setPlanResult] = useState(initialPlan);
  const [loading, setLoading] = useState(false);
  const [saveMessage, setSaveMessage] = useState('');

  useEffect(() => {
    if (initialPlan) {
      setPlanResult(initialPlan);
      setAllocatedMbps(initialPlan.parameters.allocated_bandwidth_mbps);
      setThresholdPct(initialPlan.parameters.utilization_threshold_pct);
      setUserGrowthPct(initialPlan.parameters.expected_user_growth_pct);
      setHeadroomPct(initialPlan.parameters.safety_headroom_pct);
    }
  }, [initialPlan]);

  const handleSimulate = async (persist = false) => {
    setLoading(true);
    setSaveMessage('');
    try {
      const res = await runCapacityPlan({
        allocated_bandwidth_mbps: Number(allocatedMbps),
        utilization_threshold_pct: Number(thresholdPct),
        expected_user_growth_pct: Number(userGrowthPct),
        safety_headroom_pct: Number(headroomPct),
        horizon,
        persist_settings: persist,
      });
      setPlanResult(res.capacity_plan);
      if (persist) {
        setSaveMessage('Capacity plan parameters saved to SQLite & alerts updated.');
        if (onPlanSaved) onPlanSaved();
      }
    } catch (err) {
      console.error('Capacity simulation error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (!planResult) return null;

  const p = planResult.parameters;
  const hMetrics = planResult.historical_metrics;
  const fMetrics = planResult.forecast_metrics;
  const gap = planResult.capacity_gap_analysis;
  const recTier = planResult.recommended_upgrade_tier;

  const comparisonChartData = [
    { name: 'Historical Peak', mbps: hMetrics.peak_mbps, color: '#3B82F6' },
    { name: 'Base ML Peak', mbps: fMetrics.base_peak_mbps, color: '#0EA5E9' },
    { name: 'Growth-Adj Peak', mbps: fMetrics.growth_adjusted_peak_mbps, color: '#06B6D4' },
    { name: '90% Burst Peak', mbps: fMetrics.upper_90_peak_mbps, color: '#F59E0B' },
    { name: 'SLA Safe Ceiling', mbps: p.effective_sla_ceiling_mbps, color: '#10B981' },
    { name: 'Allocated Link', mbps: p.allocated_bandwidth_mbps, color: '#6366F1' },
    { name: 'Required Target', mbps: gap.required_link_capacity_mbps, color: '#EC4899' },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Gauge className="w-5 h-5 text-cyan-400" />
            <span>Interactive Network Capacity Planner & Upgrade Simulator</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Simulate link bandwidth, SLA utilization ceilings, user traffic growth, and burst headroom to compute
            exact capacity deficits and optical upgrade tiers
          </p>
        </div>

        <a
          href={getCapacityCsvUrl(horizon)}
          download
          className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-emerald-500/15 hover:bg-emerald-500/25 text-xs font-semibold text-emerald-300 border border-emerald-500/30 transition self-start"
        >
          <Download className="w-4 h-4" />
          <span>Download Capacity Audit CSV</span>
        </a>
      </div>

      {/* Main Grid: Left Controls + Right Results */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Editable Parameters Panel */}
        <div className="lg:col-span-5 bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90 space-y-4">
          <h2 className="text-sm font-bold text-white border-b border-slate-800 pb-2.5">
            1. Editable Infrastructure & SLA Parameters
          </h2>

          {/* Allocated Bandwidth */}
          <div>
            <div className="flex justify-between text-xs mb-1.5">
              <label className="font-medium text-slate-300">Allocated Link Bandwidth (Mbps)</label>
              <span className="font-mono font-bold text-cyan-300">{allocatedMbps} Mbps</span>
            </div>
            <input
              type="number"
              min="50"
              max="200000"
              step="50"
              value={allocatedMbps}
              onChange={(e) => setAllocatedMbps(Number(e.target.value))}
              className="w-full px-3 py-2 rounded-xl bg-[#111C35] border border-slate-700 text-sm font-mono text-white focus:outline-none focus:border-cyan-500"
            />
            <div className="flex flex-wrap gap-1.5 mt-2">
              {QUICK_TIERS.map((tier) => (
                <button
                  key={tier}
                  type="button"
                  onClick={() => setAllocatedMbps(tier)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-mono border transition ${
                    Number(allocatedMbps) === tier
                      ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                      : 'bg-slate-800/80 text-slate-400 border-slate-700 hover:text-white'
                  }`}
                >
                  {tier >= 1000 ? `${tier / 1000} Gbps` : `${tier} Mbps`}
                </button>
              ))}
            </div>
          </div>

          {/* SLA Utilization Threshold */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <label className="font-medium text-slate-300">SLA Max Utilization Threshold (%)</label>
              <span className="font-mono font-bold text-amber-300">
                {thresholdPct}% ({Math.round((allocatedMbps * thresholdPct) / 100)} Mbps Ceiling)
              </span>
            </div>
            <input
              type="range"
              min="50"
              max="95"
              step="1"
              value={thresholdPct}
              onChange={(e) => setThresholdPct(Number(e.target.value))}
              className="w-full accent-amber-400 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-500">
              <span>50% (Strict Core)</span>
              <span>80% (Standard SLA)</span>
              <span>95% (Aggressive)</span>
            </div>
          </div>

          {/* Expected User Growth */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <label className="font-medium text-slate-300">Expected User / Traffic Growth (%)</label>
              <span className="font-mono font-bold text-cyan-300">
                {userGrowthPct >= 0 ? `+${userGrowthPct}%` : `${userGrowthPct}%`}
              </span>
            </div>
            <input
              type="range"
              min="-20"
              max="120"
              step="1"
              value={userGrowthPct}
              onChange={(e) => setUserGrowthPct(Number(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer"
            />
          </div>

          {/* Safety Burst Headroom */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <label className="font-medium text-slate-300">Safety Burst Headroom Buffer (%)</label>
              <span className="font-mono font-bold text-emerald-300">+{headroomPct}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="50"
              step="1"
              value={headroomPct}
              onChange={(e) => setHeadroomPct(Number(e.target.value))}
              className="w-full accent-emerald-400 cursor-pointer"
            />
          </div>

          {/* Forecast Horizon */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">Planning Forecast Horizon</label>
            <div className="grid grid-cols-3 gap-2">
              {['24h', '7d', '30d'].map((h) => {
                const disabled = activeDataset?.frequency === 'daily' && h === '24h';
                return (
                  <button
                    key={h}
                    type="button"
                    disabled={disabled}
                    onClick={() => setHorizon(h)}
                    className={`py-2 rounded-xl font-mono text-xs font-semibold border transition ${
                      horizon === h
                        ? 'bg-cyan-500 text-slate-950 border-cyan-400'
                        : disabled
                        ? 'bg-slate-900 text-slate-600 border-slate-800 cursor-not-allowed'
                        : 'bg-[#111C35] text-slate-300 border-slate-700 hover:bg-slate-800'
                    }`}
                  >
                    {h.toUpperCase()}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Action Buttons */}
          <div className="pt-2 grid grid-cols-2 gap-3">
            <button
              type="button"
              onClick={() => handleSimulate(false)}
              disabled={loading}
              className="flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-[#111C35] hover:bg-slate-800 text-cyan-300 border border-cyan-500/40 font-semibold text-xs transition"
            >
              <Play className="w-3.5 h-3.5" />
              <span>{loading ? 'Running...' : 'Simulate Only'}</span>
            </button>
            <button
              type="button"
              onClick={() => handleSimulate(true)}
              disabled={loading}
              className="flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs transition"
            >
              <Save className="w-3.5 h-3.5" />
              <span>Apply & Save Plan</span>
            </button>
          </div>

          {saveMessage && (
            <p className="text-xs text-emerald-400 font-medium bg-emerald-500/10 border border-emerald-500/30 p-2.5 rounded-xl">
              {saveMessage}
            </p>
          )}
        </div>

        {/* Right Side: Computed Results & Chart */}
        <div className="lg:col-span-7 space-y-6">
          {/* Quantitative Capacity Gap Summary Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3.5">
            <div className="bg-[#0B1325] p-4 rounded-2xl border border-slate-800/90">
              <span className="text-[11px] font-mono uppercase text-slate-400">Effective SLA Ceiling</span>
              <p className="text-xl font-bold font-mono text-amber-300 mt-1">{p.effective_sla_ceiling_mbps} Mbps</p>
              <p className="text-[11px] text-slate-400 mt-1">
                {p.allocated_bandwidth_mbps}M × {p.utilization_threshold_pct}%
              </p>
            </div>

            <div className="bg-[#0B1325] p-4 rounded-2xl border border-slate-800/90">
              <span className="text-[11px] font-mono uppercase text-slate-400">Projected Peak Demand</span>
              <p className="text-xl font-bold font-mono text-cyan-300 mt-1">
                {fMetrics.growth_adjusted_peak_mbps} Mbps
              </p>
              <p className="text-[11px] text-slate-400 mt-1">
                {fMetrics.peak_utilization_pct}% of physical link
              </p>
            </div>

            <div className="bg-[#0B1325] p-4 rounded-2xl border border-slate-800/90">
              <span className="text-[11px] font-mono uppercase text-slate-400">SLA Capacity Gap</span>
              <p
                className={`text-xl font-bold font-mono mt-1 ${
                  gap.sla_capacity_gap_mbps > 0 ? 'text-rose-400' : 'text-emerald-400'
                }`}
              >
                {gap.sla_capacity_gap_mbps > 0
                  ? `+${gap.sla_capacity_gap_mbps} Mbps`
                  : `0 Mbps (Safe)`}
              </p>
              <p className="text-[11px] text-slate-400 mt-1">
                {gap.sla_breach_intervals_count} of {gap.total_forecast_intervals} steps breach
              </p>
            </div>

            <div className="bg-[#0B1325] p-4 rounded-2xl border border-slate-800/90">
              <span className="text-[11px] font-mono uppercase text-slate-400">Required Link Capacity</span>
              <p className="text-xl font-bold font-mono text-pink-400 mt-1">
                {gap.required_link_capacity_mbps} Mbps
              </p>
              <p className="text-[11px] text-slate-400 mt-1">Includes +{p.safety_headroom_pct}% safety buffer</p>
            </div>

            <div className="bg-[#0B1325] p-4 rounded-2xl border border-slate-800/90">
              <span className="text-[11px] font-mono uppercase text-slate-400">Additional BW Needed</span>
              <p className="text-xl font-bold font-mono text-white mt-1">
                +{gap.additional_bandwidth_needed_mbps} Mbps
              </p>
              <p className="text-[11px] text-slate-400 mt-1">Above current {p.allocated_bandwidth_mbps}M</p>
            </div>

            <div className="bg-[#0B1325] p-4 rounded-2xl border border-slate-800/90">
              <span className="text-[11px] font-mono uppercase text-slate-400">Recommended Tier</span>
              <p className="text-xl font-bold font-mono text-emerald-400 mt-1">{recTier.tier_mbps} Mbps</p>
              <p className="text-[11px] text-slate-400 mt-1">Est. ${recTier.typical_monthly_usd}/mo</p>
            </div>
          </div>

          {/* Capacity Benchmark Bar Chart */}
          <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90">
            <h3 className="text-sm font-bold text-white mb-1">
              Bandwidth Demand vs. SLA Ceiling & Required Upgrade Capacity (Mbps)
            </h3>
            <p className="text-xs text-slate-400 mb-4">
              Formula: Required Capacity = [Peak Forecast × (1 + Growth) × (1 + Headroom)] ÷ (SLA Threshold)
            </p>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonChartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis dataKey="name" stroke="#64748B" tick={{ fontSize: 11, fill: '#94A3B8' }} />
                  <YAxis stroke="#64748B" tick={{ fontSize: 11, fill: '#94A3B8' }} unit=" M" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0F172A',
                      borderColor: '#334155',
                      borderRadius: '10px',
                      fontSize: '12px',
                    }}
                  />
                  <Bar dataKey="mbps" name="Bandwidth (Mbps)" radius={[6, 6, 0, 0]}>
                    {comparisonChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Recommended Carrier Hardware Upgrade & Action Plan */}
      <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Server className="w-4 h-4 text-cyan-400" />
              <span>Recommended Carrier Tier: {recTier.label}</span>
            </h3>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              Optical / Interface Specification: {recTier.interface_spec}
            </p>
          </div>
          <span className="text-xs font-mono px-3 py-1 rounded-full bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
            Status: {gap.status_label}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {planResult.recommendations.map((rec, idx) => (
            <div key={idx} className="p-4 rounded-xl bg-[#111C35] border border-slate-800 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-mono font-semibold text-cyan-300 uppercase">{rec.category}</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-amber-300">
                  {rec.priority}
                </span>
              </div>
              <h4 className="text-sm font-bold text-white">{rec.title}</h4>
              <p className="text-xs text-slate-300 leading-relaxed">{rec.detail}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
