import React from 'react';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Cpu,
  Database,
  Download,
  Gauge,
  Network,
  Settings,
  ShieldCheck,
  TrendingUp,
} from 'lucide-react';
import { getCapacityCsvUrl, getForecastCsvUrl } from './api';

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Analytics Dashboard', icon: Activity, badge: null },
  { id: 'forecast', label: 'Forecast Explorer', icon: TrendingUp, badge: '24h/7d/30d' },
  { id: 'capacity', label: 'Capacity Planner', icon: Gauge, badge: 'Simulator' },
  { id: 'evaluation', label: 'Model Evaluation', icon: Cpu, badge: 'RF vs Naive' },
  { id: 'datasets', label: 'Dataset Manager', icon: Database, badge: 'CSV Upload' },
  { id: 'settings', label: 'Alerts & Settings', icon: Settings, badge: null },
];

export default function Sidebar({
  activeTab,
  setActiveTab,
  unacknowledgedAlerts = 0,
  activeDataset,
  horizon = '24h',
}) {
  return (
    <aside className="w-72 shrink-0 bg-[#0B1325] border-r border-slate-800/80 flex flex-col justify-between min-h-screen select-none">
      <div>
        {/* Brand Header */}
        <div className="p-5 border-b border-slate-800/80 flex items-center gap-3.5">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Network className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-base tracking-tight text-white">NetCapacity AI</span>
              <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                v1.0
              </span>
            </div>
            <p className="text-xs text-slate-400">Intelligent Bandwidth Planning</p>
          </div>
        </div>

        {/* Active Dataset Pill */}
        {activeDataset && (
          <div className="mx-4 mt-4 p-3 rounded-xl bg-[#111C35] border border-slate-800/90">
            <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
              <span className="uppercase tracking-wider font-semibold">Active Telemetry</span>
              <span
                className={`px-1.5 py-0.5 rounded font-mono text-[10px] ${
                  activeDataset.is_synthetic
                    ? 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
                    : 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30'
                }`}
              >
                {activeDataset.is_synthetic ? 'SYNTHETIC' : 'UPLOADED CSV'}
              </span>
            </div>
            <p className="text-xs font-medium text-slate-200 truncate" title={activeDataset.name}>
              {activeDataset.name}
            </p>
            <div className="mt-1.5 flex items-center justify-between text-[11px] font-mono text-slate-400">
              <span>{activeDataset.row_count} rows</span>
              <span className="uppercase text-cyan-400">{activeDataset.frequency}</span>
            </div>
          </div>
        )}

        {/* Navigation Links */}
        <nav className="p-3 space-y-1 mt-2">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            const showAlertCount = item.id === 'settings' && unacknowledgedAlerts > 0;

            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-500/20 to-blue-600/10 text-cyan-300 border border-cyan-500/30 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                </div>
                {showAlertCount ? (
                  <span className="px-2 py-0.5 text-[11px] font-mono font-semibold rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/40 flex items-center gap-1">
                    <AlertTriangle className="w-3 h-3" />
                    {unacknowledgedAlerts}
                  </span>
                ) : item.badge ? (
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                    {item.badge}
                  </span>
                ) : null}
              </button>
            );
          })}
        </nav>

        {/* Quick Report Exports */}
        <div className="px-4 mt-4">
          <div className="p-3.5 rounded-xl bg-[#111C35]/90 border border-slate-800/80 space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
              <Download className="w-3.5 h-3.5 text-cyan-400" />
              <span>Export CSV Reports</span>
            </div>
            <div className="grid grid-cols-1 gap-1.5 pt-1">
              <a
                href={getForecastCsvUrl(horizon)}
                download
                className="flex items-center justify-between px-2.5 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700/80 text-xs text-slate-200 border border-slate-700/70 transition"
              >
                <span>Forecast Series ({horizon})</span>
                <span className="font-mono text-[10px] text-cyan-400">.CSV</span>
              </a>
              <a
                href={getCapacityCsvUrl(horizon)}
                download
                className="flex items-center justify-between px-2.5 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700/80 text-xs text-slate-200 border border-slate-700/70 transition"
              >
                <span>Executive Capacity Audit</span>
                <span className="font-mono text-[10px] text-emerald-400">.CSV</span>
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* Footer Academic Credit */}
      <div className="p-4 border-t border-slate-800/80 bg-[#080F1E]">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
          <div>
            <p className="font-semibold text-slate-300">B.Tech AI & DS Project</p>
            <p className="text-[11px] text-slate-500">Random Forest Lag Forecasting</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
