import React, { useEffect, useState } from 'react';
import {
  AlertTriangle,
  Bell,
  CheckCircle2,
  Save,
  Settings,
} from 'lucide-react';
import {
  acknowledgeAlertById,
  fetchAlerts,
  fetchSettings,
  updateSettings,
} from './api';

export default function AlertsAndSettingsPage({ onSettingsUpdated }) {
  const [alerts, setAlerts] = useState([]);
  const [settings, setSettings] = useState(null);
  const [saving, setSaving] = useState(false);
  const [statusMsg, setStatusMsg] = useState('');

  const loadData = async () => {
    try {
      const [alertsRes, settingsRes] = await Promise.all([fetchAlerts(), fetchSettings()]);
      setAlerts(alertsRes.alerts || []);
      setSettings(settingsRes);
    } catch (err) {
      console.error('Failed to load alerts/settings:', err);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleAck = async (alertId) => {
    try {
      const res = await acknowledgeAlertById(alertId);
      setAlerts(res.alerts || []);
      if (onSettingsUpdated) onSettingsUpdated();
    } catch (err) {
      console.error('Failed to acknowledge alert:', err);
    }
  };

  const handleSaveSettings = async (e) => {
    e.preventDefault();
    if (!settings) return;
    setSaving(true);
    setStatusMsg('');
    try {
      const updated = await updateSettings({
        organization_name: settings.organization_name,
        allocated_bandwidth_mbps: Number(settings.allocated_bandwidth_mbps),
        utilization_threshold_pct: Number(settings.utilization_threshold_pct),
        expected_user_growth_pct: Number(settings.expected_user_growth_pct),
        safety_headroom_pct: Number(settings.safety_headroom_pct),
        alert_latency_threshold_ms: Number(settings.alert_latency_threshold_ms),
        alert_packet_loss_threshold_pct: Number(settings.alert_packet_loss_threshold_pct),
        cost_per_100mbps_usd: Number(settings.cost_per_100mbps_usd),
      });
      setSettings(updated);
      const alertsRes = await fetchAlerts();
      setAlerts(alertsRes.alerts || []);
      setStatusMsg('Settings persisted to SQLite database and alerts recalculated.');
      if (onSettingsUpdated) onSettingsUpdated();
    } catch (err) {
      console.error('Error saving settings:', err);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90">
        <h1 className="text-xl font-bold text-white flex items-center gap-2">
          <Settings className="w-5 h-5 text-cyan-400" />
          <span>Capacity Alert Center & Persistent System Settings</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Review and acknowledge capacity/QoS anomaly alerts and configure global SQLite-backed system parameters
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Alerts Management */}
        <div className="lg:col-span-6 bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-white flex items-center gap-2">
              <Bell className="w-4 h-4 text-amber-400" />
              <span>Detected Capacity & QoS Alerts ({alerts.length})</span>
            </h2>
          </div>

          {alerts.length === 0 ? (
            <p className="text-xs text-slate-400 py-8 text-center">No alerts logged for the active dataset.</p>
          ) : (
            <div className="space-y-3">
              {alerts.map((a) => {
                const isCrit = a.severity === 'CRITICAL';
                const isWarn = a.severity === 'WARNING';
                return (
                  <div
                    key={a.id}
                    className={`p-4 rounded-xl border ${
                      a.acknowledged
                        ? 'bg-slate-900/40 border-slate-800 opacity-65'
                        : isCrit
                        ? 'bg-rose-500/10 border-rose-500/30'
                        : isWarn
                        ? 'bg-amber-500/10 border-amber-500/30'
                        : 'bg-sky-500/10 border-sky-500/30'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[10px] font-mono font-bold uppercase px-2 py-0.5 rounded bg-slate-800 text-cyan-300">
                        {a.severity} • {a.category}
                      </span>
                      <span className="text-[11px] font-mono text-slate-400">{a.event_timestamp}</span>
                    </div>
                    <h3 className="text-xs font-bold text-white mt-2">{a.title}</h3>
                    <p className="text-xs text-slate-300 mt-1 leading-relaxed">{a.message}</p>
                    <div className="mt-3 flex items-center justify-between">
                      <span className="text-[11px] font-mono text-slate-400">
                        Metric: {a.metric_value} (Threshold: {a.threshold_value})
                      </span>
                      {a.acknowledged ? (
                        <span className="text-[11px] font-mono text-emerald-400 flex items-center gap-1">
                          <CheckCircle2 className="w-3.5 h-3.5" /> Acknowledged
                        </span>
                      ) : (
                        <button
                          onClick={() => handleAck(a.id)}
                          className="px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-cyan-300 border border-slate-700 transition"
                        >
                          Acknowledge
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Persistent Settings Form */}
        {settings && (
          <form
            onSubmit={handleSaveSettings}
            className="lg:col-span-6 bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90 space-y-4"
          >
            <h2 className="text-sm font-bold text-white">Persistent SQLite Configuration</h2>

            <div>
              <label className="block text-xs text-slate-300 mb-1">Organization / Network Node Label</label>
              <input
                type="text"
                value={settings.organization_name}
                onChange={(e) => setSettings({ ...settings, organization_name: e.target.value })}
                className="w-full px-3 py-2 rounded-xl bg-[#111C35] border border-slate-700 text-xs text-white"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-slate-300 mb-1">Allocated Bandwidth (Mbps)</label>
                <input
                  type="number"
                  value={settings.allocated_bandwidth_mbps}
                  onChange={(e) => setSettings({ ...settings, allocated_bandwidth_mbps: e.target.value })}
                  className="w-full px-3 py-2 rounded-xl bg-[#111C35] border border-slate-700 text-xs font-mono text-white"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-300 mb-1">SLA Threshold (%)</label>
                <input
                  type="number"
                  value={settings.utilization_threshold_pct}
                  onChange={(e) => setSettings({ ...settings, utilization_threshold_pct: e.target.value })}
                  className="w-full px-3 py-2 rounded-xl bg-[#111C35] border border-slate-700 text-xs font-mono text-white"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-300 mb-1">Default User Growth (%)</label>
                <input
                  type="number"
                  value={settings.expected_user_growth_pct}
                  onChange={(e) => setSettings({ ...settings, expected_user_growth_pct: e.target.value })}
                  className="w-full px-3 py-2 rounded-xl bg-[#111C35] border border-slate-700 text-xs font-mono text-white"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-300 mb-1">Safety Headroom Buffer (%)</label>
                <input
                  type="number"
                  value={settings.safety_headroom_pct}
                  onChange={(e) => setSettings({ ...settings, safety_headroom_pct: e.target.value })}
                  className="w-full px-3 py-2 rounded-xl bg-[#111C35] border border-slate-700 text-xs font-mono text-white"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-300 mb-1">Latency Alert Threshold (ms)</label>
                <input
                  type="number"
                  step="0.5"
                  value={settings.alert_latency_threshold_ms}
                  onChange={(e) => setSettings({ ...settings, alert_latency_threshold_ms: e.target.value })}
                  className="w-full px-3 py-2 rounded-xl bg-[#111C35] border border-slate-700 text-xs font-mono text-white"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-300 mb-1">Packet Loss Alert Ceiling (%)</label>
                <input
                  type="number"
                  step="0.05"
                  value={settings.alert_packet_loss_threshold_pct}
                  onChange={(e) => setSettings({ ...settings, alert_packet_loss_threshold_pct: e.target.value })}
                  className="w-full px-3 py-2 rounded-xl bg-[#111C35] border border-slate-700 text-xs font-mono text-white"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={saving}
              className="w-full py-2.5 px-4 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs flex items-center justify-center gap-2 transition"
            >
              <Save className="w-4 h-4" />
              <span>{saving ? 'Saving to SQLite...' : 'Save Persistent Settings'}</span>
            </button>

            {statusMsg && (
              <p className="text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/30 p-2.5 rounded-xl">
                {statusMsg}
              </p>
            )}
          </form>
        )}
      </div>
    </div>
  );
}
