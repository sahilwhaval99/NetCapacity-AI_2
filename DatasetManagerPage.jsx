import React, { useEffect, useState } from 'react';
import {
  CheckCircle2,
  Database,
  FileSpreadsheet,
  Trash2,
  UploadCloud,
  AlertCircle,
} from 'lucide-react';
import {
  activateDataset,
  deleteDataset,
  fetchDatasets,
  uploadDatasetCsv,
} from './api';

export default function DatasetManagerPage({ onDatasetChanged }) {
  const [datasets, setDatasets] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [loading, setLoading] = useState(false);

  // Upload form state
  const [selectedFile, setSelectedFile] = useState(null);
  const [customName, setCustomName] = useState('');
  const [customDesc, setCustomDesc] = useState('');
  const [imputationMethod, setImputationMethod] = useState('linear');
  const [uploading, setUploading] = useState(false);
  const [uploadReport, setUploadReport] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');

  const loadList = async () => {
    setLoading(true);
    try {
      const res = await fetchDatasets();
      setDatasets(res.datasets || []);
      setActiveId(res.active_dataset_id);
    } catch (err) {
      console.error('Failed to list datasets:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadList();
  }, []);

  const handleActivate = async (id) => {
    setLoading(true);
    setErrorMsg('');
    try {
      await activateDataset(id);
      setActiveId(id);
      await loadList();
      if (onDatasetChanged) onDatasetChanged();
    } catch (err) {
      setErrorMsg(err.response?.data?.detail || 'Failed to activate dataset.');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    setErrorMsg('');
    try {
      await deleteDataset(id);
      await loadList();
      if (onDatasetChanged) onDatasetChanged();
    } catch (err) {
      setErrorMsg(err.response?.data?.detail || 'Cannot delete built-in dataset.');
    }
  };

  const handleUploadSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setErrorMsg('Please select a CSV file to upload.');
      return;
    }
    setUploading(true);
    setErrorMsg('');
    setUploadReport(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      if (customName.trim()) formData.append('name', customName.trim());
      if (customDesc.trim()) formData.append('description', customDesc.trim());
      formData.append('imputation_method', imputationMethod);
      formData.append('activate_immediately', 'true');

      const res = await uploadDatasetCsv(formData);
      setUploadReport(res.preprocessing_report);
      setSelectedFile(null);
      setCustomName('');
      setCustomDesc('');
      await loadList();
      if (onDatasetChanged) onDatasetChanged();
    } catch (err) {
      setErrorMsg(
        err.response?.data?.detail ||
          'CSV validation failed. Ensure your CSV contains timestamp and bandwidth_mbps columns with at least 35 rows.'
      );
    } finally {
      setUploading(false);
    }
  };

  const currentDs = datasets.find((d) => d.id === activeId);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90">
        <h1 className="text-xl font-bold text-white flex items-center gap-2">
          <Database className="w-5 h-5 text-cyan-400" />
          <span>Telemetry Dataset Manager & CSV Ingestion Pipeline</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Switch between reproducible synthetic enterprise/ISP datasets or upload custom network telemetry CSVs with
          automatic timestamp continuity checks, deduplication, and missing-value imputation
        </p>
      </div>

      {errorMsg && (
        <div className="p-4 rounded-xl bg-rose-500/15 border border-rose-500/40 text-xs text-rose-200 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Grid: Left Registered Datasets + Right CSV Upload */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Registered Datasets List */}
        <div className="lg:col-span-7 bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90 space-y-4">
          <h2 className="text-sm font-bold text-white">
            Registered Network Telemetry Datasets ({datasets.length})
          </h2>

          <div className="space-y-3">
            {datasets.map((ds) => {
              const isActive = ds.id === activeId;
              return (
                <div
                  key={ds.id}
                  className={`p-4 rounded-xl border transition ${
                    isActive
                      ? 'bg-cyan-500/10 border-cyan-500/40'
                      : 'bg-[#111C35] border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-bold text-white">{ds.name}</span>
                        <span
                          className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                            ds.is_synthetic
                              ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
                              : 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                          }`}
                        >
                          {ds.is_synthetic ? 'SYNTHETIC' : 'UPLOADED CSV'}
                        </span>
                        {isActive && (
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500 text-slate-950 font-bold">
                            ACTIVE
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-slate-400 mt-1">{ds.description}</p>
                    </div>

                    <div className="flex items-center gap-2 shrink-0">
                      {!isActive && (
                        <button
                          onClick={() => handleActivate(ds.id)}
                          disabled={loading}
                          className="px-3 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs transition"
                        >
                          Activate
                        </button>
                      )}
                      {!ds.is_synthetic && (
                        <button
                          onClick={() => handleDelete(ds.id)}
                          title="Delete uploaded dataset"
                          className="p-1.5 rounded-lg bg-rose-500/15 hover:bg-rose-500/30 text-rose-300 border border-rose-500/30 transition"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </div>

                  <div className="mt-3 pt-2.5 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono text-slate-400">
                    <div>
                      Rows: <strong className="text-slate-200">{ds.row_count}</strong> ({ds.frequency})
                    </div>
                    <div>
                      Default Link: <strong className="text-cyan-300">{ds.default_capacity_mbps} Mbps</strong>
                    </div>
                    <div>
                      Imputed NaNs: <strong className="text-amber-300">{ds.missing_values_filled}</strong>
                    </div>
                    <div>
                      File: <strong className="text-slate-300">{ds.filename}</strong>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* CSV Upload Form */}
        <form
          onSubmit={handleUploadSubmit}
          className="lg:col-span-5 bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90 space-y-4"
        >
          <h2 className="text-sm font-bold text-white flex items-center gap-2">
            <UploadCloud className="w-4 h-4 text-cyan-400" />
            <span>Upload Custom Network Traffic CSV</span>
          </h2>

          <div className="p-3.5 rounded-xl bg-[#111C35] border border-dashed border-slate-700 text-xs space-y-2">
            <p className="font-semibold text-slate-200">Supported CSV Schema:</p>
            <p className="text-slate-400">
              Required columns: <code className="text-cyan-300">timestamp</code> (or <code className="text-cyan-300">date</code>/
              <code className="text-cyan-300">datetime</code>) and{' '}
              <code className="text-cyan-300">bandwidth_mbps</code> (or{' '}
              <code className="text-cyan-300">traffic_mbps</code>).
            </p>
            <p className="text-slate-400">
              Optional columns: <code className="text-slate-300">active_users</code>,{' '}
              <code className="text-slate-300">latency_ms</code>,{' '}
              <code className="text-slate-300">packet_loss_pct</code>,{' '}
              <code className="text-slate-300">allocated_capacity_mbps</code>.
            </p>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">Select CSV File (.csv)</label>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
              className="w-full text-xs text-slate-300 file:mr-3 file:py-2 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-cyan-500/20 file:text-cyan-300 hover:file:bg-cyan-500/30"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">Dataset Display Name (Optional)</label>
            <input
              type="text"
              placeholder="e.g., Mumbai Core Router Q3 Telemetry"
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
              className="w-full px-3 py-2 rounded-xl bg-[#111C35] border border-slate-700 text-xs text-white"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">Description (Optional)</label>
            <input
              type="text"
              placeholder="e.g., Hourly SNMP interface counters from router Gi0/0/1"
              value={customDesc}
              onChange={(e) => setCustomDesc(e.target.value)}
              className="w-full px-3 py-2 rounded-xl bg-[#111C35] border border-slate-700 text-xs text-white"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">
              Missing-Value Imputation Method
            </label>
            <select
              value={imputationMethod}
              onChange={(e) => setImputationMethod(e.target.value)}
              className="w-full px-3 py-2 rounded-xl bg-[#111C35] border border-slate-700 text-xs text-white"
            >
              <option value="linear">Linear Interpolation (Recommended for continuous traffic)</option>
              <option value="ffill">Forward Fill + Backward Fill</option>
              <option value="median">7-Step Rolling Median</option>
            </select>
          </div>

          <button
            type="submit"
            disabled={uploading}
            className="w-full py-2.5 px-4 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs transition"
          >
            {uploading ? 'Validating, Imputing & Training RF Model...' : 'Upload, Validate & Train Model'}
          </button>

          {uploadReport && (
            <div className="p-3.5 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-xs space-y-1">
              <div className="font-bold text-emerald-300 flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4" />
                <span>CSV Preprocessed & Activated Successfully!</span>
              </div>
              <p className="text-slate-300 font-mono text-[11px]">
                Clean Rows: {uploadReport.clean_rows} ({uploadReport.frequency}) • Imputed NaNs:{' '}
                {uploadReport.total_missing_imputed} • Duplicates Removed: {uploadReport.duplicates_removed}
              </p>
            </div>
          )}
        </form>
      </div>

      {/* Active Dataset Preprocessing & Statistical Audit Report */}
      {currentDs && currentDs.summary && (
        <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90">
          <h3 className="text-sm font-bold text-white mb-3">
            Active Dataset Preprocessing & Quality Audit: {currentDs.name}
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3 text-xs font-mono">
            <div className="p-3 rounded-xl bg-[#111C35] border border-slate-800">
              <span className="text-slate-400 block text-[10px]">CLEAN OBSERVATIONS</span>
              <span className="text-white font-bold text-sm">{currentDs.summary.clean_rows}</span>
            </div>
            <div className="p-3 rounded-xl bg-[#111C35] border border-slate-800">
              <span className="text-slate-400 block text-[10px]">MISSING VALUES IMPUTED</span>
              <span className="text-amber-300 font-bold text-sm">{currentDs.summary.total_missing_imputed}</span>
            </div>
            <div className="p-3 rounded-xl bg-[#111C35] border border-slate-800">
              <span className="text-slate-400 block text-[10px]">MEAN BANDWIDTH</span>
              <span className="text-cyan-300 font-bold text-sm">{currentDs.summary.stats?.mean_mbps} Mbps</span>
            </div>
            <div className="p-3 rounded-xl bg-[#111C35] border border-slate-800">
              <span className="text-slate-400 block text-[10px]">P95 BANDWIDTH</span>
              <span className="text-blue-300 font-bold text-sm">{currentDs.summary.stats?.p95_mbps} Mbps</span>
            </div>
            <div className="p-3 rounded-xl bg-[#111C35] border border-slate-800">
              <span className="text-slate-400 block text-[10px]">MAX BANDWIDTH</span>
              <span className="text-emerald-300 font-bold text-sm">{currentDs.summary.stats?.max_mbps} Mbps</span>
            </div>
            <div className="p-3 rounded-xl bg-[#111C35] border border-slate-800">
              <span className="text-slate-400 block text-[10px]">AVG LATENCY</span>
              <span className="text-white font-bold text-sm">{currentDs.summary.stats?.avg_latency_ms} ms</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
