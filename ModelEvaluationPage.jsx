import React, { useEffect, useState } from 'react';
import {
  Award,
  BarChart2,
  CheckCircle,
  Cpu,
  GitBranch,
  RefreshCw,
  ShieldCheck,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { fetchModelEvaluation, retrainModel } from './api';

export default function ModelEvaluationPage({ activeDataset, onModelRetrained }) {
  const [evalData, setEvalData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [retraining, setRetraining] = useState(false);

  // Hyperparameter controls
  const [nEstimators, setNEstimators] = useState(50);
  const [maxDepth, setMaxDepth] = useState(10);
  const [testSplitRatio, setTestSplitRatio] = useState(0.2);
  const [imputationMethod, setImputationMethod] = useState('linear');

  const loadEvaluation = async () => {
    setLoading(true);
    try {
      const res = await fetchModelEvaluation();
      setEvalData(res);
      if (res.hyperparameters) {
        setNEstimators(res.hyperparameters.n_estimators);
        setMaxDepth(res.hyperparameters.max_depth);
        setTestSplitRatio(res.hyperparameters.test_split_ratio);
      }
    } catch (err) {
      console.error('Error loading model evaluation:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadEvaluation();
  }, [activeDataset?.id]);

  const handleRetrain = async (e) => {
    e.preventDefault();
    setRetraining(true);
    try {
      const updated = await retrainModel({
        n_estimators: Number(nEstimators),
        max_depth: Number(maxDepth),
        test_split_ratio: Number(testSplitRatio),
        imputation_method: imputationMethod,
      });
      setEvalData(updated);
      if (onModelRetrained) onModelRetrained();
    } catch (err) {
      console.error('Error retraining model:', err);
    } finally {
      setRetraining(false);
    }
  };

  if (!evalData || loading) {
    return (
      <div className="p-8 bg-[#0B1325] rounded-2xl border border-slate-800 text-sm text-slate-400">
        Loading chronological train/test evaluation and feature importances...
      </div>
    );
  }

  const { metrics, feature_importances, test_comparison_series } = evalData;
  const rfTest = metrics.random_forest_test;
  const rfTrain = metrics.random_forest_train;
  const pers = metrics.persistence_baseline;
  const seas = metrics.seasonal_naive_baseline;

  const modelComparisonRows = [
    {
      model: 'Random Forest Regressor (Lag + Rolling + Cyclical)',
      badge: 'PRIMARY ML',
      mae: rfTest.mae,
      rmse: rfTest.rmse,
      r2: rfTest.r2,
      mape: rfTest.mape,
      highlight: true,
    },
    {
      model: 'Persistence Baseline (Naive Walk: y_hat_t = y_{t-1})',
      badge: 'BASELINE 1',
      mae: pers.mae,
      rmse: pers.rmse,
      r2: pers.r2,
      mape: pers.mape,
      highlight: false,
    },
    {
      model: `Seasonal-Naive Baseline (y_hat_t = y_{t-${metrics.seasonal_lag_used}})`,
      badge: 'BASELINE 2',
      mae: seas.mae,
      rmse: seas.rmse,
      r2: seas.r2,
      mape: seas.mape,
      highlight: false,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header & Leak-Free Guarantee Badge */}
      <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-mono px-2.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>CHRONOLOGICAL SPLIT • ZERO DATA LEAKAGE VERIFIED</span>
            </span>
            <span className="text-xs font-mono px-2.5 py-0.5 rounded-full bg-slate-800 text-cyan-300">
              {evalData.engine_used}
            </span>
          </div>
          <h1 className="text-xl font-bold text-white">
            Machine Learning Evaluation & Baseline Comparison
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Train Window: <strong className="text-slate-200 font-mono">{evalData.train_window}</strong> (
            {evalData.train_rows} rows) → Test Window:{' '}
            <strong className="text-cyan-300 font-mono">{evalData.test_window}</strong> ({evalData.test_rows} rows)
          </p>
        </div>

        <div className="flex items-center gap-3 text-xs font-mono">
          <div className="px-3.5 py-2 rounded-xl bg-[#111C35] border border-slate-800">
            <span className="text-slate-400 block text-[10px]">SKILL VS PERSISTENCE</span>
            <span className="text-emerald-400 font-bold text-sm">
              +{metrics.rmse_improvement_over_persistence_pct}% RMSE Reduction
            </span>
          </div>
          <div className="px-3.5 py-2 rounded-xl bg-[#111C35] border border-slate-800">
            <span className="text-slate-400 block text-[10px]">SKILL VS SEASONAL-NAIVE</span>
            <span className="text-cyan-400 font-bold text-sm">
              +{metrics.rmse_improvement_over_seasonal_pct}% RMSE Reduction
            </span>
          </div>
        </div>
      </div>

      {/* Model Comparison Table */}
      <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90">
        <h2 className="text-sm font-bold text-white mb-1">
          Held-Out Chronological Test Set Performance (Random Forest vs. Baselines)
        </h2>
        <p className="text-xs text-slate-400 mb-4">
          All metrics are computed strictly on the unseen chronological test partition ({evalData.test_rows} samples)
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead className="bg-[#111C35] text-slate-300 font-mono uppercase text-[11px] border-b border-slate-700">
              <tr>
                <th className="py-3 px-4">Forecasting Model</th>
                <th className="py-3 px-4">Role</th>
                <th className="py-3 px-4 text-right">MAE (Mbps) ↓</th>
                <th className="py-3 px-4 text-right">RMSE (Mbps) ↓</th>
                <th className="py-3 px-4 text-right">R² Score ↑</th>
                <th className="py-3 px-4 text-right">MAPE (%) ↓</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 font-mono">
              {modelComparisonRows.map((row, idx) => (
                <tr
                  key={idx}
                  className={row.highlight ? 'bg-cyan-500/10 text-white font-semibold' : 'text-slate-300'}
                >
                  <td className="py-3 px-4 font-sans">{row.model}</td>
                  <td className="py-3 px-4">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] ${
                        row.highlight
                          ? 'bg-cyan-500 text-slate-950 font-bold'
                          : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      {row.badge}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-right">{row.mae} Mbps</td>
                  <td className="py-3 px-4 text-right text-cyan-300">{row.rmse} Mbps</td>
                  <td className="py-3 px-4 text-right text-emerald-400">{row.r2}</td>
                  <td className="py-3 px-4 text-right">{row.mape}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Actual vs Predicted on Held-Out Test Split */}
      <div className="bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90">
        <div className="mb-4">
          <h3 className="text-sm font-bold text-white">
            Chronological Test Split: Observed Actual vs. Random Forest Predicted vs. Seasonal Baseline
          </h3>
          <p className="text-xs text-slate-400">
            Visualizing out-of-sample generalization on the held-out test set (Train MAE: {rfTrain.mae} Mbps, Test MAE:{' '}
            {rfTest.mae} Mbps)
          </p>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={test_comparison_series} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis
                dataKey="timestamp"
                stroke="#64748B"
                tick={{ fontSize: 11, fill: '#94A3B8' }}
                minTickGap={35}
              />
              <YAxis stroke="#64748B" tick={{ fontSize: 11, fill: '#94A3B8' }} unit=" M" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0F172A',
                  borderColor: '#334155',
                  borderRadius: '12px',
                  fontSize: '12px',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '12px' }} />
              <Line
                type="monotone"
                dataKey="actual_mbps"
                name="Actual Test Traffic (Mbps)"
                stroke="#3B82F6"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="rf_predicted_mbps"
                name="Random Forest Predicted (Mbps)"
                stroke="#10B981"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="seasonal_naive_mbps"
                name="Seasonal-Naive Baseline (Mbps)"
                stroke="#94A3B8"
                strokeWidth={1.3}
                strokeDasharray="4 4"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Bottom Row: Feature Importances + Interactive Model Retraining */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Feature Importance BarChart */}
        <div className="lg:col-span-7 bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90">
          <h3 className="text-sm font-bold text-white mb-1">
            Random Forest Feature Importance Ranking (Impurity Variance Reduction %)
          </h3>
          <p className="text-xs text-slate-400 mb-4">
            Autoregressive lags (lag_1, lag_24, lag_168), causal rolling windows, and diurnal sine/cosine harmonics
          </p>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={feature_importances.slice(0, 10)}
                layout="vertical"
                margin={{ top: 5, right: 20, left: 45, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                <XAxis type="number" stroke="#64748B" tick={{ fontSize: 11, fill: '#94A3B8' }} unit="%" />
                <YAxis
                  type="category"
                  dataKey="feature"
                  stroke="#94A3B8"
                  tick={{ fontSize: 11, fill: '#E2E8F0', fontFamily: 'monospace' }}
                  width={120}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0F172A',
                    borderColor: '#334155',
                    borderRadius: '10px',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="importance_pct" name="Importance (%)" fill="#06B6D4" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Interactive Hyperparameter Tuning & Retraining Panel */}
        <form
          onSubmit={handleRetrain}
          className="lg:col-span-5 bg-[#0B1325] p-5 rounded-2xl border border-slate-800/90 flex flex-col justify-between space-y-4"
        >
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Cpu className="w-4 h-4 text-cyan-400" />
                <span>Interactive Random Forest Hyperparameter Lab</span>
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Adjust ensemble size, tree depth, or chronological test split ratio and retrain live on the backend
              </p>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <label className="text-slate-300">Number of Trees (n_estimators)</label>
                <span className="font-mono font-bold text-cyan-300">{nEstimators} trees</span>
              </div>
              <input
                type="range"
                min="15"
                max="120"
                step="5"
                value={nEstimators}
                onChange={(e) => setNEstimators(Number(e.target.value))}
                className="w-full accent-cyan-400"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <label className="text-slate-300">Max Tree Depth (max_depth)</label>
                <span className="font-mono font-bold text-cyan-300">{maxDepth} levels</span>
              </div>
              <input
                type="range"
                min="4"
                max="18"
                step="1"
                value={maxDepth}
                onChange={(e) => setMaxDepth(Number(e.target.value))}
                className="w-full accent-cyan-400"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <label className="text-slate-300">Chronological Test Split Ratio</label>
                <span className="font-mono font-bold text-amber-300">
                  {Math.round((1 - testSplitRatio) * 100)}% Train / {Math.round(testSplitRatio * 100)}% Test
                </span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.30"
                step="0.05"
                value={testSplitRatio}
                onChange={(e) => setTestSplitRatio(Number(e.target.value))}
                className="w-full accent-amber-400"
              />
            </div>

            <div>
              <label className="block text-xs text-slate-300 mb-1">Missing-Value Imputation Strategy</label>
              <select
                value={imputationMethod}
                onChange={(e) => setImputationMethod(e.target.value)}
                className="w-full px-3 py-2 rounded-xl bg-[#111C35] border border-slate-700 text-xs text-white"
              >
                <option value="linear">Linear Interpolation (Default)</option>
                <option value="ffill">Forward Fill + Backward Boundary Fill</option>
                <option value="median">7-Step Rolling Median Imputation</option>
              </select>
            </div>
          </div>

          <button
            type="submit"
            disabled={retraining}
            className="w-full py-2.5 px-4 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs flex items-center justify-center gap-2 transition"
          >
            <RefreshCw className={`w-4 h-4 ${retraining ? 'animate-spin' : ''}`} />
            <span>{retraining ? 'Training Random Forest Ensemble...' : 'Retrain & Evaluate Model'}</span>
          </button>
        </form>
      </div>
    </div>
  );
}
