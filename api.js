/**
 * Backend REST API Client for NetCapacity AI
 * Connects React components to FastAPI endpoints.
 */
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export async function fetchHealth() {
  const { data } = await apiClient.get('/health');
  return data;
}

export async function fetchDashboard(horizon = null) {
  const params = horizon ? { horizon } : {};
  const { data } = await apiClient.get('/dashboard', { params });
  return data;
}

export async function fetchForecast(horizon = '24h', userGrowthPct = null) {
  const params = { horizon };
  if (userGrowthPct !== null && userGrowthPct !== undefined) {
    params.user_growth_pct = userGrowthPct;
  }
  const { data } = await apiClient.get('/forecast', { params });
  return data;
}

export async function runCapacityPlan(payload) {
  const { data } = await apiClient.post('/capacity/plan', payload);
  return data;
}

export async function fetchModelEvaluation() {
  const { data } = await apiClient.get('/model/evaluation');
  return data;
}

export async function retrainModel(payload) {
  const { data } = await apiClient.post('/model/train', payload);
  return data;
}

export async function fetchDatasets() {
  const { data } = await apiClient.get('/datasets');
  return data;
}

export async function uploadDatasetCsv(formData) {
  const { data } = await apiClient.post('/datasets/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function activateDataset(datasetId) {
  const { data } = await apiClient.post(`/datasets/${datasetId}/activate`);
  return data;
}

export async function deleteDataset(datasetId) {
  const { data } = await apiClient.delete(`/datasets/${datasetId}`);
  return data;
}

export async function fetchAlerts() {
  const { data } = await apiClient.get('/alerts');
  return data;
}

export async function acknowledgeAlertById(alertId) {
  const { data } = await apiClient.post(`/alerts/${alertId}/acknowledge`);
  return data;
}

export async function fetchSettings() {
  const { data } = await apiClient.get('/settings');
  return data;
}

export async function updateSettings(payload) {
  const { data } = await apiClient.put('/settings', payload);
  return data;
}

export function getForecastCsvUrl(horizon = '24h') {
  return `${API_BASE}/reports/forecast-csv?horizon=${encodeURIComponent(horizon)}`;
}

export function getCapacityCsvUrl(horizon = '24h') {
  return `${API_BASE}/reports/capacity-csv?horizon=${encodeURIComponent(horizon)}`;
}
