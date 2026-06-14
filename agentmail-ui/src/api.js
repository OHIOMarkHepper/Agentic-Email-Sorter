// api.js
// All HTTP calls to the FastAPI server live here so components stay clean
// and the API surface is easy to change in one place.

const BASE = "http://localhost:8000";

async function handleResponse(res) {
  // 204 No Content has no body to parse
  if (res.status === 204) return null;

  const data = await res.json();

  if (!res.ok) {
    // FastAPI's HTTPException puts the message in `detail`
    throw new Error(data.detail || `Request failed with status ${res.status}`);
  }

  return data;
}

// ------------------------------------------------------------------
// Training
// ------------------------------------------------------------------

export function trainModel({ filepath, strategy, categories = null, useAiLabels = true }) {
  return fetch(`${BASE}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filepath,
      strategy,
      categories,
      use_ai_labels: useAiLabels,
    }),
  }).then(handleResponse);
}

export function retrainModel(k) {
  return fetch(`${BASE}/train/retrain?k=${k}`, {
    method: "POST",
  }).then(handleResponse);
}

// ------------------------------------------------------------------
// Classification
// ------------------------------------------------------------------

export function classifyEmail(text) {
  return fetch(`${BASE}/classify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  }).then(handleResponse);
}

// ------------------------------------------------------------------
// Clusters
// ------------------------------------------------------------------

export function getClusters() {
  return fetch(`${BASE}/clusters`).then(handleResponse);
}

export function relabelCluster(id, newLabel) {
  return fetch(`${BASE}/clusters/${id}/relabel`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ new_label: newLabel }),
  }).then(handleResponse);
}

// ------------------------------------------------------------------
// Analysis
// ------------------------------------------------------------------

export function getAnalysis() {
  return fetch(`${BASE}/analysis`).then(handleResponse);
}

// ------------------------------------------------------------------
// Database / saved emails
// ------------------------------------------------------------------

export function saveEmails(dbPath) {
  return fetch(`${BASE}/emails/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ db_path: dbPath }),
  }).then(handleResponse);
}

export function getEmailsByCluster(cluster, dbPath) {
  const params = new URLSearchParams({ cluster });
  if (dbPath) params.set("db_path", dbPath);
  return fetch(`${BASE}/emails?${params}`).then(handleResponse);
}

export function getDbClusters(dbPath) {
  const params = dbPath ? `?db_path=${encodeURIComponent(dbPath)}` : "";
  return fetch(`${BASE}/emails/clusters${params}`).then(handleResponse);
}