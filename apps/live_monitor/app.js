import {
  calculateKpis,
  inspectionWebSocketUrl,
  mergeInspections,
  parseInspectionEvent,
  parseInspectionHistory,
  reconnectDelayMs,
} from "./state.js";

const elements = {
  connection: document.querySelector("#connection-state"),
  lastSync: document.querySelector("#last-sync"),
  visible: document.querySelector("#kpi-visible"),
  normal: document.querySelector("#kpi-normal"),
  anomaly: document.querySelector("#kpi-anomaly"),
  ratio: document.querySelector("#kpi-ratio"),
  latest: document.querySelector("#latest-inspection"),
  feed: document.querySelector("#inspection-feed"),
  empty: document.querySelector("#empty-state"),
  detail: document.querySelector("#inspection-detail"),
  detailBody: document.querySelector("#detail-body"),
  detailTitle: document.querySelector("#detail-title"),
  detailClose: document.querySelector("#detail-close"),
};

let inspections = [];
let socket = null;
let reconnectTimer = null;
let reconnectAttempt = 0;
let generation = 0;
let bufferedInspections = [];
let isSynchronizing = false;
let stopped = false;
let syncAbortController = null;

function setConnectionState(value) {
  elements.connection.textContent = value;
  elements.connection.dataset.state = value.toLowerCase();
}

function formatScore(value) {
  return Number(value).toFixed(3);
}

function formatTimestamp(value) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(new Date(value));
}

function resultLabel(inspection) {
  return inspection.is_anomaly ? "ANOMALY" : "NORMAL";
}

function appendValue(container, label, value, className = "") {
  const wrapper = document.createElement("div");
  wrapper.className = `value-pair ${className}`.trim();
  const term = document.createElement("span");
  term.className = "value-label";
  term.textContent = label;
  const content = document.createElement("strong");
  content.textContent = value;
  wrapper.append(term, content);
  container.append(wrapper);
}

function detailButton(inspectionId, label = "View lineage") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "detail-button";
  button.textContent = label;
  button.addEventListener("click", () => showInspectionDetail(inspectionId));
  return button;
}

function renderLatest() {
  elements.latest.replaceChildren();
  const latest = inspections[0];
  if (latest === undefined) {
    const message = document.createElement("p");
    message.className = "muted";
    message.textContent = "Waiting for the first persisted inspection.";
    elements.latest.append(message);
    return;
  }

  const result = document.createElement("span");
  result.className = `result-badge ${latest.is_anomaly ? "anomaly" : "normal"}`;
  result.textContent = resultLabel(latest);
  elements.latest.append(result);
  appendValue(elements.latest, "Score", formatScore(latest.anomaly_score), "score-primary");
  appendValue(elements.latest, "Threshold", formatScore(latest.threshold));
  appendValue(
    elements.latest,
    "Decision",
    `${formatScore(latest.anomaly_score)} ${latest.is_anomaly ? ">" : "≤"} ${formatScore(latest.threshold)}`,
  );
  appendValue(elements.latest, "Model", latest.model_name);
  appendValue(elements.latest, "Category", latest.category);
  appendValue(elements.latest, "Device", latest.device.toUpperCase());
  appendValue(elements.latest, "Observed", formatTimestamp(latest.created_at));
  appendValue(elements.latest, "Inspection ID", latest.inspection_id);
  elements.latest.append(detailButton(latest.inspection_id));
}

function renderFeed() {
  elements.feed.replaceChildren();
  elements.empty.hidden = inspections.length !== 0;
  for (const inspection of inspections) {
    const item = document.createElement("li");
    item.className = `feed-item ${inspection.is_anomaly ? "anomaly" : "normal"}`;
    item.dataset.inspectionId = inspection.inspection_id;

    const summary = document.createElement("div");
    summary.className = "feed-summary";
    const badge = document.createElement("span");
    badge.className = "feed-result";
    badge.textContent = resultLabel(inspection);
    const time = document.createElement("time");
    time.dateTime = inspection.created_at;
    time.textContent = formatTimestamp(inspection.created_at);
    summary.append(badge, time);

    const measurement = document.createElement("div");
    measurement.className = "feed-measurement";
    measurement.textContent = `${formatScore(inspection.anomaly_score)} > ${formatScore(inspection.threshold)}`;
    if (!inspection.is_anomaly) {
      measurement.textContent = `${formatScore(inspection.anomaly_score)} ≤ ${formatScore(inspection.threshold)}`;
    }

    const identity = document.createElement("div");
    identity.className = "feed-identity";
    identity.textContent = `${inspection.category} · ${inspection.device.toUpperCase()}`;
    item.append(summary, measurement, identity, detailButton(inspection.inspection_id, "Details"));
    elements.feed.append(item);
  }
}

function render() {
  const kpis = calculateKpis(inspections);
  elements.visible.textContent = String(kpis.visible);
  elements.normal.textContent = String(kpis.normal);
  elements.anomaly.textContent = String(kpis.anomaly);
  elements.ratio.textContent = `${(kpis.anomalyRatio * 100).toFixed(1)}%`;
  renderLatest();
  renderFeed();
}

async function showInspectionDetail(inspectionId) {
  elements.detailTitle.textContent = "Inspection lineage";
  elements.detailBody.replaceChildren();
  appendValue(elements.detailBody, "Status", "Loading…");
  elements.detail.showModal();
  try {
    const response = await fetch(`/v1/inspections/${encodeURIComponent(inspectionId)}`, {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error(`Detail request failed (${response.status}).`);
    }
    const detail = await response.json();
    elements.detailBody.replaceChildren();
    appendValue(elements.detailBody, "Inspection ID", detail.inspection_id);
    appendValue(elements.detailBody, "Device", detail.device);
    appendValue(elements.detailBody, "Image SHA-256", detail.image_sha256);
    appendValue(elements.detailBody, "Model SHA-256", detail.model_sha256);
    appendValue(elements.detailBody, "Metadata SHA-256", detail.artifact_metadata_sha256);
    appendValue(elements.detailBody, "Threshold SHA-256", detail.threshold_artifact_sha256);
    appendValue(elements.detailBody, "Manifest SHA-256", detail.manifest_sha256);
  } catch (error) {
    elements.detailBody.replaceChildren();
    appendValue(elements.detailBody, "Status", error.message);
  }
}

function acceptLiveMessage(rawValue) {
  let event;
  try {
    event = parseInspectionEvent(JSON.parse(rawValue));
  } catch {
    return;
  }
  if (event === null) {
    return;
  }
  if (isSynchronizing) {
    bufferedInspections.push(event);
    return;
  }
  inspections = mergeInspections(inspections, [event]);
  render();
}

function scheduleReconnect() {
  if (stopped || reconnectTimer !== null) {
    return;
  }
  if (!navigator.onLine) {
    setConnectionState("OFFLINE");
    return;
  }
  setConnectionState("RECONNECTING");
  const delay = reconnectDelayMs(reconnectAttempt);
  reconnectAttempt += 1;
  reconnectTimer = window.setTimeout(() => {
    reconnectTimer = null;
    connectAndSynchronize();
  }, delay);
}

// WebSocket을 먼저 연결해 event를 buffering한 뒤 REST snapshot과 atomic merge한다.
function connectAndSynchronize() {
  if (stopped) {
    return;
  }
  generation += 1;
  const activeGeneration = generation;
  isSynchronizing = true;
  bufferedInspections = [];
  syncAbortController?.abort();
  syncAbortController = null;
  setConnectionState(reconnectAttempt === 0 ? "CONNECTING" : "RECONNECTING");

  const connection = new WebSocket(inspectionWebSocketUrl(window.location));
  socket = connection;
  connection.addEventListener("message", (message) => acceptLiveMessage(message.data));
  connection.addEventListener("open", async () => {
    if (activeGeneration !== generation || stopped) {
      connection.close();
      return;
    }
    syncAbortController = new AbortController();
    try {
      const response = await fetch("/v1/inspections?limit=100&offset=0", {
        headers: { Accept: "application/json" },
        signal: syncAbortController.signal,
      });
      if (!response.ok) {
        throw new Error(`History request failed (${response.status}).`);
      }
      const history = parseInspectionHistory(await response.json());
      if (activeGeneration !== generation || stopped) {
        return;
      }
      inspections = mergeInspections(history, bufferedInspections);
      bufferedInspections = [];
      isSynchronizing = false;
      reconnectAttempt = 0;
      elements.lastSync.textContent = `Synced ${new Date().toLocaleTimeString()}`;
      render();
      setConnectionState("LIVE");
    } catch (error) {
      if (error.name !== "AbortError" && activeGeneration === generation) {
        connection.close();
      }
    }
  });
  connection.addEventListener("error", () => connection.close());
  connection.addEventListener("close", () => {
    if (activeGeneration !== generation) {
      return;
    }
    syncAbortController?.abort();
    isSynchronizing = false;
    scheduleReconnect();
  });
}

function stopConnections() {
  stopped = true;
  generation += 1;
  syncAbortController?.abort();
  if (reconnectTimer !== null) {
    window.clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  socket?.close();
}

elements.detailClose.addEventListener("click", () => elements.detail.close());
window.addEventListener("offline", () => {
  if (reconnectTimer !== null) {
    window.clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  setConnectionState("OFFLINE");
  socket?.close();
});
window.addEventListener("online", () => {
  if (!stopped) {
    reconnectAttempt = 0;
    connectAndSynchronize();
  }
});
window.addEventListener("beforeunload", stopConnections);

Object.defineProperty(window, "__liveMonitorDebug", {
  value: Object.freeze({
    snapshot: () => ({
      connectionState: elements.connection.textContent,
      inspections: structuredClone(inspections),
      reconnectAttempt,
    }),
  }),
  writable: false,
});

render();
connectAndSynchronize();
