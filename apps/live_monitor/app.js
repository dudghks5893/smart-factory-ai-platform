import {
  calculateCombinedKpis,
  calculateKpis,
  calculateKnownDefectKpis,
  combinedInspectionDetailUrl,
  combinedInspectionWebSocketUrl,
  decisionReasonLabel,
  inspectionWebSocketUrl,
  knownDefectDetailUrl,
  knownDefectWebSocketUrl,
  mergeCombinedInspections,
  mergeInspections,
  mergeKnownDefectInspections,
  parseCombinedInspectionEvent,
  parseCombinedInspectionHistory,
  parseInspectionEvent,
  parseInspectionHistory,
  parseKnownDefectEvent,
  parseKnownDefectHistory,
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
  knownConnection: document.querySelector("#known-connection-state"),
  knownLastSync: document.querySelector("#known-last-sync"),
  knownVisible: document.querySelector("#known-kpi-visible"),
  knownEmptyCount: document.querySelector("#known-kpi-empty"),
  knownDefectCount: document.querySelector("#known-kpi-defect"),
  knownInstanceCount: document.querySelector("#known-kpi-instances"),
  knownLatest: document.querySelector("#known-latest-inspection"),
  knownFeed: document.querySelector("#known-defect-feed"),
  knownEmpty: document.querySelector("#known-empty-state"),
  knownDetail: document.querySelector("#known-defect-detail"),
  knownDetailBody: document.querySelector("#known-detail-body"),
  knownDetailTitle: document.querySelector("#known-detail-title"),
  knownDetailClose: document.querySelector("#known-detail-close"),
  combinedConnection: document.querySelector("#combined-connection-state"),
  combinedLastSync: document.querySelector("#combined-last-sync"),
  combinedVisible: document.querySelector("#combined-kpi-visible"),
  combinedPass: document.querySelector("#combined-kpi-pass"),
  combinedReview: document.querySelector("#combined-kpi-review"),
  combinedReject: document.querySelector("#combined-kpi-reject"),
  combinedLatest: document.querySelector("#combined-latest-inspection"),
  combinedFeed: document.querySelector("#combined-inspection-feed"),
  combinedEmpty: document.querySelector("#combined-empty-state"),
  combinedDetail: document.querySelector("#combined-inspection-detail"),
  combinedDetailBody: document.querySelector("#combined-detail-body"),
  combinedDetailTitle: document.querySelector("#combined-detail-title"),
  combinedDetailClose: document.querySelector("#combined-detail-close"),
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
let knownDefectInspections = [];
let knownSocket = null;
let knownReconnectTimer = null;
let knownReconnectAttempt = 0;
let knownGeneration = 0;
let bufferedKnownDefects = [];
let isKnownSynchronizing = false;
let knownSyncAbortController = null;
let combinedInspections = [];
let combinedSocket = null;
let combinedReconnectTimer = null;
let combinedReconnectAttempt = 0;
let combinedGeneration = 0;
let bufferedCombinedInspections = [];
let isCombinedSynchronizing = false;
let combinedSyncAbortController = null;

function setConnectionState(value) {
  elements.connection.textContent = value;
  elements.connection.dataset.state = value.toLowerCase();
}

// ADD 2026-08-26: YOLO channel status를 PatchCore indicator와 독립적으로 갱신한다.
function setKnownConnectionState(value) {
  elements.knownConnection.textContent = value;
  elements.knownConnection.dataset.state = value.toLowerCase();
}

// ADD 2026-08-26: Manufacturing decision channel status를 child model indicators와 분리한다.
function setCombinedConnectionState(value) {
  elements.combinedConnection.textContent = value;
  elements.combinedConnection.dataset.state = value.toLowerCase();
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

function appendNote(container, value) {
  const note = document.createElement("p");
  note.className = "diagnostic-note";
  note.textContent = value;
  container.append(note);
}

function detailButton(inspectionId, label = "View lineage") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "detail-button";
  button.textContent = label;
  button.addEventListener("click", () => showInspectionDetail(inspectionId));
  return button;
}

// ADD 2026-08-26: Known-defect detail은 explicit interaction에서만 REST로 요청한다.
function knownDetailButton(inspectionId, label = "View instances and lineage") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "detail-button";
  button.textContent = label;
  button.addEventListener("click", () => showKnownDefectDetail(inspectionId));
  return button;
}

// ADD 2026-08-26: Combined detail fetch를 explicit operator interaction까지 지연한다.
function combinedDetailButton(combinedInspectionId, label = "View decision evidence") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "detail-button";
  button.textContent = label;
  button.addEventListener("click", () => showCombinedInspectionDetail(combinedInspectionId));
  return button;
}

function knownDefectLabel(inspection) {
  return inspection.instance_count === 0 ? "NO KNOWN DEFECT" : "KNOWN DEFECTS";
}

// ADD 2026-08-26: Empty, compact event와 parent-only history의 class 표현을 구분한다.
function knownClassSummary(inspection) {
  if (inspection.instance_count === 0) {
    return "None observed";
  }
  if (inspection.classes.length === 0) {
    return "Available in details";
  }
  return inspection.classes.join(", ");
}

function formatOptionalLatency(value) {
  return value === null ? "Available in details" : `${Number(value).toFixed(3)} ms`;
}

// ADD 2026-08-26: Combined summary의 empty/history/event class evidence를 구분해 표시한다.
function combinedClassSummary(inspection) {
  if (inspection.known_defect_instance_count === 0) {
    return "None observed";
  }
  if (inspection.known_defect_classes.length === 0) {
    return "Available in details";
  }
  return inspection.known_defect_classes.join(", ");
}

// ADD 2026-08-26: Latest persisted manufacturing decision과 child evidence summary를 함께 표시한다.
function renderCombinedLatest() {
  elements.combinedLatest.replaceChildren();
  const latest = combinedInspections[0];
  if (latest === undefined) {
    const message = document.createElement("p");
    message.className = "muted";
    message.textContent = "Waiting for the first persisted combined inspection.";
    elements.combinedLatest.append(message);
    return;
  }

  const disposition = document.createElement("span");
  disposition.className = `result-badge decision-${latest.disposition.toLowerCase()}`;
  disposition.textContent = latest.disposition;
  elements.combinedLatest.append(disposition);
  appendValue(
    elements.combinedLatest,
    "Reason",
    decisionReasonLabel(latest.reason_code),
    "decision-reason-primary",
  );
  appendValue(elements.combinedLatest, "PatchCore evidence", latest.patchcore_prediction);
  appendValue(
    elements.combinedLatest,
    "YOLO evidence",
    `${latest.known_defect_instance_count} instance${
      latest.known_defect_instance_count === 1 ? "" : "s"
    }`,
  );
  appendValue(elements.combinedLatest, "Known-defect classes", combinedClassSummary(latest));
  appendValue(
    elements.combinedLatest,
    "Policy",
    `${latest.policy_name} v${latest.policy_version}`,
  );
  appendValue(elements.combinedLatest, "Observed", formatTimestamp(latest.created_at));
  appendValue(elements.combinedLatest, "Combined ID", latest.combined_inspection_id);
  appendNote(
    elements.combinedLatest,
    "Experimental model-agreement baseline. Not production calibrated.",
  );
  elements.combinedLatest.append(combinedDetailButton(latest.combined_inspection_id));
}

// ADD 2026-08-26: Combined summary event/history를 newest-first clickable feed로 렌더링한다.
function renderCombinedFeed() {
  elements.combinedFeed.replaceChildren();
  elements.combinedEmpty.hidden = combinedInspections.length !== 0;
  for (const inspection of combinedInspections) {
    const item = document.createElement("li");
    item.className = `feed-item combined-feed-item decision-${inspection.disposition.toLowerCase()}`;
    item.dataset.combinedInspectionId = inspection.combined_inspection_id;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "combined-feed-button";
    button.addEventListener("click", () =>
      showCombinedInspectionDetail(inspection.combined_inspection_id),
    );

    const summary = document.createElement("span");
    summary.className = "feed-summary";
    const badge = document.createElement("strong");
    badge.className = "feed-result";
    badge.textContent = inspection.disposition;
    const time = document.createElement("time");
    time.dateTime = inspection.created_at;
    time.textContent = formatTimestamp(inspection.created_at);
    summary.append(badge, time);

    const reason = document.createElement("span");
    reason.className = "feed-measurement";
    reason.textContent = decisionReasonLabel(inspection.reason_code);

    const evidence = document.createElement("span");
    evidence.className = "feed-identity";
    evidence.textContent = `PatchCore ${inspection.patchcore_prediction} · YOLO ${
      inspection.known_defect_instance_count
    } · ${combinedClassSummary(inspection)}`;

    const shortId = document.createElement("span");
    shortId.className = "feed-short-id";
    shortId.textContent = `…${inspection.combined_inspection_id.slice(-8)}`;
    button.append(summary, reason, evidence, shortId);
    item.append(button);
    elements.combinedFeed.append(item);
  }
}

// ADD 2026-08-26: Combined visible window의 backend-provided disposition counts와 cards를 갱신한다.
function renderCombinedInspections() {
  const kpis = calculateCombinedKpis(combinedInspections);
  elements.combinedVisible.textContent = String(kpis.visible);
  elements.combinedPass.textContent = String(kpis.pass);
  elements.combinedReview.textContent = String(kpis.review);
  elements.combinedReject.textContent = String(kpis.reject);
  renderCombinedLatest();
  renderCombinedFeed();
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

// ADD 2026-08-26: Latest YOLO parent summary를 empty prediction도 valid observation으로 표시한다.
function renderKnownLatest() {
  elements.knownLatest.replaceChildren();
  const latest = knownDefectInspections[0];
  if (latest === undefined) {
    const message = document.createElement("p");
    message.className = "muted";
    message.textContent = "Waiting for the first persisted YOLO inspection.";
    elements.knownLatest.append(message);
    return;
  }

  const result = document.createElement("span");
  const hasKnownDefect = latest.instance_count > 0;
  result.className = `result-badge ${hasKnownDefect ? "known" : "normal"}`;
  result.textContent = knownDefectLabel(latest);
  elements.knownLatest.append(result);
  appendValue(
    elements.knownLatest,
    "Instances",
    `${latest.instance_count} instance${latest.instance_count === 1 ? "" : "s"}`,
    "score-primary",
  );
  appendValue(elements.knownLatest, "Classes", knownClassSummary(latest));
  appendValue(
    elements.knownLatest,
    "Diagnostic confidence",
    Number(latest.diagnostic_confidence).toFixed(2),
  );
  appendValue(elements.knownLatest, "Inference latency", formatOptionalLatency(latest.inference_ms));
  appendValue(elements.knownLatest, "Model", latest.model_name);
  appendValue(elements.knownLatest, "Category", latest.category);
  appendValue(elements.knownLatest, "Device", latest.device.toUpperCase());
  appendValue(elements.knownLatest, "Observed", formatTimestamp(latest.created_at));
  appendValue(elements.knownLatest, "Inspection ID", latest.inspection_id);
  appendNote(elements.knownLatest, "Diagnostic operating point; not production-calibrated.");
  elements.knownLatest.append(knownDetailButton(latest.inspection_id));
}

// ADD 2026-08-26: Known-defect summary feed를 newest-first로 렌더링하고 detail fetch는 click까지 지연한다.
function renderKnownFeed() {
  elements.knownFeed.replaceChildren();
  elements.knownEmpty.hidden = knownDefectInspections.length !== 0;
  for (const inspection of knownDefectInspections) {
    const item = document.createElement("li");
    item.className = `feed-item known-feed-item ${
      inspection.instance_count > 0 ? "known" : "normal"
    }`;
    item.dataset.inspectionId = inspection.inspection_id;

    const button = document.createElement("button");
    button.type = "button";
    button.className = "known-feed-button";
    button.addEventListener("click", () => showKnownDefectDetail(inspection.inspection_id));

    const summary = document.createElement("span");
    summary.className = "feed-summary";
    const badge = document.createElement("strong");
    badge.className = "feed-result";
    badge.textContent = knownDefectLabel(inspection);
    const time = document.createElement("time");
    time.dateTime = inspection.created_at;
    time.textContent = formatTimestamp(inspection.created_at);
    summary.append(badge, time);

    const measurement = document.createElement("span");
    measurement.className = "feed-measurement";
    measurement.textContent = `${inspection.instance_count} instance${
      inspection.instance_count === 1 ? "" : "s"
    } · ${knownClassSummary(inspection)}`;

    const identity = document.createElement("span");
    identity.className = "feed-identity";
    identity.textContent = `${inspection.model_name} · ${inspection.device.toUpperCase()}`;

    const shortId = document.createElement("span");
    shortId.className = "feed-short-id";
    shortId.textContent = `…${inspection.inspection_id.slice(-8)}`;
    button.append(summary, measurement, identity, shortId);
    item.append(button);
    elements.knownFeed.append(item);
  }
}

// ADD 2026-08-26: PatchCore와 별도 YOLO window에서 four KPI와 latest/feed를 갱신한다.
function renderKnownDefects() {
  const kpis = calculateKnownDefectKpis(knownDefectInspections);
  elements.knownVisible.textContent = String(kpis.visible);
  elements.knownEmptyCount.textContent = String(kpis.noKnownDefect);
  elements.knownDefectCount.textContent = String(kpis.knownDefect);
  elements.knownInstanceCount.textContent = String(kpis.totalInstances);
  renderKnownLatest();
  renderKnownFeed();
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

// ADD 2026-08-26: Raw mask 없이 한 persisted child의 class, bbox와 mask summary를 렌더링한다.
function appendKnownDefectInstance(container, instance) {
  const card = document.createElement("article");
  card.className = "instance-card";
  const heading = document.createElement("h3");
  heading.textContent = `Instance ${instance.instance_index + 1} · ${instance.class_name}`;
  card.append(heading);
  appendValue(card, "Instance ID", instance.instance_id);
  appendValue(card, "Class ID", String(instance.class_id));
  appendValue(card, "Confidence", Number(instance.confidence).toFixed(6));
  appendValue(
    card,
    "Bounding box",
    `${Number(instance.box.x_min).toFixed(2)}, ${Number(instance.box.y_min).toFixed(2)} → ${Number(
      instance.box.x_max,
    ).toFixed(2)}, ${Number(instance.box.y_max).toFixed(2)}`,
  );
  appendValue(card, "Mask pixels", String(instance.mask.pixel_count));
  appendValue(card, "Mask area ratio", Number(instance.mask.area_ratio).toFixed(6));
  container.append(card);
}

// ADD 2026-08-26: Explicit row interaction에서 parent provenance와 ordered compact instances를 복원한다.
async function showKnownDefectDetail(inspectionId) {
  elements.knownDetailTitle.textContent = "Known-defect lineage";
  elements.knownDetailBody.replaceChildren();
  appendValue(elements.knownDetailBody, "Status", "Loading…");
  elements.knownDetail.showModal();
  try {
    const response = await fetch(knownDefectDetailUrl(inspectionId), {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error(`Detail request failed (${response.status}).`);
    }
    const detail = await response.json();
    elements.knownDetailBody.replaceChildren();
    appendValue(elements.knownDetailBody, "Inspection ID", detail.inspection_id);
    appendValue(elements.knownDetailBody, "Created", formatTimestamp(detail.created_at));
    appendValue(elements.knownDetailBody, "Model", detail.model.name);
    appendValue(elements.knownDetailBody, "Category", detail.model.category);
    appendValue(elements.knownDetailBody, "Device", detail.model.device.toUpperCase());
    appendValue(
      elements.knownDetailBody,
      "Diagnostic confidence",
      Number(detail.diagnostic_confidence).toFixed(2),
    );
    appendValue(
      elements.knownDetailBody,
      "Inference latency",
      `${Number(detail.inference_ms).toFixed(3)} ms`,
    );
    appendValue(
      elements.knownDetailBody,
      "Image dimensions",
      `${detail.image.width} × ${detail.image.height}`,
    );
    appendValue(elements.knownDetailBody, "Image SHA-256", detail.image_sha256);
    appendValue(elements.knownDetailBody, "Model SHA-256", detail.model_sha256);
    appendValue(
      elements.knownDetailBody,
      "Metadata SHA-256",
      detail.artifact_metadata_sha256,
    );
    appendValue(
      elements.knownDetailBody,
      "Dataset Manifest SHA-256",
      detail.dataset_manifest_sha256,
    );
    appendValue(
      elements.knownDetailBody,
      "Dataset fingerprint SHA-256",
      detail.dataset_semantic_fingerprint_sha256,
    );
    appendValue(elements.knownDetailBody, "Instance count", String(detail.instance_count));
    appendNote(elements.knownDetailBody, "Diagnostic operating point; not production-calibrated.");

    const instanceSection = document.createElement("section");
    instanceSection.className = "instance-list";
    const heading = document.createElement("h3");
    heading.textContent = "Instances";
    instanceSection.append(heading);
    if (detail.instances.length === 0) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "NO KNOWN DEFECT · no instance children were persisted.";
      instanceSection.append(empty);
    } else {
      for (const instance of detail.instances) {
        appendKnownDefectInstance(instanceSection, instance);
      }
    }
    elements.knownDetailBody.append(instanceSection);
  } catch (error) {
    elements.knownDetailBody.replaceChildren();
    appendValue(elements.knownDetailBody, "Status", error.message);
  }
}

// ADD 2026-08-26: Combined detail의 decision/evidence/timing block 경계를 표시한다.
function appendDetailHeading(container, value) {
  const heading = document.createElement("h3");
  heading.className = "detail-section-title";
  heading.textContent = value;
  container.append(heading);
}

// ADD 2026-08-26: Persisted combined detail에서 decision, child evidence, lineage와 timings를 복원한다.
async function showCombinedInspectionDetail(combinedInspectionId) {
  elements.combinedDetailTitle.textContent = "Combined manufacturing decision";
  elements.combinedDetailBody.replaceChildren();
  appendValue(elements.combinedDetailBody, "Status", "Loading…");
  elements.combinedDetail.showModal();
  try {
    const response = await fetch(combinedInspectionDetailUrl(combinedInspectionId), {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error(`Detail request failed (${response.status}).`);
    }
    const detail = await response.json();
    const decision = detail.decision;
    const patchcore = detail.patchcore;
    const knownDefects = detail.known_defects;
    elements.combinedDetailBody.replaceChildren();

    appendDetailHeading(elements.combinedDetailBody, "Decision");
    appendValue(
      elements.combinedDetailBody,
      "Disposition",
      decision.disposition,
      `decision-${decision.disposition.toLowerCase()}`,
    );
    appendValue(elements.combinedDetailBody, "Reason", decision.reason);
    appendValue(elements.combinedDetailBody, "Reason code", decision.reason_code);
    appendValue(
      elements.combinedDetailBody,
      "Policy",
      `${decision.policy.name} v${decision.policy.version}`,
    );
    appendNote(
      elements.combinedDetailBody,
      "Experimental model-agreement baseline. Not production calibrated.",
    );

    appendDetailHeading(elements.combinedDetailBody, "Combined inspection");
    appendValue(elements.combinedDetailBody, "Combined ID", detail.combined_inspection_id);
    appendValue(elements.combinedDetailBody, "Created", formatTimestamp(detail.created_at));
    appendValue(
      elements.combinedDetailBody,
      "Image dimensions",
      `${detail.image.width} × ${detail.image.height}`,
    );
    appendValue(elements.combinedDetailBody, "Image SHA-256", detail.image.sha256);

    appendDetailHeading(elements.combinedDetailBody, "PatchCore evidence");
    appendValue(elements.combinedDetailBody, "Child inspection ID", patchcore.inspection_id);
    appendValue(elements.combinedDetailBody, "Model", patchcore.model_name);
    appendValue(elements.combinedDetailBody, "Category", patchcore.category);
    appendValue(elements.combinedDetailBody, "Device", patchcore.device.toUpperCase());
    appendValue(
      elements.combinedDetailBody,
      "Model output",
      decision.evidence.patchcore.prediction,
    );
    appendValue(elements.combinedDetailBody, "Anomaly score", formatScore(patchcore.anomaly_score));
    appendValue(elements.combinedDetailBody, "Threshold", formatScore(patchcore.threshold));
    appendValue(
      elements.combinedDetailBody,
      "Strict comparison",
      `${formatScore(patchcore.anomaly_score)} ${patchcore.is_anomaly ? ">" : "≤"} ${formatScore(
        patchcore.threshold,
      )}`,
    );

    appendDetailHeading(elements.combinedDetailBody, "YOLO evidence");
    appendValue(elements.combinedDetailBody, "Child inspection ID", knownDefects.inspection_id);
    appendValue(elements.combinedDetailBody, "Model", knownDefects.model.name);
    appendValue(elements.combinedDetailBody, "Device", knownDefects.model.device.toUpperCase());
    appendValue(
      elements.combinedDetailBody,
      "Diagnostic confidence",
      Number(knownDefects.diagnostic_confidence).toFixed(2),
    );
    appendValue(elements.combinedDetailBody, "Instance count", String(knownDefects.instance_count));
    appendValue(
      elements.combinedDetailBody,
      "Known-defect classes",
      decision.evidence.known_defects.classes.length === 0
        ? "None observed"
        : decision.evidence.known_defects.classes.join(", "),
    );

    const instanceSection = document.createElement("section");
    instanceSection.className = "instance-list combined-instance-list";
    const instanceHeading = document.createElement("h3");
    instanceHeading.textContent = "Persisted YOLO instances";
    instanceSection.append(instanceHeading);
    if (knownDefects.instances.length === 0) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "No known-defect instance children were persisted.";
      instanceSection.append(empty);
    } else {
      for (const instance of knownDefects.instances) {
        appendKnownDefectInstance(instanceSection, instance);
      }
    }
    elements.combinedDetailBody.append(instanceSection);

    appendDetailHeading(elements.combinedDetailBody, "Timing observations");
    appendValue(
      elements.combinedDetailBody,
      "PatchCore inference",
      `${Number(detail.timings.patchcore_inference_ms).toFixed(3)} ms`,
    );
    appendValue(
      elements.combinedDetailBody,
      "YOLO inference",
      `${Number(detail.timings.yolo_inference_ms).toFixed(3)} ms`,
    );
    appendValue(
      elements.combinedDetailBody,
      "Parallel orchestration",
      `${Number(detail.timings.orchestration_ms).toFixed(3)} ms`,
    );
  } catch (error) {
    elements.combinedDetailBody.replaceChildren();
    appendValue(elements.combinedDetailBody, "Status", error.message);
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

// ADD 2026-08-26: Known-defect event를 synchronization 중에는 buffer하고 이후에는 bounded window에 반영한다.
function acceptKnownLiveMessage(rawValue) {
  let event;
  try {
    event = parseKnownDefectEvent(JSON.parse(rawValue));
  } catch {
    return;
  }
  if (event === null) {
    return;
  }
  if (isKnownSynchronizing) {
    bufferedKnownDefects.push(event);
    return;
  }
  knownDefectInspections = mergeKnownDefectInspections(knownDefectInspections, [event]);
  renderKnownDefects();
}

// ADD 2026-08-26: YOLO reconnect backoff와 indicator를 PatchCore lifecycle과 격리한다.
function scheduleKnownReconnect() {
  if (stopped || knownReconnectTimer !== null) {
    return;
  }
  if (!navigator.onLine) {
    setKnownConnectionState("OFFLINE");
    return;
  }
  setKnownConnectionState("RECONNECTING");
  const delay = reconnectDelayMs(knownReconnectAttempt);
  knownReconnectAttempt += 1;
  knownReconnectTimer = window.setTimeout(() => {
    knownReconnectTimer = null;
    connectKnownAndSynchronize();
  }, delay);
}

// ADD 2026-08-26: YOLO WebSocket-first buffer와 REST snapshot을 atomic merge해 event gap을 막는다.
function connectKnownAndSynchronize() {
  if (stopped) {
    return;
  }
  knownGeneration += 1;
  const activeGeneration = knownGeneration;
  isKnownSynchronizing = true;
  bufferedKnownDefects = [];
  knownSyncAbortController?.abort();
  knownSyncAbortController = null;
  setKnownConnectionState(knownReconnectAttempt === 0 ? "CONNECTING" : "RECONNECTING");

  const connection = new WebSocket(knownDefectWebSocketUrl(window.location));
  knownSocket = connection;
  connection.addEventListener("message", (message) => acceptKnownLiveMessage(message.data));
  connection.addEventListener("open", async () => {
    if (activeGeneration !== knownGeneration || stopped) {
      connection.close();
      return;
    }
    knownSyncAbortController = new AbortController();
    try {
      const response = await fetch("/v1/known-defects?limit=100&offset=0", {
        headers: { Accept: "application/json" },
        signal: knownSyncAbortController.signal,
      });
      if (!response.ok) {
        throw new Error(`History request failed (${response.status}).`);
      }
      const history = parseKnownDefectHistory(await response.json());
      if (activeGeneration !== knownGeneration || stopped) {
        return;
      }
      knownDefectInspections = mergeKnownDefectInspections(
        history,
        knownDefectInspections,
        bufferedKnownDefects,
      );
      bufferedKnownDefects = [];
      isKnownSynchronizing = false;
      knownReconnectAttempt = 0;
      elements.knownLastSync.textContent = `Synced ${new Date().toLocaleTimeString()}`;
      renderKnownDefects();
      setKnownConnectionState("LIVE");
    } catch (error) {
      if (error.name !== "AbortError" && activeGeneration === knownGeneration) {
        connection.close();
      }
    }
  });
  connection.addEventListener("error", () => connection.close());
  connection.addEventListener("close", () => {
    if (activeGeneration !== knownGeneration) {
      return;
    }
    knownSyncAbortController?.abort();
    isKnownSynchronizing = false;
    scheduleKnownReconnect();
  });
}

// ADD 2026-08-26: Combined event를 initial/reconnect sync 중 buffer하고 이후 summary window에 반영한다.
function acceptCombinedLiveMessage(rawValue) {
  let event;
  try {
    event = parseCombinedInspectionEvent(JSON.parse(rawValue));
  } catch {
    return;
  }
  if (event === null) {
    return;
  }
  if (isCombinedSynchronizing) {
    bufferedCombinedInspections.push(event);
    return;
  }
  combinedInspections = mergeCombinedInspections(combinedInspections, [event]);
  renderCombinedInspections();
}

// ADD 2026-08-26: Combined reconnect backoff와 indicator를 두 child channel에서 격리한다.
function scheduleCombinedReconnect() {
  if (stopped || combinedReconnectTimer !== null) {
    return;
  }
  if (!navigator.onLine) {
    setCombinedConnectionState("OFFLINE");
    return;
  }
  setCombinedConnectionState("RECONNECTING");
  const delay = reconnectDelayMs(combinedReconnectAttempt);
  combinedReconnectAttempt += 1;
  combinedReconnectTimer = window.setTimeout(() => {
    combinedReconnectTimer = null;
    connectCombinedAndSynchronize();
  }, delay);
}

// ADD 2026-08-26: Combined WebSocket-first buffer와 durable history를 race-safe하게 병합한다.
function connectCombinedAndSynchronize() {
  if (stopped) {
    return;
  }
  combinedGeneration += 1;
  const activeGeneration = combinedGeneration;
  isCombinedSynchronizing = true;
  bufferedCombinedInspections = [];
  combinedSyncAbortController?.abort();
  combinedSyncAbortController = null;
  setCombinedConnectionState(
    combinedReconnectAttempt === 0 ? "CONNECTING" : "RECONNECTING",
  );

  const connection = new WebSocket(combinedInspectionWebSocketUrl(window.location));
  combinedSocket = connection;
  connection.addEventListener("message", (message) => acceptCombinedLiveMessage(message.data));
  connection.addEventListener("open", async () => {
    if (activeGeneration !== combinedGeneration || stopped) {
      connection.close();
      return;
    }
    combinedSyncAbortController = new AbortController();
    try {
      const response = await fetch("/v1/combined-inspections?limit=100&offset=0", {
        headers: { Accept: "application/json" },
        signal: combinedSyncAbortController.signal,
      });
      if (!response.ok) {
        throw new Error(`History request failed (${response.status}).`);
      }
      const history = parseCombinedInspectionHistory(await response.json());
      if (activeGeneration !== combinedGeneration || stopped) {
        return;
      }
      combinedInspections = mergeCombinedInspections(
        history,
        combinedInspections,
        bufferedCombinedInspections,
      );
      bufferedCombinedInspections = [];
      isCombinedSynchronizing = false;
      combinedReconnectAttempt = 0;
      elements.combinedLastSync.textContent = `Synced ${new Date().toLocaleTimeString()}`;
      renderCombinedInspections();
      setCombinedConnectionState("LIVE");
    } catch (error) {
      if (error.name !== "AbortError" && activeGeneration === combinedGeneration) {
        connection.close();
      }
    }
  });
  connection.addEventListener("error", () => connection.close());
  connection.addEventListener("close", () => {
    if (activeGeneration !== combinedGeneration) {
      return;
    }
    combinedSyncAbortController?.abort();
    isCombinedSynchronizing = false;
    scheduleCombinedReconnect();
  });
}

// MODIFY 2026-08-26: Page teardown에서 세 domain의 socket, timer와 sync request를 모두 정리한다.
function stopConnections() {
  stopped = true;
  generation += 1;
  knownGeneration += 1;
  combinedGeneration += 1;
  syncAbortController?.abort();
  knownSyncAbortController?.abort();
  combinedSyncAbortController?.abort();
  if (reconnectTimer !== null) {
    window.clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (knownReconnectTimer !== null) {
    window.clearTimeout(knownReconnectTimer);
    knownReconnectTimer = null;
  }
  if (combinedReconnectTimer !== null) {
    window.clearTimeout(combinedReconnectTimer);
    combinedReconnectTimer = null;
  }
  socket?.close();
  knownSocket?.close();
  combinedSocket?.close();
}

elements.detailClose.addEventListener("click", () => elements.detail.close());
elements.knownDetailClose.addEventListener("click", () => elements.knownDetail.close());
elements.combinedDetailClose.addEventListener("click", () => elements.combinedDetail.close());
window.addEventListener("offline", () => {
  if (reconnectTimer !== null) {
    window.clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (knownReconnectTimer !== null) {
    window.clearTimeout(knownReconnectTimer);
    knownReconnectTimer = null;
  }
  if (combinedReconnectTimer !== null) {
    window.clearTimeout(combinedReconnectTimer);
    combinedReconnectTimer = null;
  }
  setConnectionState("OFFLINE");
  setKnownConnectionState("OFFLINE");
  setCombinedConnectionState("OFFLINE");
  socket?.close();
  knownSocket?.close();
  combinedSocket?.close();
});
window.addEventListener("online", () => {
  if (!stopped) {
    reconnectAttempt = 0;
    knownReconnectAttempt = 0;
    combinedReconnectAttempt = 0;
    connectAndSynchronize();
    connectKnownAndSynchronize();
    connectCombinedAndSynchronize();
  }
});
window.addEventListener("beforeunload", stopConnections);

Object.defineProperty(window, "__liveMonitorDebug", {
  value: Object.freeze({
    snapshot: () => ({
      connectionState: elements.connection.textContent,
      inspections: structuredClone(inspections),
      reconnectAttempt,
      knownConnectionState: elements.knownConnection.textContent,
      knownDefectInspections: structuredClone(knownDefectInspections),
      knownReconnectAttempt,
      combinedConnectionState: elements.combinedConnection.textContent,
      combinedInspections: structuredClone(combinedInspections),
      combinedReconnectAttempt,
    }),
  }),
  writable: false,
});

render();
renderKnownDefects();
renderCombinedInspections();
connectCombinedAndSynchronize();
connectAndSynchronize();
connectKnownAndSynchronize();
