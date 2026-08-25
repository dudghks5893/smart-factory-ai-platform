const INSPECTION_ID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export const MAX_VISIBLE_INSPECTIONS = 100;

function isNonEmptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

// ADD 2026-08-25: REST/WS inspection을 UI가 사용할 compact validated value로 제한한다.
export function normalizeInspection(value) {
  if (
    value === null ||
    typeof value !== "object" ||
    !INSPECTION_ID_PATTERN.test(value.inspection_id ?? "") ||
    !isNonEmptyString(value.model_name) ||
    !isNonEmptyString(value.category) ||
    typeof value.is_anomaly !== "boolean" ||
    !isFiniteNumber(value.anomaly_score) ||
    !isFiniteNumber(value.threshold) ||
    value.comparison_operator !== ">" ||
    !isNonEmptyString(value.device) ||
    !isNonEmptyString(value.created_at) ||
    !Number.isFinite(Date.parse(value.created_at))
  ) {
    return null;
  }

  return {
    inspection_id: value.inspection_id,
    model_name: value.model_name,
    category: value.category,
    is_anomaly: value.is_anomaly,
    anomaly_score: value.anomaly_score,
    threshold: value.threshold,
    comparison_operator: ">",
    device: value.device,
    created_at: value.created_at,
  };
}

// ADD 2026-08-25: Versioned inspection.created event만 parsing하고 unknown/malformed event를 무시한다.
export function parseInspectionEvent(value) {
  if (
    value === null ||
    typeof value !== "object" ||
    value.schema_version !== "1" ||
    value.type !== "inspection.created"
  ) {
    return null;
  }
  return normalizeInspection(value.inspection);
}

// ADD 2026-08-25: History response의 bounded list contract를 검증하고 compact inspection으로 변환한다.
export function parseInspectionHistory(value) {
  if (value === null || typeof value !== "object" || !Array.isArray(value.items)) {
    throw new TypeError("Inspection history response is malformed.");
  }
  const items = value.items.map(normalizeInspection);
  if (items.some((item) => item === null)) {
    throw new TypeError("Inspection history contains a malformed item.");
  }
  return items;
}

// ADD 2026-08-25: REST snapshot과 buffered/live event를 ID로 dedupe하고 timestamp newest-first window로 병합한다.
// MODIFY 2026-08-26: 동일 timestamp에서도 inspection ID로 deterministic order를 보장한다.
export function mergeInspections(...groups) {
  const byId = new Map();
  for (const group of groups) {
    for (const candidate of group) {
      const inspection = normalizeInspection(candidate);
      if (inspection !== null) {
        byId.set(inspection.inspection_id, inspection);
      }
    }
  }
  return [...byId.values()]
    .sort(
      (left, right) =>
        Date.parse(right.created_at) - Date.parse(left.created_at) ||
        right.inspection_id.localeCompare(left.inspection_id),
    )
    .slice(0, MAX_VISIBLE_INSPECTIONS);
}

// ADD 2026-08-25: 현재 표시 window만 기준으로 normal/anomaly KPI를 계산한다.
export function calculateKpis(inspections) {
  const anomaly = inspections.filter((inspection) => inspection.is_anomaly).length;
  const normal = inspections.length - anomaly;
  return {
    visible: inspections.length,
    normal,
    anomaly,
    anomalyRatio: inspections.length === 0 ? 0 : anomaly / inspections.length,
  };
}

// ADD 2026-08-25: Browser location에서 same-origin ws/wss inspection endpoint를 구성한다.
export function inspectionWebSocketUrl(locationValue) {
  const protocol = locationValue.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${locationValue.host}/v1/ws/inspections`;
}

// ADD 2026-08-25: Reconnect delay를 0.5초부터 exponential로 늘리되 5초로 제한한다.
export function reconnectDelayMs(attempt) {
  return Math.min(500 * 2 ** Math.max(0, attempt), 5000);
}

// ADD 2026-08-26: Event class names를 compact deterministic unique summary로 제한한다.
function normalizedClassNames(value) {
  if (!Array.isArray(value)) {
    return [];
  }
  return [...new Set(value.filter(isNonEmptyString).map((item) => item.trim()))].sort();
}

// ADD 2026-08-26: Nested REST model과 flattened event identity를 같은 shape으로 읽는다.
function normalizedKnownDefectModel(value) {
  if (value?.model !== null && typeof value?.model === "object") {
    return {
      model_name: value.model.name,
      category: value.model.category,
      device: value.model.device,
    };
  }
  return {
    model_name: value?.model_name,
    category: value?.category,
    device: value?.device,
  };
}

// ADD 2026-08-26: REST parent summary와 compact WS payload를 하나의 YOLO UI value로 정규화한다.
export function normalizeKnownDefectInspection(value) {
  const model = normalizedKnownDefectModel(value);
  const inferenceMs = value?.inference_ms ?? null;
  const imageWidth = value?.image?.width ?? value?.image_width ?? null;
  const imageHeight = value?.image?.height ?? value?.image_height ?? null;
  if (
    value === null ||
    typeof value !== "object" ||
    !INSPECTION_ID_PATTERN.test(value.inspection_id ?? "") ||
    !isNonEmptyString(model.model_name) ||
    !isNonEmptyString(model.category) ||
    !isNonEmptyString(model.device) ||
    !isFiniteNumber(value.diagnostic_confidence) ||
    value.diagnostic_confidence <= 0 ||
    value.diagnostic_confidence >= 1 ||
    !Number.isInteger(value.instance_count) ||
    value.instance_count < 0 ||
    (inferenceMs !== null && (!isFiniteNumber(inferenceMs) || inferenceMs < 0)) ||
    (imageWidth !== null && (!Number.isInteger(imageWidth) || imageWidth <= 0)) ||
    (imageHeight !== null && (!Number.isInteger(imageHeight) || imageHeight <= 0)) ||
    !isNonEmptyString(value.created_at) ||
    !Number.isFinite(Date.parse(value.created_at))
  ) {
    return null;
  }

  return {
    inspection_id: value.inspection_id,
    model_name: model.model_name,
    category: model.category,
    device: model.device,
    diagnostic_confidence: value.diagnostic_confidence,
    inference_ms: inferenceMs,
    image_width: imageWidth,
    image_height: imageHeight,
    instance_count: value.instance_count,
    classes: normalizedClassNames(value.classes),
    created_at: value.created_at,
  };
}

// ADD 2026-08-26: Versioned known_defect.created summary만 parsing하고 다른 domain event를 무시한다.
export function parseKnownDefectEvent(value) {
  if (
    value === null ||
    typeof value !== "object" ||
    value.schema_version !== "1" ||
    value.type !== "known_defect.created"
  ) {
    return null;
  }
  return normalizeKnownDefectInspection(value.inspection);
}

// ADD 2026-08-26: Known-defect REST history의 bounded parent list를 검증한다.
export function parseKnownDefectHistory(value) {
  if (value === null || typeof value !== "object" || !Array.isArray(value.items)) {
    throw new TypeError("Known-defect history response is malformed.");
  }
  const items = value.items.map(normalizeKnownDefectInspection);
  if (items.some((item) => item === null)) {
    throw new TypeError("Known-defect history contains a malformed item.");
  }
  return items;
}

// ADD 2026-08-26: Duplicate summary 병합 시 REST latency/image와 event class 정보를 함께 보존한다.
function mergeKnownDefectValue(previous, incoming) {
  if (previous === undefined) {
    return incoming;
  }
  return {
    ...previous,
    ...incoming,
    inference_ms: incoming.inference_ms ?? previous.inference_ms,
    image_width: incoming.image_width ?? previous.image_width,
    image_height: incoming.image_height ?? previous.image_height,
    classes: normalizedClassNames([...previous.classes, ...incoming.classes]),
  };
}

// ADD 2026-08-26: YOLO REST/WS summary를 ID로 dedupe하고 richer fields를 보존해 newest-first 100으로 제한한다.
export function mergeKnownDefectInspections(...groups) {
  const byId = new Map();
  for (const group of groups) {
    for (const candidate of group) {
      const inspection = normalizeKnownDefectInspection(candidate);
      if (inspection !== null) {
        byId.set(
          inspection.inspection_id,
          mergeKnownDefectValue(byId.get(inspection.inspection_id), inspection),
        );
      }
    }
  }
  return [...byId.values()]
    .sort(
      (left, right) =>
        Date.parse(right.created_at) - Date.parse(left.created_at) ||
        right.inspection_id.localeCompare(left.inspection_id),
    )
    .slice(0, MAX_VISIBLE_INSPECTIONS);
}

// ADD 2026-08-26: 현재 YOLO window의 empty/defect inspections와 total instance 수를 계산한다.
export function calculateKnownDefectKpis(inspections) {
  const knownDefect = inspections.filter((inspection) => inspection.instance_count > 0).length;
  return {
    visible: inspections.length,
    noKnownDefect: inspections.length - knownDefect,
    knownDefect,
    totalInstances: inspections.reduce(
      (total, inspection) => total + inspection.instance_count,
      0,
    ),
  };
}

// ADD 2026-08-26: Browser location에서 독립 known-defect ws/wss endpoint를 구성한다.
export function knownDefectWebSocketUrl(locationValue) {
  const protocol = locationValue.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${locationValue.host}/v1/ws/known-defects`;
}

// ADD 2026-08-26: UUID를 encode해 same-origin known-defect detail endpoint를 구성한다.
export function knownDefectDetailUrl(inspectionId) {
  return `/v1/known-defects/${encodeURIComponent(inspectionId)}`;
}
