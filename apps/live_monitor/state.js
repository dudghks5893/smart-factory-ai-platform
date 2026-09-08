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

const COMBINED_DISPOSITIONS = new Set(["PASS", "REVIEW", "REJECT"]);
const PATCHCORE_PREDICTIONS = new Set(["NORMAL", "ANOMALY"]);
const DECISION_REASON_LABELS = Object.freeze({
  NO_ANOMALY_EVIDENCE: "No anomaly evidence",
  UNKNOWN_ANOMALY: "Unknown anomaly requires review",
  MODEL_DISAGREEMENT: "Model disagreement requires review",
  CONFIRMED_KNOWN_DEFECT: "Confirmed known-defect evidence",
});

// ADD 2026-08-26: History의 nested policy와 event의 flattened policy를 같은 UI identity로 읽는다.
function normalizedDecisionPolicy(value) {
  if (value?.policy !== null && typeof value?.policy === "object") {
    return {
      policy_name: value.policy.name,
      policy_version: value.policy.version,
    };
  }
  return {
    policy_name: value?.policy_name,
    policy_version: value?.policy_version,
  };
}

// ADD 2026-08-26: Backend decision summary를 client-side policy 계산 없는 compact UI value로 검증한다.
export function normalizeCombinedInspection(value) {
  const policy = normalizedDecisionPolicy(value);
  const classes = normalizedClassNames(value?.known_defect_classes ?? value?.classes);
  if (
    value === null ||
    typeof value !== "object" ||
    !INSPECTION_ID_PATTERN.test(value.combined_inspection_id ?? "") ||
    !isNonEmptyString(value.created_at) ||
    !Number.isFinite(Date.parse(value.created_at)) ||
    !PATCHCORE_PREDICTIONS.has(value.patchcore_prediction) ||
    !Number.isInteger(value.known_defect_instance_count) ||
    value.known_defect_instance_count < 0 ||
    !COMBINED_DISPOSITIONS.has(value.disposition) ||
    !Object.hasOwn(DECISION_REASON_LABELS, value.reason_code) ||
    !isNonEmptyString(policy.policy_name) ||
    !isNonEmptyString(policy.policy_version)
  ) {
    return null;
  }
  return {
    combined_inspection_id: value.combined_inspection_id,
    created_at: value.created_at,
    patchcore_prediction: value.patchcore_prediction,
    known_defect_instance_count: value.known_defect_instance_count,
    known_defect_classes: classes,
    disposition: value.disposition,
    reason_code: value.reason_code,
    policy_name: policy.policy_name,
    policy_version: policy.policy_version,
  };
}

// ADD 2026-08-26: Versioned combined decision event만 parsing하고 child-domain event를 무시한다.
export function parseCombinedInspectionEvent(value) {
  if (
    value === null ||
    typeof value !== "object" ||
    value.schema_version !== "1" ||
    value.type !== "combined_inspection.created"
  ) {
    return null;
  }
  return normalizeCombinedInspection(value.inspection);
}

// ADD 2026-08-26: PostgreSQL-backed combined history의 bounded summary list를 검증한다.
export function parseCombinedInspectionHistory(value) {
  if (value === null || typeof value !== "object" || !Array.isArray(value.items)) {
    throw new TypeError("Combined inspection history response is malformed.");
  }
  const items = value.items.map(normalizeCombinedInspection);
  if (items.some((item) => item === null)) {
    throw new TypeError("Combined inspection history contains a malformed item.");
  }
  return items;
}

// ADD 2026-08-26: REST summary와 event가 같은 ID면 event class summary를 잃지 않고 병합한다.
function mergeCombinedValue(previous, incoming) {
  if (previous === undefined) {
    return incoming;
  }
  return {
    ...previous,
    ...incoming,
    known_defect_classes: normalizedClassNames([
      ...previous.known_defect_classes,
      ...incoming.known_defect_classes,
    ]),
  };
}

// ADD 2026-08-26: Combined summary를 UUID dedupe, deterministic newest-first, visible 100으로 제한한다.
export function mergeCombinedInspections(...groups) {
  const byId = new Map();
  for (const group of groups) {
    for (const candidate of group) {
      const inspection = normalizeCombinedInspection(candidate);
      if (inspection !== null) {
        byId.set(
          inspection.combined_inspection_id,
          mergeCombinedValue(byId.get(inspection.combined_inspection_id), inspection),
        );
      }
    }
  }
  return [...byId.values()]
    .sort(
      (left, right) =>
        Date.parse(right.created_at) - Date.parse(left.created_at) ||
        right.combined_inspection_id.localeCompare(left.combined_inspection_id),
    )
    .slice(0, MAX_VISIBLE_INSPECTIONS);
}

// ADD 2026-08-26: 현재 combined visible window의 persisted disposition count만 집계한다.
export function calculateCombinedKpis(inspections) {
  return {
    visible: inspections.length,
    pass: inspections.filter((inspection) => inspection.disposition === "PASS").length,
    review: inspections.filter((inspection) => inspection.disposition === "REVIEW").length,
    reject: inspections.filter((inspection) => inspection.disposition === "REJECT").length,
  };
}

// ADD 2026-08-26: Stable reason code를 operator-facing label로 변환하되 code 자체는 보존한다.
export function decisionReasonLabel(reasonCode) {
  return DECISION_REASON_LABELS[reasonCode] ?? "Unknown decision reason";
}

// ADD 2026-08-26: Browser location에서 독립 combined ws/wss endpoint를 구성한다.
export function combinedInspectionWebSocketUrl(locationValue) {
  const protocol = locationValue.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${locationValue.host}/v1/ws/combined-inspections`;
}

// ADD 2026-08-26: Combined UUID를 encode해 same-origin persisted detail endpoint를 구성한다.
export function combinedInspectionDetailUrl(combinedInspectionId) {
  return `/v1/combined-inspections/${encodeURIComponent(combinedInspectionId)}`;
}

// ADD 2026-09-07: Non-persisted DeepStream observation을 compact browser value로 strict하게 검증한다. → MODIFY 2026-09-08: raw/normalized 값을 모두 수용하고 bbox를 보존한다.
export function normalizeStreamingObservation(value) {
  if (value === null || typeof value !== "object") {
    return null;
  }

  const imageWidth = value.image?.width ?? value.image_width;
  const imageHeight = value.image?.height ?? value.image_height;
  const decoderId = value.runtime?.decoder_id ?? value.decoder_id;

  if (
    !INSPECTION_ID_PATTERN.test(value.observation_id ?? "") ||
    !INSPECTION_ID_PATTERN.test(value.stream_session_id ?? "") ||
    !isNonEmptyString(value.source_id) ||
    !Number.isInteger(value.frame_number) ||
    value.frame_number < 0 ||
    !Number.isInteger(value.pts_ns) ||
    value.pts_ns < 0 ||
    !isNonEmptyString(value.observed_at) ||
    !Number.isFinite(Date.parse(value.observed_at)) ||
    !Number.isInteger(imageWidth) ||
    imageWidth <= 0 ||
    !Number.isInteger(imageHeight) ||
    imageHeight <= 0 ||
    !isFiniteNumber(value.diagnostic_confidence) ||
    value.diagnostic_confidence !== 0.25 ||
    !Array.isArray(value.instances) ||
    !isNonEmptyString(decoderId)
  ) {
    return null;
  }

  const instances = [];
  for (const instance of value.instances) {
    const expectedClass = { 0: "bent", 1: "color", 2: "scratch" }[instance?.class_id];
    const maskPixelCount = instance?.mask?.pixel_count ?? instance?.mask_pixel_count;
    const maskAreaRatio = instance?.mask?.area_ratio ?? instance?.mask_area_ratio;
    const box = instance?.box;
    if (
      instance === null ||
      typeof instance !== "object" ||
      !Number.isInteger(instance.class_id) ||
      instance.class_name !== expectedClass ||
      !isFiniteNumber(instance.confidence) ||
      instance.confidence < 0 ||
      instance.confidence > 1 ||
      box === null ||
      typeof box !== "object" ||
      !isFiniteNumber(box.x_min) ||
      !isFiniteNumber(box.y_min) ||
      !isFiniteNumber(box.x_max) ||
      !isFiniteNumber(box.y_max) ||
      box.x_min < 0 || box.y_min < 0 ||
      box.x_min > box.x_max || box.y_min > box.y_max ||
      box.x_max > imageWidth || box.y_max > imageHeight ||
      !Number.isInteger(maskPixelCount) || maskPixelCount <= 0 ||
      !isFiniteNumber(maskAreaRatio) || maskAreaRatio <= 0 || maskAreaRatio > 1
    ) {
      return null;
    }
    instances.push({
      class_id: instance.class_id,
      class_name: instance.class_name,
      confidence: instance.confidence,
      box: {
        x_min: box.x_min,
        y_min: box.y_min,
        x_max: box.x_max,
        y_max: box.y_max,
      },
      mask: { pixel_count: maskPixelCount, area_ratio: maskAreaRatio },
      mask_pixel_count: maskPixelCount,
      mask_area_ratio: maskAreaRatio,
    });
  }

  return {
    observation_id: value.observation_id,
    source_id: value.source_id,
    stream_session_id: value.stream_session_id,
    frame_number: value.frame_number,
    pts_ns: value.pts_ns,
    observed_at: value.observed_at,
    image_width: imageWidth,
    image_height: imageHeight,
    diagnostic_confidence: value.diagnostic_confidence,
    decoder_id: decoderId,
    instances,
  };
}

// ADD 2026-09-07: Persisted YOLO event와 분리된 streaming_known_defect.observed만 수용한다.
export function parseStreamingObservationEvent(value) {
  if (
    value === null ||
    typeof value !== "object" ||
    value.schema_version !== "1" ||
    value.type !== "streaming_known_defect.observed"
  ) {
    return null;
  }
  return normalizeStreamingObservation(value.observation);
}

// ADD 2026-09-07: Browser-memory streaming frame을 observation UUID로 dedupe하고 newest-first 100으로 제한한다.
export function mergeStreamingObservations(...groups) {
  const byId = new Map();
  for (const group of groups) {
    for (const candidate of group) {
      const observation = normalizeStreamingObservation(candidate);
      if (observation !== null) {
        byId.set(observation.observation_id, observation);
      }
    }
  }
  return [...byId.values()]
    .sort(
      (left, right) =>
        Date.parse(right.observed_at) - Date.parse(left.observed_at) ||
        right.frame_number - left.frame_number ||
        right.observation_id.localeCompare(left.observation_id),
    )
    .slice(0, MAX_VISIBLE_INSPECTIONS);
}

// ADD 2026-09-07: Non-persisted visible frame window의 defect-frame/instance/latest-frame KPI를 계산한다.
export function calculateStreamingKpis(observations) {
  return {
    visible: observations.length,
    defectFrames: observations.filter((observation) => observation.instances.length > 0).length,
    totalInstances: observations.reduce(
      (total, observation) => total + observation.instances.length,
      0,
    ),
    latestFrame: observations.length === 0 ? null : observations[0].frame_number,
  };
}

// ADD 2026-09-07: Browser location에서 dedicated live-only DeepStream ws/wss endpoint를 구성한다.
export function streamingKnownDefectWebSocketUrl(locationValue) {
  const protocol = locationValue.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${locationValue.host}/v1/ws/streaming-known-defects`;
}
