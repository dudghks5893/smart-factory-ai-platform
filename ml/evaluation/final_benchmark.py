"""Final benchmark aggregation, cross-source lineage, and artifact contracts."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ml.evaluation.rag import load_rag_evaluation_artifact
from shared.hashing import is_sha256_digest, sha256_file

FINAL_BENCHMARK_SCHEMA_VERSION = 2
FINAL_BENCHMARK_NAME = "smart_factory_final_benchmark"
FINAL_BENCHMARK_FILENAME = "benchmark.json"
DEFAULT_FINAL_BENCHMARK_ROOT = Path("outputs/benchmarks/final")
API_V1_LABEL = "API Benchmark — schema v1 / pre-persistence"
API_V2_LABEL = "API Benchmark — schema v2 / persistence-inclusive"
RAG_DEMO_LABEL = "Demo / deterministic RAG benchmark"
_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_VERIFICATION_STATUSES = {"verified", "partially_verified", "pending"}


@dataclass(frozen=True)
class FinalBenchmarkSources:
    """Required official evidence snapshots plus optional current API benchmark."""

    vision_quality_path: Path
    model_runtime_path: Path
    api_v1_path: Path
    rag_evaluation_dir: Path
    platform_verification_path: Path
    api_v2_path: Path | None = None


@dataclass(frozen=True)
class FinalBenchmarkResult:
    """Generated final benchmark artifact and validated payload."""

    output_dir: Path
    benchmark_path: Path
    payload: Mapping[str, Any]


GitCommandRunner = Callable[[Sequence[str], Path], str]


@dataclass(frozen=True)
class RepositoryProvenance:
    """Git HEAD and whether non-ignored working-tree changes existed at build time."""

    git_commit: str
    working_tree_dirty: bool

    # ADD 2026-08-21: Repository SHA와 dirty flag의 strict types를 검증한다.
    def validate(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{40}", self.git_commit):
            raise ValueError("Repository git_commit must be a full lowercase SHA-1 hash.")
        if type(self.working_tree_dirty) is not bool:
            raise TypeError("Repository working_tree_dirty must be boolean.")

    # ADD 2026-08-21: Repository state를 final artifact JSON provenance로 변환한다.
    def to_json_dict(self) -> dict[str, str | bool]:
        self.validate()
        return {
            "git_commit": self.git_commit,
            "working_tree_dirty": self.working_tree_dirty,
        }


# ADD 2026-08-21: Git에서 build 시작 시점의 HEAD와 non-ignored dirty state를 조회한다.
def resolve_repository_provenance(
    repository_root: Path,
    *,
    run_git: GitCommandRunner | None = None,
) -> RepositoryProvenance:
    """Resolve repository state without allowing a caller-supplied commit claim."""
    runner = run_git or _run_git
    git_commit = runner(("rev-parse", "HEAD"), repository_root).strip()
    working_tree_status = runner(
        ("status", "--porcelain", "--untracked-files=normal"),
        repository_root,
    )
    provenance = RepositoryProvenance(
        git_commit=git_commit,
        working_tree_dirty=bool(working_tree_status.strip()),
    )
    provenance.validate()
    return provenance


# ADD 2026-08-21: Official evidence와 RAG artifact를 lineage-safe final benchmark로 결합한다.
# MODIFY 2026-08-21: Caller SHA를 validated repository HEAD/dirty provenance로 대체했다.
def build_final_benchmark(
    *,
    sources: FinalBenchmarkSources,
    output_root: Path,
    benchmark_id: str,
    created_at: str,
    repository_provenance: RepositoryProvenance,
) -> FinalBenchmarkResult:
    """Aggregate existing evidence without rerunning or tuning any benchmark."""
    _validate_identifier(benchmark_id, "benchmark_id")
    _parse_aware_datetime(created_at)
    repository_provenance.validate()
    final_dir = output_root / benchmark_id
    if final_dir.exists():
        raise FileExistsError(f"Final benchmark output already exists: {final_dir}")

    # Required official snapshots와 actual RAG evaluation artifact를 모두 fail-fast load한다.
    quality = _load_official_snapshot(
        sources.vision_quality_path,
        expected_name="patchcore_quality_evaluation",
    )
    model_runtime = _load_official_snapshot(
        sources.model_runtime_path,
        expected_name="patchcore_inference",
    )
    api_v1 = _load_official_snapshot(
        sources.api_v1_path,
        expected_name="patchcore_fastapi_http_e2e",
    )
    _validate_api_v1(api_v1)
    rag_summary, _ = load_rag_evaluation_artifact(sources.rag_evaluation_dir)
    platform = _load_platform_verification(sources.platform_verification_path)
    api_v2 = _load_api_v2(sources.api_v2_path) if sources.api_v2_path is not None else None

    # Vision quality/model/API evidence가 같은 lineage인지 확인한다.
    _validate_vision_lineage(quality, model_runtime, api_v1, api_v2)
    source_hashes = _source_hashes(sources)
    sections = _build_sections(
        quality=quality,
        model_runtime=model_runtime,
        api_v1=api_v1,
        api_v2=api_v2,
        rag_summary=rag_summary,
        platform=platform,
    )
    payload = {
        "schema_version": FINAL_BENCHMARK_SCHEMA_VERSION,
        "benchmark_name": FINAL_BENCHMARK_NAME,
        "benchmark_id": benchmark_id,
        "created_at": created_at,
        "repository": repository_provenance.to_json_dict(),
        "sources": source_hashes,
        "sections": sections,
        "environment_matrix": _environment_matrix(sections),
        "limitations": [
            (
                "This artifact aggregates measurements from different environments and "
                "boundaries; it does not define an overall score."
            ),
            (
                "Historical approved STEP 3/4 compact evidence snapshots preserve documented "
                "values because ignored raw runtime artifacts are not committed."
            ),
            (
                "The API schema v1 result excludes inspection persistence and is not "
                "production network latency."
            ),
            (
                "The RAG result uses a fictional public demo corpus and deterministic "
                "evaluation-only providers."
            ),
            (
                "No model, threshold, retrieval, database, or container performance tuning "
                "was performed in STEP 15."
            ),
        ],
    }
    _validate_final_payload(payload)

    # Temporary directory에서 full validation을 완료한 뒤 immutable final directory로 commit한다.
    output_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{benchmark_id}.tmp-", dir=output_root))
    try:
        benchmark_path = temp_dir / FINAL_BENCHMARK_FILENAME
        benchmark_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        load_final_benchmark_artifact(benchmark_path)
        temp_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return FinalBenchmarkResult(
        output_dir=final_dir,
        benchmark_path=final_dir / FINAL_BENCHMARK_FILENAME,
        payload=payload,
    )


# ADD 2026-08-21: Final benchmark schema, labels, hashes와 finite values를 검증해 로드한다.
def load_final_benchmark_artifact(path: Path) -> Mapping[str, Any]:
    """Load one generated final summary and reject corrupt or ambiguous evidence."""
    if not path.is_file():
        raise FileNotFoundError(f"Final benchmark artifact not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("final benchmark must be an object")
        _validate_final_payload(raw)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Cannot load final benchmark artifact.") from exc
    return raw


# ADD 2026-08-21: Versioned official evidence snapshot의 taxonomy와 finite result를 검증한다.
def _load_official_snapshot(path: Path, *, expected_name: str) -> Mapping[str, Any]:
    raw = _load_json_mapping(path, "official benchmark snapshot")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("Official benchmark snapshot schema_version must be 1.")
    if raw.get("benchmark_name") != expected_name or raw.get("official_result") is not True:
        raise ValueError("Official benchmark snapshot identity is invalid.")
    for field in ("evidence_origin", "category", "measurement_boundary"):
        _required_string(raw.get(field), field)
    environment = _mapping(raw.get("environment"), "environment")
    _required_string(environment.get("environment_id"), "environment_id")
    _required_string(environment.get("label"), "environment label")
    _required_string(environment.get("device"), "environment device")
    warmup = _mapping(raw.get("warmup"), "warmup")
    if type(warmup.get("count")) is not int or warmup["count"] < 0:
        raise ValueError("Official benchmark warmup count is invalid.")
    iterations = _mapping(raw.get("iterations"), "iterations")
    if type(iterations.get("count")) is not int or iterations["count"] <= 0:
        raise ValueError("Official benchmark iterations are invalid.")
    batch_size = raw.get("batch_size")
    if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
        raise ValueError("Official benchmark batch_size is invalid.")
    _validate_provenance(_mapping(raw.get("provenance"), "provenance"))
    _validate_finite_values(raw.get("results"))
    return raw


# ADD 2026-08-21: STEP 4 API snapshot이 명시적 pre-persistence schema v1인지 검증한다.
def _validate_api_v1(api_v1: Mapping[str, Any]) -> None:
    if api_v1.get("benchmark_label") != API_V1_LABEL:
        raise ValueError("Official API benchmark must use the pre-persistence schema v1 label.")
    if api_v1.get("inspection_persistence_included") is not False:
        raise ValueError("Official API schema v1 must exclude inspection persistence.")
    environment = _mapping(api_v1.get("environment"), "API environment")
    if environment.get("transport") != "in_process_asgi_testclient":
        raise ValueError("Official API schema v1 transport is invalid.")


# ADD 2026-08-21: Optional current API artifact가 real persistence-inclusive schema v2인지 검증한다.
def _load_api_v2(path: Path) -> Mapping[str, Any]:
    raw = _load_json_mapping(path, "API schema v2 benchmark")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 2:
        raise ValueError("Persistence-inclusive API benchmark must use schema_version 2.")
    if raw.get("benchmark_name") != "patchcore_fastapi_http_e2e":
        raise ValueError("Persistence-inclusive API benchmark name is invalid.")
    if raw.get("inspection_persistence_included") is not True:
        raise ValueError("API schema v2 must include inspection persistence.")
    _required_string(raw.get("category"), "API v2 category")
    _required_string(raw.get("device"), "API v2 device")
    _required_string(raw.get("latency_definition"), "API v2 measurement boundary")
    _mapping(raw.get("runtime"), "API v2 runtime")
    provenance = _mapping(raw.get("provenance"), "API v2 provenance")
    _validate_provenance(provenance, require_category=False)
    for field in ("threshold_artifact_sha256", "artifact_metadata_sha256"):
        value = provenance.get(field)
        if not isinstance(value, str) or not is_sha256_digest(value):
            raise ValueError(f"Persistence-inclusive API benchmark {field} is invalid.")
    _validate_finite_values(raw.get("metrics"))
    return raw


# ADD 2026-08-21: Vision source의 category/model/manifest/threshold hash 일치를 검증한다.
def _validate_vision_lineage(
    quality: Mapping[str, Any],
    model_runtime: Mapping[str, Any],
    api_v1: Mapping[str, Any],
    api_v2: Mapping[str, Any] | None,
) -> None:
    sources = [quality, model_runtime, api_v1]
    if api_v2 is not None:
        sources.append(api_v2)
    categories = {source["category"] for source in sources}
    provenances = [_mapping(source["provenance"], "provenance") for source in sources]
    provenance_categories = {
        provenance["category"]
        for provenance in provenances
        if provenance.get("category") is not None
    }
    manifests = {provenance["manifest_sha256"] for provenance in provenances}
    models = {provenance["model_sha256"] for provenance in provenances}
    if (
        len(categories) != 1
        or len(provenance_categories) != 1
        or provenance_categories != categories
        or len(manifests) != 1
        or len(models) != 1
    ):
        raise ValueError("Vision benchmark source category/model/manifest lineage mismatch.")
    threshold_values = {
        provenance["threshold_artifact_sha256"]
        for provenance in provenances
        if provenance.get("threshold_artifact_sha256") is not None
    }
    if len(threshold_values) != 1:
        raise ValueError("Vision benchmark threshold artifact lineage mismatch.")
    metadata_values = {
        provenance["artifact_metadata_sha256"]
        for provenance in provenances
        if provenance.get("artifact_metadata_sha256") is not None
    }
    if len(metadata_values) > 1:
        raise ValueError("Vision benchmark artifact metadata lineage mismatch.")


# ADD 2026-08-21: Source artifacts를 final report의 여섯 독립 measurement section으로 구성한다.
def _build_sections(
    *,
    quality: Mapping[str, Any],
    model_runtime: Mapping[str, Any],
    api_v1: Mapping[str, Any],
    api_v2: Mapping[str, Any] | None,
    rag_summary: Mapping[str, Any],
    platform: Mapping[str, Any],
) -> dict[str, Any]:
    quality_results = _mapping(quality["results"], "quality results")
    api_v2_section: dict[str, Any]
    if api_v2 is None:
        api_v2_section = {
            "label": API_V2_LABEL,
            "status": "not_available",
            "reason": (
                "No real PostgreSQL plus production PatchCore GPU schema v2 artifact was supplied; "
                "SQLite/fake/CPU latency was not substituted."
            ),
        }
    else:
        api_v2_conditions = _mapping(api_v2["conditions"], "API v2 conditions")
        api_v2_section = {
            "benchmark_name": api_v2["benchmark_name"],
            "schema_version": api_v2["schema_version"],
            "label": API_V2_LABEL,
            "status": "available",
            "environment": {
                "environment_id": "api-schema-v2-source-runtime",
                "label": "Source artifact runtime",
                **_mapping(api_v2["runtime"], "API v2 runtime"),
                "device": api_v2["device"],
            },
            "measurement_boundary": api_v2["latency_definition"],
            "conditions": {
                "warmup": {
                    "count": api_v2_conditions["warmup_count"],
                    "included": False,
                },
                "iterations": {
                    "count": api_v2_conditions["measured_count"],
                    "unit": "measured_requests",
                },
                "batch_size": api_v2_conditions["request_batch_size"],
                "source": api_v2_conditions,
            },
            "results": api_v2["metrics"],
            "provenance": api_v2["provenance"],
        }
    return {
        "vision_quality": {
            "benchmark_name": "patchcore_image_quality",
            "schema_version": quality["schema_version"],
            "label": "Vision image-level quality / official STEP 3",
            "environment": quality["environment"],
            "measurement_boundary": quality["measurement_boundary"],
            "conditions": {
                "warmup": quality["warmup"],
                "iterations": quality["iterations"],
                "batch_size": quality["batch_size"],
                "threshold": quality["threshold"],
            },
            "results": quality_results["image_level"],
            "per_defect": quality_results["per_defect"],
            "provenance": quality["provenance"],
        },
        "pixel_localization_quality": {
            "benchmark_name": "patchcore_pixel_localization_quality",
            "schema_version": quality["schema_version"],
            "label": "Pixel localization quality / official STEP 3",
            "environment": quality["environment"],
            "measurement_boundary": quality["measurement_boundary"],
            "conditions": {
                "warmup": quality["warmup"],
                "iterations": quality["iterations"],
                "batch_size": quality["batch_size"],
                "threshold": quality["threshold"],
            },
            "results": quality_results["pixel_level"],
            "provenance": quality["provenance"],
        },
        "model_runtime_performance": {
            "benchmark_name": model_runtime["benchmark_name"],
            "schema_version": model_runtime["schema_version"],
            "label": "T4 model inference benchmark / official STEP 3",
            "environment": model_runtime["environment"],
            "measurement_boundary": model_runtime["measurement_boundary"],
            "excluded": model_runtime["excluded"],
            "conditions": {
                "warmup": model_runtime["warmup"],
                "iterations": model_runtime["iterations"],
                "measured_batch_count": model_runtime["measured_batch_count"],
                "batch_size": model_runtime["batch_size"],
                "num_workers": model_runtime["num_workers"],
            },
            "results": model_runtime["results"],
            "provenance": model_runtime["provenance"],
        },
        "api_application_performance_v1": {
            "benchmark_name": api_v1["benchmark_name"],
            "schema_version": api_v1["schema_version"],
            "label": API_V1_LABEL,
            "environment": api_v1["environment"],
            "measurement_boundary": api_v1["measurement_boundary"],
            "excluded": api_v1["excluded"],
            "conditions": {
                "warmup": api_v1["warmup"],
                "iterations": api_v1["iterations"],
                "batch_size": api_v1["batch_size"],
                "inspection_persistence_included": False,
            },
            "results": api_v1["results"],
            "provenance": api_v1["provenance"],
        },
        "api_application_performance_v2": api_v2_section,
        "rag_quality": {
            "benchmark_name": "rag_quality_evaluation",
            "schema_version": rag_summary["schema_version"],
            "label": RAG_DEMO_LABEL,
            "environment": {
                "environment_id": "local-deterministic-rag-step14",
                "label": "Local deterministic public demo evaluation",
                "device": "CPU/NumPy exact retrieval",
            },
            "measurement_boundary": (
                "Versioned nine-case public demo dataset over one existing immutable index; "
                "deterministic extraction and lexical support evaluation."
            ),
            "conditions": {
                "warmup": {"count": 0, "note": "not applicable to quality evaluation"},
                "iterations": {
                    "count": rag_summary["dataset"]["case_count"],
                    "unit": "evaluation_cases",
                },
                "batch_size": None,
            },
            "dataset": rag_summary["dataset"],
            "index_lineage": rag_summary["index_lineage"],
            "provenance": {
                "dataset_sha256": rag_summary["dataset"]["sha256"],
                "index_metadata_sha256": rag_summary["index_lineage"]["metadata_sha256"],
            },
            "configuration": rag_summary["configuration"],
            "results": rag_summary["metrics"],
            "score_analysis": rag_summary["score_analysis"],
        },
        "platform_verification": {
            "label": "Platform engineering verification matrix",
            "verification_id": platform["verification_id"],
            "entries": platform["entries"],
        },
    }


# ADD 2026-08-21: Section별 independent environment를 합치지 않고 matrix row로 보존한다.
def _environment_matrix(sections: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for section_name in (
        "vision_quality",
        "pixel_localization_quality",
        "model_runtime_performance",
        "api_application_performance_v1",
        "rag_quality",
    ):
        section = _mapping(sections[section_name], section_name)
        rows.append(
            {
                "section": section_name,
                "label": section["label"],
                "environment": section["environment"],
                "measurement_boundary": section["measurement_boundary"],
            }
        )
    api_v2 = _mapping(sections["api_application_performance_v2"], "API v2 section")
    rows.append(
        {
            "section": "api_application_performance_v2",
            "label": api_v2["label"],
            "status": api_v2["status"],
            "environment": api_v2.get("environment"),
            "measurement_boundary": api_v2.get("measurement_boundary"),
        }
    )
    return rows


# ADD 2026-08-21: Platform verification config의 status vocabulary와 unique area를 검증한다.
def _load_platform_verification(path: Path) -> Mapping[str, Any]:
    raw = _load_json_mapping(path, "platform verification")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("Platform verification schema_version must be 1.")
    _validate_identifier(raw.get("verification_id"), "verification_id")
    entries = raw.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Platform verification entries must be non-empty.")
    areas = []
    for entry in entries:
        item = _mapping(entry, "platform verification entry")
        areas.append(_required_string(item.get("area"), "verification area"))
        if item.get("status") not in _VERIFICATION_STATUSES:
            raise ValueError("Platform verification status is invalid.")
        _required_string(item.get("environment"), "verification environment")
        _required_string(item.get("evidence"), "verification evidence")
    if len(areas) != len(set(areas)):
        raise ValueError("Platform verification areas must be unique.")
    return raw


# ADD 2026-08-21: Final builder가 읽은 모든 source file SHA를 path-free mapping으로 기록한다.
def _source_hashes(sources: FinalBenchmarkSources) -> dict[str, Any]:
    values: dict[str, Path | None] = {
        "vision_quality": sources.vision_quality_path,
        "model_runtime": sources.model_runtime_path,
        "api_v1": sources.api_v1_path,
        "rag_evaluation": sources.rag_evaluation_dir / "evaluation.json",
        "rag_cases": sources.rag_evaluation_dir / "cases.jsonl",
        "platform_verification": sources.platform_verification_path,
        "api_v2": sources.api_v2_path,
    }
    return {
        name: (
            {"filename": path.name, "sha256": sha256_file(path)}
            if path is not None
            else {"status": "not_available"}
        )
        for name, path in values.items()
    }


# ADD 2026-08-21: Final artifact identity, required labels/source hashes와 finite metric을 검증한다.
# MODIFY 2026-08-21: Schema v2 repository HEAD/dirty provenance를 strict validation한다.
def _validate_final_payload(payload: Mapping[str, Any]) -> None:
    if (
        type(payload.get("schema_version")) is not int
        or payload["schema_version"] != FINAL_BENCHMARK_SCHEMA_VERSION
        or payload.get("benchmark_name") != FINAL_BENCHMARK_NAME
    ):
        raise ValueError("Final benchmark schema identity is invalid.")
    _validate_identifier(payload.get("benchmark_id"), "benchmark_id")
    _parse_aware_datetime(payload.get("created_at"))
    repository = _mapping(payload.get("repository"), "repository provenance")
    working_tree_dirty = repository.get("working_tree_dirty")
    if type(working_tree_dirty) is not bool:
        raise TypeError("Final benchmark repository working_tree_dirty must be boolean.")
    RepositoryProvenance(
        git_commit=_required_string(repository.get("git_commit"), "repository git_commit"),
        working_tree_dirty=working_tree_dirty,
    ).validate()
    sources = _mapping(payload.get("sources"), "sources")
    for name in (
        "vision_quality",
        "model_runtime",
        "api_v1",
        "rag_evaluation",
        "rag_cases",
        "platform_verification",
    ):
        source = _mapping(sources.get(name), f"source {name}")
        source_sha256 = source.get("sha256")
        if not isinstance(source_sha256, str) or not is_sha256_digest(source_sha256):
            raise ValueError(f"Final benchmark source hash is invalid: {name}.")
    sections = _mapping(payload.get("sections"), "sections")
    for section_name in (
        "vision_quality",
        "pixel_localization_quality",
        "model_runtime_performance",
        "api_application_performance_v1",
        "rag_quality",
    ):
        _validate_benchmark_taxonomy(_mapping(sections.get(section_name), section_name))
    if (
        _mapping(sections.get("api_application_performance_v1"), "API v1").get("label")
        != API_V1_LABEL
    ):
        raise ValueError("Final benchmark API v1 label is invalid.")
    api_v1 = _mapping(sections["api_application_performance_v1"], "API v1")
    if (
        _mapping(api_v1["conditions"], "API v1 conditions").get("inspection_persistence_included")
        is not False
    ):
        raise ValueError("Final benchmark API v1 persistence boundary is invalid.")
    api_v2 = _mapping(sections.get("api_application_performance_v2"), "API v2")
    if api_v2.get("label") != API_V2_LABEL or api_v2.get("status") not in {
        "available",
        "not_available",
    }:
        raise ValueError("Final benchmark API v2 taxonomy is invalid.")
    if api_v2["status"] == "available":
        _validate_benchmark_taxonomy(api_v2)
    if _mapping(sections.get("rag_quality"), "RAG quality").get("label") != RAG_DEMO_LABEL:
        raise ValueError("Final benchmark RAG demo label is invalid.")
    environment_matrix = payload.get("environment_matrix")
    if not isinstance(environment_matrix, list) or len(environment_matrix) != 6:
        raise ValueError("Final benchmark environment matrix is invalid.")
    _validate_finite_values(payload)


# ADD 2026-08-21: Available benchmark section의 공통 taxonomy fields를 검증한다.
def _validate_benchmark_taxonomy(section: Mapping[str, Any]) -> None:
    _required_string(section.get("benchmark_name"), "section benchmark_name")
    if type(section.get("schema_version")) is not int or section["schema_version"] <= 0:
        raise ValueError("Final benchmark section schema_version is invalid.")
    environment = _mapping(section.get("environment"), "section environment")
    _required_string(environment.get("environment_id"), "section environment_id")
    _required_string(environment.get("device"), "section device")
    _required_string(section.get("measurement_boundary"), "section measurement_boundary")
    conditions = _mapping(section.get("conditions"), "section conditions")
    for field in ("warmup", "iterations", "batch_size"):
        if field not in conditions:
            raise ValueError(f"Final benchmark section conditions omit {field}.")
    _mapping(section.get("results"), "section results")
    _mapping(section.get("provenance"), "section provenance")


# ADD 2026-08-21: Vision provenance의 required/optional SHA fields를 검증한다.
def _validate_provenance(
    provenance: Mapping[str, Any],
    *,
    require_category: bool = True,
) -> None:
    if require_category:
        _required_string(provenance.get("category"), "provenance category")
    for field in ("manifest_sha256", "model_sha256"):
        value = provenance.get(field)
        if not isinstance(value, str) or not is_sha256_digest(value):
            raise ValueError(f"Benchmark {field} is invalid.")
    for field in ("threshold_artifact_sha256", "artifact_metadata_sha256"):
        value = provenance.get(field)
        if value is not None and not is_sha256_digest(value):
            raise ValueError(f"Benchmark {field} is invalid.")


# ADD 2026-08-21: JSON source file을 required object mapping으로 로드한다.
def _load_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}.") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return raw


# ADD 2026-08-21: Nested benchmark payload의 NaN/Inf values를 재귀적으로 거부한다.
def _validate_finite_values(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Benchmark payload contains a non-finite value.")
    if isinstance(value, dict):
        for nested in value.values():
            _validate_finite_values(nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_finite_values(nested)


# ADD 2026-08-21: Final benchmark identifier가 safe single path component인지 검증한다.
def _validate_identifier(value: object, field: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"Final benchmark {field} must be a safe 1-128 character identifier.")


# ADD 2026-08-21: Final benchmark timestamp가 timezone-aware ISO-8601인지 검증한다.
def _parse_aware_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Final benchmark created_at must be an ISO-8601 string.")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Final benchmark created_at must include a timezone offset.")
    return parsed


# ADD 2026-08-21: Required benchmark JSON string을 non-empty contract로 검증한다.
def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Benchmark {field} must be non-empty text.")
    return value.strip()


# ADD 2026-08-21: Benchmark JSON object field를 typed mapping으로 좁힌다.
def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Benchmark {field} must be an object.")
    return value


# ADD 2026-08-21: Git subprocess output을 provenance resolver에 제공한다.
def _run_git(arguments: Sequence[str], repository_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ValueError("Cannot resolve final benchmark Git repository provenance.") from exc
    return completed.stdout
