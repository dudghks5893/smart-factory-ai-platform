"""Deterministic offline RAG evaluation metrics, lineage, and artifacts."""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from services.rag.generation import RagCitation, validate_generated_answer
from services.rag.index import METADATA_FILENAME, RagIndex, load_rag_index
from services.rag.providers import AnswerGenerator, EmbeddingProvider, GenerationContext
from services.rag.retrieval import ExactCosineRetriever, RetrievalResult
from shared.hashing import is_sha256_digest, sha256_file

RAG_EVALUATION_SCHEMA_VERSION = 1
EVALUATION_FILENAME = "evaluation.json"
CASES_FILENAME = "cases.jsonl"
DEFAULT_RAG_EVALUATION_ROOT = Path("outputs/evaluation/rag")
_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_CITATION_MARKER = re.compile(r"\[(C[1-9][0-9]*)\]")


@dataclass(frozen=True)
class ReferenceFact:
    """Reference correctness criterion kept separate from groundedness."""

    fact_id: str
    text: str
    required_terms: tuple[str, ...]


@dataclass(frozen=True)
class RagEvaluationCase:
    """One answerable or unanswerable public evaluation question."""

    case_id: str
    question: str
    expected_document_ids: tuple[str, ...]
    expected_chunk_ids: tuple[str, ...]
    reference_facts: tuple[ReferenceFact, ...]
    answerable: bool
    language: str


@dataclass(frozen=True)
class RagEvaluationDataset:
    """Validated ordered cases plus source file identity."""

    path: Path
    sha256: str
    cases: tuple[RagEvaluationCase, ...]


@dataclass(frozen=True)
class RagEvaluationConfig:
    """Retrieval cutoffs and operational threshold used by one evaluation."""

    top_k_values: tuple[int, ...] = (1, 3, 5)
    minimum_retrieval_score: float = 0.2

    # ADD 2026-08-21: Recall cutoffs와 threshold가 finite/bounded인지 검증한다.
    def validate(self) -> None:
        if (
            not self.top_k_values
            or tuple(sorted(set(self.top_k_values))) != self.top_k_values
            or not all(1 <= value <= 100 for value in self.top_k_values)
        ):
            raise ValueError(
                "RAG evaluation top_k values must be sorted unique integers in [1, 100]."
            )
        if (
            not math.isfinite(self.minimum_retrieval_score)
            or not -1 <= self.minimum_retrieval_score <= 1
        ):
            raise ValueError("RAG evaluation minimum score must be finite and in [-1, 1].")


@dataclass(frozen=True)
class CitationMetrics:
    """Exact expected-chunk citation precision and recall."""

    precision: float
    recall: float


@dataclass(frozen=True)
class FaithfulnessMetrics:
    """Lexically supported cited-claim ratio for deterministic baseline answers."""

    score: float
    supported_claims: int
    total_claims: int


@dataclass(frozen=True)
class RagEvaluationResult:
    """Committed evaluation artifact and aggregate metric payload."""

    output_dir: Path
    evaluation_path: Path
    cases_path: Path
    metrics: Mapping[str, Any]


# ADD 2026-08-21: JSONL evaluation dataset을 strict schema와 stable ordering으로 로드한다.
def load_rag_evaluation_dataset(path: Path) -> RagEvaluationDataset:
    """Load public evaluation cases and reject malformed or duplicate records."""
    if not path.is_file():
        raise FileNotFoundError(f"RAG evaluation dataset not found: {path}")
    cases: list[RagEvaluationCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"RAG evaluation dataset has a blank line at {line_number}.")
        try:
            raw = json.loads(line)
            cases.append(_parse_evaluation_case(raw))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid RAG evaluation case at line {line_number}.") from exc
    if not cases:
        raise ValueError("RAG evaluation dataset must not be empty.")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("RAG evaluation case_id values must be unique.")
    if case_ids != sorted(case_ids):
        raise ValueError("RAG evaluation cases must be sorted by case_id.")
    return RagEvaluationDataset(path=path, sha256=sha256_file(path), cases=tuple(cases))


# ADD 2026-08-21: Expected evidence가 loaded immutable index에 실제로 존재하는지 검증한다.
def validate_evaluation_references(dataset: RagEvaluationDataset, index: RagIndex) -> None:
    """Reject dataset references that do not belong to the evaluated index lineage."""
    document_ids = {chunk.document_id for chunk in index.chunks}
    chunks_by_id = {chunk.chunk_id: chunk for chunk in index.chunks}
    for case in dataset.cases:
        unknown_documents = set(case.expected_document_ids) - document_ids
        unknown_chunks = set(case.expected_chunk_ids) - set(chunks_by_id)
        if unknown_documents or unknown_chunks:
            raise ValueError(f"Evaluation case {case.case_id} references unknown index evidence.")
        if any(
            chunks_by_id[chunk_id].document_id not in case.expected_document_ids
            for chunk_id in case.expected_chunk_ids
        ):
            raise ValueError(
                f"Evaluation case {case.case_id} expected chunk/document references disagree."
            )


# ADD 2026-08-21: Expected evidence set의 case-level fractional Recall@K를 계산한다.
def calculate_recall_at_k(
    expected_ids: Sequence[str],
    retrieved_ids: Sequence[str],
    k: int,
) -> float:
    """Return the fraction of unique expected evidence present in the first K results."""
    expected = set(expected_ids)
    if not expected:
        raise ValueError("Recall@K requires at least one expected evidence identifier.")
    if k <= 0:
        raise ValueError("Recall@K requires a positive K.")
    return len(expected & set(retrieved_ids[:k])) / len(expected)


# ADD 2026-08-21: Cited chunks와 expected chunks에서 citation precision/recall을 계산한다.
def calculate_citation_metrics(
    expected_chunk_ids: Sequence[str],
    citations: Sequence[RagCitation],
) -> CitationMetrics:
    """Treat exact expected chunks as relevant citation evidence."""
    expected = set(expected_chunk_ids)
    if not expected:
        raise ValueError("Citation metrics require expected chunks for an answerable case.")
    cited = {citation.chunk_id for citation in citations}
    relevant = expected & cited
    return CitationMetrics(
        precision=len(relevant) / len(cited) if cited else 0.0,
        recall=len(relevant) / len(expected),
    )


# ADD 2026-08-21: Answer line별 citation source가 claim text를 직접 포함하는지 평가한다.
def calculate_deterministic_faithfulness(
    answer: str,
    citations: Sequence[RagCitation],
    index: RagIndex,
) -> FaithfulnessMetrics:
    """Measure lexical grounding without conflating it with answer correctness."""
    citations_by_id = {citation.citation_id: citation for citation in citations}
    chunks_by_id = {chunk.chunk_id: chunk for chunk in index.chunks}
    claims = tuple(line.strip() for line in answer.splitlines() if line.strip())
    if not claims:
        return FaithfulnessMetrics(score=0.0, supported_claims=0, total_claims=0)
    supported = 0
    for claim in claims:
        marker_ids = _CITATION_MARKER.findall(claim)
        claim_text = _normalize_claim(_CITATION_MARKER.sub("", claim))
        source_texts = []
        for citation_id in marker_ids:
            citation = citations_by_id.get(citation_id)
            chunk = chunks_by_id.get(citation.chunk_id) if citation is not None else None
            if chunk is not None:
                source_texts.append(_normalize_claim(chunk.text))
        if claim_text and any(claim_text in source_text for source_text in source_texts):
            supported += 1
    return FaithfulnessMetrics(
        score=supported / len(claims),
        supported_claims=supported,
        total_claims=len(claims),
    )


# ADD 2026-08-21: Reference fact term의 answer 포함률을 correctness diagnostic으로 계산한다.
def calculate_reference_fact_recall(
    answer: str,
    reference_facts: Sequence[ReferenceFact],
) -> float:
    """Measure deterministic reference-fact coverage separately from faithfulness."""
    if not reference_facts:
        raise ValueError("Reference fact recall requires answerable-case facts.")
    normalized_answer = _normalize_claim(answer)
    covered = sum(
        all(_normalize_claim(term) in normalized_answer for term in fact.required_terms)
        for fact in reference_facts
    )
    return covered / len(reference_facts)


# ADD 2026-08-21: Existing index/dataset 평가를 immutable artifact로 atomic commit한다.
def evaluate_rag(
    *,
    index_dir: Path,
    dataset_path: Path,
    output_root: Path,
    evaluation_id: str,
    embedding_provider: EmbeddingProvider,
    answer_generator: AnswerGenerator,
    config: RagEvaluationConfig,
    created_at: str,
) -> RagEvaluationResult:
    """Evaluate retrieval, citations, faithfulness, and abstention without rebuilding the index."""
    _validate_identifier(evaluation_id, "evaluation_id")
    _parse_aware_datetime(created_at)
    config.validate()
    final_dir = output_root / evaluation_id
    if final_dir.exists():
        raise FileExistsError(f"RAG evaluation artifact already exists: {final_dir}")

    # Existing immutable index와 versioned dataset을 먼저 load하고 cross-reference를 검증한다.
    index = load_rag_index(index_dir)
    dataset = load_rag_evaluation_dataset(dataset_path)
    validate_evaluation_references(dataset, index)
    if (
        embedding_provider.provider_name != index.embedding_provider
        or embedding_provider.model_name != index.embedding_model
    ):
        raise ValueError("Evaluation embedding provider/model does not match the index lineage.")
    maximum_k = max(config.top_k_values)
    if maximum_k > 100:
        raise ValueError("RAG evaluation maximum K exceeds the retrieval contract.")
    raw_retriever = ExactCosineRetriever(
        index=index,
        embedding_provider=embedding_provider,
        max_top_k=min(len(index.chunks), 100),
        minimum_score=-1.0,
    )

    # 각 case의 raw ranking, thresholded evidence와 deterministic answer metric을 계산한다.
    case_records = tuple(
        _evaluate_case(
            case,
            index=index,
            retriever=raw_retriever,
            answer_generator=answer_generator,
            config=config,
        )
        for case in dataset.cases
    )
    metrics = _aggregate_metrics(case_records, config.top_k_values)
    score_analysis = _score_analysis(case_records)
    lineage = _load_index_lineage(index_dir)

    # Cases hash를 summary에 연결하고 full reload validation 뒤 final directory로 rename한다.
    output_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{evaluation_id}.tmp-", dir=output_root))
    try:
        cases_path = temp_dir / CASES_FILENAME
        cases_path.write_text(
            "".join(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
                for record in case_records
            ),
            encoding="utf-8",
        )
        summary = {
            "schema_version": RAG_EVALUATION_SCHEMA_VERSION,
            "evaluation_id": evaluation_id,
            "created_at": created_at,
            "dataset": {
                "filename": dataset_path.name,
                "sha256": dataset.sha256,
                "case_count": len(dataset.cases),
            },
            "index_lineage": lineage,
            "configuration": {
                "top_k_values": list(config.top_k_values),
                "minimum_retrieval_score": config.minimum_retrieval_score,
                "generator_provider": answer_generator.provider_name,
                "generator_model": answer_generator.model_name,
                "faithfulness_evaluator": "deterministic-extractive-support-v1",
                "external_judge": None,
            },
            "metrics": metrics,
            "score_analysis": score_analysis,
            "artifacts": {"cases_sha256": sha256_file(cases_path)},
        }
        evaluation_path = temp_dir / EVALUATION_FILENAME
        evaluation_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        load_rag_evaluation_artifact(temp_dir)
        temp_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return RagEvaluationResult(
        output_dir=final_dir,
        evaluation_path=final_dir / EVALUATION_FILENAME,
        cases_path=final_dir / CASES_FILENAME,
        metrics=metrics,
    )


# ADD 2026-08-21: Evaluation summary/cases hash, count와 finite metric을 fail-fast 검증한다.
def load_rag_evaluation_artifact(
    artifact_dir: Path,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    """Load one immutable evaluation result and validate its reproducibility contract."""
    evaluation_path = artifact_dir / EVALUATION_FILENAME
    cases_path = artifact_dir / CASES_FILENAME
    try:
        root = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if not isinstance(root, dict):
            raise TypeError("evaluation summary must be an object")
        if (
            type(root["schema_version"]) is not int
            or root["schema_version"] != RAG_EVALUATION_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported RAG evaluation schema_version.")
        _validate_identifier(root["evaluation_id"], "evaluation_id")
        _parse_aware_datetime(root["created_at"])
        dataset = _mapping(root["dataset"], "dataset")
        artifacts = _mapping(root["artifacts"], "artifacts")
        if not is_sha256_digest(dataset["sha256"]):
            raise ValueError("RAG evaluation dataset SHA is invalid.")
        if not is_sha256_digest(artifacts["cases_sha256"]):
            raise ValueError("RAG evaluation cases SHA is invalid.")
        if sha256_file(cases_path) != artifacts["cases_sha256"]:
            raise ValueError("RAG evaluation cases artifact hash mismatch.")
        case_records = _read_case_records(cases_path)
        if type(dataset["case_count"]) is not int or dataset["case_count"] != len(case_records):
            raise ValueError("RAG evaluation case count mismatch.")
        case_ids = [record.get("case_id") for record in case_records]
        if any(not isinstance(case_id, str) for case_id in case_ids) or len(case_ids) != len(
            set(case_ids)
        ):
            raise ValueError("RAG evaluation artifact case IDs are invalid.")
        _validate_finite_values(root)
        _validate_index_lineage(_mapping(root["index_lineage"], "index_lineage"))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("Cannot load RAG evaluation artifact.") from exc
    return root, case_records


# ADD 2026-08-21: One case의 retrieval/generation/evidence metrics를 deterministic record로 만든다.
def _evaluate_case(
    case: RagEvaluationCase,
    *,
    index: RagIndex,
    retriever: ExactCosineRetriever,
    answer_generator: AnswerGenerator,
    config: RagEvaluationConfig,
) -> dict[str, Any]:
    raw_results = retriever.retrieve(case.question, top_k=min(len(index.chunks), 100))
    selected = tuple(
        result for result in raw_results if result.score >= config.minimum_retrieval_score
    )[: max(config.top_k_values)]
    generated_called = False
    citations: tuple[RagCitation, ...] = ()
    if selected:
        contexts = tuple(
            GenerationContext(
                citation_id=f"C{rank}",
                chunk_id=result.chunk.chunk_id,
                title=result.chunk.title,
                section=result.chunk.section,
                source_path=result.chunk.source_path,
                page=result.chunk.page,
                text=result.chunk.text,
            )
            for rank, result in enumerate(selected, start=1)
        )
        generated_called = True
        generated = answer_generator.generate(case.question, contexts)
        citation_ids = validate_generated_answer(generated, contexts)
        result_by_citation = dict(
            zip((context.citation_id for context in contexts), selected, strict=True)
        )
        citations = tuple(
            _rag_citation(citation_id, result_by_citation[citation_id])
            for citation_id in citation_ids
        )
        status = "answered"
        answer = generated.answer
    else:
        status = "insufficient_context"
        answer = "The supplied SOP does not contain enough information to answer this question."

    retrieved_document_ids = [result.chunk.document_id for result in selected]
    retrieved_chunk_ids = [result.chunk.chunk_id for result in selected]
    document_recall = (
        {
            str(k): calculate_recall_at_k(
                case.expected_document_ids,
                retrieved_document_ids,
                k,
            )
            for k in config.top_k_values
        }
        if case.answerable
        else None
    )
    chunk_recall = (
        {
            str(k): calculate_recall_at_k(case.expected_chunk_ids, retrieved_chunk_ids, k)
            for k in config.top_k_values
        }
        if case.answerable
        else None
    )
    citation_metrics = (
        calculate_citation_metrics(case.expected_chunk_ids, citations) if case.answerable else None
    )
    faithfulness = (
        calculate_deterministic_faithfulness(answer, citations, index)
        if status == "answered"
        else None
    )
    fact_recall = (
        calculate_reference_fact_recall(answer, case.reference_facts)
        if case.answerable and status == "answered"
        else 0.0
        if case.answerable
        else None
    )
    abstention_correct = (
        status == "insufficient_context" and not generated_called and not citations
        if not case.answerable
        else None
    )
    expected_ranks = {
        chunk_id: next(
            (result.rank for result in raw_results if result.chunk.chunk_id == chunk_id),
            None,
        )
        for chunk_id in case.expected_chunk_ids
    }
    reciprocal_rank = (
        1.0 / min(rank for rank in expected_ranks.values() if rank is not None)
        if case.answerable and any(rank is not None for rank in expected_ranks.values())
        else 0.0
        if case.answerable
        else None
    )
    answerability_correct = (status == "answered") if case.answerable else bool(abstention_correct)
    return {
        "case_id": case.case_id,
        "question": case.question,
        "language": case.language,
        "answerable": case.answerable,
        "expected_document_ids": list(case.expected_document_ids),
        "expected_chunk_ids": list(case.expected_chunk_ids),
        "reference_facts": [asdict(fact) for fact in case.reference_facts],
        "raw_top_score": raw_results[0].score if raw_results else None,
        "expected_chunk_ranks": expected_ranks,
        "retrieval": [_retrieval_record(result) for result in selected],
        "answer_status": status,
        "answer": answer,
        "generator_called": generated_called,
        "citations": [asdict(citation) for citation in citations],
        "metrics": {
            "document_recall_at_k": document_recall,
            "chunk_recall_at_k": chunk_recall,
            "reciprocal_rank": reciprocal_rank,
            "citation_precision": citation_metrics.precision if citation_metrics else None,
            "citation_recall": citation_metrics.recall if citation_metrics else None,
            "faithfulness": faithfulness.score if faithfulness else None,
            "supported_claims": faithfulness.supported_claims if faithfulness else None,
            "total_claims": faithfulness.total_claims if faithfulness else None,
            "reference_fact_recall": fact_recall,
            "abstention_correct": abstention_correct,
            "answerability_correct": answerability_correct,
        },
    }


# ADD 2026-08-21: Case-level metric을 answerable macro average와 abstention accuracy로 집계한다.
def _aggregate_metrics(
    case_records: Sequence[Mapping[str, Any]],
    top_k_values: Sequence[int],
) -> dict[str, Any]:
    answerable = [record for record in case_records if record["answerable"]]
    unanswerable = [record for record in case_records if not record["answerable"]]
    document_recall = {
        str(k): _mean([record["metrics"]["document_recall_at_k"][str(k)] for record in answerable])
        for k in top_k_values
    }
    chunk_recall = {
        str(k): _mean([record["metrics"]["chunk_recall_at_k"][str(k)] for record in answerable])
        for k in top_k_values
    }
    return {
        "aggregation": "case_macro_answerable",
        "answerable_case_count": len(answerable),
        "unanswerable_case_count": len(unanswerable),
        "document_recall_at_k": document_recall,
        "chunk_recall_at_k": chunk_recall,
        "mean_reciprocal_rank": _mean(
            [record["metrics"]["reciprocal_rank"] for record in answerable]
        ),
        "citation_precision": _mean(
            [record["metrics"]["citation_precision"] for record in answerable]
        ),
        "citation_recall": _mean([record["metrics"]["citation_recall"] for record in answerable]),
        "faithfulness": _mean(
            [
                record["metrics"]["faithfulness"]
                for record in answerable
                if record["metrics"]["faithfulness"] is not None
            ]
        ),
        "reference_fact_recall": _mean(
            [record["metrics"]["reference_fact_recall"] for record in answerable]
        ),
        "unanswerable_abstention_accuracy": _mean(
            [float(record["metrics"]["abstention_correct"]) for record in unanswerable]
        ),
        "answerability_accuracy": _mean(
            [float(record["metrics"]["answerability_correct"]) for record in case_records]
        ),
    }


# ADD 2026-08-21: Answerable/unanswerable raw top-score를 calibration evidence로 요약한다.
def _score_analysis(case_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "answerable_raw_top_score": _distribution(
            [
                record["raw_top_score"]
                for record in case_records
                if record["answerable"] and record["raw_top_score"] is not None
            ]
        ),
        "unanswerable_raw_top_score": _distribution(
            [
                record["raw_top_score"]
                for record in case_records
                if not record["answerable"] and record["raw_top_score"] is not None
            ]
        ),
        "policy": "analysis_only_no_automatic_threshold_update",
    }


# ADD 2026-08-21: Index metadata SHA와 corpus/chunk/embedding lineage를 복사한다.
def _load_index_lineage(index_dir: Path) -> dict[str, Any]:
    metadata_path = index_dir / METADATA_FILENAME
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("RAG index metadata must be an object.")
    return {
        "index_id": raw["index_id"],
        "metadata_sha256": sha256_file(metadata_path),
        "document_count": raw["document_count"],
        "chunk_count": raw["chunk_count"],
        "documents": raw["documents"],
        "chunking": raw["chunking"],
        "embedding": raw["embedding"],
        "index_artifacts": raw["artifacts"],
    }


# ADD 2026-08-21: Dataset JSON object를 typed evaluation case로 strict parsing한다.
def _parse_evaluation_case(raw: object) -> RagEvaluationCase:
    expected_fields = {
        "case_id",
        "question",
        "expected_document_ids",
        "expected_chunk_ids",
        "reference_facts",
        "answerable",
        "language",
    }
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise ValueError("RAG evaluation case fields do not match the schema.")
    case_id = _required_string(raw["case_id"], "case_id")
    _validate_identifier(case_id, "case_id")
    question = _required_string(raw["question"], "question")
    if len(question) > 2000:
        raise ValueError("RAG evaluation question exceeds 2000 characters.")
    answerable = raw["answerable"]
    if type(answerable) is not bool:
        raise TypeError("RAG evaluation answerable must be boolean.")
    language = _required_string(raw["language"], "language")
    if language not in {"en", "ko"}:
        raise ValueError("RAG evaluation language must be en or ko.")
    document_ids = _string_tuple(raw["expected_document_ids"], "expected_document_ids")
    chunk_ids = _string_tuple(raw["expected_chunk_ids"], "expected_chunk_ids")
    facts_raw = raw["reference_facts"]
    if not isinstance(facts_raw, list):
        raise TypeError("RAG evaluation reference_facts must be an array.")
    facts = tuple(_parse_reference_fact(item) for item in facts_raw)
    if answerable and (not document_ids or not chunk_ids or not facts):
        raise ValueError("Answerable RAG evaluation cases require evidence and reference facts.")
    if not answerable and (document_ids or chunk_ids or facts):
        raise ValueError("Unanswerable RAG evaluation cases must not declare expected evidence.")
    return RagEvaluationCase(
        case_id=case_id,
        question=question,
        expected_document_ids=document_ids,
        expected_chunk_ids=chunk_ids,
        reference_facts=facts,
        answerable=answerable,
        language=language,
    )


# ADD 2026-08-21: Reference fact ID/text/required terms를 strict typed contract로 parsing한다.
def _parse_reference_fact(raw: object) -> ReferenceFact:
    if not isinstance(raw, dict) or set(raw) != {"fact_id", "text", "required_terms"}:
        raise ValueError("RAG reference fact fields do not match the schema.")
    terms = _string_tuple(raw["required_terms"], "required_terms")
    if not terms:
        raise ValueError("RAG reference fact requires at least one term.")
    return ReferenceFact(
        fact_id=_required_string(raw["fact_id"], "fact_id"),
        text=_required_string(raw["text"], "text"),
        required_terms=terms,
    )


# ADD 2026-08-21: Retrieval result를 full text 없는 evaluation evidence record로 변환한다.
def _retrieval_record(result: RetrievalResult) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "score": result.score,
        "chunk_id": result.chunk.chunk_id,
        "document_id": result.chunk.document_id,
        "title": result.chunk.title,
        "section": result.chunk.section,
        "source_path": result.chunk.source_path,
        "page": result.chunk.page,
    }


# ADD 2026-08-21: Controlled citation ID를 실제 retrieved chunk provenance에 연결한다.
def _rag_citation(citation_id: str, result: RetrievalResult) -> RagCitation:
    return RagCitation(
        citation_id=citation_id,
        chunk_id=result.chunk.chunk_id,
        document_id=result.chunk.document_id,
        title=result.chunk.title,
        section=result.chunk.section,
        page=result.chunk.page,
        source_path=result.chunk.source_path,
        retrieval_score=result.score,
    )


# ADD 2026-08-21: Evaluation cases JSONL을 strict object records로 복원한다.
def _read_case_records(path: Path) -> tuple[Mapping[str, Any], ...]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            raise ValueError("RAG evaluation cases artifact contains a blank line.")
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise TypeError("RAG evaluation case artifact must contain JSON objects.")
        records.append(raw)
    if not records:
        raise ValueError("RAG evaluation cases artifact must not be empty.")
    return tuple(records)


# ADD 2026-08-21: Summary의 index lineage identifiers와 SHA fields를 검증한다.
def _validate_index_lineage(lineage: Mapping[str, Any]) -> None:
    _validate_identifier(lineage["index_id"], "index_id")
    if not is_sha256_digest(lineage["metadata_sha256"]):
        raise ValueError("RAG evaluation index metadata SHA is invalid.")
    if type(lineage["document_count"]) is not int or lineage["document_count"] <= 0:
        raise ValueError("RAG evaluation index document count is invalid.")
    if type(lineage["chunk_count"]) is not int or lineage["chunk_count"] <= 0:
        raise ValueError("RAG evaluation index chunk count is invalid.")


# ADD 2026-08-21: Nested evaluation payload에서 NaN/Inf metric을 재귀적으로 거부한다.
def _validate_finite_values(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("RAG evaluation artifact contains a non-finite number.")
    if isinstance(value, dict):
        for nested in value.values():
            _validate_finite_values(nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_finite_values(nested)


# ADD 2026-08-21: Score vector의 count/min/mean/max를 deterministic finite summary로 계산한다.
def _distribution(values: Sequence[float]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "minimum": None, "mean": None, "maximum": None}
    if not all(math.isfinite(value) for value in values):
        raise ValueError("RAG evaluation score distribution contains non-finite values.")
    return {
        "count": len(values),
        "minimum": min(values),
        "mean": sum(values) / len(values),
        "maximum": max(values),
    }


# ADD 2026-08-21: Non-empty finite numeric values의 arithmetic mean을 계산한다.
def _mean(values: Sequence[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("RAG evaluation mean requires non-empty finite values.")
    return sum(values) / len(values)


# ADD 2026-08-21: Human-readable claim 비교를 위해 case/whitespace를 정규화한다.
def _normalize_claim(value: str) -> str:
    return " ".join(value.lower().split()).strip()


# ADD 2026-08-21: Evaluation/index identifier가 safe single path segment인지 검증한다.
def _validate_identifier(value: object, field: str) -> None:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"RAG evaluation {field} must be a safe 1-128 character identifier.")


# ADD 2026-08-21: Evaluation artifact timestamp가 timezone-aware ISO-8601인지 검증한다.
def _parse_aware_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("RAG evaluation created_at must be an ISO-8601 string.")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("RAG evaluation created_at must include a timezone offset.")
    return parsed


# ADD 2026-08-21: JSON field가 non-empty unique string array인지 검증한다.
def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise TypeError(f"RAG evaluation {field} must be a string array.")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"RAG evaluation {field} must contain unique values.")
    return result


# ADD 2026-08-21: Required JSON string field를 non-empty contract로 검증한다.
def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"RAG evaluation {field} must be non-empty text.")
    return value.strip()


# ADD 2026-08-21: Artifact JSON object field를 typed mapping으로 좁힌다.
def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"RAG evaluation {field} must be an object.")
    return value
