"""Deterministic offline providers for the public demo RAG evaluation."""

from __future__ import annotations

import re
from collections.abc import Sequence

from services.rag.providers import GeneratedAnswer, GenerationContext

DEMO_EMBEDDING_PROVIDER = "evaluation-demo-semantic"
DEMO_EMBEDDING_MODEL = "evaluation-demo-semantic-v1"
EXTRACTIVE_GENERATOR_PROVIDER = "evaluation-deterministic"
EXTRACTIVE_GENERATOR_MODEL = "extractive-citation-v1"

_TOKEN_PATTERN = re.compile(r"[a-z]+|[가-힣]+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")
_CONCEPT_GROUPS = (
    frozenset(
        {
            "camera",
            "lens",
            "lighting",
            "illumination",
            "image",
            "blurred",
            "focus",
            "optical",
            "카메라",
            "렌즈",
            "조명",
            "이미지",
        }
    ),
    frozenset(
        {
            "anomaly",
            "quarantine",
            "containment",
            "accepted",
            "affected",
            "격리",
            "이상",
            "차단",
        }
    ),
    frozenset(
        {
            "reinspection",
            "reinspect",
            "rotate",
            "orientation",
            "second",
            "original",
            "재검사",
            "회전",
            "원본",
        }
    ),
    frozenset(
        {
            "escalate",
            "escalation",
            "reviewer",
            "maintenance",
            "notify",
            "human",
            "에스컬레이션",
            "검토자",
            "정비",
            "알림",
        }
    ),
    frozenset(
        {
            "record",
            "records",
            "identifier",
            "identifiers",
            "timestamps",
            "score",
            "scores",
            "lineage",
            "기록",
            "식별자",
            "점수",
        }
    ),
    frozenset(
        {
            "equipment",
            "bracket",
            "unstable",
            "cleaning",
            "secure",
            "dust",
            "residue",
            "장비",
            "브래킷",
            "청소",
            "불안정",
        }
    ),
    frozenset(
        {
            "defect",
            "classification",
            "screening",
            "prediction",
            "confirmed",
            "결함",
            "분류",
            "판정",
            "예측",
        }
    ),
    frozenset(
        {
            "release",
            "disposition",
            "discard",
            "rework",
            "accepted",
            "해제",
            "폐기",
            "재작업",
        }
    ),
    frozenset(
        {
            "inspection",
            "inspect",
            "metal",
            "nut",
            "product",
            "item",
            "검사",
            "제품",
            "너트",
        }
    ),
    frozenset(
        {
            "threshold",
            "model",
            "configured",
            "change",
            "임계값",
            "모델",
            "변경",
        }
    ),
    frozenset(
        {
            "procedure",
            "factory",
            "quality",
            "demo",
            "demonstration",
            "document",
            "checklist",
            "절차",
            "공장",
            "품질",
        }
    ),
    frozenset(
        {
            "torque",
            "temperature",
            "conveyor",
            "speed",
            "password",
            "토크",
            "온도",
            "컨베이어",
            "속도",
        }
    ),
)


class DemoSemanticEmbeddingProvider:
    """Evaluation-only bilingual concept vectorizer for the public demo corpus."""

    provider_name = DEMO_EMBEDDING_PROVIDER
    model_name = DEMO_EMBEDDING_MODEL

    # ADD 2026-08-21: Demo documents를 고정 bilingual concept vector space로 변환한다.
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [self._vector(text) for text in texts]

    # ADD 2026-08-21: Demo evaluation question을 동일한 concept vector space로 변환한다.
    def embed_query(self, text: str) -> Sequence[float]:
        return self._vector(text)

    # ADD 2026-08-21: Case ID와 무관한 fixed vocabulary의 binary concept vector를 계산한다.
    def _vector(self, text: str) -> tuple[float, ...]:
        tokens = set(_TOKEN_PATTERN.findall(text.lower()))
        vector = tuple(float(bool(tokens & group)) for group in _CONCEPT_GROUPS)
        if not any(vector):
            raise ValueError("Demo evaluation text contains no supported semantic concepts.")
        return vector


class ExtractiveCitationGenerator:
    """Evaluation-only generator that quotes one source sentence per retrieved context."""

    provider_name = EXTRACTIVE_GENERATOR_PROVIDER
    model_name = EXTRACTIVE_GENERATOR_MODEL

    # ADD 2026-08-21: Evaluation generator call count를 process-local state로 초기화한다.
    def __init__(self) -> None:
        self.call_count = 0

    # ADD 2026-08-21: Retrieved sentence를 추출해 deterministic citation answer를 만든다.
    # MODIFY 2026-08-21: Source 줄바꿈 보존 → claim당 한 줄이 되도록 whitespace를 정규화한다.
    def generate(
        self,
        question: str,
        contexts: Sequence[GenerationContext],
    ) -> GeneratedAnswer:
        if not contexts:
            raise ValueError("Extractive evaluation generation requires retrieved contexts.")
        self.call_count += 1
        claims = []
        citation_ids = []
        for context in contexts:
            sentence = " ".join(
                _SENTENCE_BOUNDARY.split(context.text.strip(), maxsplit=1)[0].split()
            )
            if not sentence:
                raise ValueError("Retrieved context contains no extractive sentence.")
            claims.append(f"{sentence} [{context.citation_id}]")
            citation_ids.append(context.citation_id)
        return GeneratedAnswer(answer="\n".join(claims), citation_ids=tuple(citation_ids))
