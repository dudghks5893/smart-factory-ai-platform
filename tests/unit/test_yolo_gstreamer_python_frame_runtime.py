"""Unit tests for canonical C6-3 Python GStreamer frame validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from services.streaming.yolo_gstreamer_python_frame_runtime import (
    EXPECTED_VALIDATION_ID,
    PythonFrameSourceConfig,
    PythonFrameValidationConfig,
    build_python_frame_pipeline,
    load_python_frame_validation_config,
)


def _valid_config(tmp_path: Path) -> PythonFrameValidationConfig:
    return PythonFrameValidationConfig(
        schema_version=1,
        validation_id=EXPECTED_VALIDATION_ID,
        source=PythonFrameSourceConfig(
            num_buffers=1,
            pattern="ball",
            width=320,
            height=240,
        ),
        timeout_seconds=5,
        output_root=Path("outputs/streaming/yolo_gstreamer/c6_3_python_frame"),
        config_path=tmp_path / "config.yaml",
    )


# ADD 2026-09-04: Canonical C6-3 Python frame config가 strict schema로 load되는지 검증한다.
def test_load_python_frame_validation_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "validation_id": EXPECTED_VALIDATION_ID,
                "source": {
                    "num_buffers": 1,
                    "pattern": "ball",
                    "width": 320,
                    "height": 240,
                },
                "timeout_seconds": 5,
                "output_root": "outputs/streaming/yolo_gstreamer/c6_3_python_frame",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = load_python_frame_validation_config(path)

    assert config.validation_id == EXPECTED_VALIDATION_ID
    assert config.source.width == 320
    assert config.source.height == 240
    assert config.timeout_seconds == 5


# ADD 2026-09-04: Unknown config field가 canonical evidence semantics를 바꾸지 못하게 차단한다.
def test_load_python_frame_validation_config_rejects_unknown_field(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "validation_id": EXPECTED_VALIDATION_ID,
                "source": {
                    "num_buffers": 1,
                    "pattern": "ball",
                    "width": 320,
                    "height": 240,
                },
                "timeout_seconds": 5,
                "output_root": "outputs/streaming/yolo_gstreamer/c6_3_python_frame",
                "unexpected": True,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fields"):
        load_python_frame_validation_config(path)


# ADD 2026-09-04: Source contract 변경을 validation 단계에서 fail-closed로 거부한다.
def test_python_frame_source_config_is_frozen(tmp_path: Path) -> None:
    config = _valid_config(tmp_path)
    config.validate()

    with pytest.raises(ValueError, match="exactly one"):
        PythonFrameSourceConfig(
            num_buffers=2,
            pattern="ball",
            width=320,
            height=240,
        ).validate()

    with pytest.raises(ValueError, match="pattern"):
        PythonFrameSourceConfig(
            num_buffers=1,
            pattern="smpte",
            width=320,
            height=240,
        ).validate()


# ADD 2026-09-04: Pipeline string이 BGR/appsink/latest-frame-wins contract를 유지하는지 검증한다.
def test_build_python_frame_pipeline_contract(tmp_path: Path) -> None:
    pipeline = build_python_frame_pipeline(_valid_config(tmp_path))

    assert "videotestsrc num-buffers=1 pattern=ball" in pipeline
    assert "video/x-raw,format=BGR,width=320,height=240" in pipeline
    assert "queue max-size-buffers=1" in pipeline
    assert "leaky=downstream" in pipeline
    assert "appsink name=framesink" in pipeline
    assert "max-buffers=1 drop=true sync=false wait-on-eos=false" in pipeline
