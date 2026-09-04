from __future__ import annotations

from dataclasses import replace

import pytest

from services.streaming.yolo_rtsp_reconnect_exhaustion_smoke import (
    DEFAULT_RTSP_RECONNECT_EXHAUSTION_CONFIG,
    EXPECTED_BACKOFF_MS,
    build_reconnect_exhaustion_plan,
    load_rtsp_reconnect_exhaustion_config,
)


def test_repository_reconnect_exhaustion_config_loads() -> None:
    config = load_rtsp_reconnect_exhaustion_config(DEFAULT_RTSP_RECONNECT_EXHAUSTION_CONFIG)

    assert config.smoke_id == "c6_4c_rtsp_reconnect_exhaustion_v1"
    assert config.required_c6_4b_acceptance_commit == "028c86264aef6859ca4b68cf5d561f26fa341f95"
    assert (
        config.required_c6_4b_smoke_sha256
        == "f6f140bfee3ac6cf71ab71f5e03c8043ad539f9a775cce6146f593f6b209683e"
    )
    assert (
        config.required_c6_4b_archive_sha256
        == "a6497cf91438981c9fbf23f65e391f030d1b19cfc09815d918e32dc163b1ab17"
    )
    assert config.runtime.expected_max_reconnect_attempts == 5
    assert config.runtime.expected_backoff_ms == EXPECTED_BACKOFF_MS
    assert config.runtime.keep_fixture_offline_after_fault is True
    assert config.scope.final_test_used is False


def test_exhaustion_runtime_rejects_backoff_mutation() -> None:
    config = load_rtsp_reconnect_exhaustion_config(DEFAULT_RTSP_RECONNECT_EXHAUSTION_CONFIG)
    mutated = replace(
        config.runtime,
        expected_backoff_ms=(500, 1000, 2000, 4000, 7000),
    )

    with pytest.raises(ValueError, match="backoff schedule"):
        mutated.validate()


def test_exhaustion_runtime_rejects_attempt_budget_mutation() -> None:
    config = load_rtsp_reconnect_exhaustion_config(DEFAULT_RTSP_RECONNECT_EXHAUSTION_CONFIG)
    mutated = replace(config.runtime, expected_max_reconnect_attempts=4)

    with pytest.raises(ValueError, match="attempt boundary"):
        mutated.validate()


def test_reconnect_exhaustion_plan_matches_frozen_policy() -> None:
    assert build_reconnect_exhaustion_plan() == (
        (1, 500),
        (2, 1000),
        (3, 2000),
        (4, 4000),
        (5, 8000),
    )


def test_scope_rejects_external_camera_or_final_test() -> None:
    config = load_rtsp_reconnect_exhaustion_config(DEFAULT_RTSP_RECONNECT_EXHAUSTION_CONFIG)

    with pytest.raises(ValueError, match="scope changed"):
        replace(config.scope, external_camera_used=True).validate()

    with pytest.raises(ValueError, match="scope changed"):
        replace(config.scope, final_test_used=True).validate()


def test_fixture_must_stay_offline_after_fault() -> None:
    config = load_rtsp_reconnect_exhaustion_config(DEFAULT_RTSP_RECONNECT_EXHAUSTION_CONFIG)

    with pytest.raises(ValueError, match="remain offline"):
        replace(config.runtime, keep_fixture_offline_after_fault=False).validate()
