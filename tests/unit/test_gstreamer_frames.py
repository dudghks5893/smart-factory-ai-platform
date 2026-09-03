"""Unit tests for the C6 GStreamer-to-NumPy frame adapter."""

from __future__ import annotations

import numpy as np
import pytest

from services.streaming.gstreamer_frames import (
    BgrFrameSpec,
    bgr_buffer_to_numpy,
    validate_bgr_numpy_frame,
)


# ADD 2026-09-04: Packed BGR bytes가 owned uint8 HWC contiguous frame으로 복사되는지 검증한다.
def test_bgr_buffer_to_numpy_packed_frame() -> None:
    source = bytes(range(18))

    frame = bgr_buffer_to_numpy(source, width=3, height=2)

    assert frame.shape == (2, 3, 3)
    assert frame.dtype == np.uint8
    assert frame.flags.c_contiguous
    assert frame.flags.owndata
    assert frame.reshape(-1).tolist() == list(range(18))


# ADD 2026-09-04: GstVideo stride padding을 제거하고 pixel bytes만 보존하는지 검증한다.
def test_bgr_buffer_to_numpy_removes_row_padding() -> None:
    first_row = bytes([1, 2, 3, 4, 5, 6, 99, 99])
    second_row = bytes([7, 8, 9, 10, 11, 12, 88, 88])

    frame = bgr_buffer_to_numpy(
        first_row + second_row,
        width=2,
        height=2,
        stride=8,
    )

    assert frame.tolist() == [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ]
    assert frame.flags.c_contiguous


# ADD 2026-09-04: Declared stride보다 작은 native buffer를 fail-closed로 거부한다.
def test_bgr_buffer_to_numpy_rejects_short_buffer() -> None:
    with pytest.raises(ValueError, match="smaller"):
        bgr_buffer_to_numpy(bytes(15), width=2, height=2, stride=8)


# ADD 2026-09-04: BGR frame contract가 dtype, shape, contiguous 조건을 강제하는지 검증한다.
def test_validate_bgr_numpy_frame_contract() -> None:
    valid = np.zeros((2, 3, 3), dtype=np.uint8)
    validate_bgr_numpy_frame(valid, width=3, height=2)

    with pytest.raises(ValueError, match="uint8"):
        validate_bgr_numpy_frame(valid.astype(np.float32), width=3, height=2)

    with pytest.raises(ValueError, match="shape"):
        validate_bgr_numpy_frame(valid, width=2, height=2)

    non_contiguous = valid[:, ::-1, :]
    assert not non_contiguous.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        validate_bgr_numpy_frame(non_contiguous, width=3, height=2)


# ADD 2026-09-04: Invalid BGR stride contract를 constructor validation 단계에서 차단한다.
def test_bgr_frame_spec_rejects_stride_smaller_than_row() -> None:
    with pytest.raises(ValueError, match="stride"):
        BgrFrameSpec(width=3, height=2, stride=8).validate()
