"""Tests for PatchCore image and mask preprocessing."""

import torch

from ml.training.preprocessing import PatchCorePreprocessingConfig, PatchCorePreprocessor


# ADD 2026-08-19: 테스트용 PatchCore configuration을 생성한다.
def _config() -> PatchCorePreprocessingConfig:
    return PatchCorePreprocessingConfig(
        resize_size=(256, 256),
        center_crop_size=(224, 224),
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
    )


# ADD 2026-08-19: preprocessor produces expected shapes and binary mask 테스트 시나리오를 검증한다.
def test_preprocessor_produces_expected_shapes_and_binary_mask() -> None:
    images = torch.rand(2, 3, 700, 700)
    masks = torch.zeros(2, 1, 700, 700)
    masks[:, :, 175:525, 175:525] = 1

    transformed_images, transformed_masks = PatchCorePreprocessor(_config())(images, masks)

    assert transformed_images.shape == (2, 3, 224, 224)
    assert transformed_masks is not None
    assert transformed_masks.shape == (2, 1, 224, 224)
    assert set(torch.unique(transformed_masks).tolist()) == {0.0, 1.0}


# ADD 2026-08-19: image and mask geometric regions remain aligned 테스트 시나리오를 검증한다.
def test_image_and_mask_geometric_regions_remain_aligned() -> None:
    masks = torch.zeros(1, 1, 700, 700)
    masks[:, :, 210:490, 245:525] = 1
    images = masks.repeat(1, 3, 1, 1)
    config = _config()

    transformed_images, transformed_masks = PatchCorePreprocessor(config)(images, masks)

    assert transformed_masks is not None
    denormalized_first_channel = (
        transformed_images[:, :1] * config.image_std[0] + config.image_mean[0]
    )
    image_region = denormalized_first_channel.gt(0.5)
    mask_region = transformed_masks.bool()
    intersection = torch.logical_and(image_region, mask_region).sum()
    union = torch.logical_or(image_region, mask_region).sum()
    assert float(intersection / union) > 0.99


# ADD 2026-08-19: mask is not image normalized 테스트 시나리오를 검증한다.
def test_mask_is_not_image_normalized() -> None:
    images = torch.zeros(1, 3, 700, 700)
    masks = torch.ones(1, 1, 700, 700)

    transformed_images, transformed_masks = PatchCorePreprocessor(_config())(images, masks)

    assert transformed_masks is not None
    assert torch.all(transformed_masks == 1)
    assert not torch.all(transformed_images == 1)


# ADD 2026-08-19: preprocessor rejects misaligned image and mask 테스트 시나리오를 검증한다.
def test_preprocessor_rejects_misaligned_image_and_mask() -> None:
    images = torch.zeros(1, 3, 700, 700)
    masks = torch.zeros(1, 1, 699, 700)

    try:
        PatchCorePreprocessor(_config())(images, masks)
    except ValueError as exc:
        assert "matching" in str(exc)
    else:
        raise AssertionError("Expected mismatched image and mask dimensions to fail.")
