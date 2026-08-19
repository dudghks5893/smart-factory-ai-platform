"""Constants for supported datasets."""

MVTEC_AD_CATEGORIES: tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

GOOD_DIR_NAME = "good"
IMAGE_SUFFIX = ".png"
MASK_SUFFIX = "_mask.png"
MVTEC_SPLITS: tuple[str, ...] = ("train", "validation", "test")
MVTEC_SOURCE_SPLITS: tuple[str, ...] = ("train", "test")
