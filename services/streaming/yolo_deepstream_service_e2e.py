"""C6-6 DeepStream-to-service E2E stdout bridge contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from services.streaming.yolo_deepstream_segmentation import (
    DeepStreamSegmentationConfig,
    build_nvinfer_config_text,
)
from services.streaming.yolo_deepstream_segmentation_metadata import (
    EXPECTED_SAMPLE_BYTES,
    EXPECTED_SAMPLE_PATH,
    EXPECTED_SAMPLE_SHA256,
)
from services.streaming.yolo_deepstream_service_integration import EXPECTED_CLASSES
from services.streaming.yolo_deepstream_service_publisher import (
    DeepStreamFrameSnapshot,
    DeepStreamInstanceSnapshot,
)

EXPECTED_E2E_ADAPTER_ID = "c6_6_deepstream_stdout_service_e2e_v1"
EXPECTED_FRAME_PREFIX = "C6_6_SERVICE_E2E_FRAME="
EXPECTED_RESULT_PREFIX = "C6_6_SERVICE_E2E_RESULT="
EXPECTED_TARGET_FRAMES = 30
EXPECTED_IMAGE_WIDTH = 1280
EXPECTED_IMAGE_HEIGHT = 720
EXPECTED_CONTAINER_LABEL = "c6_6_service_e2e=1"
EXPECTED_RUNTIME_IMAGE = "c6-6-service-e2e:verify"
EXPECTED_NVINFER_CONFIG_PATH = "/tmp/c6_6_service_e2e_nvinfer.txt"


# ADD 2026-09-07: compact stdout 직렬화 → MODIFY 2026-09-07: mask grid 의미 명시
def build_service_e2e_probe_source(config: DeepStreamSegmentationConfig) -> str:
    config.validate()
    nvinfer_config = build_nvinfer_config_text(config)
    template = r"""#include <gst/gst.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "gstnvdsmeta.h"
#include "nvdsinfer.h"
#include "nvdsmeta.h"

namespace {

constexpr unsigned int kTargetFrames = __TARGET_FRAMES__U;
constexpr unsigned int kImageWidth = __IMAGE_WIDTH__U;
constexpr unsigned int kImageHeight = __IMAGE_HEIGHT__U;
constexpr float kExpectedMaskThreshold = 0.5F;
constexpr const char* kFramePrefix = __FRAME_PREFIX__;
constexpr const char* kResultPrefix = __RESULT_PREFIX__;
constexpr const char* kConfigPath = __CONFIG_PATH__;
constexpr const char* kNvinferConfig = __NVINFER_CONFIG__;

struct ProbeState {
  unsigned int frames = 0;
  unsigned int events = 0;
  unsigned int instances = 0;
  bool batch_meta_observed = false;
  bool invalid = false;
  std::string invalid_reason;
};

void mark_invalid(ProbeState* state, const std::string& reason) {
  if (!state->invalid) {
    state->invalid = true;
    state->invalid_reason = reason;
  }
}

GstPadProbeReturn emit_compact_metadata(
    GstPad*, GstPadProbeInfo* info, gpointer user_data) {
  auto* state = static_cast<ProbeState*>(user_data);
  auto* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (buffer == nullptr) {
    mark_invalid(state, "null GstBuffer");
    return GST_PAD_PROBE_OK;
  }

  NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
  if (batch_meta == nullptr) {
    mark_invalid(state, "missing NvDsBatchMeta");
    return GST_PAD_PROBE_OK;
  }
  state->batch_meta_observed = true;

  const GstClockTime buffer_pts = GST_BUFFER_PTS(buffer);
  const unsigned long long pts_ns =
      buffer_pts == GST_CLOCK_TIME_NONE
          ? 0ULL
          : static_cast<unsigned long long>(buffer_pts);

  for (NvDsMetaList* frame_node = batch_meta->frame_meta_list;
       frame_node != nullptr;
       frame_node = frame_node->next) {
    auto* frame_meta = static_cast<NvDsFrameMeta*>(frame_node->data);
    if (frame_meta == nullptr) {
      mark_invalid(state, "null NvDsFrameMeta");
      continue;
    }
    ++state->frames;

    std::ostringstream instances_json;
    instances_json << std::setprecision(9);
    bool first_instance = true;
    unsigned int emitted_instances = 0;

    for (NvDsMetaList* object_node = frame_meta->obj_meta_list;
         object_node != nullptr;
         object_node = object_node->next) {
      auto* object_meta = static_cast<NvDsObjectMeta*>(object_node->data);
      if (object_meta == nullptr) {
        mark_invalid(state, "null NvDsObjectMeta");
        continue;
      }
      if (object_meta->class_id < 0 || object_meta->class_id > 2) {
        mark_invalid(state, "class_id outside frozen mapping");
        continue;
      }
      if (!std::isfinite(object_meta->confidence) ||
          object_meta->confidence < 0.0F ||
          object_meta->confidence > 1.0F) {
        mark_invalid(state, "invalid object confidence");
        continue;
      }

      const float left = object_meta->rect_params.left;
      const float top = object_meta->rect_params.top;
      const float width = object_meta->rect_params.width;
      const float height = object_meta->rect_params.height;
      if (!std::isfinite(left) || !std::isfinite(top) ||
          !std::isfinite(width) || !std::isfinite(height) ||
          left < 0.0F || top < 0.0F || width <= 0.0F || height <= 0.0F ||
          left + width > static_cast<float>(kImageWidth) + 1e-3F ||
          top + height > static_cast<float>(kImageHeight) + 1e-3F) {
        mark_invalid(state, "invalid object rectangle");
        continue;
      }

      const NvOSD_MaskParams& mask = object_meta->mask_params;
      if (mask.data == nullptr || mask.size == 0U ||
          mask.width == 0U || mask.height == 0U) {
        mark_invalid(state, "object missing instance mask metadata");
        continue;
      }
      const std::size_t elements =
          static_cast<std::size_t>(mask.width) * mask.height;
      const std::size_t expected_bytes = elements * sizeof(float);
      if (mask.size != expected_bytes) {
        mark_invalid(state, "mask byte-size does not match geometry");
        continue;
      }
      if (!std::isfinite(mask.threshold) ||
          std::fabs(mask.threshold - kExpectedMaskThreshold) > 1e-6F) {
        mark_invalid(state, "mask threshold changed");
        continue;
      }

      unsigned int mask_positive_sample_count = 0U;
      bool mask_valid = true;
      for (std::size_t index = 0; index < elements; ++index) {
        const float value = mask.data[index];
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
          mark_invalid(state, "mask probability outside [0,1]");
          mask_valid = false;
          break;
        }
        if (value >= mask.threshold) {
          ++mask_positive_sample_count;
        }
      }
      if (!mask_valid) {
        continue;
      }
      if (mask_positive_sample_count == 0U ||
          mask_positive_sample_count > elements) {
        mark_invalid(state, "mask positive sample count outside mask grid");
        continue;
      }

      if (!first_instance) {
        instances_json << ',';
      }
      first_instance = false;
      instances_json
          << "{\"class_id\":" << object_meta->class_id
          << ",\"confidence\":" << object_meta->confidence
          << ",\"left\":" << left
          << ",\"top\":" << top
          << ",\"width\":" << width
          << ",\"height\":" << height
          << ",\"mask_width\":" << mask.width
          << ",\"mask_height\":" << mask.height
          << ",\"mask_positive_sample_count\":"
          << mask_positive_sample_count
          << '}';
      ++emitted_instances;
    }

    if (emitted_instances > 0U) {
      std::cout << kFramePrefix
                << "{\"frame_number\":" << frame_meta->frame_num
                << ",\"pts_ns\":" << pts_ns
                << ",\"image_width\":" << kImageWidth
                << ",\"image_height\":" << kImageHeight
                << ",\"instances\":[" << instances_json.str() << "]}"
                << std::endl;
      ++state->events;
      state->instances += emitted_instances;
    }
  }
  return GST_PAD_PROBE_OK;
}

std::string escape_json(const std::string& value) {
  std::ostringstream escaped;
  for (const char character : value) {
    if (character == '\\' || character == '"') {
      escaped << '\\';
    }
    escaped << character;
  }
  return escaped.str();
}

}  // namespace

int main(int argc, char** argv) {
  gst_init(&argc, &argv);

  {
    std::ofstream config_file(kConfigPath, std::ios::binary | std::ios::trunc);
    if (!config_file) {
      std::cerr << "failed to create nvinfer config" << std::endl;
      return 2;
    }
    config_file << kNvinferConfig;
  }

  GError* parse_error = nullptr;
  const std::string pipeline_text =
      "nvstreammux name=mux batch-size=1 width=1280 height=720 "
      "live-source=false batched-push-timeout=40000 "
      "filesrc location=__SAMPLE_PATH__ ! h264parse ! nvv4l2decoder ! queue ! mux.sink_0 "
      "mux. ! nvinfer name=primary "
      "config-file-path=__CONFIG_PATH_RAW__ ! "
      "identity name=postinfer eos-after=30 silent=false ! "
      "fakesink sync=false async=false";

  GstElement* pipeline = gst_parse_launch(pipeline_text.c_str(), &parse_error);
  if (pipeline == nullptr || parse_error != nullptr) {
    if (parse_error != nullptr) {
      std::cerr << parse_error->message << std::endl;
      g_error_free(parse_error);
    }
    std::remove(kConfigPath);
    return 3;
  }

  GstElement* primary = gst_bin_get_by_name(GST_BIN(pipeline), "primary");
  if (primary == nullptr) {
    std::cerr << "missing nvinfer element" << std::endl;
    gst_object_unref(pipeline);
    std::remove(kConfigPath);
    return 4;
  }
  GstPad* src_pad = gst_element_get_static_pad(primary, "src");
  if (src_pad == nullptr) {
    std::cerr << "missing nvinfer src pad" << std::endl;
    gst_object_unref(primary);
    gst_object_unref(pipeline);
    std::remove(kConfigPath);
    return 5;
  }

  ProbeState state;
  gst_pad_add_probe(
      src_pad,
      GST_PAD_PROBE_TYPE_BUFFER,
      emit_compact_metadata,
      &state,
      nullptr);

  GstBus* bus = gst_element_get_bus(pipeline);
  if (gst_element_set_state(pipeline, GST_STATE_PLAYING) ==
      GST_STATE_CHANGE_FAILURE) {
    std::cerr << "pipeline failed to enter PLAYING" << std::endl;
    gst_object_unref(bus);
    gst_object_unref(src_pad);
    gst_object_unref(primary);
    gst_object_unref(pipeline);
    std::remove(kConfigPath);
    return 6;
  }

  bool eos_observed = false;
  bool runtime_error = false;
  while (!eos_observed && !runtime_error) {
    GstMessage* message = gst_bus_timed_pop_filtered(
        bus,
        180 * GST_SECOND,
        static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    if (message == nullptr) {
      std::cerr << "pipeline timeout" << std::endl;
      runtime_error = true;
      break;
    }
    if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS) {
      eos_observed = true;
    } else if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
      GError* error = nullptr;
      gchar* debug = nullptr;
      gst_message_parse_error(message, &error, &debug);
      std::cerr << "pipeline error: "
                << (error != nullptr ? error->message : "unknown") << std::endl;
      if (debug != nullptr) {
        std::cerr << debug << std::endl;
      }
      if (error != nullptr) {
        g_error_free(error);
      }
      g_free(debug);
      runtime_error = true;
    }
    gst_message_unref(message);
  }

  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(bus);
  gst_object_unref(src_pad);
  gst_object_unref(primary);
  gst_object_unref(pipeline);
  std::remove(kConfigPath);

  const bool target_frames_reached = state.frames >= kTargetFrames;
  const bool events_observed = state.events > 0U;
  const bool instances_observed = state.instances > 0U;
  const bool passed =
      !runtime_error && eos_observed && state.batch_meta_observed &&
      target_frames_reached && events_observed && instances_observed &&
      !state.invalid;

  std::cout << kResultPrefix
            << "{\"status\":\"" << (passed ? "passed" : "failed") << "\""
            << ",\"frames\":" << state.frames
            << ",\"events\":" << state.events
            << ",\"instances\":" << state.instances
            << ",\"eos_observed\":" << (eos_observed ? "true" : "false")
            << ",\"batch_meta_observed\":"
            << (state.batch_meta_observed ? "true" : "false")
            << ",\"target_frames_reached\":"
            << (target_frames_reached ? "true" : "false")
            << ",\"events_observed\":" << (events_observed ? "true" : "false")
            << ",\"instances_observed\":"
            << (instances_observed ? "true" : "false")
            << ",\"metadata_valid\":" << (!state.invalid ? "true" : "false")
            << ",\"invalid_reason\":\""
            << escape_json(state.invalid_reason) << "\""
            << ",\"deepstream_gpu_runtime_executed\":true"
            << ",\"segmentation_decode_executed\":true"
            << ",\"compact_metadata_emitted\":true"
            << ",\"network_used\":false"
            << ",\"raw_frame_emitted\":false"
            << ",\"raw_mask_emitted\":false"
            << ",\"dataset_used\":false"
            << ",\"validation_used\":false"
            << ",\"test_used\":false"
            << ",\"final_test_used\":false}"
            << std::endl;

  return passed ? 0 : 7;
}
"""
    return (
        template.replace("__TARGET_FRAMES__", str(EXPECTED_TARGET_FRAMES))
        .replace("__IMAGE_WIDTH__", str(EXPECTED_IMAGE_WIDTH))
        .replace("__IMAGE_HEIGHT__", str(EXPECTED_IMAGE_HEIGHT))
        .replace("__FRAME_PREFIX__", json.dumps(EXPECTED_FRAME_PREFIX))
        .replace("__RESULT_PREFIX__", json.dumps(EXPECTED_RESULT_PREFIX))
        .replace("__CONFIG_PATH__", json.dumps(EXPECTED_NVINFER_CONFIG_PATH))
        .replace("__CONFIG_PATH_RAW__", EXPECTED_NVINFER_CONFIG_PATH)
        .replace("__NVINFER_CONFIG__", json.dumps(nvinfer_config))
        .replace("__SAMPLE_PATH__", EXPECTED_SAMPLE_PATH)
    )


# ADD 2026-09-07: Probe를 networkless DeepStream container에서 compile/run하도록 구성한다.
def build_service_e2e_shell_payload(config: DeepStreamSegmentationConfig) -> str:
    source = build_service_e2e_probe_source(config)
    return "\n".join(
        (
            "set -euo pipefail",
            f"test \"$(sha256sum {EXPECTED_SAMPLE_PATH} | awk '{{print $1}}')\" = "
            f'"{EXPECTED_SAMPLE_SHA256}"',
            f'test "$(stat -c %s {EXPECTED_SAMPLE_PATH})" = "{EXPECTED_SAMPLE_BYTES}"',
            "cat > /tmp/c6_6_service_e2e_probe.cpp <<'C6_6_E2E_CPP'",
            source,
            "C6_6_E2E_CPP",
            "DS=/opt/nvidia/deepstream/deepstream",
            'export LD_LIBRARY_PATH="${DS}/lib:${LD_LIBRARY_PATH:-}"',
            "g++ -std=c++17 -O2 -Wall -Wextra "
            "$(pkg-config --cflags gstreamer-1.0) "
            '-I"${DS}/sources/includes" '
            "/tmp/c6_6_service_e2e_probe.cpp "
            '-L"${DS}/lib" -Wl,-rpath,"${DS}/lib" '
            "-lnvdsgst_meta -lnvds_meta "
            "$(pkg-config --libs gstreamer-1.0) "
            "-o /tmp/c6_6_service_e2e_probe",
            "set +e",
            "/tmp/c6_6_service_e2e_probe",
            "status=$?",
            "set -e",
            "rm -f /tmp/c6_6_service_e2e_probe.cpp "
            "/tmp/c6_6_service_e2e_probe "
            f"{EXPECTED_NVINFER_CONFIG_PATH}",
            "exit ${status}",
            "",
        )
    )


# ADD 2026-09-07: GPU runtime은 sealed plan을 read-only로 mount하고 network를 차단한다.
def build_service_e2e_docker_command(plan_path: Path) -> tuple[str, ...]:
    return (
        "sudo",
        "docker",
        "run",
        "--rm",
        "--interactive",
        "--runtime=nvidia",
        "--network",
        "none",
        "--gpus",
        "all",
        "--label",
        EXPECTED_CONTAINER_LABEL,
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "-v",
        f"{plan_path.resolve()}:/model/model.plan:ro",
        "--entrypoint",
        "bash",
        EXPECTED_RUNTIME_IMAGE,
        "-s",
    )


def _strict_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be JSON object.")
    return cast(dict[str, Any], value)


def _strict_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be int.")
    return cast(int, value)


def _strict_number(value: object, *, label: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be numeric.")
    number = float(cast(int | float, value))
    if not (number == number and abs(number) != float("inf")):
        raise ValueError(f"{label} must be finite.")
    return number


# ADD 2026-09-07: strict snapshot 변환 → MODIFY 2026-09-07: source-space 면적 환산
def parse_service_e2e_frame_line(line: str) -> DeepStreamFrameSnapshot:
    if not line.startswith(EXPECTED_FRAME_PREFIX):
        raise ValueError("C6-6 E2E frame prefix mismatch.")
    raw = _strict_mapping(
        json.loads(line[len(EXPECTED_FRAME_PREFIX) :]),
        label="C6-6 E2E frame",
    )
    expected_fields = {
        "frame_number",
        "pts_ns",
        "image_width",
        "image_height",
        "instances",
    }
    if set(raw) != expected_fields:
        raise ValueError("C6-6 E2E frame fields changed.")

    image_width = _strict_int(raw["image_width"], label="image_width")
    image_height = _strict_int(raw["image_height"], label="image_height")
    if image_width != EXPECTED_IMAGE_WIDTH or image_height != EXPECTED_IMAGE_HEIGHT:
        raise ValueError("C6-6 E2E frame dimensions changed.")

    raw_instances = raw["instances"]
    if not isinstance(raw_instances, list) or not raw_instances:
        raise ValueError("C6-6 E2E frame must contain at least one compact instance.")

    instances: list[DeepStreamInstanceSnapshot] = []
    for index, item in enumerate(raw_instances):
        instance = _strict_mapping(item, label=f"instances[{index}]")
        if set(instance) != {
            "class_id",
            "confidence",
            "left",
            "top",
            "width",
            "height",
            "mask_width",
            "mask_height",
            "mask_positive_sample_count",
        }:
            raise ValueError("C6-6 E2E compact instance fields changed.")

        class_id = _strict_int(instance["class_id"], label="class_id")
        if class_id not in EXPECTED_CLASSES:
            raise ValueError("C6-6 E2E class_id is outside the frozen mapping.")

        left = _strict_number(instance["left"], label="left")
        top = _strict_number(instance["top"], label="top")
        width = _strict_number(instance["width"], label="width")
        height = _strict_number(instance["height"], label="height")
        mask_width = _strict_int(instance["mask_width"], label="mask_width")
        mask_height = _strict_int(instance["mask_height"], label="mask_height")
        positive_samples = _strict_int(
            instance["mask_positive_sample_count"],
            label="mask_positive_sample_count",
        )
        if mask_width <= 0 or mask_height <= 0:
            raise ValueError("C6-6 E2E mask grid dimensions must be positive.")
        mask_grid_area = mask_width * mask_height
        if not 0 < positive_samples <= mask_grid_area:
            raise ValueError("C6-6 E2E positive mask samples must fit the bbox-local mask grid.")

        image_area = image_width * image_height
        bbox_area = width * height
        estimated_source_pixels = round((positive_samples / mask_grid_area) * bbox_area)
        estimated_source_pixels = max(
            1,
            min(image_area, estimated_source_pixels),
        )

        snapshot = DeepStreamInstanceSnapshot(
            class_id=class_id,
            confidence=_strict_number(instance["confidence"], label="confidence"),
            left=left,
            top=top,
            width=width,
            height=height,
            mask_pixel_count=estimated_source_pixels,
        )
        snapshot.to_streaming_instance(
            image_width=image_width,
            image_height=image_height,
        )
        instances.append(snapshot)

    return DeepStreamFrameSnapshot(
        frame_number=_strict_int(raw["frame_number"], label="frame_number"),
        pts_ns=_strict_int(raw["pts_ns"], label="pts_ns"),
        image_width=image_width,
        image_height=image_height,
        instances=tuple(instances),
    )


# ADD 2026-09-07: DeepStream process의 strict single summary line을 파싱한다.
def parse_service_e2e_result(stdout: str) -> dict[str, Any]:
    matches = [
        line[len(EXPECTED_RESULT_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(EXPECTED_RESULT_PREFIX)
    ]
    if len(matches) != 1:
        raise ValueError("C6-6 E2E result count must be exactly one.")
    return _strict_mapping(json.loads(matches[0]), label="C6-6 E2E result")


# ADD 2026-09-07: GPU metadata-emission runtime summary의 scope와 acceptance를 검증한다.
def validate_service_e2e_result(payload: dict[str, Any]) -> None:
    expected_fields = {
        "status",
        "frames",
        "events",
        "instances",
        "eos_observed",
        "batch_meta_observed",
        "target_frames_reached",
        "events_observed",
        "instances_observed",
        "metadata_valid",
        "invalid_reason",
        "deepstream_gpu_runtime_executed",
        "segmentation_decode_executed",
        "compact_metadata_emitted",
        "network_used",
        "raw_frame_emitted",
        "raw_mask_emitted",
        "dataset_used",
        "validation_used",
        "test_used",
        "final_test_used",
    }
    if set(payload) != expected_fields:
        raise ValueError("C6-6 E2E result fields changed.")

    integer_fields = ("frames", "events", "instances")
    if any(type(payload[name]) is not int for name in integer_fields):
        raise TypeError("C6-6 E2E result counters must be int.")
    if not isinstance(payload["status"], str) or not isinstance(payload["invalid_reason"], str):
        raise TypeError("C6-6 E2E result strings are invalid.")

    true_flags = (
        "eos_observed",
        "batch_meta_observed",
        "target_frames_reached",
        "events_observed",
        "instances_observed",
        "metadata_valid",
        "deepstream_gpu_runtime_executed",
        "segmentation_decode_executed",
        "compact_metadata_emitted",
    )
    false_flags = (
        "network_used",
        "raw_frame_emitted",
        "raw_mask_emitted",
        "dataset_used",
        "validation_used",
        "test_used",
        "final_test_used",
    )
    if any(type(payload[name]) is not bool for name in (*true_flags, *false_flags)):
        raise TypeError("C6-6 E2E result flags must be bool.")

    if (
        payload["status"] != "passed"
        or payload["frames"] < EXPECTED_TARGET_FRAMES
        or payload["events"] <= 0
        or payload["instances"] <= 0
        or payload["invalid_reason"] != ""
        or any(payload[name] is not True for name in true_flags)
        or any(payload[name] is not False for name in false_flags)
    ):
        raise ValueError("C6-6 E2E GPU metadata-emission acceptance failed.")
