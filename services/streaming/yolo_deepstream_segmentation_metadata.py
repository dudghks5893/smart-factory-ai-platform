"""C6-5D DeepStream instance-mask metadata runtime contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from services.streaming.yolo_deepstream_segmentation import (
    DeepStreamSegmentationConfig,
    build_nvinfer_config_text,
)

EXPECTED_RUNTIME_ID = "c6_5d_instance_metadata_runtime_v1"
EXPECTED_SAMPLE_PATH = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.h264"
EXPECTED_SAMPLE_SHA256 = "5f29353a6ec4727bd49fb523efc207d643e6638f4e5c56f060e1b61291aa6ea2"
EXPECTED_SAMPLE_BYTES = 14_759_548
EXPECTED_TARGET_FRAMES = 30
EXPECTED_RESULT_PREFIX = "C6_5D_INSTANCE_METADATA_RESULT="
EXPECTED_CONTAINER_LABEL = "c6_5d_instance_metadata=1"
EXPECTED_RUNTIME_IMAGE = "c6-5d-instance-metadata:verify"


# ADD 2026-09-06: DeepStream object/mask metadata용 deterministic C++ pad-probe를 생성한다.
def build_instance_metadata_probe_source(config: DeepStreamSegmentationConfig) -> str:
    config.validate()
    nvinfer_config = build_nvinfer_config_text(config)
    template = r"""#include <gst/gst.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "gstnvdsmeta.h"
#include "nvdsinfer.h"
#include "nvdsmeta.h"

namespace {

constexpr unsigned int kTargetFrames = __TARGET_FRAMES__U;
constexpr float kExpectedMaskThreshold = 0.5F;
constexpr const char* kResultPrefix = __RESULT_PREFIX__;
constexpr const char* kConfigPath = "/tmp/c6_5d_instance_metadata_nvinfer.txt";
constexpr const char* kNvinferConfig = __NVINFER_CONFIG__;

struct ProbeState {
  unsigned int frames = 0;
  unsigned int objects = 0;
  unsigned int masks = 0;
  unsigned int tensor_meta_frames = 0;
  unsigned int class_0 = 0;
  unsigned int class_1 = 0;
  unsigned int class_2 = 0;
  unsigned long long mask_elements = 0;
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

GstPadProbeReturn inspect_metadata(
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

  for (NvDsMetaList* frame_node = batch_meta->frame_meta_list;
       frame_node != nullptr;
       frame_node = frame_node->next) {
    auto* frame_meta = static_cast<NvDsFrameMeta*>(frame_node->data);
    if (frame_meta == nullptr) {
      mark_invalid(state, "null NvDsFrameMeta");
      continue;
    }
    ++state->frames;

    bool tensor_meta_observed = false;
    for (NvDsMetaList* user_node = frame_meta->frame_user_meta_list;
         user_node != nullptr;
         user_node = user_node->next) {
      auto* user_meta = static_cast<NvDsUserMeta*>(user_node->data);
      if (user_meta != nullptr &&
          user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
        tensor_meta_observed = true;
      }
    }
    if (tensor_meta_observed) {
      ++state->tensor_meta_frames;
    }

    for (NvDsMetaList* object_node = frame_meta->obj_meta_list;
         object_node != nullptr;
         object_node = object_node->next) {
      auto* object_meta = static_cast<NvDsObjectMeta*>(object_node->data);
      if (object_meta == nullptr) {
        mark_invalid(state, "null NvDsObjectMeta");
        continue;
      }
      ++state->objects;

      if (object_meta->class_id == 0) {
        ++state->class_0;
      } else if (object_meta->class_id == 1) {
        ++state->class_1;
      } else if (object_meta->class_id == 2) {
        ++state->class_2;
      } else {
        mark_invalid(state, "class_id outside frozen mapping");
      }

      if (!std::isfinite(object_meta->confidence) ||
          object_meta->confidence < 0.0F || object_meta->confidence > 1.0F) {
        mark_invalid(state, "invalid object confidence");
      }
      if (object_meta->rect_params.width <= 0.0F ||
          object_meta->rect_params.height <= 0.0F) {
        mark_invalid(state, "invalid object rectangle");
      }

      const NvOSD_MaskParams& mask = object_meta->mask_params;
      if (mask.data == nullptr || mask.size == 0U ||
          mask.width == 0U || mask.height == 0U) {
        mark_invalid(state, "object missing instance mask metadata");
        continue;
      }
      const std::size_t expected_bytes =
          static_cast<std::size_t>(mask.width) * mask.height * sizeof(float);
      if (mask.size != expected_bytes) {
        mark_invalid(state, "mask byte-size does not match geometry");
        continue;
      }
      if (!std::isfinite(mask.threshold) ||
          std::fabs(mask.threshold - kExpectedMaskThreshold) > 1e-6F) {
        mark_invalid(state, "mask threshold changed");
      }

      const std::size_t elements =
          static_cast<std::size_t>(mask.width) * mask.height;
      for (std::size_t index = 0; index < elements; ++index) {
        const float value = mask.data[index];
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
          mark_invalid(state, "mask probability outside [0,1]");
          break;
        }
      }
      ++state->masks;
      state->mask_elements += static_cast<unsigned long long>(elements);
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
      "config-file-path=/tmp/c6_5d_instance_metadata_nvinfer.txt ! "
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
      inspect_metadata,
      &state,
      nullptr);

  GstBus* bus = gst_element_get_bus(pipeline);
  if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
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
  const bool object_metadata_observed = state.objects > 0U;
  const bool instance_masks_observed = state.masks > 0U;
  const bool masks_cover_objects = state.objects > 0U && state.masks == state.objects;
  const bool tensor_meta_observed = state.tensor_meta_frames > 0U;
  const bool passed =
      !runtime_error && eos_observed && state.batch_meta_observed &&
      target_frames_reached && tensor_meta_observed && object_metadata_observed &&
      instance_masks_observed && masks_cover_objects && !state.invalid;

  std::cout << kResultPrefix
            << "{\"status\":\"" << (passed ? "passed" : "failed") << "\""
            << ",\"frames\":" << state.frames
            << ",\"objects\":" << state.objects
            << ",\"masks\":" << state.masks
            << ",\"mask_elements\":" << state.mask_elements
            << ",\"tensor_meta_frames\":" << state.tensor_meta_frames
            << ",\"class_0\":" << state.class_0
            << ",\"class_1\":" << state.class_1
            << ",\"class_2\":" << state.class_2
            << ",\"eos_observed\":" << (eos_observed ? "true" : "false")
            << ",\"batch_meta_observed\":"
            << (state.batch_meta_observed ? "true" : "false")
            << ",\"target_frames_reached\":"
            << (target_frames_reached ? "true" : "false")
            << ",\"tensor_meta_observed\":"
            << (tensor_meta_observed ? "true" : "false")
            << ",\"object_metadata_observed\":"
            << (object_metadata_observed ? "true" : "false")
            << ",\"instance_masks_observed\":"
            << (instance_masks_observed ? "true" : "false")
            << ",\"masks_cover_objects\":"
            << (masks_cover_objects ? "true" : "false")
            << ",\"metadata_valid\":" << (!state.invalid ? "true" : "false")
            << ",\"invalid_reason\":\"" << escape_json(state.invalid_reason) << "\""
            << ",\"segmentation_decode_executed\":true"
            << ",\"instance_metadata_executed\":true"
            << ",\"overlay_executed\":false"
            << ",\"annotated_video_written\":false"
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
        .replace("__RESULT_PREFIX__", json.dumps(EXPECTED_RESULT_PREFIX))
        .replace("__NVINFER_CONFIG__", json.dumps(nvinfer_config))
        .replace("__SAMPLE_PATH__", EXPECTED_SAMPLE_PATH)
    )


# ADD 2026-09-06: Probe를 networkless container에서 compile/run하는 shell payload를 생성한다.
def build_instance_metadata_shell_payload(config: DeepStreamSegmentationConfig) -> str:
    source = build_instance_metadata_probe_source(config)
    return "\n".join(
        (
            "set -euo pipefail",
            f"test \"$(sha256sum {EXPECTED_SAMPLE_PATH} | awk '{{print $1}}')\" = "
            f'"{EXPECTED_SAMPLE_SHA256}"',
            f'test "$(stat -c %s {EXPECTED_SAMPLE_PATH})" = "{EXPECTED_SAMPLE_BYTES}"',
            "cat > /tmp/c6_5d_instance_metadata_probe.cpp <<'C6_5D_CPP'",
            source,
            "C6_5D_CPP",
            "DS=/opt/nvidia/deepstream/deepstream",
            'export LD_LIBRARY_PATH="${DS}/lib:${LD_LIBRARY_PATH:-}"',
            "g++ -std=c++17 -O2 -Wall -Wextra "
            "$(pkg-config --cflags gstreamer-1.0) "
            '-I"${DS}/sources/includes" '
            "/tmp/c6_5d_instance_metadata_probe.cpp "
            '-L"${DS}/lib" -Wl,-rpath,"${DS}/lib" '
            "-lnvdsgst_meta -lnvds_meta "
            "$(pkg-config --libs gstreamer-1.0) "
            "-o /tmp/c6_5d_instance_metadata_probe",
            "set +e",
            "/tmp/c6_5d_instance_metadata_probe",
            "status=$?",
            "set -e",
            "rm -f /tmp/c6_5d_instance_metadata_probe.cpp "
            "/tmp/c6_5d_instance_metadata_probe "
            "/tmp/c6_5d_instance_metadata_nvinfer.txt",
            "exit ${status}",
            "",
        )
    )


# ADD 2026-09-06: Runtime container를 GPU/no-network/read-only-plan boundary로 고정한다.
def build_instance_metadata_docker_command(plan_path: Path) -> tuple[str, ...]:
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


# ADD 2026-09-06: Container stdout의 single strict JSON result를 파싱한다.
def parse_instance_metadata_payload(
    stdout: str,
    *,
    prefix: str = EXPECTED_RESULT_PREFIX,
) -> dict[str, Any]:
    matches = [line[len(prefix) :] for line in stdout.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError("C6-5D instance metadata result count must be exactly one.")
    raw: object = json.loads(matches[0])
    if not isinstance(raw, dict):
        raise ValueError("C6-5D instance metadata result must be object.")
    return cast(dict[str, Any], raw)


# ADD 2026-09-06: Runtime payload가 decode/metadata-only acceptance boundary를 만족하는지 검증한다.
def validate_instance_metadata_payload(payload: dict[str, Any]) -> None:
    expected_fields = {
        "status",
        "frames",
        "objects",
        "masks",
        "mask_elements",
        "tensor_meta_frames",
        "class_0",
        "class_1",
        "class_2",
        "eos_observed",
        "batch_meta_observed",
        "target_frames_reached",
        "tensor_meta_observed",
        "object_metadata_observed",
        "instance_masks_observed",
        "masks_cover_objects",
        "metadata_valid",
        "invalid_reason",
        "segmentation_decode_executed",
        "instance_metadata_executed",
        "overlay_executed",
        "annotated_video_written",
        "dataset_used",
        "validation_used",
        "test_used",
        "final_test_used",
    }
    if set(payload) != expected_fields:
        raise ValueError("C6-5D instance metadata payload fields changed.")

    integer_fields = (
        "frames",
        "objects",
        "masks",
        "mask_elements",
        "tensor_meta_frames",
        "class_0",
        "class_1",
        "class_2",
    )
    if any(type(payload[name]) is not int for name in integer_fields):
        raise TypeError("C6-5D instance metadata counters must be int.")
    if not isinstance(payload["status"], str) or not isinstance(payload["invalid_reason"], str):
        raise TypeError("C6-5D instance metadata strings are invalid.")

    true_flags = (
        "eos_observed",
        "batch_meta_observed",
        "target_frames_reached",
        "tensor_meta_observed",
        "object_metadata_observed",
        "instance_masks_observed",
        "masks_cover_objects",
        "metadata_valid",
        "segmentation_decode_executed",
        "instance_metadata_executed",
    )
    false_flags = (
        "overlay_executed",
        "annotated_video_written",
        "dataset_used",
        "validation_used",
        "test_used",
        "final_test_used",
    )
    if any(type(payload[name]) is not bool for name in (*true_flags, *false_flags)):
        raise TypeError("C6-5D instance metadata flags must be bool.")

    if (
        payload["status"] != "passed"
        or payload["frames"] < EXPECTED_TARGET_FRAMES
        or payload["objects"] <= 0
        or payload["masks"] != payload["objects"]
        or payload["mask_elements"] <= 0
        or payload["tensor_meta_frames"] <= 0
        or payload["class_0"] + payload["class_1"] + payload["class_2"] != payload["objects"]
        or any(payload[name] is not True for name in true_flags)
        or any(payload[name] is not False for name in false_flags)
        or payload["invalid_reason"] != ""
    ):
        raise ValueError("C6-5D instance metadata runtime acceptance failed.")
