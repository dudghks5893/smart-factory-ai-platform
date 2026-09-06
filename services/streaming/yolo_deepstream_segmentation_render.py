"""C6-5D DeepStream instance-mask rendering runtime contracts."""

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
    EXPECTED_TARGET_FRAMES,
)

EXPECTED_RENDER_RUNTIME_ID = "c6_5d_mask_rendering_runtime_v1"
EXPECTED_RENDER_RESULT_PREFIX = "C6_5D_MASK_RENDER_RESULT="
EXPECTED_RENDER_CONTAINER_LABEL = "c6_5d_mask_render=1"
EXPECTED_RENDER_RUNTIME_IMAGE = "c6-5d-mask-render:verify"


# ADD 2026-09-06: Instance mask metadata를 nvdsosd에 전달하는 runtime probe를 생성한다.
def build_mask_render_probe_source(config: DeepStreamSegmentationConfig) -> str:
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
#include "nvdsmeta.h"

namespace {

constexpr unsigned int kTargetFrames = __TARGET_FRAMES__U;
constexpr const char* kResultPrefix = __RESULT_PREFIX__;
constexpr const char* kConfigPath = "/tmp/c6_5d_mask_render_nvinfer.txt";
constexpr const char* kNvinferConfig = __NVINFER_CONFIG__;

struct RenderState {
  unsigned int pre_osd_frames = 0;
  unsigned int post_osd_frames = 0;
  unsigned int objects = 0;
  unsigned int masks = 0;
  unsigned long long mask_elements = 0;
  bool batch_meta_observed = false;
  bool invalid = false;
  std::string invalid_reason;
};

void mark_invalid(RenderState* state, const std::string& reason) {
  if (!state->invalid) {
    state->invalid = true;
    state->invalid_reason = reason;
  }
}

GstPadProbeReturn inspect_pre_osd(
    GstPad*, GstPadProbeInfo* info, gpointer user_data) {
  auto* state = static_cast<RenderState*>(user_data);
  auto* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (buffer == nullptr) {
    mark_invalid(state, "null pre-OSD GstBuffer");
    return GST_PAD_PROBE_OK;
  }

  NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
  if (batch_meta == nullptr) {
    mark_invalid(state, "missing pre-OSD NvDsBatchMeta");
    return GST_PAD_PROBE_OK;
  }
  state->batch_meta_observed = true;

  for (NvDsMetaList* frame_node = batch_meta->frame_meta_list;
       frame_node != nullptr;
       frame_node = frame_node->next) {
    auto* frame_meta = static_cast<NvDsFrameMeta*>(frame_node->data);
    if (frame_meta == nullptr) {
      mark_invalid(state, "null pre-OSD NvDsFrameMeta");
      continue;
    }
    ++state->pre_osd_frames;

    for (NvDsMetaList* object_node = frame_meta->obj_meta_list;
         object_node != nullptr;
         object_node = object_node->next) {
      auto* object_meta = static_cast<NvDsObjectMeta*>(object_node->data);
      if (object_meta == nullptr) {
        mark_invalid(state, "null pre-OSD NvDsObjectMeta");
        continue;
      }
      ++state->objects;

      const NvOSD_MaskParams& mask = object_meta->mask_params;
      if (mask.data == nullptr || mask.size == 0U ||
          mask.width == 0U || mask.height == 0U) {
        mark_invalid(state, "object missing pre-OSD mask metadata");
        continue;
      }

      const std::size_t expected_bytes =
          static_cast<std::size_t>(mask.width) * mask.height * sizeof(float);
      if (mask.size != expected_bytes) {
        mark_invalid(state, "pre-OSD mask byte-size mismatch");
        continue;
      }

      const std::size_t elements =
          static_cast<std::size_t>(mask.width) * mask.height;
      for (std::size_t index = 0; index < elements; ++index) {
        const float value = mask.data[index];
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
          mark_invalid(state, "pre-OSD mask probability invalid");
          break;
        }
      }

      ++state->masks;
      state->mask_elements += static_cast<unsigned long long>(elements);
    }
  }

  return GST_PAD_PROBE_OK;
}

GstPadProbeReturn inspect_post_osd(
    GstPad*, GstPadProbeInfo* info, gpointer user_data) {
  auto* state = static_cast<RenderState*>(user_data);
  auto* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (buffer == nullptr) {
    mark_invalid(state, "null post-OSD GstBuffer");
    return GST_PAD_PROBE_OK;
  }

  NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
  if (batch_meta == nullptr) {
    mark_invalid(state, "missing post-OSD NvDsBatchMeta");
    return GST_PAD_PROBE_OK;
  }

  for (NvDsMetaList* frame_node = batch_meta->frame_meta_list;
       frame_node != nullptr;
       frame_node = frame_node->next) {
    auto* frame_meta = static_cast<NvDsFrameMeta*>(frame_node->data);
    if (frame_meta == nullptr) {
      mark_invalid(state, "null post-OSD NvDsFrameMeta");
      continue;
    }
    ++state->post_osd_frames;
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
      "config-file-path=/tmp/c6_5d_mask_render_nvinfer.txt ! "
      "nvvideoconvert ! video/x-raw(memory:NVMM),format=RGBA ! "
      "nvdsosd name=osd display-mask=1 display-bbox=0 display-text=0 ! "
      "identity name=postrender eos-after=30 silent=false ! "
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

  GstElement* osd = gst_bin_get_by_name(GST_BIN(pipeline), "osd");
  if (osd == nullptr) {
    std::cerr << "missing nvdsosd element" << std::endl;
    gst_object_unref(pipeline);
    std::remove(kConfigPath);
    return 4;
  }

  gboolean display_mask = FALSE;
  g_object_get(G_OBJECT(osd), "display-mask", &display_mask, nullptr);
  if (display_mask != TRUE) {
    std::cerr << "nvdsosd display-mask is not enabled" << std::endl;
    gst_object_unref(osd);
    gst_object_unref(pipeline);
    std::remove(kConfigPath);
    return 5;
  }

  GstPad* sink_pad = gst_element_get_static_pad(osd, "sink");
  GstPad* src_pad = gst_element_get_static_pad(osd, "src");
  if (sink_pad == nullptr || src_pad == nullptr) {
    std::cerr << "missing nvdsosd pad" << std::endl;
    if (sink_pad != nullptr) {
      gst_object_unref(sink_pad);
    }
    if (src_pad != nullptr) {
      gst_object_unref(src_pad);
    }
    gst_object_unref(osd);
    gst_object_unref(pipeline);
    std::remove(kConfigPath);
    return 6;
  }

  RenderState state;
  gst_pad_add_probe(
      sink_pad,
      GST_PAD_PROBE_TYPE_BUFFER,
      inspect_pre_osd,
      &state,
      nullptr);
  gst_pad_add_probe(
      src_pad,
      GST_PAD_PROBE_TYPE_BUFFER,
      inspect_post_osd,
      &state,
      nullptr);

  GstBus* bus = gst_element_get_bus(pipeline);
  if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    std::cerr << "pipeline failed to enter PLAYING" << std::endl;
    gst_object_unref(bus);
    gst_object_unref(src_pad);
    gst_object_unref(sink_pad);
    gst_object_unref(osd);
    gst_object_unref(pipeline);
    std::remove(kConfigPath);
    return 7;
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
                << (error != nullptr ? error->message : "unknown")
                << std::endl;
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
  gst_object_unref(sink_pad);
  gst_object_unref(osd);
  gst_object_unref(pipeline);
  std::remove(kConfigPath);

  const bool pre_frames_reached = state.pre_osd_frames >= kTargetFrames;
  const bool post_frames_reached = state.post_osd_frames >= kTargetFrames;
  const bool mask_metadata_observed = state.objects > 0U && state.masks > 0U;
  const bool masks_cover_objects =
      state.objects > 0U && state.masks == state.objects;
  const bool passed =
      !runtime_error && eos_observed && state.batch_meta_observed &&
      pre_frames_reached && post_frames_reached &&
      mask_metadata_observed && masks_cover_objects &&
      state.mask_elements > 0U && !state.invalid;

  std::cout << kResultPrefix
            << "{\"status\":\"" << (passed ? "passed" : "failed") << "\""
            << ",\"pre_osd_frames\":" << state.pre_osd_frames
            << ",\"post_osd_frames\":" << state.post_osd_frames
            << ",\"objects\":" << state.objects
            << ",\"masks\":" << state.masks
            << ",\"mask_elements\":" << state.mask_elements
            << ",\"eos_observed\":" << (eos_observed ? "true" : "false")
            << ",\"batch_meta_observed\":"
            << (state.batch_meta_observed ? "true" : "false")
            << ",\"display_mask_enabled\":true"
            << ",\"pre_osd_frames_reached\":"
            << (pre_frames_reached ? "true" : "false")
            << ",\"post_osd_frames_reached\":"
            << (post_frames_reached ? "true" : "false")
            << ",\"mask_metadata_observed\":"
            << (mask_metadata_observed ? "true" : "false")
            << ",\"masks_cover_objects\":"
            << (masks_cover_objects ? "true" : "false")
            << ",\"metadata_valid\":" << (!state.invalid ? "true" : "false")
            << ",\"invalid_reason\":\""
            << escape_json(state.invalid_reason) << "\""
            << ",\"segmentation_decode_executed\":true"
            << ",\"instance_metadata_executed\":true"
            << ",\"mask_rendering_executed\":true"
            << ",\"overlay_executed\":true"
            << ",\"annotated_video_written\":false"
            << ",\"dataset_used\":false"
            << ",\"validation_used\":false"
            << ",\"test_used\":false"
            << ",\"final_test_used\":false}"
            << std::endl;

  return passed ? 0 : 8;
}
"""
    return (
        template.replace("__TARGET_FRAMES__", str(EXPECTED_TARGET_FRAMES))
        .replace("__RESULT_PREFIX__", json.dumps(EXPECTED_RENDER_RESULT_PREFIX))
        .replace("__NVINFER_CONFIG__", json.dumps(nvinfer_config))
        .replace("__SAMPLE_PATH__", EXPECTED_SAMPLE_PATH)
    )


# ADD 2026-09-06: Mask rendering probe를 networkless container에서 compile/run한다.
def build_mask_render_shell_payload(config: DeepStreamSegmentationConfig) -> str:
    source = build_mask_render_probe_source(config)
    return "\n".join(
        (
            "set -euo pipefail",
            f"test \"$(sha256sum {EXPECTED_SAMPLE_PATH} | awk '{{print $1}}')\" = "
            f'"{EXPECTED_SAMPLE_SHA256}"',
            f'test "$(stat -c %s {EXPECTED_SAMPLE_PATH})" = "{EXPECTED_SAMPLE_BYTES}"',
            "gst-inspect-1.0 nvvideoconvert >/dev/null",
            "gst-inspect-1.0 nvdsosd >/dev/null",
            "cat > /tmp/c6_5d_mask_render_probe.cpp <<'C6_5D_CPP'",
            source,
            "C6_5D_CPP",
            "DS=/opt/nvidia/deepstream/deepstream",
            'export LD_LIBRARY_PATH="${DS}/lib:${LD_LIBRARY_PATH:-}"',
            "g++ -std=c++17 -O2 -Wall -Wextra "
            "$(pkg-config --cflags gstreamer-1.0) "
            '-I"${DS}/sources/includes" '
            "/tmp/c6_5d_mask_render_probe.cpp "
            '-L"${DS}/lib" -Wl,-rpath,"${DS}/lib" '
            "-lnvdsgst_meta -lnvds_meta "
            "$(pkg-config --libs gstreamer-1.0) "
            "-o /tmp/c6_5d_mask_render_probe",
            "set +e",
            "/tmp/c6_5d_mask_render_probe",
            "status=$?",
            "set -e",
            "rm -f /tmp/c6_5d_mask_render_probe.cpp "
            "/tmp/c6_5d_mask_render_probe "
            "/tmp/c6_5d_mask_render_nvinfer.txt",
            "exit ${status}",
            "",
        )
    )


# ADD 2026-09-06: Rendering container를 GPU/no-network/read-only-plan 경계로 고정한다.
def build_mask_render_docker_command(plan_path: Path) -> tuple[str, ...]:
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
        EXPECTED_RENDER_CONTAINER_LABEL,
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics",
        "-v",
        f"{plan_path.resolve()}:/model/model.plan:ro",
        "--entrypoint",
        "bash",
        EXPECTED_RENDER_RUNTIME_IMAGE,
        "-s",
    )


# ADD 2026-09-06: Container stdout의 single strict rendering JSON을 파싱한다.
def parse_mask_render_payload(
    stdout: str,
    *,
    prefix: str = EXPECTED_RENDER_RESULT_PREFIX,
) -> dict[str, Any]:
    matches = [line[len(prefix) :] for line in stdout.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError("C6-5D mask rendering result count must be exactly one.")

    raw: object = json.loads(matches[0])
    if not isinstance(raw, dict):
        raise ValueError("C6-5D mask rendering result must be object.")
    return cast(dict[str, Any], raw)


# ADD 2026-09-06: Rendering payload가 mask-only overlay acceptance를 만족하는지 검증한다.
def validate_mask_render_payload(payload: dict[str, Any]) -> None:
    expected_fields = {
        "status",
        "pre_osd_frames",
        "post_osd_frames",
        "objects",
        "masks",
        "mask_elements",
        "eos_observed",
        "batch_meta_observed",
        "display_mask_enabled",
        "pre_osd_frames_reached",
        "post_osd_frames_reached",
        "mask_metadata_observed",
        "masks_cover_objects",
        "metadata_valid",
        "invalid_reason",
        "segmentation_decode_executed",
        "instance_metadata_executed",
        "mask_rendering_executed",
        "overlay_executed",
        "annotated_video_written",
        "dataset_used",
        "validation_used",
        "test_used",
        "final_test_used",
    }
    if set(payload) != expected_fields:
        raise ValueError("C6-5D mask rendering payload fields changed.")

    integer_fields = (
        "pre_osd_frames",
        "post_osd_frames",
        "objects",
        "masks",
        "mask_elements",
    )
    if any(type(payload[name]) is not int for name in integer_fields):
        raise TypeError("C6-5D mask rendering counters must be int.")

    if not isinstance(payload["status"], str):
        raise TypeError("C6-5D mask rendering status must be str.")
    if not isinstance(payload["invalid_reason"], str):
        raise TypeError("C6-5D mask rendering invalid_reason must be str.")

    true_flags = (
        "eos_observed",
        "batch_meta_observed",
        "display_mask_enabled",
        "pre_osd_frames_reached",
        "post_osd_frames_reached",
        "mask_metadata_observed",
        "masks_cover_objects",
        "metadata_valid",
        "segmentation_decode_executed",
        "instance_metadata_executed",
        "mask_rendering_executed",
        "overlay_executed",
    )
    false_flags = (
        "annotated_video_written",
        "dataset_used",
        "validation_used",
        "test_used",
        "final_test_used",
    )
    if any(type(payload[name]) is not bool for name in (*true_flags, *false_flags)):
        raise TypeError("C6-5D mask rendering flags must be bool.")

    if (
        payload["status"] != "passed"
        or payload["pre_osd_frames"] < EXPECTED_TARGET_FRAMES
        or payload["post_osd_frames"] < EXPECTED_TARGET_FRAMES
        or payload["objects"] <= 0
        or payload["masks"] != payload["objects"]
        or payload["mask_elements"] <= 0
        or any(payload[name] is not True for name in true_flags)
        or any(payload[name] is not False for name in false_flags)
        or payload["invalid_reason"] != ""
    ):
        raise ValueError("C6-5D mask rendering runtime acceptance failed.")
