#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {

constexpr int kNetworkWidth = 640;
constexpr int kNetworkHeight = 640;
constexpr int kPredictionChannels = 39;
constexpr int kAnchorCount = 8400;
constexpr int kClassCount = 3;
constexpr int kMaskChannels = 32;
constexpr int kPrototypeHeight = 160;
constexpr int kPrototypeWidth = 160;
constexpr float kNmsIouThreshold = 0.7F;
constexpr std::size_t kMaxDetections = 300;

struct Candidate {
  int anchor = -1;
  int class_id = -1;
  float confidence = 0.0F;
  float x1 = 0.0F;
  float y1 = 0.0F;
  float x2 = 0.0F;
  float y2 = 0.0F;
  std::array<float, kMaskChannels> mask_coefficients{};
};

// ADD 2026-09-05: Tensor value가 finite인지 공통으로 검증한다.
bool finite_value(float value) {
  return std::isfinite(value);
}

// ADD 2026-09-05: Decoder geometry를 bounded network coordinate로 제한한다.
float clamp_value(float value, float minimum, float maximum) {
  return std::min(maximum, std::max(minimum, value));
}

// ADD 2026-09-05: Mask logit을 overflow-safe sigmoid probability로 변환한다.
float stable_sigmoid(float value) {
  if (value >= 0.0F) {
    const float exponent = std::exp(-value);
    return 1.0F / (1.0F + exponent);
  }
  const float exponent = std::exp(value);
  return exponent / (1.0F + exponent);
}

// ADD 2026-09-05: Exact output layer name을 중복 없이 찾는다.
const NvDsInferLayerInfo& find_layer(
    const std::vector<NvDsInferLayerInfo>& layers,
    const char* expected_name) {
  const NvDsInferLayerInfo* match = nullptr;
  for (const auto& layer : layers) {
    if (layer.layerName == nullptr ||
        std::strcmp(layer.layerName, expected_name) != 0) {
      continue;
    }
    if (match != nullptr) {
      throw std::runtime_error("duplicate output layer name");
    }
    match = &layer;
  }
  if (match == nullptr) {
    throw std::runtime_error("required output layer missing");
  }
  return *match;
}

// ADD 2026-09-05: Raw output names, FLOAT dtype, dimensions와 network geometry를 검증한다.
void validate_layers(
    const NvDsInferLayerInfo& output0,
    const NvDsInferLayerInfo& output1,
    const NvDsInferNetworkInfo& network_info) {
  if (network_info.width != kNetworkWidth ||
      network_info.height != kNetworkHeight) {
    throw std::runtime_error("network geometry changed");
  }
  if (output0.buffer == nullptr || output1.buffer == nullptr) {
    throw std::runtime_error("output buffer is null");
  }
  if (output0.dataType != FLOAT || output1.dataType != FLOAT) {
    throw std::runtime_error("output data type must be FLOAT");
  }
  if (output0.inferDims.numDims != 2 ||
      output0.inferDims.d[0] != kPredictionChannels ||
      output0.inferDims.d[1] != kAnchorCount) {
    throw std::runtime_error("output0 dimensions changed");
  }
  if (output1.inferDims.numDims != 3 ||
      output1.inferDims.d[0] != kMaskChannels ||
      output1.inferDims.d[1] != kPrototypeHeight ||
      output1.inferDims.d[2] != kPrototypeWidth) {
    throw std::runtime_error("output1 dimensions changed");
  }
}

// ADD 2026-09-05: YOLO center-width-height box를 clipped network-space xyxy로 변환한다.
std::array<float, 4> xywh_to_xyxy(
    float center_x,
    float center_y,
    float width,
    float height) {
  if (!finite_value(center_x) || !finite_value(center_y) ||
      !finite_value(width) || !finite_value(height) ||
      width <= 0.0F || height <= 0.0F) {
    throw std::runtime_error("invalid YOLO box geometry");
  }
  const float half_width = width * 0.5F;
  const float half_height = height * 0.5F;
  return {
      clamp_value(center_x - half_width, 0.0F, static_cast<float>(kNetworkWidth)),
      clamp_value(center_y - half_height, 0.0F, static_cast<float>(kNetworkHeight)),
      clamp_value(center_x + half_width, 0.0F, static_cast<float>(kNetworkWidth)),
      clamp_value(center_y + half_height, 0.0F, static_cast<float>(kNetworkHeight)),
  };
}

// ADD 2026-09-05: Class-aware NMS용 axis-aligned box IoU를 계산한다.
float box_iou(const Candidate& left, const Candidate& right) {
  const float intersection_width =
      std::max(0.0F, std::min(left.x2, right.x2) - std::max(left.x1, right.x1));
  const float intersection_height =
      std::max(0.0F, std::min(left.y2, right.y2) - std::max(left.y1, right.y1));
  const float intersection = intersection_width * intersection_height;
  const float left_area =
      std::max(0.0F, left.x2 - left.x1) * std::max(0.0F, left.y2 - left.y1);
  const float right_area =
      std::max(0.0F, right.x2 - right.x1) * std::max(0.0F, right.y2 - right.y1);
  const float union_area = left_area + right_area - intersection;
  return union_area > 0.0F ? intersection / union_area : 0.0F;
}

// ADD 2026-09-05: Channel-major output0에서 confidence-filtered candidates를 복원한다.
std::vector<Candidate> decode_candidates(
    const float* output0,
    const NvDsInferParseDetectionParams& detection_params) {
  if (output0 == nullptr) {
    throw std::runtime_error("output0 pointer is null");
  }
  if (detection_params.numClassesConfigured != kClassCount ||
      detection_params.perClassPreclusterThreshold.size() != kClassCount) {
    throw std::runtime_error("configured class thresholds changed");
  }

  std::vector<Candidate> candidates;
  candidates.reserve(128);

  for (int anchor = 0; anchor < kAnchorCount; ++anchor) {
    int best_class = 0;
    float best_score = output0[4 * kAnchorCount + anchor];

    for (int class_id = 0; class_id < kClassCount; ++class_id) {
      const float score = output0[(4 + class_id) * kAnchorCount + anchor];
      if (!finite_value(score) || score < 0.0F || score > 1.0F) {
        throw std::runtime_error("class probability is invalid");
      }
      if (score > best_score) {
        best_score = score;
        best_class = class_id;
      }
    }

    if (best_score < detection_params.perClassPreclusterThreshold[best_class]) {
      continue;
    }

    const auto box = xywh_to_xyxy(
        output0[0 * kAnchorCount + anchor],
        output0[1 * kAnchorCount + anchor],
        output0[2 * kAnchorCount + anchor],
        output0[3 * kAnchorCount + anchor]);

    if (box[2] <= box[0] || box[3] <= box[1]) {
      continue;
    }

    Candidate candidate;
    candidate.anchor = anchor;
    candidate.class_id = best_class;
    candidate.confidence = best_score;
    candidate.x1 = box[0];
    candidate.y1 = box[1];
    candidate.x2 = box[2];
    candidate.y2 = box[3];

    for (int coeff = 0; coeff < kMaskChannels; ++coeff) {
      const float value = output0[(7 + coeff) * kAnchorCount + anchor];
      if (!finite_value(value)) {
        throw std::runtime_error("mask coefficient is non-finite");
      }
      candidate.mask_coefficients[coeff] = value;
    }

    candidates.push_back(candidate);
  }

  std::stable_sort(
      candidates.begin(),
      candidates.end(),
      [](const Candidate& left, const Candidate& right) {
        if (left.confidence != right.confidence) {
          return left.confidence > right.confidence;
        }
        return left.anchor < right.anchor;
      });
  return candidates;
}

// ADD 2026-09-05: Confidence order를 보존하는 class-aware IoU NMS와 max-detection cap을 적용한다.
std::vector<Candidate> class_aware_nms(const std::vector<Candidate>& candidates) {
  std::vector<Candidate> kept;
  kept.reserve(std::min(candidates.size(), kMaxDetections));

  for (const auto& candidate : candidates) {
    bool suppressed = false;
    for (const auto& accepted : kept) {
      if (candidate.class_id == accepted.class_id &&
          box_iou(candidate, accepted) > kNmsIouThreshold) {
        suppressed = true;
        break;
      }
    }
    if (suppressed) {
      continue;
    }
    kept.push_back(candidate);
    if (kept.size() == kMaxDetections) {
      break;
    }
  }
  return kept;
}

// ADD 2026-09-05: CHW prototype plane에서 clamped half-pixel bilinear sample을 계산한다.
float bilinear_sample(
    const float* prototypes,
    int channel,
    float prototype_x,
    float prototype_y) {
  if (prototypes == nullptr || channel < 0 || channel >= kMaskChannels ||
      !finite_value(prototype_x) || !finite_value(prototype_y)) {
    throw std::runtime_error("prototype sample request is invalid");
  }

  const float x = clamp_value(
      prototype_x,
      0.0F,
      static_cast<float>(kPrototypeWidth - 1));
  const float y = clamp_value(
      prototype_y,
      0.0F,
      static_cast<float>(kPrototypeHeight - 1));

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const int x1 = std::min(x0 + 1, kPrototypeWidth - 1);
  const int y1 = std::min(y0 + 1, kPrototypeHeight - 1);
  const float wx = x - static_cast<float>(x0);
  const float wy = y - static_cast<float>(y0);
  const std::size_t plane =
      static_cast<std::size_t>(kPrototypeHeight) * kPrototypeWidth;
  const std::size_t offset = static_cast<std::size_t>(channel) * plane;

  const auto at = [&](int row, int column) {
    const float value =
        prototypes[offset + static_cast<std::size_t>(row) * kPrototypeWidth + column];
    if (!finite_value(value)) {
      throw std::runtime_error("prototype contains non-finite value");
    }
    return value;
  };

  const float top = at(y0, x0) * (1.0F - wx) + at(y0, x1) * wx;
  const float bottom = at(y1, x0) * (1.0F - wx) + at(y1, x1) * wx;
  return top * (1.0F - wy) + bottom * wy;
}

// ADD 2026-09-05: Candidate coefficient×prototype logits를 bbox-local sigmoid mask probability로 복원한다.
NvDsInferInstanceMaskInfo build_instance_mask(
    const Candidate& candidate,
    const float* prototypes) {
  const float box_width = candidate.x2 - candidate.x1;
  const float box_height = candidate.y2 - candidate.y1;
  if (box_width <= 0.0F || box_height <= 0.0F) {
    throw std::runtime_error("cannot build mask for empty box");
  }

  const int mask_width = std::max(
      1,
      static_cast<int>(std::ceil(
          box_width * static_cast<float>(kPrototypeWidth) / kNetworkWidth)));
  const int mask_height = std::max(
      1,
      static_cast<int>(std::ceil(
          box_height * static_cast<float>(kPrototypeHeight) / kNetworkHeight)));
  const std::size_t mask_elements =
      static_cast<std::size_t>(mask_width) * mask_height;

  std::vector<float> mask_values(mask_elements);
  for (int row = 0; row < mask_height; ++row) {
    const float network_y =
        candidate.y1 +
        (static_cast<float>(row) + 0.5F) * box_height /
            static_cast<float>(mask_height);
    const float prototype_y =
        network_y * static_cast<float>(kPrototypeHeight) / kNetworkHeight - 0.5F;

    for (int column = 0; column < mask_width; ++column) {
      const float network_x =
          candidate.x1 +
          (static_cast<float>(column) + 0.5F) * box_width /
              static_cast<float>(mask_width);
      const float prototype_x =
          network_x * static_cast<float>(kPrototypeWidth) / kNetworkWidth - 0.5F;

      float logit = 0.0F;
      for (int channel = 0; channel < kMaskChannels; ++channel) {
        logit += candidate.mask_coefficients[channel] *
                 bilinear_sample(prototypes, channel, prototype_x, prototype_y);
      }
      if (!finite_value(logit)) {
        throw std::runtime_error("decoded mask logit is non-finite");
      }
      mask_values[static_cast<std::size_t>(row) * mask_width + column] =
          stable_sigmoid(logit);
    }
  }

  NvDsInferInstanceMaskInfo info{};
  info.classId = candidate.class_id;
  info.left = candidate.x1;
  info.top = candidate.y1;
  info.width = box_width;
  info.height = box_height;
  info.detectionConfidence = candidate.confidence;
  info.mask_width = static_cast<unsigned int>(mask_width);
  info.mask_height = static_cast<unsigned int>(mask_height);
  info.mask_size =
      static_cast<unsigned int>(mask_elements * sizeof(float));
  info.mask = new float[mask_elements];
  std::copy(mask_values.begin(), mask_values.end(), info.mask);
  return info;
}

// ADD 2026-09-05: Error/self-test path에서 parser-owned mask allocations를 회수한다.
void release_masks(std::vector<NvDsInferInstanceMaskInfo>& objects) {
  for (auto& object : objects) {
    delete[] object.mask;
    object.mask = nullptr;
    object.mask_size = 0;
  }
  objects.clear();
}

}  // namespace

extern "C" bool NvDsInferParseYolo11Seg(
    const std::vector<NvDsInferLayerInfo>& outputLayersInfo,
    const NvDsInferNetworkInfo& networkInfo,
    const NvDsInferParseDetectionParams& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList) {
  std::vector<NvDsInferInstanceMaskInfo> decoded;
  try {
    const auto& output0 = find_layer(outputLayersInfo, "output0");
    const auto& output1 = find_layer(outputLayersInfo, "output1");
    validate_layers(output0, output1, networkInfo);

    const auto candidates = decode_candidates(
        static_cast<const float*>(output0.buffer),
        detectionParams);
    const auto retained = class_aware_nms(candidates);
    const auto* prototypes = static_cast<const float*>(output1.buffer);

    decoded.reserve(retained.size());
    for (const auto& candidate : retained) {
      decoded.push_back(build_instance_mask(candidate, prototypes));
    }

    objectList = std::move(decoded);
    return true;
  } catch (const std::exception& error) {
    release_masks(decoded);
    std::cerr << "C6-5D parser error: " << error.what() << std::endl;
    return false;
  }
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo11Seg);

#ifdef C6_5D_PARSER_SELF_TEST

namespace {

// ADD 2026-09-05: Synthetic overlapping boxes로 class-aware IoU NMS behavior를 검증한다.
void test_class_aware_nms() {
  Candidate first;
  first.anchor = 1;
  first.class_id = 0;
  first.confidence = 0.9F;
  first.x1 = 100.0F;
  first.y1 = 100.0F;
  first.x2 = 300.0F;
  first.y2 = 300.0F;

  Candidate duplicate = first;
  duplicate.anchor = 2;
  duplicate.confidence = 0.8F;
  duplicate.x1 = 110.0F;
  duplicate.y1 = 110.0F;
  duplicate.x2 = 290.0F;
  duplicate.y2 = 290.0F;

  Candidate other_class = duplicate;
  other_class.anchor = 3;
  other_class.class_id = 1;
  other_class.confidence = 0.7F;

  const auto kept = class_aware_nms({first, duplicate, other_class});
  if (kept.size() != 2 || kept[0].anchor != 1 || kept[1].anchor != 3) {
    throw std::runtime_error("class-aware NMS self-test failed");
  }
  std::cout << "C6_5D_CPP_NMS_SELF_TEST=PASS" << std::endl;
}

// ADD 2026-09-05: Constant synthetic prototype로 bbox-local sigmoid mask geometry/value를 검증한다.
void test_mask_decode() {
  Candidate candidate;
  candidate.anchor = 4;
  candidate.class_id = 2;
  candidate.confidence = 0.75F;
  candidate.x1 = 100.0F;
  candidate.y1 = 120.0F;
  candidate.x2 = 300.0F;
  candidate.y2 = 280.0F;
  candidate.mask_coefficients[0] = 1.0F;

  std::vector<float> prototypes(
      static_cast<std::size_t>(kMaskChannels) *
          kPrototypeHeight * kPrototypeWidth,
      0.0F);
  std::fill(
      prototypes.begin(),
      prototypes.begin() +
          static_cast<std::ptrdiff_t>(kPrototypeHeight * kPrototypeWidth),
      1.0F);

  auto mask = build_instance_mask(candidate, prototypes.data());
  const float expected = stable_sigmoid(1.0F);
  const std::size_t elements =
      static_cast<std::size_t>(mask.mask_width) * mask.mask_height;

  if (mask.mask_width != 50 || mask.mask_height != 40 ||
      elements != 2000 || mask.mask == nullptr) {
    delete[] mask.mask;
    throw std::runtime_error("mask geometry self-test failed");
  }

  for (std::size_t index = 0; index < elements; ++index) {
    if (std::fabs(mask.mask[index] - expected) > 1e-6F) {
      delete[] mask.mask;
      throw std::runtime_error("mask probability self-test failed");
    }
  }

  delete[] mask.mask;
  mask.mask = nullptr;
  std::cout << "C6_5D_CPP_MASK_SELF_TEST=PASS" << std::endl;
}

}  // namespace

// ADD 2026-09-05: Decoder foundation C++ pure postprocess self-tests를 실행한다.
int main() {
  try {
    test_class_aware_nms();
    test_mask_decode();
    std::cout << "C6_5D_CPP_PARSER_SELF_TEST=PASS" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "C6_5D_CPP_PARSER_SELF_TEST=FAIL:" << error.what() << std::endl;
    return 1;
  }
}

#endif
