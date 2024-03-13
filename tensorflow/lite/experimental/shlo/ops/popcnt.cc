/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/experimental/shlo/ops/popcnt.h"

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct Popcnt {
  template <class T>
  T operator()(T v) const {
    constexpr T one = static_cast<T>(1);
    constexpr T zero = static_cast<T>(0);
    return v < zero ? -one : (v > zero ? one : v);
  }

  template <>
  F16 operator()(F16 v) const {
    return static_cast<F16>(operator()(static_cast<float>(v)));
  }

  template <>
  BF16 operator()(BF16 v) const {
    return static_cast<BF16>(operator()(static_cast<float>(v)));
  }
};

PopcntOp Create(PopcntOp::Attributes) { return {}; }

absl::Status Prepare(PopcntOp& op, const Tensor& input, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));

  if (!input.IsQuantized() && input.StorageType() == DataType::kI1) {
    return absl::FailedPreconditionError(
        "stablehlo.popcnt does not support boolean tensors.");
  }
  if (input.IsPerAxisQuantized()) {
    return absl::FailedPreconditionError(
        "stablehlo.popcnt does not support per axis quantization.");
  }
  if (BaselineType(input.element_type()) !=
      BaselineType(output.element_type())) {
    return absl::FailedPreconditionError(
        "stablehlo.popcnt constraint (C1) is not satisfied (incompatible "
        "baseline types).");
  }
  return absl::OkStatus();
}

absl::Status Evaluate(PopcntOp& op, const Tensor& input, Tensor& output) {
  Popcnt popcnt;
  if (input.IsPerTensorQuantized()) {
    DISPATCH_QUANTIZED(detail::DequantizeOpQuantizePerTensor,
                       input.quantized_tensor_element_type().StorageType(),
                       input.quantized_tensor_element_type().ExpressedType(),
                       popcnt, input, output)
  } else {
    DISPATCH_INT_FLOAT(detail::EvaluateNoQuantization,
                       input.tensor_element_type(), popcnt, input, output);
  }
  return absl::OkStatus();
}

};  // namespace shlo_ref
