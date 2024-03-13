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

#include "tensorflow/lite/experimental/shlo/ops/log_plus_one.h"

#include <cmath>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/bf16.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/f16.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct LogPlusOne {
  template <class T>
  T operator()(T v) const {
    return std::log1p(v);
  }

  template <>
  F16 operator()(F16 v) const {
    return F16(operator()(static_cast<float>(v)));
  }

  template <>
  BF16 operator()(BF16 v) const {
    return BF16(operator()(static_cast<float>(v)));
  }
};

LogPlusOneOp Create(LogPlusOneOp::Attributes) { return {}; }

absl::Status Prepare(LogPlusOneOp& op, const Tensor& input, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));
  if (!input.IsQuantized() && IsInteger(input.StorageType())) {
    return absl::FailedPreconditionError(
        "stablehlo.log_plus_one does not support integer tensor types.");
  }
  if (input.IsPerAxisQuantized()) {
    return absl::FailedPreconditionError(
        "stablehlo.log_plus_one does not support per axis quantization.");
  }
  if (BaselineType(input.element_type()) !=
      BaselineType(output.element_type())) {
    return absl::FailedPreconditionError(
        "stablehlo.log_plus_one constraint (C1) is not satisfied (incompatible "
        "baseline types).");
  }
  return absl::OkStatus();
}

absl::Status Evaluate(LogPlusOneOp& op, const Tensor& input, Tensor& output) {
  LogPlusOne log_plus_one;
  if (input.IsPerTensorQuantized()) {
    DISPATCH_QUANTIZED(detail::DequantizeOpQuantizePerTensor,
                       input.quantized_tensor_element_type().StorageType(),
                       input.quantized_tensor_element_type().ExpressedType(),
                       log_plus_one, input, output)
  } else {
    DISPATCH_FLOAT(detail::EvaluateNoQuantization, input.tensor_element_type(),
                   log_plus_one, input, output);
  }
  return absl::OkStatus();
}

};  // namespace shlo_ref
