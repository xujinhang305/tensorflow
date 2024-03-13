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

#include "tensorflow/lite/experimental/shlo/ops/count_leading_zeros.h"

#include <cstdint>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct CountLeadingZeros {
  template <class T>
  T Impl(T x) const {
    if (x < 0) {
      return 0;
    } else if (x == 0) {
      return 8 * sizeof(T);
    }
    T n = 8 * sizeof(T);
    T shift = n >> 1;
    while (x && shift) {
      T y = x >> shift;
      if (y != 0) {
        n -= shift;
        x = y;
      }
      shift >>= 1;
    }
    return n - x;
  }

  int64_t operator()(int64_t v) const { return Impl<int64_t>(v); }
  int32_t operator()(int32_t v) const { return Impl<int32_t>(v); }
  int16_t operator()(int16_t v) const { return Impl<int16_t>(v); }
  int8_t operator()(int8_t v) const { return Impl<int8_t>(v); }
};

CountLeadingZerosOp Create(CountLeadingZerosOp::Attributes) { return {}; }

absl::Status Prepare(CountLeadingZerosOp& op, const Tensor& input,
                     Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));

  if (input.IsQuantized()) {
    return absl::FailedPreconditionError(
        "stablehlo.count_leading_zeros does not support quantization.");
  }
  if (!IsInteger(input.StorageType())) {
    return absl::FailedPreconditionError(
        "stablehlo.count_leading_zeros only supports integer tensors.");
  }
  if (input.StorageType() != output.StorageType()) {
    return absl::FailedPreconditionError(
        "stablehlo.count_leading_zeros constraint (C1) is not satisfied "
        "(different tensor types).");
  }
  return absl::OkStatus();
}

absl::Status Evaluate(CountLeadingZerosOp& op, const Tensor& input,
                      Tensor& output) {
  CountLeadingZeros count_leading_zeros;
  DISPATCH_INT(detail::EvaluateNoQuantization, input.tensor_element_type(),
               count_leading_zeros, input, output);
  return absl::OkStatus();
}

};  // namespace shlo_ref
