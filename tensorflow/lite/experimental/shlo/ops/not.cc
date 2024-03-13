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

#include "tensorflow/lite/experimental/shlo/ops/not.h"

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/unary_elementwise.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

struct Not {
  template <class T>
  T operator()(T v) const {
    return ~v;
  }
  template <>
  bool operator()(bool v) const {
    return !v;
  }
};

NotOp Create(NotOp::Attributes) { return {}; }

absl::Status Prepare(NotOp& op, const Tensor& input, Tensor& output) {
  SHLO_REF_RETURN_ON_ERROR(Propagate(input.shape(), output.shape()));

  if (!input.IsQuantized() && IsFloat(input.StorageType())) {
    return absl::FailedPreconditionError(
        "stablehlo.not does not support floating point tensors.");
  }
  if (input.IsQuantized()) {
    return absl::FailedPreconditionError(
        "stablehlo.not does not support quantization.");
  }
  if (BaselineType(input.element_type()) !=
      BaselineType(output.element_type())) {
    return absl::FailedPreconditionError(
        "stablehlo.not constraint (C1) is not satisfied (different "
        "baseline types).");
  }
  return absl::OkStatus();
}

absl::Status Evaluate(NotOp& op, const Tensor& input, Tensor& output) {
  Not not_func;
  DISPATCH_BOOL_INT(detail::EvaluateNoQuantization, input.tensor_element_type(),
                    not_func, input, output);
  return absl::OkStatus();
}

};  // namespace shlo_ref
