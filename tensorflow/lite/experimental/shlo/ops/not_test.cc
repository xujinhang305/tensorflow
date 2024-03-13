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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using shlo_ref::testing::StatusIs;
using testing::ElementsAreArray;
using testing::NanSensitiveFloatEq;
using testing::Pointwise;

namespace shlo_ref {

namespace {

struct Not {
  template <class T>
  T operator()(T v) const {
    return ~v;
  }
  template <>
  bool operator()(bool v) const {
    return !v;
  }
} not_ref;

template <class T>
struct BoolAndIntNotTest : ::testing::Test {};

TYPED_TEST_SUITE(BoolAndIntNotTest, NonQuantizedBoolIntTestTypes,
                 TestParamNames);

TYPED_TEST(BoolAndIntNotTest, BoolAndIntTensorsWork) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape({2, 3, 4});
  Vector<StorageT> input_data = RandomBuffer<TypeParam::kStorage>(shape);
  Vector<StorageT> output_data(shape.NumElements());

  Tensor input_tensor{
      .type = TensorType{.shape = shape, .element_type = TypeParam::kStorage},
      .data = input_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  Vector<StorageT> expected_data(shape.NumElements());
  absl::c_transform(input_data, expected_data.begin(), not_ref);

  auto op = Create(NotOp::Attributes{});
  ASSERT_OK(Prepare(op, input_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, input_tensor, output_tensor));
  EXPECT_THAT(output_data, Pointwise(NanSensitiveFloatEq(), expected_data));
}

template <class T>
struct FloatNotTest : ::testing::Test {};

TYPED_TEST_SUITE(FloatNotTest, NonQuantizedFloatTestTypes, TestParamNames);

TYPED_TEST(FloatNotTest, Fails) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape({2, 3, 4});
  Vector<StorageT> input_data = RandomBuffer<TypeParam::kStorage>(shape);
  Vector<StorageT> output_data(shape.NumElements());

  Tensor input_tensor{
      .type = TensorType{.shape = shape, .element_type = TypeParam::kStorage},
      .data = input_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  auto op = Create(NotOp::Attributes{});
  EXPECT_THAT(Prepare(op, input_tensor, output_tensor),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

template <class T>
struct QuantizedNotTest : ::testing::Test {};

TYPED_TEST_SUITE(QuantizedNotTest, QuantizedTestTypes, TestParamNames);

TYPED_TEST(QuantizedNotTest, PerTensorFails) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape({2, 3, 4});
  Vector<StorageT> input_data = RandomBuffer<TypeParam::kStorage>(shape);
  Vector<StorageT> output_data(shape.NumElements());
  const ExpressedT scale = static_cast<ExpressedT>(1.5);
  const StorageT zero_point = static_cast<StorageT>(5);
  const QuantizedTensorElementType tensor_type =
      QuantizedTensorElementType::PerTensor<TypeParam::kStorage,
                                            TypeParam::kExpressed>(scale,
                                                                   zero_point);
  Tensor input_tensor{
      .type = QuantizedTensorType{.shape = shape, .element_type = tensor_type},
      .data = input_data.data()};
  Tensor output_tensor{
      .type = QuantizedTensorType{.shape = shape, .element_type = tensor_type},
      .data = output_data.data()};

  auto op = Create(NotOp::Attributes{});
  EXPECT_THAT(Prepare(op, input_tensor, output_tensor),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TYPED_TEST(QuantizedNotTest, PerAxisFails) {
  using StorageT = typename TypeParam::StorageT;
  using ExpressedT = typename TypeParam::ExpressedT;

  const Shape shape({4, 3, 2});
  const int quantized_dimension = 2;
  Vector<ExpressedT> empty_scales;
  Vector<StorageT> empty_zero_points;
  const QuantizedTensorElementType tensor_type =
      QuantizedTensorElementType::PerAxis<TypeParam::kStorage,
                                          TypeParam::kExpressed>(
          empty_scales, empty_zero_points, quantized_dimension);
  Tensor input_tensor{
      .type = QuantizedTensorType{.shape = shape, .element_type = tensor_type},
      .data = nullptr};
  Tensor output_tensor = input_tensor;

  auto op = Create(NotOp::Attributes{});
  EXPECT_THAT(Prepare(op, input_tensor, output_tensor),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace shlo_ref
