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
#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/ops/test_util.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/status_matcher.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

using shlo_ref::testing::StatusIs;
using testing::NanSensitiveFloatEq;
using testing::Pointwise;

namespace shlo_ref {

namespace {

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
} count_leading_zeros_ref;

template <class T>
struct CountLeadingZerosFunctorTest : ::testing::Test {};

using CountLeadingZerosTypes = ::testing::Types<int32_t, int16_t, int8_t>;

TYPED_TEST_SUITE(CountLeadingZerosFunctorTest, CountLeadingZerosTypes);

TYPED_TEST(CountLeadingZerosFunctorTest, GivesCorrectResults) {
  constexpr TypeParam byte_count = 8 * sizeof(TypeParam);
  EXPECT_EQ(count_leading_zeros_ref(std::numeric_limits<TypeParam>::lowest()),
            0);
  EXPECT_EQ(count_leading_zeros_ref(static_cast<TypeParam>(-1)), 0);
  EXPECT_EQ(count_leading_zeros_ref(static_cast<TypeParam>(0)), byte_count);
  EXPECT_EQ(count_leading_zeros_ref(static_cast<TypeParam>(1)), byte_count - 1);
  EXPECT_EQ(count_leading_zeros_ref(static_cast<TypeParam>(2)), byte_count - 2);
  EXPECT_EQ(count_leading_zeros_ref(std::numeric_limits<TypeParam>::max()), 1);
}

template <class T>
struct NonQuantizedCountLeadingZerosTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedCountLeadingZerosTest, NonQuantizedIntTestTypes,
                 TestParamNames);

TYPED_TEST(NonQuantizedCountLeadingZerosTest, IntTensorsWork) {
  using StorageT = typename TypeParam::StorageT;

  const Shape shape({2, 3, 4});
  Vector<StorageT> input_data = IotaBuffer<TypeParam::kStorage>(shape, -12);
  Vector<StorageT> output_data(shape.NumElements());

  Tensor input_tensor{
      .type = TensorType{.shape = shape, .element_type = TypeParam::kStorage},
      .data = input_data.data()};
  Tensor output_tensor{
      .type = TensorType{.shape = shape, .element_type = TypeParam::kStorage},
      .data = output_data.data()};

  Vector<StorageT> expected_data(shape.NumElements());
  absl::c_transform(input_data, expected_data.begin(), count_leading_zeros_ref);

  auto op = Create(CountLeadingZerosOp::Attributes{});
  ASSERT_OK(Prepare(op, input_tensor, output_tensor));
  ASSERT_OK(Evaluate(op, input_tensor, output_tensor));
  EXPECT_THAT(output_data, Pointwise(NanSensitiveFloatEq(), expected_data));
}

template <class T>
struct NonQuantizedFloatCountLeadingZerosTest : ::testing::Test {};

TYPED_TEST_SUITE(NonQuantizedFloatCountLeadingZerosTest,
                 NonQuantizedFloatTestTypes, TestParamNames);

TYPED_TEST(NonQuantizedFloatCountLeadingZerosTest, FloatTensorsFails) {
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

  auto op = Create(CountLeadingZerosOp::Attributes{});
  EXPECT_THAT(Prepare(op, input_tensor, output_tensor),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

template <class T>
struct QuantizedCountLeadingZerosTest : ::testing::Test {};

TYPED_TEST_SUITE(QuantizedCountLeadingZerosTest, QuantizedTestTypes,
                 TestParamNames);

TYPED_TEST(QuantizedCountLeadingZerosTest, PerTensorFails) {
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

  auto op = Create(CountLeadingZerosOp::Attributes{});
  EXPECT_THAT(Prepare(op, input_tensor, output_tensor),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TYPED_TEST(QuantizedCountLeadingZerosTest, PerAxisFails) {
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

  auto op = Create(CountLeadingZerosOp::Attributes{});
  EXPECT_THAT(Prepare(op, input_tensor, output_tensor),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace shlo_ref
