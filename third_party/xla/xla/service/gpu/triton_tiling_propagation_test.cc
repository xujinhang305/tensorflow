/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/triton_tiling_propagation.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "xla/tests/hlo_test_base.h"

namespace xla::gpu {
namespace {

using TritonTilingPropagationTest = HloTestBase;

TEST_F(TritonTilingPropagationTest, DimensionOrderWithTrivialDimension) {
  triton_fusion::DimensionOrder::Fragment fragment_1(0, 97);
  triton_fusion::DimensionOrder::Fragment fragment_2(0, 1);
  triton_fusion::DimensionOrder dimension_order_1 =
      triton_fusion::DimensionOrder::FromFragments({fragment_1, fragment_2});

  triton_fusion::DimensionOrder::Fragment fragment_3(0, 97);
  triton_fusion::DimensionOrder::Fragment fragment_4(1, 1);
  triton_fusion::DimensionOrder dimension_order_2 =
      triton_fusion::DimensionOrder::FromFragments({fragment_3, fragment_4});

  // They should be equivalent because fragment_2 and fragment_4 both have count
  // 1, so they don't affect the physical representation.
  EXPECT_TRUE(dimension_order_1.IsPhysicallyEquivalent(dimension_order_2));
}

TEST_F(TritonTilingPropagationTest, TensorIterationSpecWithTrivialDimension) {
  auto get_fragment = [](int64_t stride, int64_t count, int64_t slice_start,
                         int64_t sliced_count,
                         std::vector<int64_t> subfragments)
      -> TensorIterationSpec::IterationSpecFragment {
    TensorIterationSpec::IterationSpecFragment fragment;
    fragment.stride = stride;
    fragment.count = count;
    fragment.slice_start = slice_start;
    fragment.sliced_count = sliced_count;
    fragment.subfragments = subfragments;
    return fragment;
  };

  TensorIterationSpec::IterationSpecFragment fragment_1 =
      get_fragment(1, 97, 0, 97, {97});
  TensorIterationSpec spec_1;
  spec_1[0].push_back(fragment_1);

  TensorIterationSpec::IterationSpecFragment fragment_2 =
      get_fragment(1, 97, 0, 97, {97});
  TensorIterationSpec::IterationSpecFragment fragment_3 =
      get_fragment(97, 1, 0, 1, {1});
  TensorIterationSpec spec_2;
  spec_2[0].push_back(fragment_2);
  spec_2[1].push_back(fragment_3);

  // spec_2's extra dimension is degenerate, so it should have the same physical
  // representation as spec_1.
  EXPECT_TRUE(spec_1.IsPhysicallyEquivalent(spec_2));
}

}  // namespace
}  // namespace xla::gpu
