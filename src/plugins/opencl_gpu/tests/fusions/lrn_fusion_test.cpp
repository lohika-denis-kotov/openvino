// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <opencl_gpu/primitives/input_layout.hpp>
#include <opencl_gpu/primitives/quantize.hpp>
#include <opencl_gpu/primitives/eltwise.hpp>
#include <opencl_gpu/primitives/data.hpp>
#include <opencl_gpu/primitives/lrn.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {

struct lrn_test_params {
    tensor in_shape;
    data_types data_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
    lrn_norm_region lrn_type;
    std::string kernel_name;
};

class LrnFusingTest : public ::BaseFusingTest<lrn_test_params> {
public:
    void execute(lrn_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));

        build_options options;
        implementation_desc lrn_impl = { p.input_format, p.kernel_name };
        options.set_option(build_option::optimize_data(true));
        options.set_option(build_option::force_implementations({ { "lrn_norm", lrn_impl } }));
        network network_fused(this->engine, this->topology_fused, options);
        network network_not_fused(this->engine, this->topology_non_fused, this->bo_not_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        ASSERT_FALSE(network_fused.get_primitives_info().empty());
        ASSERT_FALSE(network_not_fused.get_primitives_info().empty());

        auto find_lrn = [&](primitive_info& p) -> bool {
            if (p.original_id == "lrn_norm" || p.original_id == "reorder")
                return true;
            return false;
        };

        auto pi_fused = network_fused.get_primitives_info();
        auto pi_not_fused = network_not_fused.get_primitives_info();
        auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_lrn);
        auto info_not_fused = std::find_if(pi_not_fused.begin(), pi_not_fused.end(), find_lrn);

        ASSERT_TRUE(info_fused != pi_fused.end());
        ASSERT_TRUE(info_not_fused != pi_not_fused.end());

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(lrn_test_params& p) {
        return layout{ p.data_type, p.input_format, p.in_shape };
    }

    layout get_per_channel_layout(lrn_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.in_shape.feature[0], 1, 1 } };
    }
};

}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- LRN cases -------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

#define CASE_LRN_FP32_1 { 2, 16, 4, 4 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_LRN_FP32_2 { 8, 16, 4, 4 }, data_types::f32, format::yxfb, data_types::f32, format::yxfb
#define CASE_LRN_FP32_3 { 2, 16, 4, 4 }, data_types::f32, format::byxf, data_types::f32, format::byxf
#define CASE_LRN_FP32_4 { 2, 16, 4, 4 }, data_types::f32, format::b_fs_yx_fsv4, data_types::f32, format::bfyx
#define CASE_LRN_FP32_5 { 2, 16, 4, 4 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx

#define CASE_LRN_FP32_TO_FP16_1 { 2, 16, 5, 5 }, data_types::f32, format::bfyx, data_types::f16, format::bfyx
#define CASE_LRN_FP32_TO_FP16_2 { 2, 16, 5, 5 }, data_types::f32, format::byxf, data_types::f16, format::byxf
#define CASE_LRN_FP32_TO_FP16_3 { 8, 16, 4, 4 }, data_types::f32, format::yxfb, data_types::f16, format::byxf
#define CASE_LRN_FP32_TO_FP16_4 { 2, 16, 4, 4 }, data_types::f32, format::b_fs_yx_fsv4, data_types::f16, format::bfyx
#define CASE_LRN_FP32_TO_FP16_5 { 2, 16, 4, 4 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f16, format::bfyx

#define CASE_LRN_FP16_1 { 2, 16, 4, 4 }, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_LRN_FP16_2 { 8, 16, 4, 4 }, data_types::f16, format::yxfb, data_types::f16, format::yxfb
#define CASE_LRN_FP16_3 { 2, 16, 4, 4 }, data_types::f16, format::byxf, data_types::f16, format::byxf
#define CASE_LRN_FP16_4 { 2, 16, 4, 4 }, data_types::f16, format::b_fs_yx_fsv4, data_types::f16, format::bfyx
#define CASE_LRN_FP16_5 { 2, 16, 4, 4 }, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::bfyx

class lrn_fp32_quantize_u8_scale_activation : public LrnFusingTest {};
TEST_P(lrn_fp32_quantize_u8_scale_activation, basic) {
    auto p = GetParam();

    uint32_t size = 5;
    float k = 1.0f;
    float alpha = (float)9.9e-05;
    float beta = 0.75;

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        data("scale_data", get_mem(get_single_element_layout(p), 1.0f / 255)),
        lrn("lrn_norm", "input", size, k, alpha, beta, p.lrn_type),
        quantize("quantize", "lrn_norm", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
        scale("scale", "quantize", "scale_data"),
        activation("activation", "scale", activation_func::exp),
        reorder("reorder", "activation", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

TEST_P(lrn_fp32_quantize_u8_scale_activation, per_channel) {
    auto p = GetParam();

    uint32_t size = 5;
    float k = 1.0f;
    float alpha = (float)9.9e-05;
    float beta = 0.75;

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / 255)),
        lrn("lrn_norm", "input", size, k, alpha, beta, p.lrn_type),
        quantize("quantize", "lrn_norm", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
        scale("scale", "quantize", "scale_data"),
        activation("activation", "scale", activation_func::exp),
        reorder("reorder", "activation", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, lrn_fp32_quantize_u8_scale_activation, ::testing::ValuesIn(std::vector<lrn_test_params>{
    // InputDataType = FP32   OutputDataType = FP32
    lrn_test_params{ CASE_LRN_FP32_1, 2, 5, lrn_norm_region_across_channel, "lrn_ref" },
    lrn_test_params{ CASE_LRN_FP32_1, 2, 5, lrn_norm_region_within_channel, "lrn_gpu_within_channel_opt" },
    lrn_test_params{ CASE_LRN_FP32_1, 2, 5, lrn_norm_region_within_channel, "lrn_gpu_within_channel" },
    lrn_test_params{ CASE_LRN_FP32_1, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_ref" },
    lrn_test_params{ CASE_LRN_FP32_1, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features" },
    lrn_test_params{ CASE_LRN_FP32_2, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_yxfb_b8_opt" },
    lrn_test_params{ CASE_LRN_FP32_3, 2, 5, lrn_norm_region_within_channel, "lrn_within_channel_byxf_opt" },
    lrn_test_params{ CASE_LRN_FP32_4, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features" },
    lrn_test_params{ CASE_LRN_FP32_5, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features_fsv16" },

    // InputDataType = FP32   OutputDataType = FP16
    lrn_test_params{ CASE_LRN_FP32_TO_FP16_1, 2, 5, lrn_norm_region_across_channel, "lrn_ref" },
    lrn_test_params{ CASE_LRN_FP32_TO_FP16_1, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features" },
    lrn_test_params{ CASE_LRN_FP32_TO_FP16_1, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_ref" },
    lrn_test_params{ CASE_LRN_FP32_TO_FP16_1, 2, 5, lrn_norm_region_within_channel, "lrn_gpu_within_channel_opt" },
    lrn_test_params{ CASE_LRN_FP32_TO_FP16_1, 2, 5, lrn_norm_region_within_channel, "lrn_gpu_within_channel" },
    lrn_test_params{ CASE_LRN_FP32_TO_FP16_3, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_yxfb_b8_opt" },
    lrn_test_params{ CASE_LRN_FP32_TO_FP16_4, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features" },
    lrn_test_params{ CASE_LRN_FP32_TO_FP16_5, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features_fsv16" },
}));

class lrn_fp32_quantize_i8_scale_activation : public LrnFusingTest {};
TEST_P(lrn_fp32_quantize_i8_scale_activation, basic) {
    auto p = GetParam();

    uint32_t size = 5;
    float k = 1.0f;
    float alpha = (float)9.9e-05;
    float beta = 0.75;

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p),  127)),
        data("scale_data", get_mem(get_single_element_layout(p), 1.0f / 255)),
        lrn("lrn_norm", "input", size, k, alpha, beta, p.lrn_type),
        scale("scale", "lrn_norm", "scale_data"),
        activation("activation", "scale", activation_func::exp),
        quantize("quantize", "activation", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::i8),
        reorder("reorder", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, lrn_fp32_quantize_i8_scale_activation, ::testing::ValuesIn(std::vector<lrn_test_params>{
    // InputDataType = FP32   OutputDataType = INT8
    lrn_test_params{ CASE_LRN_FP32_1, 2, 5, lrn_norm_region_within_channel, "lrn_gpu_within_channel_opt" },
    lrn_test_params{ CASE_LRN_FP32_1, 2, 5, lrn_norm_region_within_channel, "lrn_gpu_within_channel" },
    lrn_test_params{ CASE_LRN_FP32_1, 2, 5, lrn_norm_region_across_channel, "lrn_ref" },
    lrn_test_params{ CASE_LRN_FP32_1, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features" },
    lrn_test_params{ CASE_LRN_FP32_1, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_ref" },
    lrn_test_params{ CASE_LRN_FP32_2, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_yxfb_b8_opt" },
    lrn_test_params{ CASE_LRN_FP32_3, 2, 5, lrn_norm_region_within_channel, "lrn_within_channel_byxf_opt" },
    lrn_test_params{ CASE_LRN_FP32_4, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features" },
    lrn_test_params{ CASE_LRN_FP32_5, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features_fsv16" },

    // InputDataType = FP16   OutputDataType = INT8/UINT8 can't be tested for now, because quantize
    // primitive doesn't support input type FP16 while fusing (prepare_quantization.cpp :114 -> prepare_primitive_fusing.cpp :474)
}));

class lrn_fp32_scale_activation_quantize_u8 : public LrnFusingTest {};
TEST_P(lrn_fp32_scale_activation_quantize_u8, basic) {
    auto p = GetParam();

    uint32_t size = 5;
    float k = 1.0f;
    float alpha = (float)9.9e-05;
    float beta = 0.75;

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        data("scale_data", get_mem(get_single_element_layout(p), 1.0f / 255)),
        lrn("lrn_norm", "input", size, k, alpha, beta, p.lrn_type),
        scale("scale", "lrn_norm", "scale_data"),
        activation("activation", "scale", activation_func::exp),
        quantize("quantize", "activation", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
        reorder("reorder", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, lrn_fp32_scale_activation_quantize_u8, ::testing::ValuesIn(std::vector<lrn_test_params>{
    // InputDataType = FP32   OutputDataType = UINT8
    lrn_test_params{ CASE_LRN_FP32_1, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_ref" },
    lrn_test_params{ CASE_LRN_FP32_1, 2, 5, lrn_norm_region_within_channel, "lrn_gpu_within_channel_opt" },
    lrn_test_params{ CASE_LRN_FP32_1, 2, 5, lrn_norm_region_within_channel, "lrn_gpu_within_channel" },
    lrn_test_params{ CASE_LRN_FP32_1, 2, 5, lrn_norm_region_across_channel, "lrn_ref" },
    lrn_test_params{ CASE_LRN_FP32_1, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features" },
    lrn_test_params{ CASE_LRN_FP32_2, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_yxfb_b8_opt" },
    lrn_test_params{ CASE_LRN_FP32_3, 2, 5, lrn_norm_region_within_channel, "lrn_within_channel_byxf_opt" },
    lrn_test_params{ CASE_LRN_FP32_4, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features" },
    lrn_test_params{ CASE_LRN_FP32_5, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features_fsv16" },
}));

class lrn_fp16_scale_activation : public LrnFusingTest {};
TEST_P(lrn_fp16_scale_activation, basic) {
    auto p = GetParam();

    uint32_t size = 5;
    float k = 1.0f;
    float alpha = (float)9.9e-05;
    float beta = 0.75;

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scale_data", get_mem(get_single_element_layout(p), 1.0f / 255)),
        lrn("lrn_norm", "input", size, k, alpha, beta, p.lrn_type),
        scale("scale", "lrn_norm", "scale_data"),
        activation("activation", "scale", activation_func::exp),
        reorder("reorder", "activation", p.default_format, data_types::f32)
    );

    tolerance = 1e-05f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, lrn_fp16_scale_activation, ::testing::ValuesIn(std::vector<lrn_test_params>{
    // InputDataType = FP16   OutputDataType = FP16
    lrn_test_params{ CASE_LRN_FP16_1, 2, 4, lrn_norm_region_within_channel, "lrn_gpu_within_channel_opt" },
    lrn_test_params{ CASE_LRN_FP16_1, 2, 4, lrn_norm_region_within_channel, "lrn_gpu_within_channel" },
    lrn_test_params{ CASE_LRN_FP16_1, 2, 4, lrn_norm_region_across_channel, "lrn_ref" },
    lrn_test_params{ CASE_LRN_FP16_1, 2, 4, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features" },
    lrn_test_params{ CASE_LRN_FP16_1, 2, 4, lrn_norm_region_across_channel, "lrn_gpu_across_channel_ref" },
    lrn_test_params{ CASE_LRN_FP16_3, 2, 4, lrn_norm_region_within_channel, "lrn_within_channel_byxf_opt" },
    lrn_test_params{ CASE_LRN_FP16_4, 2, 4, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features" },
    lrn_test_params{ CASE_LRN_FP16_5, 2, 4, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features_fsv16" },
}));
