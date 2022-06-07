// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <opencl_gpu/primitives/input_layout.hpp>
#include <opencl_gpu/primitives/quantize.hpp>
#include <opencl_gpu/primitives/eltwise.hpp>
#include <opencl_gpu/primitives/data.hpp>
#include <opencl_gpu/primitives/permute.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct permute_params {
    tensor in_shape;
    tensor out_shape;
    std::vector<uint16_t> permute_order;
    tensor eltw_in_shape;
    data_types data_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

struct permute_reorder_params {
    tensor in_shape;
    std::vector<uint16_t> permute_order1;
    std::vector<uint16_t> permute_order2;
    data_types permute_type;
    data_types output_type;
    format permute_format;
    format output_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class PermuteFusingTest : public ::BaseFusingTest<permute_params> {
public:

    void execute(permute_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(permute_params& p) {
        return layout{ p.data_type, p.input_format, p.in_shape, padding{} };
    }

    layout get_per_channel_layout(permute_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.out_shape.feature[0], 1, 1 } };
    }
};

class PermuteReorderFusingTest : public ::BaseFusingTest<permute_reorder_params> {
public:

    void execute(permute_reorder_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);
        compare(network_not_fused, network_fused, p, true);
    }

    layout get_input_layout(permute_reorder_params& p) {
        return layout{ p.permute_type, p.permute_format, p.in_shape, padding{} };
    }
};
}  // namespace

/* ------------------------------------------------------------------------------------------------------------ */
/* ---------------------------------------- PERMUTE FUSE cases ------------------------------------------------ */
/* ------------------------------------------------------------------------------------------------------------ */
#define CASE_PERMUTE_F32_0 { 1, 16, 2, 2 }, { 1, 16, 2, 2 }, { 0, 1, 2, 3 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_1 { 1, 15, 16, 16 }, { 1, 15, 16, 16 }, { 0, 1, 2, 3 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_2 { 1, 8, 16, 16 }, { 16, 16, 8, 1 }, { 3, 2, 1, 0 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_3 { 1, 1, 3, 4 }, { 1, 3, 4, 1 }, { 1, 2, 3, 0 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_4 { 2, 16, 16, 16 }, { 2, 16, 16, 16 }, { 0, 1, 2, 3 }, tensor{ 0 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_5 { 1, 32, 4, 5 }, { 32, 4, 5, 1 }, { 1, 2, 3, 0 }, tensor{ 0 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_6 { 1, 16, 4, 5 }, { 5, 16, 4, 1 }, { 3, 1, 2, 0 }, tensor{ 0 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_7 { 1, 16, 1, 1 }, { 1, 1, 1, 16 }, { 2, 3, 0, 1 }, tensor{ 0 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx

#define CASE_PERMUTE_F16_0 { 1, 16, 4, 5 }, { 1, 16, 4, 5 }, { 0, 1, 2, 3 }, tensor{ 0 }, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F16_1 { 2, 16, 4, 5 }, { 16, 4, 5, 2 }, { 1, 2, 3, 0 }, tensor{ 0 }, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F16_2 { 1, 32, 2, 3 }, { 2, 3, 32, 1 }, { 2, 3, 1, 0 }, tensor{ 0 }, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F16_3 { 3, 16, 1, 1 }, { 1, 1, 16, 3 }, { 3, 2, 1, 0 }, tensor{ 0 }, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F16_4 { 2, 15, 4, 5 }, { 4, 2, 5, 15 }, { 2, 0, 3, 1 }, tensor{ 0 }, data_types::f16, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_F16_5 { 1, 15, 1, 2 }, { 15, 2, 1, 1 }, { 1, 3, 2, 0 }, tensor{ 0 }, data_types::f16, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_F16_6 { 1, 15, 4, 4 }, { 4, 4, 1, 15 }, { 2, 3, 0, 1 }, tensor{ 0 }, data_types::f16, format::bfyx, data_types::f32, format::bfyx

#define CASE_PERMUTE_S8_0 { 1, 15, 4, 5 }, { 1, 15, 4, 5 }, { 0, 1, 2, 3 }, tensor{ 0 }, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_S8_1 { 1, 15, 4, 5 }, { 5, 4, 15, 1 }, { 3, 2, 1, 0 }, tensor{ 0 }, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_S8_2 { 1, 16, 1, 2 }, { 1, 1, 16, 2 }, { 2, 0, 1, 3 }, tensor{ 0 }, data_types::i8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_S8_3 { 1, 16, 2, 2 }, { 2, 2, 16, 1 }, { 2, 3, 1, 0 }, tensor{ 0 }, data_types::i8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_U8_0 { 1, 15, 4, 5 }, { 15, 5, 1, 4 }, { 1, 3, 0, 2 }, tensor{ 0 }, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_U8_1 { 1, 15, 16, 16 }, { 15, 16, 1, 16 }, { 1, 2, 0, 3 }, tensor{ 0 }, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_U8_2 { 1, 32, 5, 4 }, { 1, 32, 5, 4 }, { 0, 1, 2, 3 }, tensor{ 0 }, data_types::u8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_U8_3 { 1, 16, 4, 5 }, { 5, 4, 16, 1 }, { 3, 2, 1, 0 }, tensor{ 0 }, data_types::u8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx

// 3d
#define CASE_PERMUTE_F32_3D_0 { 1, 15, 4, 4, 5 }, { 1, 15, 4, 4, 5 }, { 0, 1, 2, 3, 4 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F32_3D_1 { 2, 15, 2, 3, 4 }, { 15, 2, 3, 4, 2 }, { 1, 2, 3, 4, 0 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F32_3D_2 { 2, 16, 4, 4, 5 }, { 4, 2, 4, 5, 16 }, { 3, 0, 2, 4, 1 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F32_3D_3 { 1, 32, 4, 2, 2 }, { 2, 2, 32, 1, 4 }, { 4, 3, 1, 0, 2 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F32_3D_4 { 1, 16, 1, 1, 1 }, { 1, 1, 1, 16, 1 }, { 2, 4, 0, 1, 3 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_PERMUTE_F16_3D_0 { 1, 15, 4, 4, 5 }, { 1, 15, 4, 4, 5 }, { 0, 1, 2, 3, 4 }, tensor{ 0 }, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F16_3D_1 { 2, 15, 4, 3, 4 }, { 4, 4, 2, 15, 3 }, { 2, 4, 0, 1, 3 }, tensor{ 0 }, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F16_3D_2 { 2, 16, 4, 4, 3 }, { 2, 4, 3, 16, 4 }, { 0, 3, 4, 1, 2 }, tensor{ 0 }, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F16_3D_3 { 1, 32, 4, 2, 1 }, { 2, 32, 4, 1, 1 }, { 3, 1, 2, 4, 0 }, tensor{ 0 }, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F16_3D_4 { 16, 16, 1, 1, 1 },{ 1, 16, 1, 1, 16 },{ 4, 0, 3, 2, 1 }, tensor{ 0 }, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_PERMUTE_S8_3D_0 { 1, 15, 4, 4, 5 }, { 1, 15, 4, 4, 5 }, { 0, 1, 2, 3, 4 }, tensor{ 0 }, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_S8_3D_1 { 2, 15, 4, 3, 4 }, { 4, 4, 15, 2, 3 }, { 4, 2, 1, 0, 3 }, tensor{ 0 }, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_S8_3D_2 { 2, 16, 4, 4, 3 }, { 2, 4, 3, 16, 4 }, { 0, 3, 4, 1, 2 }, tensor{ 0 }, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_S8_3D_3 { 1, 32, 4, 2, 1 }, { 2, 32, 4, 1, 1 }, { 3, 1, 2, 4, 0 }, tensor{ 0 }, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_U8_3D_0 { 16, 16, 1, 1, 1 }, { 1, 1, 16, 16, 1 }, { 2, 4, 0, 1, 3 }, tensor{ 0 }, data_types::u8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_U8_3D_1 { 16, 16, 1, 1, 1 }, { 1, 1, 1, 16, 16 }, { 4, 3, 2, 1, 0 }, tensor{ 0 }, data_types::u8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_U8_3D_2 { 2, 16, 4, 4, 3 }, { 4, 2, 4, 3, 16 }, { 3, 0, 2, 4, 1 }, tensor{ 0 }, data_types::u8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_U8_3D_3 { 1, 32, 4, 2, 1 }, { 1, 2, 32, 1, 4 }, { 4, 3, 1, 0, 2 }, tensor{ 0 }, data_types::u8, format::bfzyx, data_types::f32, format::bfzyx

// permute_tile_8x8_4x4
#define CASE_PERMUTE_TILE_8x8_4x4_4D_0 { 1, 8, 8, 2 }, { 1, 2, 8, 8 }, { 0, 3, 1, 2 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_TILE_8x8_4x4_4D_1 { 1, 5, 8, 2 }, { 1, 2, 5, 8 }, { 0, 3, 1, 2 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_TILE_8x8_4x4_4D_2 { 1, 8, 5, 2 }, { 1, 2, 8, 5 }, { 0, 3, 1, 2 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_TILE_8x8_4x4_4D_3 { 1, 5, 5, 2 }, { 1, 2, 5, 5 }, { 0, 3, 1, 2 }, tensor{ 0 }, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_TILE_8x8_4x4_5D_0 { 1, 8, 8, 2, 2 }, { 1, 2, 8, 8, 2 }, { 0, 4, 1, 2, 3 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_TILE_8x8_4x4_5D_1 { 1, 5, 8, 2, 2 }, { 1, 2, 5, 8, 2 }, { 0, 4, 1, 2, 3 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_TILE_8x8_4x4_5D_2 { 1, 8, 5, 2, 2 }, { 1, 2, 8, 5, 2 }, { 0, 4, 1, 2, 3 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_TILE_8x8_4x4_5D_3 { 1, 5, 5, 2, 2 }, { 1, 2, 5, 5, 2 }, { 0, 4, 1, 2, 3 }, tensor{ 0 }, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_TILE_8x8_4x4_6D_0 { 1, 8, 8, 2, 2, 2 }, { 1, 2, 8, 8, 2, 2 }, { 0, 5, 1, 2, 3, 4 }, tensor{ 0 }, data_types::f32, format::bfwzyx, data_types::f32, format::bfwzyx
#define CASE_PERMUTE_TILE_8x8_4x4_6D_1 { 1, 5, 8, 2, 2, 2 }, { 1, 2, 5, 8, 2, 2 }, { 0, 5, 1, 2, 3, 4 }, tensor{ 0 }, data_types::f32, format::bfwzyx, data_types::f32, format::bfwzyx
#define CASE_PERMUTE_TILE_8x8_4x4_6D_2 { 1, 8, 5, 2, 2, 2 }, { 1, 2, 8, 5, 2, 2 }, { 0, 5, 1, 2, 3, 4 }, tensor{ 0 }, data_types::f32, format::bfwzyx, data_types::f32, format::bfwzyx
#define CASE_PERMUTE_TILE_8x8_4x4_6D_3 { 1, 5, 5, 2, 2, 2 }, { 1, 2, 5, 5, 2, 2 }, { 0, 5, 1, 2, 3, 4 }, tensor{ 0 }, data_types::f32, format::bfwzyx, data_types::f32, format::bfwzyx

// permute_tile_8x8_4x4_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_0 { 1, 16, 16, 2 }, { 1, 2, 16, 16 }, { 0, 3, 1, 2 }, tensor{ 0 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::b_fs_yx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_1 { 1, 15, 16, 2 }, { 1, 2, 15, 16 }, { 0, 3, 1, 2 }, tensor{ 0 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::b_fs_yx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_2 { 1, 16,  3, 2 }, { 1, 2, 16,  3 }, { 0, 3, 1, 2 }, tensor{ 0 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::b_fs_yx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_3 { 1,  5,  7, 2 }, { 1, 2,  5,  7 }, { 0, 3, 1, 2 }, tensor{ 0 }, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::b_fs_yx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_0 { 1, 16, 16, 2, 2 }, { 1, 2, 16, 16, 2 }, { 0, 4, 1, 2, 3 }, tensor{ 0 }, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::b_fs_zyx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_1 { 1, 15, 16, 2, 2 }, { 1, 2, 15, 16, 2 }, { 0, 4, 1, 2, 3 }, tensor{ 0 }, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::b_fs_zyx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_2 { 1, 16,  3, 2, 2 }, { 1, 2, 16,  3, 2 }, { 0, 4, 1, 2, 3 }, tensor{ 0 }, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::b_fs_zyx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_3 { 1,  5,  7, 2, 2 }, { 1, 2,  5,  7, 2 }, { 0, 4, 1, 2, 3 }, tensor{ 0 }, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::b_fs_zyx_fsv16

class permute_activation_scale_eltwise: public PermuteFusingTest {};
TEST_P(permute_activation_scale_eltwise, basic) {
    auto p = GetParam();

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, p.out_shape })),
        data("scale_data", get_mem(get_per_channel_layout(p), 5e-1f)),
        permute("permute", "input", p.permute_order),
        scale("scale", "permute", "scale_data"),
        activation("actv", "scale", activation_func::relu),
        eltwise("eltwise", { "actv", "eltwise_data" }, eltwise_mode::sum, p.data_type),
        reorder("reorder_bfyx", "eltwise", p.default_format, p.default_type)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_activation_scale_eltwise, ::testing::ValuesIn(std::vector<permute_params>{
    permute_params{ CASE_PERMUTE_F32_0, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_1, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_2, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_3, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_4, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_5, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_6, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_7, 2, 5 },

    permute_params{ CASE_PERMUTE_F16_0, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_1, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_2, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_3, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_4, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_5, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_6, 2, 5 },

    permute_params{ CASE_PERMUTE_S8_0, 2, 5 },
    permute_params{ CASE_PERMUTE_S8_1, 2, 5 },
    permute_params{ CASE_PERMUTE_S8_2, 2, 5 },
    permute_params{ CASE_PERMUTE_S8_3, 2, 5 },

    permute_params{ CASE_PERMUTE_U8_0, 2, 5 },
    permute_params{ CASE_PERMUTE_U8_1, 2, 5 },
    permute_params{ CASE_PERMUTE_U8_2, 2, 5 },
    permute_params{ CASE_PERMUTE_U8_3, 2, 5 },

    permute_params{ CASE_PERMUTE_F32_3D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_3D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_3D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_3D_3, 2, 5 },
    permute_params{ CASE_PERMUTE_F32_3D_4, 2, 5 },

    permute_params{ CASE_PERMUTE_F16_3D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_3D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_3D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_3D_3, 2, 5 },
    permute_params{ CASE_PERMUTE_F16_3D_4, 2, 5 },

    permute_params{ CASE_PERMUTE_S8_3D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_S8_3D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_S8_3D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_S8_3D_3, 2, 5 },

    permute_params{ CASE_PERMUTE_U8_3D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_U8_3D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_U8_3D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_U8_3D_3, 2, 5 },

    // Fusing tests for permute_tile_8x8_4x4
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_4D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_4D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_4D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_4D_3, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_5D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_5D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_5D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_5D_3, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_6D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_6D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_6D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_6D_3, 2, 5 },

    // Fusing tests for permute_tile_8x8_4x4_fsv16
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_3, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_0, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_1, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_2, 2, 5 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_3, 2, 5 },
}));

class permute_quant_u8: public PermuteFusingTest {};
TEST_P(permute_quant_u8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        permute("permute", "input", p.permute_order),
        quantize("quant", "permute", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
        reorder("reorder_bfyx", "quant", p.default_format, p.default_type)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_quant_u8, ::testing::ValuesIn(std::vector<permute_params>{
    permute_params{ CASE_PERMUTE_F32_0, 2, 3 },
    permute_params{ CASE_PERMUTE_F32_1, 2, 3 },

    permute_params{ CASE_PERMUTE_F16_0, 2, 3 },
    permute_params{ CASE_PERMUTE_F16_1, 2, 3 },
}));

class permute_scale_actv_eltw_scale_actv_quant_i8: public PermuteFusingTest {};
TEST_P(permute_scale_actv_eltw_scale_actv_quant_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scale1_data", get_mem(get_per_channel_layout(p), 1e-1f)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("eltw_data", get_mem(layout(p.data_type, p.input_format, p.out_shape))),
        data("scale2_data", get_mem(get_per_channel_layout(p), 1e-1f)),
        permute("permute", "input", p.permute_order),
        scale("scale1", "permute", "scale1_data"),
        activation("actv1", "scale1", activation_func::relu),
        eltwise("eltw", { "actv1", "eltw_data" }, eltwise_mode::sum, p.data_type),
        scale("scale2", "eltw", "scale2_data"),
        activation("actv2", "scale2", activation_func::relu),
        quantize("quant", "actv2", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("out", "quant", p.default_format, p.default_type)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_scale_actv_eltw_scale_actv_quant_i8, ::testing::ValuesIn(std::vector<permute_params>{
    permute_params{ CASE_PERMUTE_F32_0, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_1, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_2, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_3, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_4, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_5, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_6, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_7, 2, 8 },

    permute_params{ CASE_PERMUTE_F16_0, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_1, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_2, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_3, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_4, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_5, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_6, 2, 8 },

    permute_params{ CASE_PERMUTE_S8_0, 2, 8 },
    permute_params{ CASE_PERMUTE_S8_1, 2, 8 },
    permute_params{ CASE_PERMUTE_S8_2, 2, 8 },
    permute_params{ CASE_PERMUTE_S8_3, 2, 8 },

    permute_params{ CASE_PERMUTE_U8_0, 2, 8 },
    permute_params{ CASE_PERMUTE_U8_1, 2, 8 },
    permute_params{ CASE_PERMUTE_U8_2, 2, 8 },
    permute_params{ CASE_PERMUTE_U8_3, 2, 8 },

    permute_params{ CASE_PERMUTE_F32_3D_0, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_3D_1, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_3D_2, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_3D_3, 2, 8 },
    permute_params{ CASE_PERMUTE_F32_3D_4, 2, 8 },

    permute_params{ CASE_PERMUTE_F16_3D_0, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_3D_1, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_3D_2, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_3D_3, 2, 8 },
    permute_params{ CASE_PERMUTE_F16_3D_4, 2, 8 },

    permute_params{ CASE_PERMUTE_S8_3D_0, 2, 8 },
    permute_params{ CASE_PERMUTE_S8_3D_1, 2, 8 },
    permute_params{ CASE_PERMUTE_S8_3D_2, 2, 8 },
    permute_params{ CASE_PERMUTE_S8_3D_3, 2, 8 },

    permute_params{ CASE_PERMUTE_U8_3D_0, 2, 8 },
    permute_params{ CASE_PERMUTE_U8_3D_1, 2, 8 },
    permute_params{ CASE_PERMUTE_U8_3D_2, 2, 8 },
    permute_params{ CASE_PERMUTE_U8_3D_3, 2, 8 },
}));

class permute_scale_eltwise_actv_scale_actv: public PermuteFusingTest {};
TEST_P(permute_scale_eltwise_actv_scale_actv, basic) {
    auto p = GetParam();

    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, p.out_shape })),
        data("scale_data1", get_mem(get_per_channel_layout(p), 1e-1f)),
        data("scale_data2", get_mem(get_per_channel_layout(p), 1e-1f)),
        permute("permute", "input", p.permute_order),
        scale("scale1", "permute", "scale_data1"),
        activation("actv1", "scale1", activation_func::relu),
        eltwise("eltwise", { "actv1", "eltwise_data" }, eltwise_mode::sum, p.default_type),
        scale("scale2", "eltwise", "scale_data2"),
        activation("actv2", "scale2", activation_func::relu),
        reorder("reorder_bfyx", "actv2", p.default_format, p.default_type)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_scale_eltwise_actv_scale_actv, ::testing::ValuesIn(std::vector<permute_params>{
    permute_params{ CASE_PERMUTE_F32_0, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_1, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_2, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_3, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_4, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_5, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_6, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_7, 2, 7 },

    permute_params{ CASE_PERMUTE_F16_0, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_1, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_2, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_3, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_4, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_5, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_6, 2, 7 },

    permute_params{ CASE_PERMUTE_S8_0, 2, 7 },
    permute_params{ CASE_PERMUTE_S8_1, 2, 7 },
    permute_params{ CASE_PERMUTE_S8_2, 2, 7 },
    permute_params{ CASE_PERMUTE_S8_3, 2, 7 },

    permute_params{ CASE_PERMUTE_U8_0, 2, 7 },
    permute_params{ CASE_PERMUTE_U8_1, 2, 7 },
    permute_params{ CASE_PERMUTE_U8_2, 2, 7 },
    permute_params{ CASE_PERMUTE_U8_3, 2, 7 },

    permute_params{ CASE_PERMUTE_F32_3D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_3D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_3D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_3D_3, 2, 7 },
    permute_params{ CASE_PERMUTE_F32_3D_4, 2, 7 },

    permute_params{ CASE_PERMUTE_F16_3D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_3D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_3D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_3D_3, 2, 7 },
    permute_params{ CASE_PERMUTE_F16_3D_4, 2, 7 },

    permute_params{ CASE_PERMUTE_S8_3D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_S8_3D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_S8_3D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_S8_3D_3, 2, 7 },

    permute_params{ CASE_PERMUTE_U8_3D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_U8_3D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_U8_3D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_U8_3D_3, 2, 7 },

    // Fusing tests for permute_tile_8x8_4x4
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_4D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_4D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_4D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_4D_3, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_5D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_5D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_5D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_5D_3, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_6D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_6D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_6D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_6D_3, 2, 7 },

    // Fusing tests for permute_tile_8x8_4x4_fsv16
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_3, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_0, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_1, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_2, 2, 7 },
    permute_params{ CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_3, 2, 7 },
}));

/* ------------------------------------------------------------------------------------------------------------ */
/* ---------------------------- PERMUTE FUSE REDUNDANT REORDER cases ------------------------------------------ */
/* ------------------------------------------------------------------------------------------------------------ */

#define CASE_PERMUTE_REORDER_F32_0 { 1, 16, 32, 2 },   { 0, 3, 2, 1 },    { 0, 3, 2, 1 },    data_types::f32, data_types::f32, format::b_fs_yx_fsv16,  format::bfyx
#define CASE_PERMUTE_REORDER_F32_1 { 2, 7, 9, 27 },  { 0, 3, 2, 1 },    { 0, 3, 2, 1 },    data_types::f32, data_types::f32, format::b_fs_yx_fsv4,   format::bfyx
#define CASE_PERMUTE_REORDER_F32_2 { 1, 16, 4, 5, 16 }, { 0, 3, 2, 1, 4 }, { 0, 3, 2, 1, 4 }, data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_F16_0 { 1, 16, 2, 4 },     { 0, 3, 2, 1 },    { 0, 3, 2, 1 },    data_types::f16, data_types::f16, format::b_fs_yx_fsv16,  format::bfyx
#define CASE_PERMUTE_REORDER_F16_1 { 1, 16, 4, 5, 16 }, { 0, 4, 2, 3, 1 }, { 0, 1, 2, 3, 4 }, data_types::f16, data_types::f16, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_F16_2 { 1, 5, 1, 2, 14 },  { 0, 3, 2, 1, 4 }, { 0, 3, 2, 1, 4 }, data_types::f16, data_types::f16, format::b_fs_zyx_fsv16, format::bfzyx

// type change
#define CASE_PERMUTE_REORDER_S8_TO_F32_0 { 1, 15, 4, 5 },    { 0, 3, 2, 1 },    { 0, 3, 2, 1 },    data_types::i8, data_types::f32, format::b_fs_yx_fsv4,   format::bfyx
#define CASE_PERMUTE_REORDER_S8_TO_F32_1 { 1, 2, 15, 4, 5 }, { 0, 3, 2, 1, 4 }, { 0, 3, 2, 1, 4 }, data_types::i8, data_types::f32, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_F32_TO_F16_0 { 1, 5, 1, 2, 14 }, { 0, 4, 2, 3, 1 }, { 0, 1, 2, 3, 4 }, data_types::f32, data_types::f16, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_U8_TO_F16_0 { 1, 17, 1, 2, 7 },  { 0, 4, 2, 3, 1 }, { 0, 1, 2, 3, 4 }, data_types::u8, data_types::f16, format::b_fs_zyx_fsv16, format::bfzyx

// dim change
#define CASE_PERMUTE_REORDER_4D_TO_5D_F32_0 { 1, 16, 8, 16 }, { 1, 2, 0, 3 }, { 0, 3, 1, 4, 2 }, data_types::f32, data_types::f32, format::bfyx, format::bfzyx
#define CASE_PERMUTE_REORDER_4D_TO_6D_F32_1 { 1, 16, 8, 16 }, { 0, 2, 3, 1 }, { 0, 3, 5, 4, 1, 2 }, data_types::f32, data_types::f32, format::bfyx, format::bfwzyx
#define CASE_PERMUTE_REORDER_5D_TO_4D_F32_0 { 1, 16, 4, 5, 18 },{ 0, 4, 2, 3, 1 }, { 0, 3, 1, 2 }, data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfyx
#define CASE_PERMUTE_REORDER_5D_TO_4D_F32_1 { 1, 16, 4, 5, 16 },{ 0, 2, 3, 4, 1 }, { 0, 3, 1, 2 }, data_types::f32, data_types::f32, format::bfzyx, format::bfyx
#define CASE_PERMUTE_REORDER_5D_TO_6D_F32_2 { 1, 16, 8, 4, 16 }, { 0, 4, 2, 3, 1 }, { 0, 3, 5, 4, 1, 2 }, data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfwzyx
#define CASE_PERMUTE_REORDER_6D_TO_4D_F32_0 { 1, 16, 4, 5, 4, 16 }, { 0, 2, 5, 3, 4, 1 }, { 0, 3, 1, 2 }, data_types::f32, data_types::f32, format::bfwzyx, format::bfyx
#define CASE_PERMUTE_REORDER_6D_TO_5D_F32_1 { 1, 16, 4, 5, 4, 16 }, { 0, 2, 5, 3, 4, 1 }, { 0, 3, 4, 1, 2 }, data_types::f32, data_types::f32, format::bfwzyx, format::bfzyx

// permute_opt for blocked format
#define CASE_PERMUTE_REORDER_TILED_F32_0 { 1, 256, 2, 64 }, { 0, 3, 1, 2 }, { 0, 2, 3, 1 },  data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F32_1 { 1, 78, 2, 259 }, { 0, 3, 1, 2 }, { 0, 2, 3, 1 },  data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F32_2 { 1, 48, 1, 3, 259 }, { 0, 4, 1, 2, 3 }, { 0, 2, 3, 4, 1 },  data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfzyx

// permute_opt for blocked format => reorder to differnt dim
#define CASE_PERMUTE_REORDER_TILED_F32_3 { 1, 45, 1, 3, 259 }, { 0, 4, 1, 2, 3 }, { 0, 2, 3, 1 },  data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F32_4 { 2, 273, 19, 19 }, { 0, 3, 1, 2 }, { 0, 2, 3, 1 },  data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F32_5 { 2, 546, 2, 2 }, { 0, 3, 1, 2 }, { 0, 2, 3, 1 },  data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx

// permute opt for blocked format => reorder to different dim/type
#define CASE_PERMUTE_REORDER_TILED_I8_4 { 1, 45, 1, 3, 259 }, { 0, 4, 1, 2, 3 }, { 0, 2, 3, 1 },  data_types::i8, data_types::f32, format::b_fs_zyx_fsv16, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F16_5 { 1, 48, 3, 256 }, { 0, 3, 1, 2 }, { 0, 2, 4, 3, 1 },  data_types::f16, data_types::f32, format::b_fs_yx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_TILED_F16_6 { 1, 48, 2, 3, 256 }, { 0, 4, 1, 2, 3 }, { 0, 2, 5, 4, 3, 1 },  data_types::f16, data_types::f32, format::b_fs_zyx_fsv16, format::bfwzyx

// permute opt for non_blocked format => reorder to differnt dim/type
#define CASE_PERMUTE_REORDER_TILED_F16_7 { 1, 48, 2, 3, 256 }, { 0, 4, 1, 2, 3 }, { 0, 2, 3, 1 },  data_types::f16, data_types::f32, format::bfzyx, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F16_8 { 1, 28, 2, 2, 3, 256 }, { 0, 5, 1, 2, 3, 4 }, { 0, 2, 3, 1 },  data_types::f16, data_types::f32, format::bfwzyx, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F16_9 { 1, 24, 2, 3, 256 }, { 0, 4, 1, 2, 3 }, { 0, 2, 3, 1 },  data_types::f16, data_types::f32, format::bfzyx, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F16_10 { 1, 35, 3, 253 }, { 0, 3, 1, 2 }, { 0, 2, 4, 3, 1 },  data_types::f16, data_types::f32, format::bfyx, format::bfzyx
#define CASE_PERMUTE_REORDER_TILED_F16_11 { 1, 32, 3, 253 }, { 0, 3, 1, 2 }, { 0, 2, 4, 5, 3, 1 },  data_types::f16, data_types::f32, format::bfyx, format::bfwzyx

class permute_redundant_reorder : public PermuteReorderFusingTest {};
TEST_P(permute_redundant_reorder, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        permute("permute1", "input",    p.permute_order1),
        reorder("reorder1", "permute1", p.output_format, p.output_type), // to be fused
        permute("permute2", "reorder1", p.permute_order2)                  // dummy last op to make reorder fused
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_redundant_reorder, ::testing::ValuesIn(std::vector<permute_reorder_params>{
    permute_reorder_params{ CASE_PERMUTE_REORDER_F32_0, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F32_1, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F32_2, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F16_0, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F16_1, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F16_2, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_S8_TO_F32_0, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_S8_TO_F32_1, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F32_TO_F16_0, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_U8_TO_F16_0, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_4D_TO_5D_F32_0, 3, 3 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_4D_TO_6D_F32_1, 3, 3 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_5D_TO_4D_F32_0, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_5D_TO_4D_F32_1, 3, 3 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_5D_TO_6D_F32_2, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_6D_TO_4D_F32_0, 3, 3 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_6D_TO_5D_F32_1, 3, 3 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F32_0, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F32_1, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F32_2, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F32_3, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F32_4, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_I8_4, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_5, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_6, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_7, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_8, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_9, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_10, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_11, 3, 4 },
}));

class permute_act_reorder : public PermuteReorderFusingTest {};

TEST_P(permute_act_reorder, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        permute("permute1", "input",    p.permute_order1),
        activation("activation", "permute1", activation_func::abs),
        reorder("reorder1", "activation", p.output_format, p.output_type),    // to be fused
        permute("permute2", "reorder1", p.permute_order2)                // dummy last op to make reorder fused
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_act_reorder, ::testing::ValuesIn(std::vector<permute_reorder_params>{
    permute_reorder_params{ CASE_PERMUTE_REORDER_F32_0, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F32_1, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F32_2, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F16_0, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F16_1, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_F16_2, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_4D_TO_5D_F32_0, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_4D_TO_6D_F32_1, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_5D_TO_4D_F32_0, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_5D_TO_4D_F32_1, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_5D_TO_6D_F32_2, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_6D_TO_4D_F32_0, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_6D_TO_5D_F32_1, 3, 4 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F32_0, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F32_1, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F32_2, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F32_3, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_5, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_6, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_7, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_8, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_9, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_10, 3, 5 },
    permute_reorder_params{ CASE_PERMUTE_REORDER_TILED_F16_11, 3, 5 },
}));
