// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "opencl_gpu/primitives/condition.hpp"
#include "opencl_gpu/primitives/loop.hpp"
#include "opencl_gpu/primitives/data.hpp"
#include "opencl_gpu/primitives/input_layout.hpp"
#include "opencl_gpu/primitives/prior_box.hpp"


namespace cldnn {
namespace common {
void register_implementations();

namespace detail {

#define REGISTER_COMMON(prim)           \
    struct attach_##prim##_common {     \
        attach_##prim##_common();       \
    }

REGISTER_COMMON(condition);
REGISTER_COMMON(data);
REGISTER_COMMON(input_layout);
REGISTER_COMMON(loop);
REGISTER_COMMON(prior_box);

#undef REGISTER_COMMON

}  // namespace detail
}  // namespace common
}  // namespace cldnn
