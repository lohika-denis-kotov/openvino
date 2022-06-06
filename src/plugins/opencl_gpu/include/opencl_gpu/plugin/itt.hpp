// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines openvino domains for tracing
 * @file itt.hpp
 */

#pragma once

#include <openvino/itt.hpp>

namespace ov {
namespace runtime {
namespace opencl_gpu {
namespace itt {
namespace domains {
    OV_ITT_DOMAIN(opencl_gpu_plugin);
}  // namespace domains
}  // namespace itt
}  // namespace opencl_gpu
}  // namespace runtime
}  // namespace ov
