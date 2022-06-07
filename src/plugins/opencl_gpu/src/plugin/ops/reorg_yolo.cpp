// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "opencl_gpu/plugin/program.hpp"
#include "opencl_gpu/plugin/common_utils.hpp"

#include "ngraph/op/reorg_yolo.hpp"

#include "opencl_gpu/primitives/reorg_yolo.hpp"

namespace ov {
namespace runtime {
namespace opencl_gpu {

static void CreateReorgYoloOp(Program& p, const std::shared_ptr<ngraph::op::v0::ReorgYolo>& op) {
    p.ValidateInputs(op, {1});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    uint32_t stride = op->get_strides()[0];

    auto reorgPrim = cldnn::reorg_yolo(layerName,
                                       inputPrimitives[0],
                                       stride,
                                       op->get_friendly_name());

    p.AddPrimitive(reorgPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, ReorgYolo);

}  // namespace opencl_gpu
}  // namespace runtime
}  // namespace ov
