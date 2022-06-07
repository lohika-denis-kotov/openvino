// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "opencl_gpu/plugin/program.hpp"
#include "opencl_gpu/plugin/common_utils.hpp"

#include "ngraph/op/grn.hpp"

#include "opencl_gpu/primitives/grn.hpp"

namespace ov {
namespace runtime {
namespace opencl_gpu {

static void CreateGRNOp(Program& p, const std::shared_ptr<ngraph::op::v0::GRN>& op) {
    p.ValidateInputs(op, {1});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto primitive = cldnn::grn(layerName,
                                inputPrimitives[0],
                                op->get_bias(),
                                DataTypeFromPrecision(op->get_output_element_type(0)),
                                op->get_friendly_name());

    p.AddPrimitive(primitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, GRN);

}  // namespace opencl_gpu
}  // namespace runtime
}  // namespace ov
