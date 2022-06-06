// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "opencl_gpu/plugin/program.hpp"
#include "opencl_gpu/plugin/common_utils.hpp"

#include "ngraph/op/reverse_sequence.hpp"

#include "opencl_gpu/primitives/reverse_sequence.hpp"

namespace ov {
namespace runtime {
namespace opencl_gpu {

static void CreateReverseSequenceOp(Program& p, const std::shared_ptr<ngraph::op::v0::ReverseSequence>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    size_t batch_axis = op->get_batch_axis();
    size_t seq_axis = op->get_sequence_axis();
    auto reverseSequencePrim = cldnn::reverse_sequence(layerName,
                                                       inputPrimitives[0],
                                                       inputPrimitives[1],
                                                       seq_axis,
                                                       batch_axis,
                                                       op->get_friendly_name());

    p.AddPrimitive(reverseSequencePrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, ReverseSequence);

}  // namespace opencl_gpu
}  // namespace runtime
}  // namespace ov
