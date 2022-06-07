// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "opencl_gpu/plugin/program.hpp"
#include "opencl_gpu/plugin/common_utils.hpp"

#include "ngraph/op/experimental_detectron_topkrois.hpp"

#include "opencl_gpu/primitives/experimental_detectron_topk_rois.hpp"
#include "opencl_gpu/primitives/arg_max_min.hpp"

namespace ov {
namespace runtime {
namespace opencl_gpu {

namespace {

using namespace cldnn;

void CreateExperimentalDetectronTopKROIsOp(Program &p,
                                           const std::shared_ptr<ngraph::op::v6::ExperimentalDetectronTopKROIs> &op) {
    p.ValidateInputs(op, {2});
    auto input_primitives = p.GetInputPrimitiveIDs(op);
    auto max_rois = op->get_max_rois();
    auto layer_name = layer_type_name_ID(op);
    auto argmax_layer_name = layer_name + "_topk";
    auto top_k_indices = arg_max_min(argmax_layer_name,
                                     {input_primitives[1]}, arg_max_min::max, max_rois, arg_max_min::batch,
                                     arg_max_min::sort_by_values, false, "", cldnn::padding(), cldnn::data_types::i32);


    p.AddPrimitive(top_k_indices);
    p.AddInnerPrimitiveToProfiler(top_k_indices, argmax_layer_name, op);

    auto experimental_detectron_topk_layer = cldnn::experimental_detectron_topk_rois(layer_name,
                                                                                     {input_primitives[0],
                                                                                      argmax_layer_name}, max_rois);

    p.AddPrimitive(experimental_detectron_topk_layer);
    p.AddPrimitiveToProfiler(experimental_detectron_topk_layer, op);
}

} // namespace

REGISTER_FACTORY_IMPL(v6, ExperimentalDetectronTopKROIs);

} // namespace opencl_gpu
} // namespace runtime
} // namespace ov
