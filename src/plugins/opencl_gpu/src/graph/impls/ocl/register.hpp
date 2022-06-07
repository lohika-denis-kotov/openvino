// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "opencl_gpu/primitives/activation.hpp"
#include "opencl_gpu/primitives/arg_max_min.hpp"
#include "opencl_gpu/primitives/average_unpooling.hpp"
#include "opencl_gpu/primitives/batch_to_space.hpp"
#include "opencl_gpu/primitives/binary_convolution.hpp"
#include "opencl_gpu/primitives/border.hpp"
#include "opencl_gpu/primitives/broadcast.hpp"
#include "opencl_gpu/primitives/concatenation.hpp"
#include "opencl_gpu/primitives/convolution.hpp"
#include "opencl_gpu/primitives/crop.hpp"
#include "opencl_gpu/primitives/custom_gpu_primitive.hpp"
#include "opencl_gpu/primitives/deconvolution.hpp"
#include "opencl_gpu/primitives/depth_to_space.hpp"
#include "opencl_gpu/primitives/detection_output.hpp"
#include "opencl_gpu/primitives/eltwise.hpp"
#include "opencl_gpu/primitives/experimental_detectron_roi_feature_extractor.hpp"
#include "opencl_gpu/primitives/experimental_detectron_topk_rois.hpp"
#include "opencl_gpu/primitives/fully_connected.hpp"
#include "opencl_gpu/primitives/gather.hpp"
#include "opencl_gpu/primitives/gather_nd.hpp"
#include "opencl_gpu/primitives/gather_elements.hpp"
#include "opencl_gpu/primitives/gemm.hpp"
#include "opencl_gpu/primitives/lrn.hpp"
#include "opencl_gpu/primitives/lstm.hpp"
#include "opencl_gpu/primitives/lstm_dynamic.hpp"
#include "opencl_gpu/primitives/max_unpooling.hpp"
#include "opencl_gpu/primitives/mutable_data.hpp"
#include "opencl_gpu/primitives/mvn.hpp"
#include "opencl_gpu/primitives/non_max_suppression.hpp"
#include "opencl_gpu/primitives/normalize.hpp"
#include "opencl_gpu/primitives/one_hot.hpp"
#include "opencl_gpu/primitives/permute.hpp"
#include "opencl_gpu/primitives/pooling.hpp"
#include "opencl_gpu/primitives/pyramid_roi_align.hpp"
#include "opencl_gpu/primitives/quantize.hpp"
#include "opencl_gpu/primitives/random_uniform.hpp"
#include "opencl_gpu/primitives/range.hpp"
#include "opencl_gpu/primitives/reduce.hpp"
#include "opencl_gpu/primitives/region_yolo.hpp"
#include "opencl_gpu/primitives/reorder.hpp"
#include "opencl_gpu/primitives/reorg_yolo.hpp"
#include "opencl_gpu/primitives/reshape.hpp"
#include "opencl_gpu/primitives/reverse_sequence.hpp"
#include "opencl_gpu/primitives/roi_align.hpp"
#include "opencl_gpu/primitives/roi_pooling.hpp"
#include "opencl_gpu/primitives/scale.hpp"
#include "opencl_gpu/primitives/scatter_update.hpp"
#include "opencl_gpu/primitives/scatter_elements_update.hpp"
#include "opencl_gpu/primitives/scatter_nd_update.hpp"
#include "opencl_gpu/primitives/select.hpp"
#include "opencl_gpu/primitives/shuffle_channels.hpp"
#include "opencl_gpu/primitives/slice.hpp"
#include "opencl_gpu/primitives/softmax.hpp"
#include "opencl_gpu/primitives/space_to_batch.hpp"
#include "opencl_gpu/primitives/strided_slice.hpp"
#include "opencl_gpu/primitives/tile.hpp"
#include "opencl_gpu/primitives/resample.hpp"
#include "opencl_gpu/primitives/gather_tree.hpp"
#include "opencl_gpu/primitives/lstm_dynamic_input.hpp"
#include "opencl_gpu/primitives/lstm_dynamic_timeloop.hpp"
#include "opencl_gpu/primitives/grn.hpp"
#include "opencl_gpu/primitives/ctc_greedy_decoder.hpp"
#include "opencl_gpu/primitives/convert_color.hpp"
#include "generic_layer.hpp"


namespace cldnn {
namespace ocl {
void register_implementations();

namespace detail {

#define REGISTER_OCL(prim)               \
    struct attach_##prim##_impl {        \
        attach_##prim##_impl();          \
    }

REGISTER_OCL(activation);
REGISTER_OCL(arg_max_min);
REGISTER_OCL(average_unpooling);
REGISTER_OCL(batch_to_space);
REGISTER_OCL(binary_convolution);
REGISTER_OCL(border);
REGISTER_OCL(broadcast);
REGISTER_OCL(concatenation);
REGISTER_OCL(convolution);
REGISTER_OCL(crop);
REGISTER_OCL(custom_gpu_primitive);
REGISTER_OCL(data);
REGISTER_OCL(deconvolution);
REGISTER_OCL(deformable_conv);
REGISTER_OCL(deformable_interp);
REGISTER_OCL(depth_to_space);
REGISTER_OCL(detection_output);
REGISTER_OCL(experimental_detectron_roi_feature_extractor);
REGISTER_OCL(experimental_detectron_topk_rois);
REGISTER_OCL(eltwise);
REGISTER_OCL(embed);
REGISTER_OCL(fully_connected);
REGISTER_OCL(gather);
REGISTER_OCL(gather_nd);
REGISTER_OCL(gather_elements);
REGISTER_OCL(gemm);
REGISTER_OCL(lrn);
REGISTER_OCL(lstm_gemm);
REGISTER_OCL(lstm_elt);
REGISTER_OCL(max_unpooling);
REGISTER_OCL(mutable_data);
REGISTER_OCL(mvn);
REGISTER_OCL(non_max_suppression);
REGISTER_OCL(normalize);
REGISTER_OCL(one_hot);
REGISTER_OCL(permute);
REGISTER_OCL(pooling);
REGISTER_OCL(pyramid_roi_align);
REGISTER_OCL(quantize);
REGISTER_OCL(random_uniform);
REGISTER_OCL(range);
REGISTER_OCL(reduce);
REGISTER_OCL(region_yolo);
REGISTER_OCL(reorder);
REGISTER_OCL(reorg_yolo);
REGISTER_OCL(reshape);
REGISTER_OCL(reverse_sequence);
REGISTER_OCL(roi_align);
REGISTER_OCL(roi_pooling);
REGISTER_OCL(scale);
REGISTER_OCL(scatter_update);
REGISTER_OCL(scatter_elements_update);
REGISTER_OCL(scatter_nd_update);
REGISTER_OCL(select);
REGISTER_OCL(shuffle_channels);
REGISTER_OCL(slice);
REGISTER_OCL(softmax);
REGISTER_OCL(space_to_batch);
REGISTER_OCL(space_to_depth);
REGISTER_OCL(strided_slice);
REGISTER_OCL(tile);
REGISTER_OCL(lstm_dynamic_input);
REGISTER_OCL(lstm_dynamic_timeloop);
REGISTER_OCL(generic_layer);
REGISTER_OCL(gather_tree);
REGISTER_OCL(resample);
REGISTER_OCL(grn);
REGISTER_OCL(ctc_greedy_decoder);
REGISTER_OCL(cum_sum);
REGISTER_OCL(embedding_bag);
REGISTER_OCL(extract_image_patches);
REGISTER_OCL(convert_color);

#undef REGISTER_OCL

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
