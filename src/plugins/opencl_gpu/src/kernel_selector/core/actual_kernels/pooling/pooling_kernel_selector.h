﻿// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class pooling_kernel_selector : public kernel_selector_base {
public:
    static pooling_kernel_selector& Instance() {
        static pooling_kernel_selector instance_;
        return instance_;
    }

    pooling_kernel_selector();

    virtual ~pooling_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params, const optional_params& options) const override;
};
}  // namespace kernel_selector