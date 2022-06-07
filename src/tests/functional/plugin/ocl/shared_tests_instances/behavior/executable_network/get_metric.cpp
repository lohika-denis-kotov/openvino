// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/executable_network/get_metric.hpp"

#ifdef _WIN32
# include "gpu/gpu_context_api_dx.hpp"
#elif defined ENABLE_LIBVA
# include <gpu/gpu_context_api_va.hpp>
#endif
#include "gpu/gpu_context_api_ocl.hpp"

#include "gpu/gpu_config.hpp"

using namespace BehaviorTestsDefinitions;

namespace {
//
// Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_OPTIMAL_NUMBER_OF_INFER_REQUESTS,
        ::testing::Values("OCL", "MULTI:OCL", "HETERO:OCL", "AUTO:OCL,CPU", "BATCH:OCL")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("OCL", "MULTI:OCL", "HETERO:OCL", "AUTO:OCL,CPU", "BATCH:OCL")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("OCL", "MULTI:OCL", "HETERO:OCL", "AUTO:OCL,CPU", "BATCH:OCL")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values("OCL", "MULTI:OCL", "HETERO:OCL", "AUTO:OCL,CPU", "BATCH:OCL")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, IEClassExecutableNetworkGetMetricTest_ThrowsUnsupported,
        ::testing::Values("OCL", "MULTI:OCL", "HETERO:OCL", "AUTO:OCL,CPU", "BATCH:OCL")
);

//
// Executable Network GetConfig / SetConfig
//

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkGetConfigTest, IEClassExecutableNetworkGetConfigTest,
        ::testing::Values("OCL")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkSetConfigTest, IEClassExecutableNetworkSetConfigTest,
        ::testing::Values("OCL")
);

//
// Hetero Executable Network GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("OCL")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("OCL")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_NETWORK_NAME,
        ::testing::Values("OCL")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassHeteroExecutableNetworlGetMetricTest, IEClassHeteroExecutableNetworkGetMetricTest_TARGET_FALLBACK,
        ::testing::Values("OCL")
);
} // namespace
