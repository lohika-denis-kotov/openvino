// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <thread>

#include "behavior/ov_plugin/core_integration.hpp"
#include "openvino/runtime/opencl_gpu/properties.hpp"

#ifdef _WIN32
#    include "gpu/gpu_context_api_dx.hpp"
#elif defined ENABLE_LIBVA
#    include <gpu/gpu_context_api_va.hpp>
#endif
#include "gpu/gpu_config.hpp"
#include "gpu/gpu_context_api_ocl.hpp"

using namespace ov::test::behavior;

namespace {
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(nightly_OVClassCommon,
        OVClassBasicTestP,
        ::testing::Values(std::make_pair("openvino_opencl_gpu_plugin", "OCL")));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassNetworkTestP, OVClassNetworkTestP, ::testing::Values("GPU"));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO", "BATCH"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO", "BATCH"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO", "BATCH"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_MAX_BATCH_SIZE,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest, OVClassGetMetricTest_DEVICE_GOPS, ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest, OVClassGetMetricTest_DEVICE_TYPE, ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_RANGE_FOR_STREAMS,
        ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_ThrowUnsupported,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO", "BATCH"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetConfigTest,
        OVClassGetConfigTest_ThrowUnsupported,
        ::testing::Values("GPU", "MULTI", "HETERO", "AUTO", "BATCH"));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetAvailableDevices, OVClassGetAvailableDevices, ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassSetModelPriorityConfigTest, OVClassSetModelPriorityConfigTest,
        ::testing::Values("MULTI", "AUTO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassSetLogLevelConfigTest, OVClassSetLogLevelConfigTest,
        ::testing::Values("MULTI", "AUTO"));

const std::vector<ov::AnyMap> multiConfigs = {
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU)},
        {ov::device::priorities(CommonTestUtils::DEVICE_OCL)},
        {ov::device::priorities(CommonTestUtils::DEVICE_CPU, CommonTestUtils::DEVICE_OCL)}};

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassSetDevicePriorityConfigTest, OVClassSetDevicePriorityConfigTest,
        ::testing::Combine(::testing::Values("MULTI", "AUTO"),
                           ::testing::ValuesIn(multiConfigs)));
//
// GPU specific metrics
//
using OVClassGetMetricTest_GPU_DEVICE_TOTAL_MEM_SIZE = OVClassBaseTestP;
TEST_P(OVClassGetMetricTest_GPU_DEVICE_TOTAL_MEM_SIZE, GetMetricAndPrintNoThrow) {
    ov::Core ie;
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_property(deviceName, GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE)));
    uint64_t t = p;

    std::cout << "GPU device total memory size: " << t << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(GPU_METRIC_KEY(DEVICE_TOTAL_MEM_SIZE));
}

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_GPU_DEVICE_TOTAL_MEM_SIZE,
        ::testing::Values("GPU"));

using OVClassGetMetricTest_GPU_UARCH_VERSION = OVClassBaseTestP;
TEST_P(OVClassGetMetricTest_GPU_UARCH_VERSION, GetMetricAndPrintNoThrow) {
    ov::Core ie;
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_property(deviceName, GPU_METRIC_KEY(UARCH_VERSION)));
    std::string t = p;

    std::cout << "GPU device uarch: " << t << std::endl;
    OV_ASSERT_PROPERTY_SUPPORTED(GPU_METRIC_KEY(UARCH_VERSION));
}

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_GPU_UARCH_VERSION,
        ::testing::Values("GPU"));

using OVClassGetMetricTest_GPU_EXECUTION_UNITS_COUNT = OVClassBaseTestP;
TEST_P(OVClassGetMetricTest_GPU_EXECUTION_UNITS_COUNT, GetMetricAndPrintNoThrow) {
    ov::Core ie;
    ov::Any p;

    ASSERT_NO_THROW(p = ie.get_property(deviceName, GPU_METRIC_KEY(EXECUTION_UNITS_COUNT)));
    int t = p;

    std::cout << "GPU EUs count: " << t << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(GPU_METRIC_KEY(EXECUTION_UNITS_COUNT));
}

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetMetricTest_GPU_EXECUTION_UNITS_COUNT,
        ::testing::Values("GPU"));

using OVClassGetPropertyTest_GPU = OVClassBaseTestP;
TEST_P(OVClassGetPropertyTest_GPU, GetMetricAvailableDevicesAndPrintNoThrow) {
    ov::Core ie;

    std::vector<std::string> properties;
    ASSERT_NO_THROW(properties = ie.get_property(deviceName, ov::available_devices));

    std::cout << "AVAILABLE_DEVICES: ";
    for (const auto& prop : properties) {
        std::cout << prop << " ";
    }
    std::cout << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::available_devices);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricRangeForAsyncInferRequestsAndPrintNoThrow) {
    ov::Core ie;

    std::tuple<unsigned int, unsigned int, unsigned int> property;
    ASSERT_NO_THROW(property = ie.get_property(deviceName, ov::range_for_async_infer_requests));

    std::cout << "RANGE_FOR_ASYNC_INFER_REQUESTS: " << std::get<0>(property) << " " <<
                                                       std::get<1>(property) << " " <<
                                                       std::get<2>(property) << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::range_for_async_infer_requests);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricRangeForStreamsAndPrintNoThrow) {
    ov::Core ie;

    std::tuple<unsigned int, unsigned int> property;
    ASSERT_NO_THROW(property = ie.get_property(deviceName, ov::range_for_streams));

    std::cout << "RANGE_FOR_STREAMS: " << std::get<0>(property) << " " <<
                                          std::get<1>(property) << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::range_for_streams);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricOptimalBatchSizeAndPrintNoThrow) {
    ov::Core ie;

    unsigned int property;
    ASSERT_NO_THROW(property = ie.get_property(deviceName, ov::optimal_batch_size));

    std::cout << "OPTIMAL_BATCH_SIZE: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::optimal_batch_size);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricFullNameAndPrintNoThrow) {
    ov::Core ie;

    std::string property;
    ASSERT_NO_THROW(property = ie.get_property(deviceName, ov::device::full_name));

    std::cout << "FULL_DEVICE_NAME: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::full_name);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricTypeAndPrintNoThrow) {
    ov::Core ie;

    ov::device::Type property = ov::device::Type::INTEGRATED;
    ASSERT_NO_THROW(property = ie.get_property(deviceName, ov::device::type));

    std::cout << "DEVICE_TYPE: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::type);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricGopsAndPrintNoThrow) {
    ov::Core ie;

    std::map<ov::element::Type, float> properties;
    ASSERT_NO_THROW(properties = ie.get_property(deviceName, ov::device::gops));

    std::cout << "DEVICE_GOPS: " << std::endl;
    for (const auto& prop : properties) {
        std::cout << "- " << prop.first << ": " << prop.second << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::gops);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricCapabilitiesAndPrintNoThrow) {
    ov::Core ie;

    std::vector<std::string> properties;
    ASSERT_NO_THROW(properties = ie.get_property(deviceName, ov::device::capabilities));

    std::cout << "OPTIMIZATION_CAPABILITIES: " << std::endl;
    for (const auto& prop : properties) {
        std::cout << "- " << prop << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::device::capabilities);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricDeviceTotalMemSizeAndPrintNoThrow) {
    ov::Core ie;

    uint64_t property;
    ASSERT_NO_THROW(property = ie.get_property(deviceName, ov::opencl_gpu::device_total_mem_size));

    std::cout << "GPU_DEVICE_TOTAL_MEM_SIZE: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::opencl_gpu::device_total_mem_size);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricUarchVersionAndPrintNoThrow) {
    ov::Core ie;

    std::string property;
    ASSERT_NO_THROW(property = ie.get_property(deviceName, ov::opencl_gpu::uarch_version));

    std::cout << "GPU_UARCH_VERSION: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::opencl_gpu::uarch_version);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricExecutionUnitsCountAndPrintNoThrow) {
    ov::Core ie;

    int32_t property = 0;
    ASSERT_NO_THROW(property = ie.get_property(deviceName, ov::opencl_gpu::execution_units_count));

    std::cout << "GPU_EXECUTION_UNITS_COUNT: " << property << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::opencl_gpu::execution_units_count);
}

TEST_P(OVClassGetPropertyTest_GPU, GetMetricMemoryStatisticsAndPrintNoThrow) {
    ov::Core ie;

    std::map<std::string, uint64_t> properties;
    ASSERT_NO_THROW(properties = ie.get_property(deviceName, ov::opencl_gpu::memory_statistics));

    std::cout << "GPU_MEMORY_STATISTICS: " << std::endl;
    for (const auto& prop : properties) {
        std::cout << " " << prop.first << " - " << prop.second << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::opencl_gpu::memory_statistics);
}

TEST_P(OVClassGetPropertyTest_GPU, GetAndSetPerformanceModeNoThrow) {
    ov::Core ie;

    ov::hint::PerformanceMode defaultMode;
    ASSERT_NO_THROW(defaultMode = ie.get_property(deviceName, ov::hint::performance_mode));

    std::cout << "Default PERFORMANCE_HINT: \"" << defaultMode << "\"" << std::endl;

    ie.set_property(deviceName, ov::hint::performance_mode(ov::hint::PerformanceMode::UNDEFINED));
    ASSERT_EQ(ov::hint::PerformanceMode::UNDEFINED, ie.get_property(deviceName, ov::hint::performance_mode));
    ie.set_property(deviceName, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    ASSERT_EQ(ov::hint::PerformanceMode::LATENCY, ie.get_property(deviceName, ov::hint::performance_mode));
    ie.set_property(deviceName, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    ASSERT_EQ(ov::hint::PerformanceMode::THROUGHPUT, ie.get_property(deviceName, ov::hint::performance_mode));

    OV_ASSERT_PROPERTY_SUPPORTED(ov::hint::performance_mode);
}

TEST_P(OVClassGetPropertyTest_GPU, GetAndSetEnableProfilingNoThrow) {
    ov::Core ie;

    bool defaultValue = false;
    ASSERT_NO_THROW(defaultValue = ie.get_property(deviceName, ov::enable_profiling));

    std::cout << "Default PERF_COUNT: " << defaultValue << std::endl;

    ie.set_property(deviceName, ov::enable_profiling(true));
    ASSERT_EQ(true, ie.get_property(deviceName, ov::enable_profiling));


    OV_ASSERT_PROPERTY_SUPPORTED(ov::enable_profiling);
}

TEST_P(OVClassGetPropertyTest_GPU, GetAndSetModelPriorityNoThrow) {
    ov::Core ie;

    ov::hint::Priority defaultValue;
    ASSERT_NO_THROW(defaultValue = ie.get_property(deviceName, ov::hint::model_priority));

    std::cout << "Default PERF_COUNT: " << defaultValue << std::endl;

    ie.set_property(deviceName, ov::hint::model_priority(ov::hint::Priority::HIGH));
    ASSERT_EQ(ov::hint::Priority::HIGH, ie.get_property(deviceName, ov::hint::model_priority));
    ASSERT_EQ(ov::hint::Priority::HIGH, ie.get_property(deviceName, ov::opencl_gpu::hint::queue_priority));
    ie.set_property(deviceName, ov::hint::model_priority(ov::hint::Priority::LOW));
    ASSERT_EQ(ov::hint::Priority::LOW, ie.get_property(deviceName, ov::hint::model_priority));
    ASSERT_EQ(ov::hint::Priority::LOW, ie.get_property(deviceName, ov::opencl_gpu::hint::queue_priority));
    ie.set_property(deviceName, ov::hint::model_priority(ov::hint::Priority::MEDIUM));
    ASSERT_EQ(ov::hint::Priority::MEDIUM, ie.get_property(deviceName, ov::hint::model_priority));
    ASSERT_EQ(ov::hint::Priority::MEDIUM, ie.get_property(deviceName, ov::opencl_gpu::hint::queue_priority));

    OV_ASSERT_PROPERTY_SUPPORTED(ov::hint::model_priority);
}

TEST_P(OVClassGetPropertyTest_GPU, GetAndSetQueuePriorityNoThrow) {
    ov::Core ie;

    ov::hint::Priority defaultValue;
    ASSERT_NO_THROW(defaultValue = ie.get_property(deviceName, ov::opencl_gpu::hint::queue_priority));

    std::cout << "Default GPU_QUEUE_PRIORITY: " << defaultValue << std::endl;

    ie.set_property(deviceName, ov::opencl_gpu::hint::queue_priority(ov::hint::Priority::HIGH));
    ASSERT_EQ(ov::hint::Priority::HIGH, ie.get_property(deviceName, ov::opencl_gpu::hint::queue_priority));
    ie.set_property(deviceName, ov::opencl_gpu::hint::queue_priority(ov::hint::Priority::LOW));
    ASSERT_EQ(ov::hint::Priority::LOW, ie.get_property(deviceName, ov::opencl_gpu::hint::queue_priority));
    ie.set_property(deviceName, ov::opencl_gpu::hint::queue_priority(ov::hint::Priority::MEDIUM));
    ASSERT_EQ(ov::hint::Priority::MEDIUM, ie.get_property(deviceName, ov::opencl_gpu::hint::queue_priority));

    OV_ASSERT_PROPERTY_SUPPORTED(ov::opencl_gpu::hint::queue_priority);
}

TEST_P(OVClassGetPropertyTest_GPU, GetAndSetThrottleLevelNoThrow) {
    ov::Core ie;

    ov::opencl_gpu::hint::ThrottleLevel defaultValue;
    ASSERT_NO_THROW(defaultValue = ie.get_property(deviceName, ov::opencl_gpu::hint::queue_throttle));

    std::cout << "Default GPU_QUEUE_THROTTLE: " << defaultValue << std::endl;

    ie.set_property(deviceName, ov::opencl_gpu::hint::queue_throttle(ov::opencl_gpu::hint::ThrottleLevel::HIGH));
    ASSERT_EQ(ov::opencl_gpu::hint::ThrottleLevel::HIGH, ie.get_property(deviceName, ov::opencl_gpu::hint::queue_throttle));
    ie.set_property(deviceName, ov::opencl_gpu::hint::queue_throttle(ov::opencl_gpu::hint::ThrottleLevel::LOW));
    ASSERT_EQ(ov::opencl_gpu::hint::ThrottleLevel::LOW, ie.get_property(deviceName, ov::opencl_gpu::hint::queue_throttle));
    ie.set_property(deviceName, ov::opencl_gpu::hint::queue_throttle(ov::opencl_gpu::hint::ThrottleLevel::MEDIUM));
    ASSERT_EQ(ov::opencl_gpu::hint::ThrottleLevel::MEDIUM, ie.get_property(deviceName, ov::opencl_gpu::hint::queue_throttle));

    OV_ASSERT_PROPERTY_SUPPORTED(ov::opencl_gpu::hint::queue_throttle);
}

TEST_P(OVClassGetPropertyTest_GPU, CanSetDefaultValueBackToPluginNewAPI) {
    ov::Core ie;

    std::vector<ov::PropertyName> properties;
    ASSERT_NO_THROW(properties = ie.get_property(deviceName, ov::supported_properties));

    std::cout << "SUPPORTED_PROPERTIES:" << std::endl;
    for (const auto& property : properties) {
        ov::Any prop;
        if (property.is_mutable()) {
            std::cout << "RW: " << property << " ";
            ASSERT_NO_THROW(prop = ie.get_property(deviceName, property));
            prop.print(std::cout);
            std::cout << std::endl;
            ASSERT_NO_THROW(ie.set_property(deviceName, {{property, prop}}));
        } else {
            std::cout << "RO: " << property << " ";
            ASSERT_NO_THROW(prop = ie.get_property(deviceName, property));
            prop.print(std::cout);
            std::cout << std::endl;
        }
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::supported_properties);
}

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetMetricTest,
        OVClassGetPropertyTest_GPU,
        ::testing::Values("GPU"));

using OVClassGetMetricTest_GPU_OPTIMAL_BATCH_SIZE = OVClassBaseTestP;
TEST_P(OVClassGetMetricTest_GPU_OPTIMAL_BATCH_SIZE, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    unsigned int p;

    ov::AnyMap _options = {ov::hint::model(simpleNetwork)};
    ASSERT_NO_THROW(p = ie.get_property(deviceName, ov::optimal_batch_size.name(), _options));

    std::cout << "GPU device optimal batch size: " << p << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::optimal_batch_size);
}

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassExecutableNetworkGetMetricTest, OVClassGetMetricTest_GPU_OPTIMAL_BATCH_SIZE,
        ::testing::Values("GPU")
);

using OVClassGetMetricTest_GPU_MAX_BATCH_SIZE_DEFAULT = OVClassBaseTestP;
TEST_P(OVClassGetMetricTest_GPU_MAX_BATCH_SIZE_DEFAULT, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    unsigned int p;

    ov::AnyMap _options = {ov::hint::model(simpleNetwork)};
    ASSERT_NO_THROW(p = ie.get_property(deviceName, ov::max_batch_size.name(), _options));

    std::cout << "GPU device max available batch size: " << p << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::max_batch_size);
}

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, OVClassGetMetricTest_GPU_MAX_BATCH_SIZE_DEFAULT,
        ::testing::Values("GPU")
);

using OVClassGetMetricTest_GPU_MAX_BATCH_SIZE_STREAM_DEVICE_MEM = OVClassBaseTestP;
TEST_P(OVClassGetMetricTest_GPU_MAX_BATCH_SIZE_STREAM_DEVICE_MEM, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    unsigned int p;
    auto exec_net1 = ie.compile_model(simpleNetwork, deviceName);

    uint32_t n_streams = 2;
    int64_t available_device_mem_size = 1073741824;
    ov::AnyMap _options = {ov::hint::model(simpleNetwork),
                           ov::num_streams(n_streams),
                           ov::opencl_gpu::hint::available_device_mem(available_device_mem_size)};

    ASSERT_NO_THROW(p = ie.get_property(deviceName, ov::max_batch_size.name(), _options));

    std::cout << "GPU device max available batch size: " << p << std::endl;

    OV_ASSERT_PROPERTY_SUPPORTED(ov::max_batch_size);
}

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassExecutableNetworkGetMetricTest, OVClassGetMetricTest_GPU_MAX_BATCH_SIZE_STREAM_DEVICE_MEM,
        ::testing::Values("GPU")
);

using OVClassGetMetricTest_GPU_MEMORY_STATISTICS_DEFAULT = OVClassBaseTestP;
TEST_P(OVClassGetMetricTest_GPU_MEMORY_STATISTICS_DEFAULT, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    std::map<std::string, uint64_t> p;

    auto exec_net = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(p = ie.get_property(deviceName, ov::opencl_gpu::memory_statistics));

    ASSERT_FALSE(p.empty());
    std::cout << "Memory Statistics: " << std::endl;
    for (auto &&kv : p) {
        ASSERT_NE(kv.second, 0);
        std::cout << kv.first << ": " << kv.second << " bytes" << std::endl;
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::opencl_gpu::memory_statistics);
}

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, OVClassGetMetricTest_GPU_MEMORY_STATISTICS_DEFAULT,
        ::testing::Values("GPU")
);

using OVClassGetMetricTest_GPU_MEMORY_STATISTICS_MULTIPLE_NETWORKS = OVClassBaseTestP;
TEST_P(OVClassGetMetricTest_GPU_MEMORY_STATISTICS_MULTIPLE_NETWORKS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    std::map<std::string, uint64_t> t1;
    std::map<std::string, uint64_t> t2;

    auto exec_net1 = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(t1 = ie.get_property(deviceName, ov::opencl_gpu::memory_statistics));

    ASSERT_FALSE(t1.empty());
    for (auto &&kv : t1) {
        ASSERT_NE(kv.second, 0);
    }

    auto exec_net2 = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(t2 = ie.get_property(deviceName, ov::opencl_gpu::memory_statistics));

    ASSERT_FALSE(t2.empty());
    for (auto &&kv : t2) {
        ASSERT_NE(kv.second, 0);
        auto iter = t1.find(kv.first);
        if (iter != t1.end()) {
            ASSERT_EQ(kv.second, t1[kv.first] * 2);
        }
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::opencl_gpu::memory_statistics);
}

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, OVClassGetMetricTest_GPU_MEMORY_STATISTICS_MULTIPLE_NETWORKS,
        ::testing::Values("GPU")
);

using OVClassGetMetricTest_GPU_MEMORY_STATISTICS_CHECK_VALUES = OVClassBaseTestP;
TEST_P(OVClassGetMetricTest_GPU_MEMORY_STATISTICS_CHECK_VALUES, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    std::map<std::string, uint64_t> t1;

    ASSERT_NO_THROW(t1 = ie.get_property(deviceName, ov::opencl_gpu::memory_statistics));
    ASSERT_TRUE(t1.empty());

    {
        auto exec_net1 = ie.compile_model(simpleNetwork, deviceName);

        std::map<std::string, uint64_t> t2;
        ASSERT_NO_THROW(t2 = ie.get_property(deviceName, ov::opencl_gpu::memory_statistics));

        ASSERT_FALSE(t2.empty());
        for (auto &&kv : t2) {
            ASSERT_NE(kv.second, 0);
        }
        {
            auto exec_net2 = ie.compile_model(actualNetwork, deviceName);

            std::map<std::string, uint64_t> t3;
            ASSERT_NO_THROW(t3 = ie.get_property(deviceName, ov::opencl_gpu::memory_statistics));

            ASSERT_FALSE(t3.empty());
            for (auto &&kv : t3) {
                ASSERT_NE(kv.second, 0);
            }
        }
        std::map<std::string, uint64_t> t4;
        ASSERT_NO_THROW(t4 = ie.get_property(deviceName, ov::opencl_gpu::memory_statistics));

        ASSERT_FALSE(t4.empty());
        for (auto &&kv : t4) {
            ASSERT_NE(kv.second, 0);
            if (kv.first.find("_cur") != std::string::npos) {
                auto iter = t2.find(kv.first);
                if (iter != t2.end()) {
                    ASSERT_EQ(t2[kv.first], kv.second);
                }
            }
        }
    }
    std::map<std::string, uint64_t> t5;
    ASSERT_NO_THROW(t5 = ie.get_property(deviceName, ov::opencl_gpu::memory_statistics));

    ASSERT_FALSE(t5.empty());
    for (auto &&kv : t5) {
        if (kv.first.find("_cur") != std::string::npos) {
            ASSERT_EQ(kv.second, 0);
        }
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::opencl_gpu::memory_statistics);
}

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, OVClassGetMetricTest_GPU_MEMORY_STATISTICS_CHECK_VALUES,
        ::testing::Values("GPU")
);

using OVClassGetMetricTest_GPU_MEMORY_STATISTICS_MULTI_THREADS = OVClassBaseTestP;
TEST_P(OVClassGetMetricTest_GPU_MEMORY_STATISTICS_MULTI_THREADS, GetMetricAndPrintNoThrow) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::Core ie;
    std::map<std::string, uint64_t> t1;
    std::map<std::string, uint64_t> t2;

    std::atomic<uint32_t> counter{0u};
    std::vector<std::thread> threads(2);
    // key: thread id, value: executable network
    std::map<uint32_t, ov::CompiledModel> exec_net_map;
    std::vector<std::shared_ptr<ngraph::Function>> networks;
    networks.emplace_back(simpleNetwork);
    networks.emplace_back(simpleNetwork);

    auto exec_net1 = ie.compile_model(simpleNetwork, deviceName);

    ASSERT_NO_THROW(t1 = ie.get_property(deviceName, ov::opencl_gpu::memory_statistics));

    ASSERT_FALSE(t1.empty());
    for (auto &&kv : t1) {
        ASSERT_NE(kv.second, 0);
    }

    for (auto & thread : threads) {
        thread = std::thread([&](){
            auto value = counter++;
            exec_net_map[value] = ie.compile_model(networks[value], deviceName);
        });
    }

    for (auto & thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    ASSERT_NO_THROW(t2 = ie.get_property(deviceName, ov::opencl_gpu::memory_statistics));

    ASSERT_FALSE(t2.empty());
    for (auto &&kv : t2) {
        ASSERT_NE(kv.second, 0);
        auto iter = t1.find(kv.first);
        if (iter != t1.end()) {
            ASSERT_EQ(kv.second, t1[kv.first] * 3);
        }
    }

    OV_ASSERT_PROPERTY_SUPPORTED(ov::opencl_gpu::memory_statistics);
}

INSTANTIATE_TEST_SUITE_P(
        nightly_IEClassGetMetricTest, OVClassGetMetricTest_GPU_MEMORY_STATISTICS_MULTI_THREADS,
        ::testing::Values("GPU")
);

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(nightly_OVClassGetConfigTest, OVClassGetConfigTest, ::testing::Values("GPU"));

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(smoke_OVClassQueryNetworkTest, OVClassQueryNetworkTest, ::testing::Values("GPU"));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(smoke_OVClassLoadNetworkTest, OVClassLoadNetworkTest, ::testing::Values("GPU"));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassHeteroExecutableNetworkGetMetricTest,
        OVClassLoadNetworkAfterCoreRecreateTest,
        ::testing::Values("GPU"));

// GetConfig / SetConfig for specific device

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSpecificDevice0Test, OVClassSpecificDeviceTestGetConfig,
        ::testing::Values("GPU.0")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSpecificDevice1Test, OVClassSpecificDeviceTestGetConfig,
        ::testing::Values("GPU.1")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSpecificDevice0Test, OVClassSpecificDeviceTestSetConfig,
        ::testing::Values("GPU.0")
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSpecificDevice1Test, OVClassSpecificDeviceTestSetConfig,
        ::testing::Values("GPU.1")
);

// Several devices case

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSeveralDevicesTest, OVClassSeveralDevicesTestLoadNetwork,
        ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"}))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSeveralDevicesTest, OVClassSeveralDevicesTestQueryNetwork,
        ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"}))
);

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSeveralDevicesTest, OVClassSeveralDevicesTestDefaultCore,
        ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"}))
);

// Set default device ID

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSetDefaultDeviceIDTest, OVClassSetDefaultDeviceIDTest,
        ::testing::Values(std::make_pair("GPU", "1"))
);

// Set config for all GPU devices

INSTANTIATE_TEST_SUITE_P(
        nightly_OVClassSetGlobalConfigTest, OVClassSetGlobalConfigTest,
        ::testing::Values("GPU")
);
}  // namespace
