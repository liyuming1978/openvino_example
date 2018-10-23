// Copyright (c) 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @brief A header that defines advanced related properties for VPU plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file vpu_plugin_config.hpp
 */

#pragma once

#include <string>
#include "../ie_plugin_config.hpp"

#define VPU_CONFIG_KEY(name) InferenceEngine::VPUConfigParams::_CONFIG_KEY(VPU_##name)
#define VPU_CONFIG_VALUE(name) InferenceEngine::VPUConfigParams::VPU_##name

#define DECLARE_VPU_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPU_##name)
#define DECLARE_VPU_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(VPU_##name)

namespace InferenceEngine {
namespace VPUConfigParams {

/**
 * @brief The key to specify desirable log level for devices.
 * This option should be used with values: CONFIG_VALUE(LOG_NONE) (default),
 * CONFIG_VALUE(LOG_WARNING), CONFIG_VALUE(LOG_INFO), CONFIG_VALUE(LOG_DEBUG)
 */
DECLARE_VPU_CONFIG_KEY(LOG_LEVEL);

/**
 * @deprecated
 * @brief The key to define normalization coefficient for the network input.
 * This option should used with be a real number. Example "255.f"
 */
DECLARE_VPU_CONFIG_KEY(INPUT_NORM);

/**
 * @deprecated
 * @brief The flag to specify Bias value that is added to each element of the network input.
 * This option should used with be a real number. Example "0.1f"
 */
DECLARE_VPU_CONFIG_KEY(INPUT_BIAS);

/**
 * @brief The flag for adding to the profiling information the time of obtaining a tensor.
 * This option should be used with values: CONFIG_VALUE(YES) or CONFIG_VALUE(NO) (default)
 */
DECLARE_VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME);

/**
 * @brief The flag to reset stalled devices: CONFIG_VALUE(YES) or CONFIG_VALUE(NO) (default)
 * This is a plugin scope option and must be used with the plugin's SetConfig method
 */
DECLARE_VPU_CONFIG_KEY(FORCE_RESET);

/**
 * @brief This option allows to pass extra configuration for executable network.
 * By default, it is empty string, which means - no configuration.
 * String format:
 * <parameter>=<name>,<option>=<value>,<option>=<value>,<parameter>=<name>,<option>=<value>,...
 * Supported parameters and options:
 *   * data : options related to data objects (input, output, intermediate)
 *     * scale : SCALE factor for data range (applicable for input and intermediate data)
 */
DECLARE_VPU_CONFIG_KEY(NETWORK_CONFIG);

/**
 * @brief This option allows to to specify input output layouts for network layers.
 * By default, this value set to VPU_CONFIG_VALUE(AUTO) value.
 * Supported values:
 *   VPU_CONFIG_VALUE(AUTO) executable network configured to use optimal layer layout depending on available HW
 *   VPU_CONFIG_VALUE(NCHW) executable network forced to use NCHW input/output layouts
 *   VPU_CONFIG_VALUE(NHCW) executable network forced to use NHWC input/output layouts
 */
DECLARE_VPU_CONFIG_KEY(COMPUTE_LAYOUT);

/**
 * @brief Supported keys definition for VPU_CONFIG_KEY(COMPUTE_LAYOUT) option.
 */
DECLARE_VPU_CONFIG_VALUE(AUTO);
DECLARE_VPU_CONFIG_VALUE(NCHW);
DECLARE_VPU_CONFIG_VALUE(NHWC);

}  // namespace VPUConfigParams
}  // namespace InferenceEngine
