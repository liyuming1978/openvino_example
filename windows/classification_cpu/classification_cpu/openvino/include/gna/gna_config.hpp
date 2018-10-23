// Copyright (c) 2018 Intel Corporation
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

namespace InferenceEngine {

namespace GNAConfigParams {

#define GNA_CONFIG_KEY(name) InferenceEngine::GNAConfigParams::_CONFIG_KEY(GNA_##name)
#define GNA_CONFIG_VALUE(name) InferenceEngine::GNAConfigParams::GNA_##name

#define DECLARE_GNA_CONFIG_KEY(name) DECLARE_CONFIG_KEY(GNA_##name)
#define DECLARE_GNA_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(GNA_##name)

/**
* @brief Scale factor that is calculated by user, in order to use static quantisation feature
* This option should be used with floating point value serialized to string with decimal separator equals to . (dot)
*/
DECLARE_GNA_CONFIG_KEY(SCALE_FACTOR);

/**
* @brief By default gna api work in Int16 precision, however this can be adjusted if necessary,
* currently supported values are I16, I8
*/
DECLARE_GNA_CONFIG_KEY(PRECISION);


/**
* @brief if turned on, dump GNA firmware model into specified file
*/
DECLARE_GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE);

/**
* @brief GNA proc_type setting that should be one of GNA_AUTO, GNA_HW, GNA_SW, GNA_SW_EXACT
*/
DECLARE_GNA_CONFIG_KEY(DEVICE_MODE);

DECLARE_GNA_CONFIG_VALUE(AUTO);
DECLARE_GNA_CONFIG_VALUE(HW);
DECLARE_GNA_CONFIG_VALUE(SW);
DECLARE_GNA_CONFIG_VALUE(SW_EXACT);

/**
* @brief if enabled produced minimum memory footprint for loaded network in GNA memory, default value is YES
*/
DECLARE_GNA_CONFIG_KEY(COMPACT_MODE);

/**
* @brief The option to enable/disable uniformly distributed PWL algorithm.
* By default (in case of NO value set) the optimized algorithm called "Recursive Descent Algorithm for Finding
* the Optimal Minimax Piecewise Linear Approximation of Convex Functions is used.
* If value is YES then simple uniform distribution used to create PWL approximation of activation functions
* Uniform distribution usually gives poor approximation with same number of segments
*/
DECLARE_GNA_CONFIG_KEY(PWL_UNIFORM_DESIGN);
}  // namespace GNAConfigParams
}  // namespace InferenceEngine
