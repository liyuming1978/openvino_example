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
 * @brief A header that defines advanced related properties for DLIA plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file dlia_plugin_config.hpp
 */

#pragma once

#include <string>
#include "../ie_plugin_config.hpp"

namespace InferenceEngine {

namespace DLIAConfigParams {

#define DLIA_CONFIG_KEY(name) InferenceEngine::DLIAConfigParams::_CONFIG_KEY(DLIA_##name)
#define DECLARE_DLIA_CONFIG_KEY(name) DECLARE_CONFIG_KEY(DLIA_##name)
#define DECLARE_DLIA_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(DLIA_##name)

/**
 * @brief The key to define the type of transformations for DLIA inputs and outputs.
 * DLIA use custom data layout for input and output blobs. IE DLIA Plugin provides custom
 * optimized version of transformation functions that do not use OpenMP and much more faster
 * than native DLIA functions. Values: "DLIA_IO_OPTIMIZED" - optimized plugin transformations
 * are used, "DLIA_IO_NATIVE" - native DLIA transformations are used.
 */
DECLARE_DLIA_CONFIG_KEY(IO_TRANSFORMATIONS);

DECLARE_DLIA_CONFIG_VALUE(IO_OPTIMIZED);
DECLARE_DLIA_CONFIG_VALUE(IO_NATIVE);

DECLARE_DLIA_CONFIG_KEY(DLA_HG);
DECLARE_DLIA_CONFIG_KEY(ARCH_ROOT_DIR);

}  // namespace DLIAConfigParams
}  // namespace InferenceEngine
