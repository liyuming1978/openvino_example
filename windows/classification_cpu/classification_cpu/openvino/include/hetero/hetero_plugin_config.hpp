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
 * @brief A header that defines advanced related properties for Heterogeneous plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file hetero_plugin_config.hpp
 */

#pragma once

#include <string>
#include "../ie_plugin_config.hpp"

namespace InferenceEngine {

namespace HeteroConfigParams {

#define HETERO_CONFIG_KEY(name) InferenceEngine::HeteroConfigParams::_CONFIG_KEY(HETERO_##name)
#define DECLARE_HETERO_CONFIG_KEY(name) DECLARE_CONFIG_KEY(HETERO_##name)
#define DECLARE_HETERO_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(HETERO_##name)

/**
 * @brief The key for enabling of dumping the topology with details of layers and details how
 * this network would be executed on different devices to the disk in GraphViz format.
 * This option should be used with values: CONFIG_VALUE(NO) (default) or CONFIG_VALUE(YES)
 */
DECLARE_HETERO_CONFIG_KEY(DUMP_GRAPH_DOT);

}  // namespace HeteroConfigParams
}  // namespace InferenceEngine
