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
 * @brief This is a header file for the ICNNNetworkStats class
 * @file ie_icnn_network_stats.hpp
 */
#pragma once

#include <string>
#include <memory>
#include <limits>
#include <vector>

namespace InferenceEngine {

/**
 * @class ICNNNetworkStats
 * @brief This is the interface to describe the NN topology scoring statistics
 */
class ICNNNetworkStats : public details::IRelease {
public:
    virtual void SaveToFile(const std::string& xmlPath, const std::string& binPath) const = 0;
    virtual void LoadFromFile(const std::string& xmlPath, const std::string& binPath) = 0;

    virtual bool isEmpty() const = 0;
};


class NetworkNodeStats;

using NetworkNodeStatsPtr = std::shared_ptr<NetworkNodeStats>;
using NetworkNodeStatsWeakPtr = std::weak_ptr<NetworkNodeStats>;

class NetworkNodeStats {
public:
    NetworkNodeStats() { }
    explicit NetworkNodeStats(int statCount) {
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::min();

        for (int i = 0; i < statCount; i++) {
            _minOutputs.push_back(min);
            _maxOutputs.push_back(max);
        }
    }

public:
    std::vector<float> _minOutputs;
    std::vector<float> _maxOutputs;
};


}  // namespace InferenceEngine
