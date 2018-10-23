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

#pragma once
#include <string>

namespace InferenceEngine {

/**
 * @brief c++ exception based error reporting wrapper of API class IMemoryState
 */
class MemoryState {
    IMemoryState::Ptr actual = nullptr;

 public:
    explicit MemoryState(IMemoryState::Ptr pState) : actual(pState) {}

    /**
     * @brief Wraps original method
     * IMemoryState::Reset
     */
     void Reset() {
        CALL_STATUS_FNC_NO_ARGS(Reset);
     }
    /**
     * @brief Wraps original method
     * IMemoryState::GetName
     */
     std::string GetName() const {
         char name[256];
         CALL_STATUS_FNC(GetName, name, sizeof(name));
         return name;
     }
    /**
     * @brief Wraps original method
     * IMemoryState::GetLastState
     */
      Blob::CPtr GetLastState() const {
         Blob::CPtr stateBlob;
         CALL_STATUS_FNC(GetLastState, stateBlob);
         return stateBlob;
     }
    /**
     * @brief Wraps original method
     * IMemoryState::SetState
     */
     void SetState(Blob::Ptr state) {
         CALL_STATUS_FNC(SetState, state);
     }
};

}  // namespace InferenceEngine