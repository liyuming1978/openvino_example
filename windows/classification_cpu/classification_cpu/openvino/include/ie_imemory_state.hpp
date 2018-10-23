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
 * @brief a header file for IMemoryState interface
 * @file ie_imemory_state.hpp
 */

#pragma once
#include <memory>
#include "details/ie_no_copy.hpp"
#include "ie_common.h"
#include "ie_blob.h"

namespace InferenceEngine {

/**
 * @brief manages data for reset operations
 */
class IMemoryState : public details::no_copy {
 public:
    using Ptr = std::shared_ptr<IMemoryState>;

    /**
     * @brief Gets name of current memory state, if length of array is not enough name is truncated by len, null terminator is inserted as well.
     * @param name preallocated buffer for receiving name
     * @param len Length of the buffer
     * @param  resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: OK (0) for success
     */
    virtual StatusCode GetName(char *name, size_t len, ResponseDesc *resp) const noexcept = 0;

    /**
     * @brief reset internal memory state for relevant iexecutable network, to a value specified in SetState
     * @param  resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: OK (0) for success*
     */
    virtual StatusCode Reset(ResponseDesc *resp) noexcept = 0;

    /**
     * @brief  Sets the new state that is used for all future Reset() operations as a base.
     * This method can fail if Blob size does not match the internal state size or precision
     * @param  newState is the data to use as base state
     * @param  resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: OK (0) for success
    */
    virtual StatusCode SetState(Blob::Ptr newState, ResponseDesc *resp) noexcept = 0;

    /**
     * @brief returns the value of the last memory state.
     * @details Since we roll memory after each infer, we can query the input state always and still get the last state.
     * @param lastState
     * @param  resp Optional: pointer to an already allocated object to contain information in case of failure
     * @return Status code of the operation: OK (0) for success
     * */
    virtual StatusCode GetLastState(Blob::CPtr & lastState, ResponseDesc *resp) const noexcept = 0;
};

}  // namespace InferenceEngine