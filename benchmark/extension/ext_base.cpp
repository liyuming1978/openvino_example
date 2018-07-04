/*
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
*/

#include "ext_base.hpp"

#include <vector>
#include <string>
#include <algorithm>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

StatusCode
ExtLayerBase::getSupportedConfigurations(std::vector<LayerConfig>& conf, ResponseDesc *resp) noexcept {
    if (!errorMsg.empty()) {
        if (resp) {
            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
        }
        return GENERAL_ERROR;
    }
    conf = confs;
    return OK;
}

StatusCode
ExtLayerBase::init(LayerConfig& config, ResponseDesc *resp) noexcept {
    for (auto& input : config.inConfs) {
        for (auto& offset : input.desc.getBlockingDesc().getOffsetPaddingToData()) {
            if (offset) {
                return GENERAL_ERROR;
            }
        }
        if (input.desc.getBlockingDesc().getOffsetPadding()) {
            return GENERAL_ERROR;
        }
        for (size_t i = 0; i < input.desc.getBlockingDesc().getOrder().size(); i++) {
            if (input.desc.getBlockingDesc().getOrder()[i] != i)
                return GENERAL_ERROR;
        }
    }
    for (auto& output : config.outConfs) {
        for (auto& offset : output.desc.getBlockingDesc().getOffsetPaddingToData()) {
            if (offset) {
                return GENERAL_ERROR;
            }
        }
        if (output.desc.getBlockingDesc().getOffsetPadding()) {
            return GENERAL_ERROR;
        }
        for (size_t i = 0; i < output.desc.getBlockingDesc().getOrder().size(); i++) {
            if (output.desc.getBlockingDesc().getOrder()[i] != i)
                return GENERAL_ERROR;
        }
    }
    return OK;
}

void ExtLayerBase::addConfig(std::vector<DataConfigurator> in_l, std::vector<DataConfigurator> out_l, bool dynBatchSupport) {
    LayerConfig config;

    if (in_l.size() != cnnLayer.insData.size())
        THROW_IE_EXCEPTION << "Incorrect number of input edges. Expected " << cnnLayer.insData.size()
                           << " but layout specification provided for " << in_l.size();
    if (out_l.size() != cnnLayer.outData.size())
        THROW_IE_EXCEPTION << "Incorrect number of input edges. Expected " << cnnLayer.outData.size()
                           << " but layout specification provided for " << out_l.size();

    // Fill tensor parameters into config
    auto fill_port = [] (std::vector<DataConfig>& port, DataConfigurator conf, const DataPtr& data) {
        if (!data) THROW_IE_EXCEPTION << "Cannot get input data!";

        DataConfig dataConfig;
        dataConfig.inPlace = conf.inplace;
        dataConfig.constant = conf.constant;

        const TensorDesc& data_desc = data->getTensorDesc();
        const SizeVector& data_dims = data_desc.getDims();

        std::vector<size_t> blocks = data_dims;
        std::vector<size_t> order(blocks.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = i;

        if (conf.layout == ConfLayout::BLK8 || conf.layout == ConfLayout::BLK16) {
            if (data_dims.size() != 4)
                THROW_IE_EXCEPTION << "Inapplicable blocking layout."
                                   << "Tensor should be 4D.";

            int blk_size = conf.layout == ConfLayout::BLK8 ? 8 : 16;

            // Blocking through Channel dimension. Like [nChwXc]
            order.push_back(1);
            blocks[1] /= blk_size;
            blocks.push_back(blk_size);
        }

        if (conf.layout == ConfLayout::ANY) {
            dataConfig.desc = TensorDesc(data_desc.getPrecision(), data_dims, InferenceEngine::Layout::ANY);
        } else {
            dataConfig.desc = TensorDesc(data_desc.getPrecision(), data_dims, {blocks, order});
        }
        port.push_back(dataConfig);
    };

    for (int i = 0; i < in_l.size(); i++)
        fill_port(config.inConfs, in_l[i], cnnLayer.insData[i].lock());

    for (int i = 0; i < out_l.size(); i++)
        fill_port(config.outConfs, out_l[i], cnnLayer.outData[i]);

    config.dynBatchSupport = dynBatchSupport;
    confs.push_back(config);
}


}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
