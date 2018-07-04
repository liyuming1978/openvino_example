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

#include "ext_list.hpp"
#include "ext_base.hpp"
#include "defs.h"
#include "softmax.h"
#include <vector>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class RegionYoloImpl: public ExtLayerBase {
public:
    explicit RegionYoloImpl(const CNNLayer* layer): ExtLayerBase(layer) {
        try {
            if (cnnLayer.insData.size() != 1 || cnnLayer.outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            classes = cnnLayer.GetParamAsInt("classes");
            coords = cnnLayer.GetParamAsInt("coords");
            num = cnnLayer.GetParamAsInt("num");

            addConfig({DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        const auto *src_data = inputs[0]->cbuffer().as<const float *>();
        auto *dst_data = outputs[0]->buffer().as<float *>();

        int IW = (inputs[0]->getTensorDesc().getDims().size() > 3) ? inputs[0]->getTensorDesc().getDims()[3] : 1;
        int IH = (inputs[0]->getTensorDesc().getDims().size() > 2) ? inputs[0]->getTensorDesc().getDims()[2] : 1;
        int IC = (inputs[0]->getTensorDesc().getDims().size() > 1) ? inputs[0]->getTensorDesc().getDims()[1] : 1;
        int B = (inputs[0]->getTensorDesc().getDims().size() > 0) ? inputs[0]->getTensorDesc().getDims()[0] : 1;

        memcpy(dst_data, src_data, B * IC * IH * IW * sizeof(float));

        int inputs_size = IH * IW * num * (classes + coords + 1);
        for (int b = 0; b < B; b++) {
            for (int n = 0; n < num; n++) {
                int index = entry_index(IW, IH, coords, classes, inputs_size, b, n * IW * IH, 0);
                for (int i = index; i < index + 2 * IW * IH; i++) {
                    dst_data[i] = logistic_activate(dst_data[i]);
                }

                index = entry_index(IW, IH, coords, classes, inputs_size, b, n * IW * IH, coords);
                for (int i = index; i < index + IW * IH; i++) {
                    dst_data[i] = logistic_activate(dst_data[i]);
                }
            }
        }

        int index = entry_index(IW, IH, coords, classes, inputs_size, 0, 0, coords + 1);
        int batch_offset = inputs_size / num;
        for (int b = 0; b < B * num; b++)
            softmax_generic(src_data + index + b * batch_offset, dst_data + index + b * batch_offset, 1, classes, IH, IW);
        return OK;
    }

private:
    int classes;
    int coords;
    int num;

    inline int entry_index(int width, int height, int coords, int classes, int outputs, int batch, int location,
                           int entry) {
        int n = location / (width * height);
        int loc = location % (width * height);
        return batch * outputs + n * width * height * (coords + classes + 1) +
               entry * width * height + loc;
    }

    inline float logistic_activate(float x) {
        return 1.f / (1.f + exp(-x));
    }
};

REG_FACTORY_FOR(ImplFactory<RegionYoloImpl>, RegionYolo);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
