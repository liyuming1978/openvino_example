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

#include <cmath>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class MVNImpl: public ExtLayerBase {
public:
    explicit MVNImpl(const CNNLayer* layer): ExtLayerBase(layer) {
        try {
            if (cnnLayer.insData.size() != 1 || cnnLayer.outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            across_channels = static_cast<bool>(cnnLayer.GetParamAsInt("across_channels"));
            normalize_variance = static_cast<bool>(cnnLayer.GetParamAsInt("normalize_variance"));
            eps = cnnLayer.GetParamAsFloat("eps");

            addConfig({{ConfLayout::PLN, false, 0}}, {{ConfLayout::PLN, false, 0}});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        float* src_data = inputs[0]->buffer();
        float* dst_data = outputs[0]->buffer();

        SizeVector dims = inputs[0]->getTensorDesc().getDims();

        int N = static_cast<int>((dims.size() > 0) ? dims[0] : 1);
        int C = static_cast<int>((dims.size() > 1) ? dims[1] : 1);
        int H = static_cast<int>((dims.size() > 2) ? dims[2] : 1);
        int W = static_cast<int>((dims.size() > 3) ? dims[3] : 1);

        for (int b = 0; b < N; b++) {
            // Calculate mean value
            if (across_channels) {
                double mean = 0;
                #pragma omp parallel for reduction(+ : mean) schedule(static)
                for (int c = 0; c < C; c++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            mean += src_data[b*C*H*W + c*H*W + h*W + w];
                        }
                    }
                }
                mean /= C*H*W;
                #pragma omp parallel for schedule(static)
                for (int c = 0; c < C; c++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            dst_data[b*C*H*W + c*H*W + h*W + w] = src_data[b*C*H*W + c*H*W + h*W + w] - mean;
                        }
                    }
                }
            } else {
                #pragma omp parallel for schedule(static)
                for (int c = 0; c < C; c++) {
                    double mean = 0;
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            mean += src_data[b*C*H*W + c*H*W + h*W + w];
                        }
                    }
                    mean /= H*W;

                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            dst_data[b*C*H*W + c*H*W + h*W + w] = src_data[b*C*H*W + c*H*W + h*W + w] - mean;
                        }
                    }
                }
            }
        }

        if (normalize_variance) {
            for (int b = 0; b < N; b++) {
                // Calculate variances value
                if (across_channels) {
                    double variance = 0;
                    #pragma omp parallel for reduction(+ : variance) schedule(static)
                    for (int c = 0; c < C; c++) {
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                variance += std::pow(dst_data[b*C*H*W + c*H*W + h*W + w], 2);
                            }
                        }
                    }
                    variance /= C*H*W;
                    variance = std::pow(variance, 0.5f);
                    variance += eps;
                    #pragma omp parallel for schedule(static)
                    for (int c = 0; c < C; c++) {
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                dst_data[b*C*H*W + c*H*W + h*W + w] /= variance;
                            }
                        }
                    }
                } else {
                    #pragma omp parallel for schedule(static)
                    for (int c = 0; c < C; c++) {
                        double variance = 0;
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                variance += std::pow(dst_data[b*C*H*W + c*H*W + h*W + w], 2);
                            }
                        }
                        variance /= H*W;
                        variance = std::pow(variance, 0.5f);
                        variance += eps;
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                dst_data[b*C*H*W + c*H*W + h*W + w] /= variance;
                            }
                        }
                    }
                }
            }
        }
        return OK;
    }

private:
    bool across_channels = false;
    bool normalize_variance = true;
    float eps = 1e-9f;
};

REG_FACTORY_FOR(ImplFactory<MVNImpl>, MVN);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
