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

#include <vector>
#include <string>
#include <algorithm>
#include <immintrin.h>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class PReLUImpl: public ExtLayerBase {
public:
    explicit PReLUImpl(const CNNLayer *layer): ExtLayerBase(layer) {
        try {
            if (cnnLayer.insData.size() != 1 || cnnLayer.outData.empty())
                THROW_IE_EXCEPTION << "Incorrect number of input/output edges!";

            DataPtr dataPtr = cnnLayer.insData[0].lock();
            weights = std::dynamic_pointer_cast<TBlob<float>>(cnnLayer.blobs["weights"]);
            if (!weights)
                THROW_IE_EXCEPTION << cnnLayer.name << " weights is empty!";

#if defined(HAVE_AVX512F)
            auto blk_layout = ConfLayout::BLK16;
#else
            auto blk_layout = ConfLayout::BLK8;
#endif
            ConfLayout fmt = dataPtr->getTensorDesc().getDims().size() != 4 ? ConfLayout::PLN : blk_layout;
            addConfig({{fmt, false, 0}}, {{fmt, false, 0}});
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        if (inputs.size() != 1 || outputs.empty()) {
            if (resp) {
                std::string errorMsg = "Incorrect number of input or output edges!";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
        const float* src_data = inputs[0]->buffer();
        float* dst_data = outputs[0]->buffer();
        float* weight_data = weights->buffer();

        if (inputs[0]->getTensorDesc().getDims().size() == 4) {  // nChw8c format
            size_t W = inputs[0]->getTensorDesc().getDims()[3];
            size_t H = inputs[0]->getTensorDesc().getDims()[2];
            size_t C = inputs[0]->getTensorDesc().getDims()[1];
            size_t B = inputs[0]->getTensorDesc().getDims()[0];

#if defined(HAVE_AVX512F)
            const int block_size = 16;
#else
            const int block_size = 8;
#endif

            // Align channel number to block size to deal with channels padding in IE with multiple blobs
            size_t CB = (C + block_size - 1) & (-block_size);

#if _MSC_VER && !__INTEL_COMPILER
            #pragma omp parallel for schedule(static)
#else
            #pragma omp parallel for collapse(2) schedule(static)
#endif
            for (int b = 0; b < B; b++) {
                for (int c_block = 0; c_block < C; c_block += block_size) {
#if defined(HAVE_AVX512F)
                    __m512 vzero  = _mm512_setzero_ps();
                    __m512 vweights = _mm512_loadu_ps(weight_data + c_block);
                    for (int i = 0; i < H*W; i++) {
                        __m512 vsrc = _mm512_loadu_ps(src_data + b*CB*H*W + c_block*H*W + block_size*i);

                        __mmask16 vmask = _mm512_cmp_ps_mask(vsrc, vzero, _CMP_LT_OS);
                        __m512 vdst = _mm512_mask_mul_ps(vsrc, vmask, vweights, vsrc);

                        _mm512_storeu_ps(dst_data + b*CB*H*W + c_block*H*W + block_size*i, vdst);
                    }
#elif defined(HAVE_AVX2)
                    __m256 vzero  = _mm256_setzero_ps();
                    __m256 vweights = _mm256_loadu_ps(weight_data + c_block);
                    for (int i = 0; i < H*W; i++) {
                        __m256 vsrc = _mm256_loadu_ps(src_data + b*CB*H*W + c_block*H*W + block_size*i);

                        __m256 vmask = _mm256_cmp_ps(vsrc, vzero, _CMP_GT_OS);
                        __m256 vmul = _mm256_mul_ps(vweights, vsrc);
                        __m256 vdst = _mm256_blendv_ps(vmul, vsrc, vmask);

                        _mm256_storeu_ps(dst_data + b*CB*H*W + c_block*H*W + block_size*i, vdst);
                    }
#elif defined(HAVE_SSE)
                    for (int i = 0; i < H*W; i++) {
                        __m128 vzero  = _mm_setzero_ps();
                        __m128 vweights0 = _mm_loadu_ps(weight_data + c_block + 0);
                        __m128 vweights1 = _mm_loadu_ps(weight_data + c_block + 4);

                        __m128 vsrc0 = _mm_loadu_ps(src_data + b*CB*H*W + c_block*H*W + block_size*i + 0);
                        __m128 vsrc1 = _mm_loadu_ps(src_data + b*CB*H*W + c_block*H*W + block_size*i + 4);

                        __m128 vmask0 = _mm_cmpgt_ps(vsrc0, vzero);
                        __m128 vmask1 = _mm_cmpgt_ps(vsrc1, vzero);

                        __m128 vmul0 = _mm_mul_ps(vweights0, vsrc0);
                        __m128 vmul1 = _mm_mul_ps(vweights1, vsrc1);

                        __m128 vdst0 = _mm_blendv_ps(vmul0, vsrc0, vmask0);
                        __m128 vdst1 = _mm_blendv_ps(vmul1, vsrc1, vmask1);

                        _mm_storeu_ps(dst_data + b*CB*H*W + c_block*H*W + block_size*i + 0, vdst0);
                        _mm_storeu_ps(dst_data + b*CB*H*W + c_block*H*W + block_size*i + 4, vdst1);
                    }
#else
                    for (int i = 0; i < H*W; i++) {
                        for (int c = 0; c < block_size; c++) {
                            int idx = b*CB*H*W + c_block*H*W + block_size*i + c;

                            dst_data[idx] = std::max<float>(src_data[idx], 0.0f) +
                                    weight_data[c_block + c] * std::min<float>(src_data[idx], 0.0f);
                        }
                    }
#endif
                }
            }
        } else {  // nc format
            size_t C = inputs[0]->getTensorDesc().getDims()[1];
            size_t B = inputs[0]->getTensorDesc().getDims()[0];

#if _MSC_VER && !__INTEL_COMPILER
            #pragma omp parallel for schedule(static)
#else
            #pragma omp parallel for collapse(2) schedule(static)
#endif
            for (int b = 0; b < B; b++) {
                for (int c = 0; c < C; c++) {
                    dst_data[b*C + c] = std::max<float>(src_data[b*C + c], 0.0f) + weight_data[c] * std::min<float>(src_data[b*C + c], 0.0f);
                }
            }
        }
        return OK;
    }

private:
    TBlob<float>::Ptr weights;
    int channel_shared = 0;
};

REG_FACTORY_FOR(ImplFactory<PReLUImpl>, PReLU);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
