/************************************************************************
 *
 * (c) 2025 Machines With Vision Ltd.
 * All Rights Reserved.
 *
 * NOTICE:  All information contained herein is, and remains
 * the property of Machines With Vision Limited and its suppliers,
 * if any.  The intellectual and technical concepts contained
 * herein are proprietary to Machines With Vision Limited
 * and its suppliers and may be covered by U.K. and Foreign Patents,
 * patents in process, and are protected by trade secret or copyright law.
 * Dissemination of this information or reproduction of this material
 * is strictly forbidden unless prior written permission is obtained
 * from Machines With Vision Limited.
 *
 ************************************************************************/

#include "match.cuh"

// NOLINTBEGIN(cppcoreguidelines-narrowing-conversions, CppVariableCanBeMadeConstexpr,
// cppcoreguidelines-pro-type-cstyle-cast)

namespace mwvcv
{
    __global__ void MatchDescriptors(unsigned char* d1, unsigned char* d2, int pitch, int nkpts_2, cv::DMatch* matches)
    {
        __shared__ int idxBest[NTHREADS_MATCH];
        __shared__ int idxSecondBest[NTHREADS_MATCH];
        __shared__ int scoreBest[NTHREADS_MATCH];
        __shared__ int scoreSecondBest[NTHREADS_MATCH];

        int p = blockIdx.x;
        int x = threadIdx.x;

        idxBest[x]         = 0;
        idxSecondBest[x]   = 0;
        scoreBest[x]       = 512;
        scoreSecondBest[x] = 512;

        __syncthreads();

        unsigned long long* d1i = (unsigned long long*)(d1 + pitch * p);

        for (int i = 0; i < nkpts_2; i += NTHREADS_MATCH) {
            unsigned long long* d2i = (unsigned long long*)(d2 + pitch * (x + i));
            if (i + x < nkpts_2) {
                // Check d1[p] with d2[i]
                int score = 0;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    score += __popcll(d1i[j] ^ d2i[j]);
                }
                if (score < scoreBest[x]) {
                    scoreSecondBest[x] = scoreBest[x];
                    scoreBest[x]       = score;
                    idxSecondBest[x]   = idxBest[x];
                    idxBest[x]         = i + x;
                } else if (score < scoreSecondBest[x]) {
                    scoreSecondBest[x] = score;
                    idxSecondBest[x]   = i + x;
                }
            }
        }

        __syncthreads();

        for (int i = NTHREADS_MATCH / 2; i >= 1; i /= 2) {
            if (x < i) {
                if (scoreBest[x + i] < scoreBest[x]) {
                    scoreSecondBest[x] = scoreBest[x];
                    scoreBest[x]       = scoreBest[x + i];
                    idxSecondBest[x]   = idxBest[x];
                    idxBest[x]         = idxBest[x + i];
                } else if (scoreBest[x + i] < scoreSecondBest[x]) {
                    scoreSecondBest[x] = scoreBest[x + i];
                    idxSecondBest[x]   = idxBest[x + i];
                }
                if (scoreSecondBest[x + i] < scoreSecondBest[x]) {
                    scoreSecondBest[x] = scoreSecondBest[x + i];
                    idxSecondBest[x]   = idxSecondBest[x + i];
                }
            }
        }

        if (x == 0) {
            matches[2 * p].queryIdx     = p;
            matches[2 * p].trainIdx     = idxBest[x];
            matches[2 * p].distance     = scoreBest[x];
            matches[2 * p + 1].queryIdx = p;
            matches[2 * p + 1].trainIdx = idxSecondBest[x];
            matches[2 * p + 1].distance = scoreSecondBest[x];
        }
    }

    void matchDescriptors(cv::Mat& desc_query, cv::Mat& desc_train, size_t pitch, unsigned char* descq_d,
                          unsigned char* desct_d, cv::DMatch* dmatches_d)
    {
        dim3 block(desc_query.rows);
        MatchDescriptors<<<block, NTHREADS_MATCH>>>(descq_d, desct_d, pitch, desc_train.rows, dmatches_d);
    }

    void matchDescriptors(cv::Mat& desc_query, cv::Mat& desc_train, std::vector<std::vector<cv::DMatch>>& dmatches)
    {
        size_t pitch1, pitch2;

        unsigned char* descq_d;
        cudaMallocPitch(&descq_d, &pitch1, 64, desc_query.rows);
        cudaMemset2D(descq_d, pitch1, 0, 64, desc_query.rows);
        cudaMemcpy2D(descq_d, pitch1, desc_query.data, desc_query.cols, desc_query.cols, desc_query.rows,
                     cudaMemcpyHostToDevice);

        unsigned char* desct_d;
        cudaMallocPitch(&desct_d, &pitch2, 64, desc_train.rows);
        cudaMemset2D(desct_d, pitch2, 0, 64, desc_train.rows);
        cudaMemcpy2D(desct_d, pitch2, desc_train.data, desc_train.cols, desc_train.cols, desc_train.rows,
                     cudaMemcpyHostToDevice);

        cv::DMatch* dmatches_d;
        cudaMalloc(&dmatches_d, desc_query.rows * 2 * sizeof(cv::DMatch));

        dim3 block(desc_query.rows);
        MatchDescriptors<<<block, NTHREADS_MATCH>>>(descq_d, desct_d, pitch1, desc_train.rows, dmatches_d);

        cv::DMatch* dmatches_h = new cv::DMatch[2 * desc_query.rows];
        cudaMemcpy(dmatches_h, dmatches_d, desc_query.rows * 2 * sizeof(cv::DMatch), cudaMemcpyDeviceToHost);

        for (int i = 0; i < desc_query.rows; ++i) {
            std::vector<cv::DMatch> tdmatch;
            tdmatch.push_back(dmatches_h[2 * i]);
            tdmatch.push_back(dmatches_h[2 * i + 1]);
            dmatches.push_back(tdmatch);
        }

        cudaFree(descq_d);
        cudaFree(desct_d);
        cudaFree(dmatches_d);

        delete[] dmatches_h;
    }

} // namespace mwvcv

// NOLINTEND(cppcoreguidelines-narrowing-conversions, CppVariableCanBeMadeConstexpr,
// cppcoreguidelines-pro-type-cstyle-cast)