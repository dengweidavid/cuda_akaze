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

#include <cuda_runtime.h>

#include "CudaMatch.h"
#include "match.cuh"

mwvcv::CudaMatch::CudaMatch()
{
    cudaMallocPitch(reinterpret_cast<void**>(&descq_d_), &pitch_, 64, maxnquery_);
    cudaMemset2D(descq_d_, pitch_, 0, 64, maxnquery_);

    cudaMallocPitch(reinterpret_cast<void**>(&desct_d_), &pitch_, 64, maxntrain_);
    cudaMemset2DAsync(desct_d_, pitch_, 0, 64, maxntrain_);

    cudaMalloc(reinterpret_cast<void**>(&dmatches_d_), maxnquery_ * 2 * sizeof(cv::DMatch));
    dmatches_h_ = new cv::DMatch[2 * maxnquery_];

    const long size = 2 * maxnquery_ * sizeof(cv::DMatch)  + 64 * maxnquery_ + 64 * maxntrain_;
    std::cout << "Allocating " << size / 1024 / 1024 << " Mbytes of gpu memory for Akaze feature matching" << std::endl;
}

mwvcv::CudaMatch::~CudaMatch()
{
    cudaFree(descq_d_);
    cudaFree(desct_d_);

    cudaFree(dmatches_d_);
    delete[] dmatches_h_;
}

void mwvcv::CudaMatch::bfmatch(cv::Mat& desc_query, cv::Mat& desc_train, std::vector<std::vector<cv::DMatch>>& dmatches) const
{
    // Lock the mutex to prevent concurrent access to this function
    std::lock_guard<std::mutex> lock(mutex_);

    cudaMemcpy2DAsync(descq_d_, pitch_, desc_query.data, desc_query.cols, desc_query.cols, desc_query.rows,
                      cudaMemcpyHostToDevice);
    cudaMemcpy2DAsync(desct_d_, pitch_, desc_train.data, desc_train.cols, desc_train.cols, desc_train.rows,
                      cudaMemcpyHostToDevice);

    matchDescriptors(desc_query, desc_train, pitch_, descq_d_, desct_d_, dmatches_d_);

    cudaMemcpy(dmatches_h_, dmatches_d_, desc_query.rows * 2 * sizeof(cv::DMatch), cudaMemcpyDeviceToHost);

    dmatches.clear();
    for (int i = 0; i < desc_query.rows; ++i) {
        std::vector<cv::DMatch> tdmatch;
        tdmatch.push_back(dmatches_h_[2 * i]);
        tdmatch.push_back(dmatches_h_[2 * i + 1]);
        dmatches.push_back(tdmatch);
    }
}