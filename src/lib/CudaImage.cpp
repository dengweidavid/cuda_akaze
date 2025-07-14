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

#include "CudaImage.h"
#include "cudaUtils.h"
#include "TimerGPU.h"

#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

mwvcv::CudaImage::CudaImage()
    : width_(0), height_(0), pitch_(0), h_data_(nullptr), d_data_(nullptr), h_internalAlloc_(false),
      d_internalAlloc_(false)
{
    ;
}

mwvcv::CudaImage::~CudaImage()
{
    if (d_internalAlloc_ && d_data_ != nullptr) {
        safeCall(cudaFree(d_data_));
    }
    d_data_ = nullptr;

    if (h_internalAlloc_ && h_data_ != nullptr) {
        free(h_data_);
    }
    h_data_ = nullptr;
}

void mwvcv::CudaImage::allocate(const int w, const int h, const int p, const bool host, float* hostmem, float* devmem)
{
    width_  = w;
    height_ = h;
    pitch_  = p;

    h_data_ = hostmem;
    d_data_ = devmem;

    if (host && hostmem == nullptr) {
        h_data_          = static_cast<float*>(malloc(sizeof(float) * pitch_ * height_));
        h_internalAlloc_ = true;
    }

    if (devmem == nullptr) {
        safeCall(cudaMallocPitch(reinterpret_cast<void**>(&d_data_), reinterpret_cast<size_t*>(&pitch_),
                                 sizeof(float) * width_, static_cast<size_t>(height_)));
        pitch_ /= sizeof(float);
        if (d_data_ == nullptr) {
            std::cerr << "Failed to allocate device data\n";
        }
        d_internalAlloc_ = true;
    }
}

double mwvcv::CudaImage::download() const
{
    const TimerGPU timer(nullptr);

    const size_t p = sizeof(float) * pitch_;
    if (d_data_ != nullptr && h_data_ != nullptr) {
        safeCall(cudaMemcpy2D(d_data_, p, h_data_, sizeof(float) * width_, sizeof(float) * width_, height_,
                              cudaMemcpyHostToDevice));
    }

    double gpuTime = timer.read();
#ifdef VERBOSE
    std::cout << "Download time = " << gpuTime << " ms\n";
#endif
    return gpuTime;
}

double mwvcv::CudaImage::readback() const
{
    const TimerGPU timer(nullptr);

    const size_t p = sizeof(float) * pitch_;
    safeCall(cudaMemcpy2D(h_data_, sizeof(float) * width_, d_data_, p, sizeof(float) * width_, height_,
                          cudaMemcpyDeviceToHost));

    double gpuTime = timer.read();
#ifdef VERBOSE
    std::cout << "Readback time = " << gpuTime << " ms\n";
#endif
    return gpuTime;
}

double mwvcv::CudaImage::copyTo(CudaImage& outimg) const // NOLINT(readability-non-const-parameter)
{
    const TimerGPU timer(nullptr);

    safeCall(cudaMemcpy2DAsync(outimg.d_data_, sizeof(float) * outimg.pitch_, d_data_, sizeof(float) * outimg.pitch_,
                               sizeof(float) * width_, height_, cudaMemcpyDeviceToDevice));

    double gpuTime = timer.read();
#ifdef VERBOSE
    std::cout << "Copy time = " << gpuTime << " ms\n";
#endif
    return gpuTime;
}