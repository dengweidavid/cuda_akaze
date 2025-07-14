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

#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <cstdio>
#include <stdexcept>
#include <cuda_runtime.h>

namespace mwvcv
{
    #define safeCall(err) safeCall_(err, __FILE__, __LINE__)
    #define safeThreadSync() safeThreadSync_(__FILE__, __LINE__)
    #define checkMsg(msg) checkMsg_(msg, __FILE__, __LINE__)

    inline void safeCall_(const cudaError err, const char* file, const int line)
    {
        if (cudaSuccess != err) {
            fprintf(stderr, "safeCall() Runtime API error in file <%s>, line %i : %s.\n", file, line, cudaGetErrorString(err));
            throw std::runtime_error("safeCall() Runtime API error.");
        }
    }

    inline void safeThreadSync_(const char* file, const int line)
    {
        if (const cudaError err = cudaDeviceSynchronize(); cudaSuccess != err) {
            fprintf(stderr, "threadSynchronize() Driver API error in file '%s' in line %i : %s.\n", file, line, cudaGetErrorString(err));
            throw std::runtime_error("threadSynchronize() Driver API error.");
        }
    }

    inline void checkMsg_(const char* errorMessage, const char* file, const int line)
    {
        if (const cudaError_t err = cudaGetLastError(); cudaSuccess != err) {
            fprintf(stderr, "checkMsg() CUDA error: %s in file <%s>, line %i : %s.\n", errorMessage, file, line, cudaGetErrorString(err));
            throw std::runtime_error("checkMsg() CUDA error.");
        }
    }

    inline bool deviceInit(int dev)
    {
        int deviceCount;
        safeCall(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
            return false;
        }

        if (dev < 0) {
            dev = 0;
        }
        if (dev > deviceCount - 1) {
            dev = deviceCount - 1;
        }

        cudaDeviceProp deviceProp{};
        safeCall(cudaGetDeviceProperties(&deviceProp, dev));
        if (deviceProp.major < 1) {
            fprintf(stderr, "error: device does not support CUDA.\n");
            return false;
        }

        safeCall(cudaSetDevice(dev));
        return true;
    }

} // namespace mwvcv

#endif // CUDAUTILS_H
