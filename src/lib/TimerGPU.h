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

#ifndef TIMERGPU_H
#define TIMERGPU_H

#include <cuda_runtime.h>

namespace mwvcv
{
    class TimerGPU
    {
      public:
        explicit TimerGPU(cudaStream_t stream = nullptr) : stream_(stream)
        {
            initializeTimer();
        }

        ~TimerGPU()
        {
            cudaEventDestroy(start_);
            cudaEventDestroy(stop_);
        }

        [[nodiscard]] float read() const
        {
            cudaEventRecord(stop_, stream_);
            cudaEventSynchronize(stop_);
            float time;
            cudaEventElapsedTime(&time, start_, stop_);
            return time;
        }

      private:

        void initializeTimer()
        {
            cudaEventCreate(&start_);
            cudaEventCreate(&stop_);
        }

        cudaEvent_t  start_{}, stop_{};
        cudaStream_t stream_;
    };

} // namespace mwvcv

#endif // TIMERGPU_H
