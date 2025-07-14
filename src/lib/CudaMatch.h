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

#ifndef CUDAMATCH_H
#define CUDAMATCH_H

#include <vector>
#include <opencv2/opencv.hpp>

namespace mwvcv
{
    class CudaMatch
    {
      public:
        CudaMatch();
        ~CudaMatch();

        void bfmatch(cv::Mat& desc_query, cv::Mat& desc_train, std::vector<std::vector<cv::DMatch>>& dmatches) const;

      private:
        int            maxnquery_ = 8192;
        unsigned char* descq_d_   = nullptr;

        int            maxntrain_ = 8192;
        unsigned char* desct_d_   = nullptr;

        cv::DMatch* dmatches_d_ = nullptr;
        cv::DMatch* dmatches_h_ = nullptr;

        size_t pitch_ = 0;

        mutable std::mutex mutex_; // Mutex ensures thread safety for bfmatch
    };

} // namespace mwvcv

#endif // CUDAMATCH_H
