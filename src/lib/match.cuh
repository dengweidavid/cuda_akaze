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

#ifndef MATCH_CUH
#define MATCH_CUH

#include <opencv2/opencv.hpp>
#include <vector>

#define NTHREADS_MATCH 32

namespace mwvcv
{
    void matchDescriptors(cv::Mat& desc_query, cv::Mat& desc_train, size_t pitch, unsigned char* descq_d,
                          unsigned char* desct_d, cv::DMatch* dmatches_d);

    void matchDescriptors(cv::Mat& desc_query, cv::Mat& desc_train, std::vector<std::vector<cv::DMatch>>& dmatches);

}

#endif //MATCH_CUH