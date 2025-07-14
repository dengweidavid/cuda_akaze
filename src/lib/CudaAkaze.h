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

#ifndef CUDAAKAZE_H
#define CUDAAKAZE_H

#include <opencv2/core/core.hpp>

#include "AkazeCuOptions.h"
#include "AkazeTiming.h"
#include "CudaImage.h"

namespace mwvcv
{
    struct TEvolution
    {
        cv::Mat Lx, Ly;        ///< First order spatial derivatives
        cv::Mat Lxx, Lxy, Lyy; ///< Second order spatial derivatives
        cv::Mat Lflow;         ///< Diffusivity image
        cv::Mat Lt;            ///< Evolution image
        cv::Mat Lsmooth;       ///< Smoothed image
        cv::Mat Lstep;         ///< Evolution step update
        cv::Mat Ldet;          ///< Detector response

        float  esigma     = 0.0f; ///< Evolution sigma. For linear diffusion t = sigma^2 / 2
        size_t sigma_size = 0;    ///< Integer sigma. For computing the feature detector responses
        float  etime      = 0.0f; ///< Evolution time
        size_t octave     = 0;    ///< Image octave
        size_t sublevel   = 0;    ///< Image sublevel in each octave
    };

    class CudaAkaze
    {
      public:
        /// Constructor with input options
        /// @param options Configuration options for AKAZE CUDA
        /// @note This constructor allocates memory for the nonlinear scale space
        explicit CudaAkaze(const AkazeCuOptions& options);

        /// Destructor
        ~CudaAkaze();

        /// This method creates the nonlinear scale space for a given image
        /// @param img Input image for which the nonlinear scale space needs to be created
        /// @return 0 if the nonlinear scale space was created successfully, -1 otherwise
        int createNonlinearScaleSpace(const cv::Mat& img);

        /// This method selects interesting keypoints through the nonlinear scale space
        /// @param kpts Vector of detected keypoints
        void featureDetection(std::vector<cv::KeyPoint>& kpts);

        /// Feature description methods
        /// @param kpts Vector of detected keypoints
        /// @param desc Matrix to store the descriptors, each row is one feature point's descriptor
        void computeDescriptors(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc);

        /// Display timing information
        void showComputationTimes() const;

      private:
        /// Allocate the memory for the nonlinear scale space
        void allocateMemoryEvolution();

        /// This function rounds float to the nearest integer
        static int fRound(const float flt) { return static_cast<int>(flt + 0.5f); } // NOLINT

        AkazeCuOptions          options_;   ///< Configuration options for AKAZE CUDA
        std::vector<TEvolution> evolution_; ///< Vector of nonlinear diffusion evolution

        /// FED parameters
        int                             ncycles_;    ///< Number of cycles
        bool                            reordering_; ///< Flag for reordering time steps
        std::vector<std::vector<float>> tsteps_;     ///< Vector of FED dynamic time steps
        std::vector<int>                nsteps_;     ///< Vector of number of steps per cycle

        /// CUDA memory buffers
        float*                 cuda_memory       = nullptr;
        cv::KeyPoint*          cuda_points       = nullptr;
        cv::KeyPoint*          cuda_bufferpoints = nullptr;
        cv::Mat                cuda_desc;
        float*                 cuda_descbuffer = nullptr;
        int*                   cuda_ptindices  = nullptr;
        CudaImage*             cuda_images     = nullptr;
        std::vector<CudaImage> cuda_buffers;

        int nump_ = 0; ///< Number of detected feature points

        /// Computation times variables in ms
        AkazeTiming timing_;
    };

} // namespace mwvcv

#endif // CUDAAKAZE_H
