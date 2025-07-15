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

#ifndef AKAZECUOPTIONS_H
#define AKAZECUOPTIONS_H

namespace mwvcv
{
    enum class DESCRIPTOR_TYPE
    {
        SURF_UPRIGHT  = 0, ///< Upright descriptors, not invariant to rotation
        SURF          = 1,
        MSURF_UPRIGHT = 2, ///< Upright descriptors, not invariant to rotation
        MSURF         = 3,
        MLDB_UPRIGHT  = 4, ///< Upright descriptors, not invariant to rotation
        MLDB          = 5
    };

    enum class DIFFUSIVITY_TYPE
    {
        PM_G1       = 0,
        PM_G2       = 1,
        WEICKERT    = 2,
        CHARBONNIER = 3
    };

    struct AkazeCuOptions
    {
        int img_width  = 0; ///< Width of the input image
        int img_height = 0; ///< Height of the input image

        int omin       = 0; ///< Initial octave level (-1 means that the size of the input image is duplicated)
        int omax       = 1; ///< Maximum octave evolution of the image 2^sigma (coarsest scale sigma units)
        int nsublevels = 1; ///< Default number of sublevels per scale level

        float kcontrast            = 0.001f; ///< The contrast factor parameter
        float kcontrast_percentile = 0.7f;   ///< Percentile level for the contrast factor
        int   kcontrast_nbins      = 300;    ///< Number of bins for the contrast factor histogram

        // This is a base scale offset to define sigma, a parameter that often relates to Gaussian blurring or
        // smoothing. Sets the stage for what features are visible in the scale-space. Smaller soffset means finer
        // feature. It influences the fundamental granularity of features seen across all octaves and sublevels
        float soffset = 1.6f;

        // This is a predefined multiplier that allows scaling of to determine the size of filters used for multiscale
        // derivative computations. It focuses on refining how individual derivative responses are computed within the
        // already established scale. Smaller values highlight finer details, as the kernel for derivative computation
        // is applied over smaller areas.
        float derivative_factor = 1.5f;

        DIFFUSIVITY_TYPE diffusivity = DIFFUSIVITY_TYPE::PM_G2; ///< Diffusivity type, used in flow function

        float dthreshold     = 0.0001f;  ///< Detector response threshold to accept point
        float min_dthreshold = 0.00001f; ///< Minimum detector threshold to accept a point

        DESCRIPTOR_TYPE descriptor = DESCRIPTOR_TYPE::MLDB; ///< Type of descriptor

        int descriptor_size     = 0; ///< Size of the descriptor in bits. 0->Full size
        int descriptor_channels = 3; ///< Number of channels in the descriptor (1, 2, 3)

        // When we generate a descriptor, we will look the image patch around the key point.
        // The actual patch size is 2*descriptor_pattern_size*point.scale
        // This image patch will be sampled as 2x2, 3x3 and 4x4
        int descriptor_pattern_size = 10;

        int ncudaimages  = 4;         ///< Number of CUDA images allocated per octave
        int maxkeypoints = 16 * 8192; ///< Maximum number of keypoints allocated

        // filterExtrema step takes a long time, but normally few keypoints are filtered
        // We can skip this step to save some time for feature detection
        bool skip_filter_extrema = true;
    };

} // namespace mwvcv

#endif // AKAZECUOPTIONS_H
