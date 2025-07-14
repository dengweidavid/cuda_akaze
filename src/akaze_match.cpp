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

#include <opencv2/imgproc/imgproc.hpp>
#include <cuda_profiler_api.h>

#include "CudaAkaze.h"
#include "CudaMatch.h"
#include "AkazeCuOptions.h"

#include "utils.h"
#include "cxxopts.hpp"

using namespace mwvcv;

// Image matching options
constexpr float MIN_H_ERROR = 2.50f; ///< Maximum error in pixels to accept an inlier
constexpr float DRATIO      = 0.80f; ///< NNDR Matching value

bool parse_input_options(AkazeCuOptions& options, std::string& img_path1, std::string& img_path2, int argc,
                         char* argv[]);

int main(int argc, char* argv[])
{
    // Parse the input command line options
    AkazeCuOptions options;
    std::string    img_path1, img_path2;
    if (!parse_input_options(options, img_path1, img_path2, argc, argv)) {
        return -1;
    }

    // Read image 1 and if necessary, convert to grayscale.
    cv::Mat img1 = cv::imread(img_path1, 0);
    if (img1.data == nullptr) {
        std::cerr << "Error loading image 1: " << img_path1 << std::endl;
        return -1;
    }

    // Read image 2 and if necessary, convert to grayscale.
    cv::Mat img2 = cv::imread(img_path2, 0);
    if (img2.data == nullptr) {
        std::cerr << "Error loading image 2: " << img_path2 << std::endl;
        return -1;
    }

    // Create the first AKAZE object
    options.img_width  = img1.cols;
    options.img_height = img1.rows;
    CudaAkaze cuAkaze1(options);

    // Create the second AKAZE object
    options.img_width  = img2.cols;
    options.img_height = img2.rows;
    CudaAkaze cuAkaze2(options);

    cudaProfilerStart();

    // Detect and compute descriptors
    std::vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat                   desc1, desc2;

    auto t1 = static_cast<double>(cv::getTickCount());

    cuAkaze1.createNonlinearScaleSpace(img1);
    cuAkaze1.featureDetection(kpts1);
    cuAkaze1.computeDescriptors(kpts1, desc1);

    cuAkaze2.createNonlinearScaleSpace(img2);
    cuAkaze2.featureDetection(kpts2);
    cuAkaze2.computeDescriptors(kpts2, desc2);

    auto   t2     = static_cast<double>(cv::getTickCount());
    double takaze = 1000.0 * (t2 - t1) / cv::getTickFrequency();

    // Matching descriptors
    std::vector<std::vector<cv::DMatch>> dmatches;

    auto t3 = static_cast<double>(cv::getTickCount());

    CudaMatch cudaMatcher;
    cudaMatcher.bfmatch(desc1, desc2, dmatches);

    auto   t4     = static_cast<double>(cv::getTickCount());
    double tmatch = 1000.0 * (t4 - t3) / cv::getTickFrequency();

    cudaProfilerStop();

    std::cout << "#matches: " << dmatches.size() << std::endl;
    std::cout << "#kptsq:   " << kpts1.size() << std::endl;
    std::cout << "#kptst:   " << kpts2.size() << std::endl;
    std::cout << "Time AKAZE: " << takaze << " ms" << std::endl;
    std::cout << "Time Match: " << tmatch << " ms" << std::endl;

    // Compute Inliers!!
    std::vector<cv::Point2f> matches, inliers;
    matches2points_nndr(kpts2, kpts1, dmatches, matches, DRATIO);
    compute_inliers_ransac(matches, inliers, MIN_H_ERROR, false);

    // Compute the inlier statistics
    size_t nmatches  = matches.size() / 2;
    size_t ninliers  = inliers.size() / 2;
    size_t noutliers = nmatches - ninliers;
    float  ratio     = 100.0f * (static_cast<float>(ninliers) / static_cast<float>(nmatches));
    std::cout << "Number of Matches after NNDR (nearest neighbour distance ratio): " << nmatches << std::endl;
    std::cout << "Number of Inliers after RANSAC: " << ninliers << std::endl;
    std::cout << "Number of Outliers: " << noutliers << std::endl;
    std::cout << "Inliers Ratio: " << ratio << std::endl << std::endl;

    // Draw key points
    auto img1_rgb = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC3);
    auto img2_rgb = cv::Mat(cv::Size(img2.cols, img1.rows), CV_8UC3);
    cvtColor(img1, img1_rgb, cv::COLOR_GRAY2BGR);
    cvtColor(img2, img2_rgb, cv::COLOR_GRAY2BGR);
    draw_keypoints(img1_rgb, kpts1, false);
    draw_keypoints(img2_rgb, kpts2, false);

    // Draw matches
    cv::Mat img_inliers;
    draw_matches_vertical(img1_rgb, img2_rgb, img_inliers, inliers, 1);
    cv::namedWindow("Inliers", cv::WINDOW_NORMAL);
    cv::imshow("Inliers", img_inliers);
    cv::waitKey(0);

    return 0;
}

bool parse_input_options(AkazeCuOptions& options, std::string& img_path1, std::string& img_path2, const int argc,
                         char* argv[])
{
    // clang-format off
    cxxopts::Options cxxOptions("akaze_feature", "AKAZE-GPU Feature Detect and Compute Tool");
    cxxOptions
        .set_width(70)
        .set_tab_expansion()
        .add_options()
        ("q, query", "Input query image path", cxxopts::value<std::string>())
        ("t, train", "Input train image path", cxxopts::value<std::string>())
        ("soffset", "Base scale offset (sigma units) \n", cxxopts::value<float>()->default_value("1.6"))
        ("omax", "Maximum octave of image evolution \n",  cxxopts::value<int>()->default_value("1"))
        ("nsublevels", "Number of sublevels per octave \n",  cxxopts::value<int>()->default_value("1"))
        ("d, diffusivity", "Diffusivity function. Possible values: \n"
                        "0 -> Perona-Malik, g1 \n"
                        "1 -> Perona-Malik, g2 \n"
                        "2 -> Weickert diffusivity \n"
                        "3 -> Charbonnier diffusivity \n", cxxopts::value<int>()->default_value("1"))
        ("dthreshold", "Feature detector threshold response for keypoints",  cxxopts::value<float>()->default_value("0.0001f"))
        ("desc", "Descriptor Type. Possible values: \n"
                      "0 -> SURF_UPRIGHT \n"
                      "1 -> SURF \n"
                      "2 -> M-SURF_UPRIGHT \n"
                      "3 -> M-SURF \n"
                      "4 -> M-LDB_UPRIGHT \n"
                      "5 -> M-LDB \n", cxxopts::value<int>()->default_value("5"))
        ("desc_channels", "Descriptor Channels for M-LDB. Valid values: \n"
                     "1 -> intensity \n"
                     "2 -> intensity + gradient magnitude \n"
                     "3 -> intensity + X and Y gradients  \n", cxxopts::value<int>()->default_value("3"))
        ("desc_size", "Descriptor size for M-LDB in bits. \n"
                            "0: means the full length descriptor (486)!! \n", cxxopts::value<int>()->default_value("0"))
        ("h,help", "Print help");
    // clang-format on

    const auto result = cxxOptions.parse(argc, argv);

    if (result.count("help") || argc == 1) {
        std::cout << cxxOptions.help() << "\n";
        return false;
    }

    img_path1 = result["query"].as<std::string>();
    img_path2 = result["train"].as<std::string>();

    options.soffset             = result["soffset"].as<float>();
    options.omax                = result["omax"].as<int>();
    options.nsublevels          = result["nsublevels"].as<int>();
    options.diffusivity         = static_cast<DIFFUSIVITY_TYPE>(result["diffusivity"].as<int>());
    options.dthreshold          = result["dthreshold"].as<float>();
    options.descriptor          = static_cast<DESCRIPTOR_TYPE>(result["desc"].as<int>());
    options.descriptor_channels = result["desc_channels"].as<int>();
    options.descriptor_size     = result["desc_size"].as<int>();

    return true;
}
