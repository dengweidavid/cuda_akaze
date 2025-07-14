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

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include "CudaAkaze.h"
#include "AkazeCuOptions.h"
#include "utils.h"

using namespace mwvcv;

int pad_size = 25;

float threshold     = 0.0001f;
int   nOctaves      = 1;
int   nOctaveLayers = 1;

// Function to extract features using OpenCV's AKAZE
std::tuple<std::vector<cv::KeyPoint>, cv::Mat, double> extractWithOpenCV(const cv::Mat& img);
// Function to extract features using CUDA-based AKAZE
std::tuple<std::vector<cv::KeyPoint>, cv::Mat, double> extractWithCudaAkaze(const cv::Mat&  img,
                                                                            AkazeCuOptions& options);
// Function to shift keypoints to account for padding
void shiftKeypoints(std::vector<cv::KeyPoint>& kpts, int rows, int cols);

// Function to compare keypoints and descriptor
std::tuple<int, int>       compareKeyPoints(const std::vector<cv::KeyPoint>& gpu_kpts,
                                            const std::vector<cv::KeyPoint>& cv_kpts, float eps = 1e-3f);
std::tuple<double, double> compareDescriptors(const std::vector<cv::KeyPoint>& gpu_kpts,
                                              const std::vector<cv::KeyPoint>& cv_kpts, const cv::Mat& gpu_desc,
                                              const cv::Mat& cv_desc, float eps = 1e-3f);

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image>" << std::endl;
        return -1;
    }

    std::string img_path = argv[1];
    cv::Mat     img      = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error reading image" << std::endl;
        return -1;
    }

    // pad the image
    cv::Mat img_pad = cv::Mat::zeros(img.rows + 2 * pad_size, img.cols + 2 * pad_size, img.type());
    img.copyTo(img_pad(cv::Range(pad_size, img.rows + pad_size), //
                       cv::Range(pad_size, img.cols + pad_size)));

    // AKAZE CUDA Options
    AkazeCuOptions options;
    options.img_width  = img_pad.cols;
    options.img_height = img_pad.rows;

    // AKAZE GPU Results
    auto [gpu_kpts, gpu_desc, gpu_st] = extractWithCudaAkaze(img_pad, options);
    shiftKeypoints(gpu_kpts, img.rows, img.cols);

    // OpenCV AKAZE Results
    auto [cv_kpts, cv_desc, cv_st] = extractWithOpenCV(img_pad);
    shiftKeypoints(cv_kpts, img.rows, img.cols);

    // Print feature extraction details
    std::cout << "CUDA AKAZE KeyPoints: " << gpu_kpts.size() << std::endl;
    std::cout << "OpenCV AKAZE KeyPoints: " << cv_kpts.size() << std::endl;
    std::cout << "CUDA AKAZE Time: " << gpu_st << " ms" << std::endl;
    std::cout << "OpenCV AKAZE Time: " << cv_st << " ms" << std::endl;

    auto [gpu_not_in_cv, cv_not_in_gpu] = compareKeyPoints(gpu_kpts, cv_kpts);
    std::cout << "CUDA KeyPoints, NOT in OpenCV, Number: " << gpu_not_in_cv << std::endl;
    std::cout << "OpenCV KeyPoints, NOT in CUDA, Number: " << cv_not_in_gpu << std::endl;

    auto [descNumRatio, descDiffRatio] = compareDescriptors(gpu_kpts, cv_kpts, gpu_desc, cv_desc);
    std::cout << "CUDA/OpenCV Descriptor Num Ratio: " << descNumRatio << std::endl;
    std::cout << "CUDA/OpenCV Descriptor Diff Ratio: " << descDiffRatio << std::endl;

    // Save keypoints in ASCII format
    save_keypoints("./gpu_keypoints.txt", gpu_kpts, gpu_desc, true);
    save_keypoints("./cv_keypoints.txt", cv_kpts, cv_desc, true);

    // Display key points for AKAZE GPU Results
    auto gpu_img_rgb = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC3);
    cvtColor(img, gpu_img_rgb, cv::COLOR_GRAY2BGR);
    draw_keypoints(gpu_img_rgb, gpu_kpts);
    cv::namedWindow("GPU-AKAZE", cv::WINDOW_AUTOSIZE);
    cv::imshow("GPU-AKAZE", gpu_img_rgb);

    // Display key points for OpenCV AKAZE Results
    auto cv_img_rgb = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC3);
    cvtColor(img, cv_img_rgb, cv::COLOR_GRAY2BGR);
    draw_keypoints(cv_img_rgb, cv_kpts);
    cv::namedWindow("OpenCV-AKAZE", cv::WINDOW_AUTOSIZE);
    cv::imshow("OpenCV-AKAZE", cv_img_rgb);

    cv::waitKey(0);

    return 0;
}

std::tuple<std::vector<cv::KeyPoint>, cv::Mat, double> extractWithOpenCV(const cv::Mat& img)
{
    const cv::Ptr<cv::AKAZE>  akaze = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, //
                                                        threshold, nOctaves, nOctaveLayers, cv::KAZE::DIFF_PM_G2);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat                   descriptors;

    const auto t1 = static_cast<double>(cv::getTickCount());
    akaze->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
    const auto t2 = static_cast<double>(cv::getTickCount());
    double     st = 1000.0 * (t2 - t1) / cv::getTickFrequency();

    return {keypoints, descriptors, st};
}

std::tuple<std::vector<cv::KeyPoint>, cv::Mat, double> extractWithCudaAkaze(const cv::Mat& img, AkazeCuOptions& options)
{
    options.dthreshold = threshold;
    options.omax       = nOctaves;
    options.nsublevels = nOctaveLayers;
    CudaAkaze cudaAkaze(options);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat                   descriptors;

    const auto t1 = static_cast<double>(cv::getTickCount());
    cudaAkaze.createNonlinearScaleSpace(img);
    cudaAkaze.featureDetection(keypoints);
    cudaAkaze.computeDescriptors(keypoints, descriptors);
    const auto t2 = static_cast<double>(cv::getTickCount());
    double     st = 1000.0 * (t2 - t1) / cv::getTickFrequency();

    return {keypoints, descriptors, st};
}

void shiftKeypoints(std::vector<cv::KeyPoint>& kpts, const int rows, const int cols)
{
    for (auto& kpt : kpts) {
        const float pt_x = kpt.pt.x - static_cast<float>(pad_size);
        const float pt_y = kpt.pt.y - static_cast<float>(pad_size);
        if (pt_x >= 0 && pt_x < static_cast<float>(cols)) {
            if (pt_y >= 0 && pt_y < static_cast<float>(rows)) {
                kpt.pt.x = pt_x;
                kpt.pt.y = pt_y;
            }
        }
    }
}

// Helper to check if a point exists in a vector within epsilon
bool containsPoint(const std::vector<cv::KeyPoint>& vec, const cv::KeyPoint& kpt, const float eps = 1e-3f)
{
    for (const auto& p : vec) {
        if (cv::norm(p.pt - kpt.pt) < eps) {
            return true;
        }
    }
    return false;
}

std::tuple<int, int> compareKeyPoints(const std::vector<cv::KeyPoint>& gpu_kpts,
                                      const std::vector<cv::KeyPoint>& cv_kpts, const float eps)
{
    // gpu_kpts \ cv_kpts
    int gpu_not_in_cv = 0;
    for (const auto& pt_gpu : gpu_kpts) {
        if (!containsPoint(cv_kpts, pt_gpu, eps)) {
            gpu_not_in_cv++;
        }
    }

    // cv_kpts \ gpu_kpts
    int cv_not_in_gpu = 0;
    for (const auto& pt_cv : cv_kpts) {
        if (!containsPoint(gpu_kpts, pt_cv, eps)) {
            cv_not_in_gpu++;
        }
    }
    return {gpu_not_in_cv, cv_not_in_gpu};
}

std::tuple<double, double> compareDescriptors(const std::vector<cv::KeyPoint>& gpu_kpts,
                                              const std::vector<cv::KeyPoint>& cv_kpts, const cv::Mat& gpu_desc,
                                              const cv::Mat& cv_desc, const float eps)
{
    const cv::BFMatcher matcher{cv::NORM_HAMMING};

    std::vector<std::vector<cv::DMatch>> dmatches;
    matcher.knnMatch(gpu_desc, cv_desc, dmatches, 1);

    float hamming_dist_diff = 0.0f;
    int   num_desc_compare  = 0;
    int   num_desc_diff     = 0;
    for (const auto& matche : dmatches) {
        const cv::DMatch dmatch = matche[0];

        const int   queryIdx     = dmatch.queryIdx;
        const int   trainIdx     = dmatch.trainIdx;
        const float hamming_dist = dmatch.distance;

        if (const float ptDist = cv::norm(gpu_kpts[queryIdx].pt - cv_kpts[trainIdx].pt); ptDist < eps) {
            num_desc_compare++;
            if (hamming_dist > 5) {
                num_desc_diff++;
            }
            hamming_dist_diff += hamming_dist;
        }
    }

    return {static_cast<double>(num_desc_diff) / num_desc_compare,
            hamming_dist_diff / (num_desc_compare * gpu_desc.cols * 8)};
}