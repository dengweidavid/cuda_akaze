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

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/cudafeatures2d.hpp>

#include "AkazeCuOptions.h"
#include "CudaAkaze.h"
#include "CudaMatch.h"
#include "utils.h"

using namespace mwvcv;

// Function to compare CUDA matches and OpenCV matches
double compareMatches(const std::vector<std::vector<cv::DMatch>> &gpu_matches,
                      const std::vector<std::vector<cv::DMatch>> &cv_matches);

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <image1> <image2>" << std::endl;
    return -1;
  }

  std::string img_path1 = argv[1];
  std::string img_path2 = argv[2];

  cv::Mat img1 = cv::imread(img_path1, 0);
  cv::Mat img2 = cv::imread(img_path2, 0);
  if (img1.empty() || img2.empty()) {
    std::cerr << "Error reading one or both images" << std::endl;
    return -1;
  }

  // Detect and compute descriptors
  std::vector<cv::KeyPoint> kpts1, kpts2;
  cv::Mat desc1, desc2;

  AkazeCuOptions options;
  options.dthreshold = 0.0001f;
  options.omax = 1;
  options.nsublevels = 1;

  options.img_width = img1.cols;
  options.img_height = img1.rows;
  CudaAkaze cuAkaze1(options);
  cuAkaze1.createNonlinearScaleSpace(img1);
  cuAkaze1.featureDetection(kpts1);
  cuAkaze1.computeDescriptors(kpts1, desc1);

  options.img_width = img2.cols;
  options.img_height = img2.rows;
  CudaAkaze cuAkaze2(options);
  cuAkaze2.createNonlinearScaleSpace(img2);
  cuAkaze2.featureDetection(kpts2);
  cuAkaze2.computeDescriptors(kpts2, desc2);

  // Matching descriptors

  CudaMatch cudaMatcher;
  auto t1 = static_cast<double>(cv::getTickCount());
  std::vector<std::vector<cv::DMatch>> gpu_dmatches;
  cudaMatcher.bfmatch(desc1, desc2, gpu_dmatches);
  auto t2 = static_cast<double>(cv::getTickCount());
  double gpu_ts = 1000.0 * (t2 - t1) / cv::getTickFrequency();

  cv::BFMatcher cv_matcher{cv::NORM_HAMMING};
  auto t3 = static_cast<double>(cv::getTickCount());
  std::vector<std::vector<cv::DMatch>> cv_dmatches;
  cv_matcher.knnMatch(desc1, desc2, cv_dmatches, 2);
  auto t4 = static_cast<double>(cv::getTickCount());
  double cv_ts = 1000.0 * (t4 - t3) / cv::getTickFrequency();

  // cv::cuda::GpuMat desc1_gpu;
  // desc1_gpu.upload(desc1);
  // cv::cuda::GpuMat desc2_gpu;
  // desc2_gpu.upload(desc2);
  // auto cv_cuda_matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
  // std::vector<std::vector<cv::DMatch>> cv_cuda_matches;
  // cv_cuda_matcher->knnMatch(desc1_gpu, desc2_gpu, cv_cuda_matches, 2);

  // Print feature extraction details
  std::cout << "CUDA AKAZE KeyPoints Image1: " << kpts1.size()
            << " Image2: " << kpts2.size() << std::endl;
  std::cout << "CUDA Matches: " << gpu_dmatches.size() << std::endl;
  std::cout << "OpenCV Matches: " << cv_dmatches.size() << std::endl;
  std::cout << "CUDA Match Time: " << gpu_ts << " ms\n";
  std::cout << "OpenCV Match Time: " << cv_ts << " ms\n";

  // Store match results
  std::cout << "CUDA/OpenCV Diff Ratio: "
            << compareMatches(gpu_dmatches, cv_dmatches) << std::endl;
  save_matches("./gpu_match.txt", gpu_dmatches);
  save_matches("./cv_match.txt", cv_dmatches);

  return 0;
}

double compareMatches(const std::vector<std::vector<cv::DMatch>> &gpu_matches,
                      const std::vector<std::vector<cv::DMatch>> &cv_matches) {
  const int total_matches = std::min(gpu_matches.size(), cv_matches.size());

  int differing_matches = 0;

  // Loop through matches to compare
  for (int i = 0; i < total_matches; i++) {
    if (gpu_matches[i].empty() || cv_matches[i].empty()) {
      differing_matches++;
      continue;
    }

    for (int j = 0; j < gpu_matches[i].size(); j++) {
      bool no_match = true;
      for (int k = 0; k < cv_matches[i].size(); k++) {
        if (gpu_matches[i][j].trainIdx == cv_matches[i][k].trainIdx) {
          no_match = false;
        }
      }
      if (no_match) {
        differing_matches++;
      }
    }
  }

  // Calculate and return the ratio of differences
  return static_cast<double>(differing_matches) /
         static_cast<double>(total_matches * 2);
}
