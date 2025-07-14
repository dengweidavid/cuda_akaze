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
#include <opencv2/highgui/highgui.hpp>
#include <cuda_profiler_api.h>

#include "CudaAkaze.h"
#include "AkazeCuOptions.h"

#include "utils.h"
#include "cxxopts.hpp"

using namespace mwvcv;

bool parse_input_options(AkazeCuOptions& options, std::string& img_path, std::string& kpts_path, int argc,
                         char* argv[]);

int main(int argc, char* argv[])
{
    // Parse the input command line options
    AkazeCuOptions options;
    std::string    img_path, kpts_path;
    if (!parse_input_options(options, img_path, kpts_path, argc, argv)) {
        return -1;
    }

    // Try to read the image and if necessary, convert to grayscale.
    const cv::Mat img = cv::imread(img_path, 0);
    if (img.data == nullptr) {
        std::cerr << "Error: cannot load image from file:" << std::endl << img_path << std::endl;
        return -1;
    }

    // Remember to specify image dimensions in AKAZE's options
    options.img_width  = img.cols;
    options.img_height = img.rows;

    CudaAkaze cuAkaze(options);

    cudaProfilerStart();

    // Extract features
    std::vector<cv::KeyPoint> kpts;

    auto t1 = static_cast<double>(cv::getTickCount());
    cuAkaze.createNonlinearScaleSpace(img);
    cuAkaze.featureDetection(kpts);
    auto   t2   = static_cast<double>(cv::getTickCount());
    double tdet = 1000.0 * (t2 - t1) / cv::getTickFrequency();

    // Compute descriptors.
    cv::Mat desc;

    auto t3 = static_cast<double>(cv::getTickCount());
    cuAkaze.computeDescriptors(kpts, desc);
    auto   t4    = static_cast<double>(cv::getTickCount());
    double tdesc = 1000.0 * (t4 - t3) / cv::getTickFrequency();

    cudaProfilerStop();

    // Summarise the computation times.
    cuAkaze.showComputationTimes();

    std::cout << "Number of points: " << kpts.size() << std::endl;
    std::cout << "Time Detector: " << tdet << " ms" << std::endl;
    std::cout << "Time Descriptor: " << tdesc << " ms" << std::endl;

    auto img_rgb = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC3);
    cvtColor(img, img_rgb, cv::COLOR_GRAY2BGR);
    draw_keypoints(img_rgb, kpts);

    cv::namedWindow("A-KAZE", cv::WINDOW_AUTOSIZE);
    cv::imshow("A-KAZE", img_rgb);
    cv::waitKey(0);

    // Save keypoints in ASCII format
    if (!kpts_path.empty()) {
        save_keypoints(kpts_path, kpts, desc, true);
    }

    return 0;
}

bool parse_input_options(AkazeCuOptions& options, std::string& img_path, std::string& kpts_path, const int argc,
                         char* argv[])
{
    // clang-format off
    cxxopts::Options cxxOptions("akaze_feature", "AKAZE-GPU Feature Detect and Compute Tool");
    cxxOptions
        .set_width(70)
        .set_tab_expansion()
        .add_options()
        ("i, image", "Input image path", cxxopts::value<std::string>())
        ("o, output", "Output keypoints path", cxxopts::value<std::string>()->default_value(""))
        ("soffset", "Base scale offset (sigma units) \n", cxxopts::value<float>()->default_value("1.6"))
        ("omax", "Maximum octave of image evolution \n",  cxxopts::value<int>()->default_value("1"))
        ("nsublevels", "Number of sublevels per octave \n",  cxxopts::value<int>()->default_value("1"))
        ("diffusivity", "Diffusivity function. Possible values: \n"
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

    img_path  = result["image"].as<std::string>();
    kpts_path = result["output"].as<std::string>();

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