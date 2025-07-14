/**
 * @file utils.cpp
 * @brief Some utilities functions
 * @date Oct 07, 2014
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 */

#include "utils.h"

#include <fstream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

/* ************************************************************************* */
void compute_min_32F(const cv::Mat& src, float& value)
{
    float aux = 1000.0;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (src.at<float>(i, j) < aux)
                aux = src.at<float>(i, j);
        }
    }
    value = aux;
}

/* ************************************************************************* */
void compute_max_32F(const cv::Mat& src, float& value)
{
    float aux = 0.0;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (src.at<float>(i, j) > aux)
                aux = src.at<float>(i, j);
        }
    }
    value = aux;
}

/* ************************************************************************* */
void convert_scale(cv::Mat& src)
{

    float min_val = 0, max_val = 0;
    compute_min_32F(src, min_val);
    src = src - min_val;
    compute_max_32F(src, max_val);
    src = src / max_val;
}

/* ************************************************************************* */
void copy_and_convert_scale(const cv::Mat& src, cv::Mat dst)
{
    float min_val = 0, max_val = 0;
    src.copyTo(dst);
    compute_min_32F(dst, min_val);
    dst = dst - min_val;
    compute_max_32F(dst, max_val);
    dst /= max_val;
}

/* ************************************************************************* */
void draw_keypoints(cv::Mat& img, const std::vector<cv::KeyPoint>& kpts, const bool draw_point_size)
{
    int   x = 0, y = 0;
    float radius = 0.0;

    for (const auto& kpt : kpts) {
        x = (int)(kpt.pt.x + .5); // NOLINT
        y = (int)(kpt.pt.y + .5); // NOLINT

        radius = fabs(kpt.size / 2.0f);
        if (draw_point_size) {
            cv::circle(img, cv::Point(x, y), radius * 2.5, cv::Scalar(0, 255, 0), 1); // NOLINT
        }
        cv::circle(img, cv::Point(x, y), 1.0, cv::Scalar(0, 0, 255), -1);         // NOLINT
    }
}

/* ************************************************************************* */
int save_keypoints(const string& outFile, const std::vector<cv::KeyPoint>& kpts, const cv::Mat& desc,
                   const bool save_desc)
{
    const size_t nkpts = kpts.size();
    const int    dsize = desc.cols;

    ofstream ipfile(outFile.c_str());

    if (!ipfile) {
        cerr << "Couldn't open file '" << outFile << "'!" << endl;
        return -1;
    }

    if (!save_desc) {
        ipfile << 1 << endl << nkpts << endl;
    } else {
        ipfile << dsize << endl << nkpts << endl;
    }

    // Save interest point with descriptor in the format of Krystian Mikolajczyk
    // for reasons of comparison with other descriptors
    for (int i = 0; i < nkpts; i++) {
        // Radius of the keypoint
        float sc = (kpts[i].size);
        sc *= sc;

        ipfile << kpts[i].pt.x                   /* x-location of the interest point */
               << " " << kpts[i].pt.y            /* y-location of the interest point */
               << " " << 1.0 / sc                /* 1/r^2 */
               << " " << 0.0 << " " << 1.0 / sc; /* 1/r^2 */

        // Here comes the descriptor
        for (int j = 0; j < dsize; j++) {
            if (desc.type() == 0) {
                ipfile << " " << static_cast<int>(desc.at<unsigned char>(i, j));
            } else {
                ipfile << " " << (desc.at<float>(i, j));
            }
        }

        ipfile << endl;
    }

    // Close the txt file
    ipfile.close();

    return 0;
}

/* ************************************************************************* */
int save_matches(const string& outFile, const std::vector<std::vector<cv::DMatch>>& matches)
{
    ofstream ipfile(outFile.c_str());

    if (!ipfile) {
        cerr << "Couldn't open file '" << outFile << "'!" << endl;
        return -1;
    }

    // Writing match details into the file
    for (const auto& match_group : matches) {
        for (const auto& match : match_group) {
            ipfile << "QueryIdx: " << match.queryIdx
                    << ", TrainIdx: " << match.trainIdx
                    << ", Distance: " << match.distance
                    << std::endl;
        }
        ipfile << "----" << std::endl; // Separator between match groups
    }

    // Close the txt file
    ipfile.close();

    return 0;
}

/* ************************************************************************* */
void matches2points_nndr(const std::vector<cv::KeyPoint>& train, const std::vector<cv::KeyPoint>& query,
                         const std::vector<std::vector<cv::DMatch>>& matches, std::vector<cv::Point2f>& pmatches,
                         const float nndr)
{

    float dist1 = 0.0, dist2 = 0.0;
    for (const auto& matche : matches) {
        const cv::DMatch dmatch = matche[0];

        dist1 = matche[0].distance;
        dist2 = matche[1].distance;

        if (dist1 < nndr * dist2) {
            pmatches.push_back(train[dmatch.queryIdx].pt);
            pmatches.push_back(query[dmatch.trainIdx].pt);
        }
    }
}

/* ************************************************************************* */
void compute_inliers_ransac(const std::vector<cv::Point2f>& matches, std::vector<cv::Point2f>& inliers,
                            const float error, const bool use_fund)
{
    vector<cv::Point2f> points1, points2;
    for (size_t i = 0; i < matches.size(); i += 2) {
        points1.push_back(matches[i]);
        points2.push_back(matches[i + 1]);
    }

    if (const int npoints = static_cast<int>(matches.size()) / 2; npoints > 8) {
        cv::Mat H      = cv::Mat::zeros(3, 3, CV_32F);
        cv::Mat status = cv::Mat::zeros(npoints, 1, CV_8UC1);
        if (use_fund == true)
            H = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, error, 0.99, status);
        else
            H = cv::findHomography(points1, points2, cv::RANSAC, error, status);
        cout << "Homography matrix: \n" << H << endl;

        for (int i = 0; i < npoints; i++) {
            if (status.at<unsigned char>(i) == 1) {
                inliers.push_back(points1[i]);
                inliers.push_back(points2[i]);
            }
        }
    }
}

/* ************************************************************************* */
void compute_inliers_homography(const std::vector<cv::Point2f>& matches, std::vector<cv::Point2f>& inliers,
                                const cv::Mat& H, const float min_error)
{
    float h11 = 0.0, h12 = 0.0, h13 = 0.0;
    float h21 = 0.0, h22 = 0.0, h23 = 0.0;
    float h31 = 0.0, h32 = 0.0, h33 = 0.0;
    float x1 = 0.0, y1 = 0.0;
    float x2 = 0.0, y2 = 0.0;
    float x2m = 0.0, y2m = 0.0;
    float dist = 0.0, s = 0.0;

    h11 = H.at<float>(0, 0);
    h12 = H.at<float>(0, 1);
    h13 = H.at<float>(0, 2);
    h21 = H.at<float>(1, 0);
    h22 = H.at<float>(1, 1);
    h23 = H.at<float>(1, 2);
    h31 = H.at<float>(2, 0);
    h32 = H.at<float>(2, 1);
    h33 = H.at<float>(2, 2);

    inliers.clear();

    for (size_t i = 0; i < matches.size(); i += 2) {
        x1 = matches[i].x;
        y1 = matches[i].y;
        x2 = matches[i + 1].x;
        y2 = matches[i + 1].y;

        s    = h31 * x1 + h32 * y1 + h33;
        x2m  = (h11 * x1 + h12 * y1 + h13) / s;
        y2m  = (h21 * x1 + h22 * y1 + h23) / s;
        dist = sqrtf(powf(x2m - x2, 2) + powf(y2m - y2, 2));

        if (dist <= min_error) {
            inliers.push_back(matches[i]);
            inliers.push_back(matches[i + 1]);
        }
    }
}

/* ************************************************************************* */
void draw_matches_vertical(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& img_com,
                           const std::vector<cv::Point2f>& ptpairs, const int color)
{
    int   x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float ufactor = 0.0, vfactor = 0.0;

    const auto rows1 = img1.rows;
    const auto cols1 = img1.cols;
    const auto rows2 = img2.rows;
    const auto cols2 = img2.cols;

    // Combine the two images vertically
    img_com = cv::Mat(rows1 + rows2, std::max(cols1, cols2), CV_8UC3, cv::Scalar::all(0));

    // Place the first image at the top
    cv::Mat top_part = img_com(cv::Rect(0, 0, cols1, rows1));
    img1.copyTo(top_part);

    // Resize the second image to match the width of the first image
    cv::Mat img2_resized;
    cv::resize(img2, img2_resized, cv::Size(cols1, static_cast<int>(rows2 * (static_cast<float>(cols1) / cols2))), 0, 0,
               cv::INTER_LINEAR);

    // Place the resized second image below the first image
    cv::Mat bottom_part = img_com(cv::Rect(0, rows1, img2_resized.cols, img2_resized.rows));
    img2_resized.copyTo(bottom_part);

    // Adjust a vertical scaling factor after resizing
    ufactor = static_cast<float>(cols1) / cols2;
    vfactor = static_cast<float>(img2_resized.rows) / rows2;

    // Draw lines between corresponding points
    for (size_t i = 0; i < ptpairs.size(); i += 2) {
        x1 = static_cast<int>(ptpairs[i].x + 0.5);
        y1 = static_cast<int>(ptpairs[i].y + 0.5);
        x2 = static_cast<int>(ptpairs[i + 1].x * ufactor + 0.5);
        y2 = static_cast<int>(ptpairs[i + 1].y * vfactor + rows1 + 0.5);
        if (color == 0)
            cv::line(img_com, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 0), 1);
        else if (color == 1)
            cv::line(img_com, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 1);
        else if (color == 2)
            cv::line(img_com, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 1);
    }
}

/* ************************************************************************* */
void draw_matches_horizontal(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& img_com,
                             const std::vector<cv::Point2f>& ptpairs, const int color)
{
    int   x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float rows1 = 0.0, cols1 = 0.0;
    float rows2 = 0.0, cols2 = 0.0;
    float ufactor = 0.0, vfactor = 0.0;

    rows1 = static_cast<float>(img1.rows);
    cols1 = static_cast<float>(img1.cols);
    rows2 = static_cast<float>(img2.rows);
    cols2 = static_cast<float>(img2.cols);

    img_com = cv::Mat(rows1, cols1 * 2, CV_8UC3, cv::Scalar::all(0));

    ufactor = cols1 / cols2;
    vfactor = rows1 / rows2;

    // This is in case the input images don't have the same resolution
    auto img_aux = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC3);
    cv::resize(img2, img_aux, cv::Size(img1.cols, img1.rows), 0, 0, cv::INTER_LINEAR);

    for (int i = 0; i < img_com.rows; i++) {
        for (int j = 0; j < img_com.cols; j++) {
            if (j < img1.cols) {
                *(img_com.ptr<unsigned char>(i) + 3 * j)     = *(img1.ptr<unsigned char>(i) + 3 * j);
                *(img_com.ptr<unsigned char>(i) + 3 * j + 1) = *(img1.ptr<unsigned char>(i) + 3 * j + 1);
                *(img_com.ptr<unsigned char>(i) + 3 * j + 2) = *(img1.ptr<unsigned char>(i) + 3 * j + 2);
            } else {
                *(img_com.ptr<unsigned char>(i) + 3 * j) = *(img2.ptr<unsigned char>(i) + 3 * (j - img_aux.cols));
                *(img_com.ptr<unsigned char>(i) + 3 * j + 1) =
                    *(img2.ptr<unsigned char>(i) + 3 * (j - img_aux.cols) + 1);
                *(img_com.ptr<unsigned char>(i) + 3 * j + 2) =
                    *(img2.ptr<unsigned char>(i) + 3 * (j - img_aux.cols) + 2);
            }
        }
    }

    for (size_t i = 0; i < ptpairs.size(); i += 2) {
        x1 = (int)(ptpairs[i].x + .5);                           // NOLINT
        y1 = (int)(ptpairs[i].y + .5);                           // NOLINT
        x2 = (int)(ptpairs[i + 1].x * ufactor + img1.cols + .5); // NOLINT
        y2 = (int)(ptpairs[i + 1].y * vfactor + .5);             // NOLINT

        if (color == 0)
            cv::line(img_com, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 0), 2);
        else if (color == 1)
            cv::line(img_com, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
        else if (color == 2)
            cv::line(img_com, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
    }
}

/* ************************************************************************* */
bool read_homography(const string& hFile, cv::Mat& H1toN)
{
    float h11 = 0.0, h12 = 0.0, h13 = 0.0;
    float h21 = 0.0, h22 = 0.0, h23 = 0.0;
    float h31 = 0.0, h32 = 0.0, h33 = 0.0;

    constexpr int tmp_buf_size = 256;
    char          tmp_buf[tmp_buf_size];

    // Allocate memory for the OpenCV matrices
    H1toN = cv::Mat::zeros(3, 3, CV_32FC1);

    ifstream pf;
    pf.open(hFile.c_str(), std::ifstream::in);

    if (!pf.is_open())
        return false;

    pf.getline(tmp_buf, tmp_buf_size);
    sscanf(tmp_buf, "%f %f %f", &h11, &h12, &h13); // NOLINT

    pf.getline(tmp_buf, tmp_buf_size);
    sscanf(tmp_buf, "%f %f %f", &h21, &h22, &h23); // NOLINT

    pf.getline(tmp_buf, tmp_buf_size);
    sscanf(tmp_buf, "%f %f %f", &h31, &h32, &h33); // NOLINT

    pf.close();

    H1toN.at<float>(0, 0) = h11 / h33;
    H1toN.at<float>(0, 1) = h12 / h33;
    H1toN.at<float>(0, 2) = h13 / h33;

    H1toN.at<float>(1, 0) = h21 / h33;
    H1toN.at<float>(1, 1) = h22 / h33;
    H1toN.at<float>(1, 2) = h23 / h33;

    H1toN.at<float>(2, 0) = h31 / h33;
    H1toN.at<float>(2, 1) = h32 / h33;
    H1toN.at<float>(2, 2) = 1;

    return true;
}
