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

#ifndef AKAZE_CUH
#define AKAZE_CUH

#include <opencv2/core/core.hpp>

#include "CudaImage.h"
#include "AkazeCuOptions.h"

#define CONVROW_W 160
#define CONVCOL_W 32
#define CONVCOL_H 40
#define CONVCOL_S 8

#define CONTRAST_W 64
#define CONTRAST_H 7
#define HISTCONT_W 64
#define HISTCONT_H 8
#define HISTCONT_R 4

#define SCHARR_W 32
#define SCHARR_H 16

#define NLDSTEP_W 32
#define NLDSTEP_H 13

#define BitonicSortThreads 1024
#define FindNeighborsThreads 32
#define FilterExtremaThreads 1024

#define ORIENT_S (13 * 16)
#define EXTRACT_S 64

// #define VERBOSE

namespace mwvcv
{
    struct Conv_t
    {
        float* d_Result;
        float* d_Data;
        int    width;
        int    pitch;
        int    height;
    };

    struct NLDStep_t
    {
        float* imgd;
        float* flod;
        float* temd;
        int    width;
        int    pitch;
        int    height;
        float  stepsize;
    };

    struct NLDUpdate_t
    {
        float* imgd;
        float* temd;
        int    width;
        int    pitch;
        int    height;
    };

    template <typename T>
    struct SortStruct_t
    {
        T     idx;
        short x;
        short y;
    };

    unsigned char* allocBuffer(int width, int height, int& pitch);
    void           freeBuffer(unsigned char* buffer);

    float* allocBuffers(int width, int height, int num, int omax, int maxpts, std::vector<CudaImage>& buffers,
                        cv::KeyPoint*& pts, cv::KeyPoint*& ptsbuffer, int*& ptindices, unsigned char*& desc,
                        float*& descbuffer, CudaImage*& ims);
    void   freeBuffers(float* buffers);

    void initCompareIndices();

    void clearPoints();
    int  getPoints(std::vector<cv::KeyPoint>& h_pts, cv::KeyPoint* d_pts, int numPts);
    void getDescriptors(cv::Mat& h_desc, cv::Mat& d_desc, int numPts);
    void waitCuda();

    void prepareSourceImage(const cv::Mat& img, unsigned char* d_img, int width, int in_pitch, int height,
                            float* d_data, int out_pitch);

    double lowPass(CudaImage& inimg, CudaImage& outimg, CudaImage& temp, double var, int kernsize);
    double contrastPercentile(CudaImage& img, CudaImage& temp, CudaImage& blur, float perc, int nbins, float& contrast);
    double halfSample(CudaImage& inimg, CudaImage& outimg);
    double flow(CudaImage& img, CudaImage& flow, DIFFUSIVITY_TYPE type, float kcontrast);
    double nLDStep(CudaImage& img, CudaImage& flow, CudaImage& temp, float stepsize);

    double hessianDeterminant(CudaImage& img, CudaImage& lx, CudaImage& ly, int step);
    double findExtrema(CudaImage& img, CudaImage& imgp, CudaImage& imgn, float border, float dthreshold, int scale,
                       int octave, float size, cv::KeyPoint* pts, int maxpts, bool reset_octave, int& nump);
    void   filterExtrema(cv::KeyPoint* pts, cv::KeyPoint* newpts, int* kptindices, int& nump);

    double findOrientation(cv::KeyPoint* d_pts, std::vector<CudaImage>& h_imgs, CudaImage* d_imgs, int numPts);
    double extractDescriptors(cv::KeyPoint* d_pts, CudaImage* cuda_images, unsigned char* desc_h, float* vals_d,
                              int patsize, int numPts);
} // namespace mwvcv

#endif // AKAZE_CUH