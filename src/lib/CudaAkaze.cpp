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

#include "CudaAkaze.h"

#include <iostream>

#include "cudaUtils.h"
#include "fed.h"
#include "akaze.cuh"

mwvcv::CudaAkaze::CudaAkaze(const AkazeCuOptions& options) : options_(options)
{
    ncycles_    = 0;
    reordering_ = true;

    allocateMemorySrcImg();
    allocateMemoryEvolution();
}

mwvcv::CudaAkaze::~CudaAkaze()
{
    evolution_.clear();

    freeBuffer(cuda_source_image_);
    freeBuffers(cuda_memory_);
}

void mwvcv::CudaAkaze::allocateMemorySrcImg()
{
    cuda_source_image_ = allocBuffer(options_.img_width, options_.img_height, cuda_source_image_pitch_);
}

void mwvcv::CudaAkaze::allocateMemoryEvolution()
{
    // Allocate the dimension of the matrices for the evolution
    for (int i = 0; i <= options_.omax - 1; i++) {
        const auto rfactor      = static_cast<float>(1.0 / pow(2.0f, i));
        const int  level_height = static_cast<int>(static_cast<float>(options_.img_height) * rfactor);
        const int  level_width  = static_cast<int>(static_cast<float>(options_.img_width) * rfactor);

        // Smallest possible octave and allow one scale if the image is small
        if ((level_width < 80 || level_height < 40) && i != 0) {
            options_.omax = i;
            break;
        }

        for (int j = 0; j < options_.nsublevels; j++) {
            TEvolution     step;
            const cv::Size size(level_width, level_height);
            step.Lx.create(size, CV_32F);
            step.Ly.create(size, CV_32F);
            step.Lxx.create(size, CV_32F);
            step.Lxy.create(size, CV_32F);
            step.Lyy.create(size, CV_32F);
            step.Lflow.create(size, CV_32F);
            step.Lt.create(size, CV_32F);
            step.Lsmooth.create(size, CV_32F);
            step.Lstep.create(size, CV_32F);
            step.Ldet.create(size, CV_32F);
            step.esigma     = options_.soffset * pow(2.0f, (float)(j) / (float)(options_.nsublevels) + i); // NOLINT
            step.sigma_size = fRound(step.esigma);
            step.etime      = 0.5f * (step.esigma * step.esigma);
            step.octave     = i;
            step.sublevel   = j;
            evolution_.push_back(step);
        }
    }

    // Allocate memory for the number of cycles and time steps
    for (size_t i = 1; i < evolution_.size(); i++) {
        std::vector<float> tau;
        const float        ttime = evolution_[i].etime - evolution_[i - 1].etime;
        constexpr float    tmax  = 0.25f;
        int                naux  = fed_tau_by_process_time(ttime, 1, tmax, reordering_, tau);
        nsteps_.push_back(naux);
        tsteps_.push_back(tau);
        ncycles_++;
    }

    // Allocate memory for CUDA buffers
    options_.ncudaimages  = 4 * options_.nsublevels;
    options_.maxkeypoints = 4 * ((options_.maxkeypoints + 3) / 4);
    unsigned char* _cuda_desc;
    cuda_memory_ = allocBuffers(evolution_[0].Lt.cols, evolution_[0].Lt.rows, options_.ncudaimages, options_.omax,
                               options_.maxkeypoints, cuda_buffers_, cuda_bufferpoints_, cuda_points_, cuda_ptindices_,
                               _cuda_desc, cuda_descbuffer_, cuda_images_);
    cuda_desc_   = cv::Mat(options_.maxkeypoints, 61, CV_8U, _cuda_desc);
}

int mwvcv::CudaAkaze::createNonlinearScaleSpace(const cv::Mat& img)
{
    if (evolution_.empty()) {
        std::cerr << "Error generating the nonlinear scale space!!" << std::endl;
        return -1;
    }

    if (img.type() != CV_8UC1) {
        std::cerr << "Input frame image type is not CV_8UC1!" << std::endl;
        return -1;
    }

    const auto t1 = static_cast<double>(cv::getTickCount());

    const TEvolution& ev = evolution_[0];

    CudaImage& Limg    = cuda_buffers_[0];
    CudaImage& Lt      = cuda_buffers_[0];
    CudaImage& Lsmooth = cuda_buffers_[1];
    CudaImage& Ltemp   = cuda_buffers_[2];

    prepareSourceImage(img, cuda_source_image_, options_.img_width, cuda_source_image_pitch_, options_.img_height,
                       Limg.d_data_, Limg.pitch_);
    timing_.prepare = 1000.0 * (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency();

    if (evolution_.size() > 1) {
        // kcontrast is only needed when we have more than 1 evolution level
        contrastPercentile(Limg, Ltemp, Lsmooth, options_.kcontrast_percentile, options_.kcontrast_nbins,
                           options_.kcontrast);
        timing_.kcontrast = 1000.0 * (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency();
    }

    const int kernelSize = static_cast<int>(2 * ceil((options_.soffset - 0.8) / 0.3) + 3);
    lowPass(Limg, Lt, Ltemp, options_.soffset * options_.soffset, kernelSize);
    Lt.copyTo(Lsmooth);

    Lt.h_data_ = reinterpret_cast<float*>(ev.Lt.data);

    // Now generate the rest of evolution levels
    for (size_t i = 1; i < evolution_.size(); i++) {
        const TEvolution& evn = evolution_[i];

        const int  num     = options_.ncudaimages;
        CudaImage& Lt      = cuda_buffers_[evn.octave * num + 0 + 4 * evn.sublevel]; // NOLINT
        CudaImage& Lsmooth = cuda_buffers_[evn.octave * num + 1 + 4 * evn.sublevel]; // NOLINT
        CudaImage& Lstep   = cuda_buffers_[evn.octave * num + 2];
        CudaImage& Lflow   = cuda_buffers_[evn.octave * num + 3];

        const TEvolution& evo = evolution_[i - 1];

        CudaImage& Ltold = cuda_buffers_[evo.octave * num + 0 + 4 * evo.sublevel];
        if (evn.octave > evo.octave) {
            halfSample(Ltold, Lt);
            options_.kcontrast = options_.kcontrast * 0.75f;
        } else {
            Ltold.copyTo(Lt);
        }

        lowPass(Lt, Lsmooth, Lstep, 1.0, 5);
        flow(Lsmooth, Lflow, options_.diffusivity, options_.kcontrast);

        for (int j = 0; j < nsteps_[i - 1]; j++) {
            nLDStep(Lt, Lflow, Lstep, tsteps_[i - 1][j]);
        }

        Lt.h_data_ = reinterpret_cast<float*>(evn.Lt.data);
    }

    timing_.scale = 1000.0 * (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency();
    return 0;
}

void mwvcv::CudaAkaze::featureDetection(std::vector<cv::KeyPoint>& kpts)
{
    const auto t1 = static_cast<double>(cv::getTickCount());

    const int num = options_.ncudaimages;
    for (size_t i = 0; i < evolution_.size(); i++) { // NOLINT(*-loop-convert)
        const TEvolution& ev = evolution_[i];

        CudaImage& Lsmooth = cuda_buffers_[ev.octave * num + 1 + 4 * ev.sublevel];
        CudaImage& Lx      = cuda_buffers_[ev.octave * num + 2 + 4 * ev.sublevel];
        CudaImage& Ly      = cuda_buffers_[ev.octave * num + 3 + 4 * ev.sublevel];

        float ratio = pow(2.0f, (float)evolution_[i].octave); // NOLINT

        const int sigma_size_ = fRound(evolution_[i].esigma * options_.derivative_factor / ratio);
        hessianDeterminant(Lsmooth, Lx, Ly, sigma_size_);

        Lx.h_data_ = reinterpret_cast<float*>(evolution_[i].Lx.data);
        Ly.h_data_ = reinterpret_cast<float*>(evolution_[i].Ly.data);
    }

    const auto t2 = static_cast<double>(cv::getTickCount());

    clearPoints();
    for (size_t i = 0; i < evolution_.size(); i++) {
        const TEvolution& ev  = evolution_[i];
        const TEvolution& evp = evolution_[(i > 0 && evolution_[i].octave == evolution_[i - 1].octave ? i - 1 : i)];
        const TEvolution& evn =
            evolution_[(i < evolution_.size() - 1 && evolution_[i].octave == evolution_[i + 1].octave ? i + 1 : i)];

        CudaImage& Ldet = cuda_buffers_[ev.octave * num + 1 + 4 * ev.sublevel];
        CudaImage& LdetP = cuda_buffers_[evp.octave * num + 1 + 4 * evp.sublevel];
        CudaImage& LdetN = cuda_buffers_[evn.octave * num + 1 + 4 * evn.sublevel];

        float smax = 1.0f;
        if (options_.descriptor == DESCRIPTOR_TYPE::SURF_UPRIGHT || options_.descriptor == DESCRIPTOR_TYPE::SURF ||
            options_.descriptor == DESCRIPTOR_TYPE::MLDB_UPRIGHT || options_.descriptor == DESCRIPTOR_TYPE::MLDB)
            smax = 10.0f * sqrtf(2.0f);
        else if (options_.descriptor == DESCRIPTOR_TYPE::MSURF_UPRIGHT || options_.descriptor == DESCRIPTOR_TYPE::MSURF)
            smax = 12.0f * sqrtf(2.0f);

        const float ratio  = pow(2.0f, (float)evolution_[i].octave);  // NOLINT
        const float size   = evolution_[i].esigma * options_.derivative_factor;
        const float border = smax * static_cast<float>(fRound(size / ratio));
        const float thresh = std::max(options_.dthreshold, options_.min_dthreshold);

        // Here we limit max working with 16 evolutions (for example, 4 octaves and 4 sublevels)
        // Inside findExtrema function, we define "d_ExtremaIdx[16]", so max 16 results are stored for findExtrema
        // cuda_points's octave value is stored with intermediate value used inside the function of filterExtrema.
        // If we aren't going to call filterExtrema in the next step, we will store proper value in.
        const bool reset_octave = options_.skip_filter_extrema;
        findExtrema(Ldet, LdetP, LdetN, border, thresh, i, evolution_[i].octave, // NOLINT
                    size, cuda_points_, options_.maxkeypoints, reset_octave, nump_);
    }

    if (!options_.skip_filter_extrema) {
        filterExtrema(cuda_points_, cuda_bufferpoints_, cuda_ptindices_, nump_);
    }

    const auto t3 = static_cast<double>(cv::getTickCount());

    timing_.derivatives = 1000.0 * (t2 - t1) / cv::getTickFrequency();
    timing_.extrema     = 1000.0 * (t3 - t2) / cv::getTickFrequency();
    timing_.detector    = 1000.0 * (t3 - t1) / cv::getTickFrequency();
}

void mwvcv::CudaAkaze::computeDescriptors(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc)
{
    const auto t1 = static_cast<double>(cv::getTickCount());

    const int kpt_size = static_cast<int>(kpts.size());

    // Allocate memory for the matrix with the descriptors
    if (options_.descriptor < DESCRIPTOR_TYPE::MLDB_UPRIGHT) {
        desc = cv::Mat::zeros(kpt_size, 64, CV_32FC1);
    } else {
        if (options_.descriptor_size == 0) {
            // grid size 2x2, has 6 comparing pairs;
            // grid size 3x3, has 36 comparing pairs;
            // grid size 4x4, has 120 comparing pairs;
            // We use the full-length binary descriptor -> 486 bits
            const int t = (6 + 36 + 120) * options_.descriptor_channels;

            desc = cv::Mat::zeros(kpt_size, ceil(t / 8.), CV_8UC1);
        } else {
            // We use the random bit selection length binary descriptor
            desc = cv::Mat::zeros(kpt_size, ceil(options_.descriptor_size / 8.), CV_8UC1);
        }
    }

    const int pattern_size = options_.descriptor_pattern_size;

    switch (options_.descriptor) {
    case DESCRIPTOR_TYPE::MLDB:
        findOrientation(cuda_points_, cuda_buffers_, cuda_images_, nump_);
        getPoints(kpts, cuda_points_, nump_);
        extractDescriptors(cuda_points_, cuda_images_, cuda_desc_.data, cuda_descbuffer_, pattern_size, nump_);
        getDescriptors(desc, cuda_desc_, nump_);
        break;
    case DESCRIPTOR_TYPE::SURF_UPRIGHT:
    case DESCRIPTOR_TYPE::SURF:
    case DESCRIPTOR_TYPE::MSURF_UPRIGHT:
    case DESCRIPTOR_TYPE::MSURF:
    case DESCRIPTOR_TYPE::MLDB_UPRIGHT:
        std::cerr << "Descriptor not implemented\n";
    }

    const auto t2 = static_cast<double>(cv::getTickCount());

    timing_.descriptor = 1000.0 * (t2 - t1) / cv::getTickFrequency();

    waitCuda();
}

void mwvcv::CudaAkaze::showComputationTimes() const
{
    std::cout << "(*) Time Scale Space: " << timing_.scale << std::endl;
    std::cout << "   - Time Prepare: " << timing_.prepare << std::endl;
    std::cout << "   - Time KContrast: " << timing_.kcontrast << std::endl;
    std::cout << "(*) Time Detector: " << timing_.detector << std::endl;
    std::cout << "   - Time Derivatives: " << timing_.derivatives << std::endl;
    std::cout << "   - Time Extrema: " << timing_.extrema << std::endl;
    std::cout << "   - Time Subpixel: " << timing_.subpixel << std::endl;
    std::cout << "(*) Time Descriptor: " << timing_.descriptor << std::endl;
    std::cout << std::endl;
}
