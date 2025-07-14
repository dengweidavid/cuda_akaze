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

#ifndef CUDAIMAGE_H
#define CUDAIMAGE_H

namespace mwvcv
{
    class CudaImage
    {
      public:
        CudaImage();
        ~CudaImage();

        void allocate(int w, int h, int p, bool host, float* hostmem = nullptr, float* devmem = nullptr);

        double download() const; // NOLINT(*-use-nodiscard)
        double readback() const; // NOLINT(*-use-nodiscard)
        double copyTo(CudaImage &outimg) const; // NOLINT(*-use-nodiscard)

        int width_, height_;
        int pitch_;

        float* h_data_;
        float* d_data_;

        bool h_internalAlloc_;
        bool d_internalAlloc_;
    };

    inline int iDivUp(const int a, const int b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }
    inline int iDivDown(const int a, const int b)
    {
        return a / b;
    }
    inline int iAlignUp(const int a, const int b)
    {
        return (a % b != 0) ? (a - a % b + b) : a;
    }
    inline int iAlignDown(const int a, const int b)
    {
        return a - a % b;
    }

} // namespace mwvcv

#endif // CUDAIMAGE_H
