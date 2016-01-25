/******************************************************************************
*  OTFFT Sixstep Version 6.5
*
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_sixstep_h
#define otfft_sixstep_h

#include "otfft/otfft_misc.h"
#include "otfft_avxdif16.h"

namespace OTFFT_Sixstep { /////////////////////////////////////////////////////

using namespace OTFFT_MISC;

#ifdef DO_SINGLE_THREAD
static const int OMP_THRESHOLD1 = 1<<30;
static const int OMP_THRESHOLD2 = 1<<30;
#else
static const int OMP_THRESHOLD1 = 1<<13;
static const int OMP_THRESHOLD2 = 1<<17;
#endif

typedef const_complex_vector weight_t;
struct index_t { int row, col; };
typedef const index_t* __restrict const const_index_vector;

} /////////////////////////////////////////////////////////////////////////////

//#include "otfft_sixstep0r.h"
//#include "otfft_sixstepnr.h"
#include "otfft_sixstep0s.h"
#include "otfft_sixstepns.h"
#include "otfft_eightstep.h"

namespace OTFFT_Sixstep { /////////////////////////////////////////////////////

using namespace OTFFT_Eightstep;

struct FFT0
{
    int N, log_N;
    simd_array<complex_t> weight;
    complex_t* __restrict W;
    simd_array<complex_t> weight_sub;
    complex_t* __restrict Ws;
    simd_array<index_t> index;
    index_t* __restrict ip;

    FFT0() noexcept : N(0), log_N(0), W(0), Ws(0), ip(0) {}
    FFT0(const int n) { setup(n); }

    void setup(int n)
    {
        for (log_N = 0; n > 1; n >>= 1) log_N++;
        setup2(log_N);
    }

    inline void setup2(const int n)
    {
        log_N = n; N = 1 << n;
        weight.setup(N+1); W = &weight; init_W(N, W);
        if (n < 4) {}
        else if ((n & 1) == 1) {
            const int m = 1 << (n/2-1);
            weight_sub.setup(m+1); Ws = &weight_sub; init_W(m, Ws);
            index.setup(m/2*(m/2+1)/2); ip = &index;
            int i = 0;
            for (int k = 0; k < m; k += 2) {
                for (int p = k; p < m; p += 2) {
                    ip[i].row = k;
                    ip[i].col = p;
                    i++;
                }
            }
        }
        else {
            const int m = 1 << n/2;
            weight_sub.setup(m+1); Ws = &weight_sub; init_W(m, Ws);
            index.setup(m/2*(m/2+1)/2); ip = &index;
            int i = 0;
            for (int k = 0; k < m; k += 2) {
                for (int p = k; p < m; p += 2) {
                    ip[i].row = k;
                    ip[i].col = p;
                    i++;
                }
            }
        }
    }

    inline void fwd(complex_vector x, complex_vector y) const noexcept
    {
        switch (log_N) {
            case  0: break;
            case  1: OTFFT_AVXDIF16::fwdnfft<(1<<1),1,0>()(x, y, W); break;
            case  2: OTFFT_AVXDIF16::fwdnfft<(1<<2),1,0>()(x, y, W); break;
            case  3: OTFFT_AVXDIF16::fwdnfft<(1<<3),1,0>()(x, y, W); break;
            case  4: fwdnffts< 4>()(ip, x, y, W, Ws); break;
            case  5: fwdnfftq< 5>()(ip, x, y, W, Ws); break;
            case  6: fwdnffts< 6>()(ip, x, y, W, Ws); break;
            case  7: fwdnfftq< 7>()(ip, x, y, W, Ws); break;
            case  8: fwdnffts< 8>()(ip, x, y, W, Ws); break;
            case  9: fwdnfftq< 9>()(ip, x, y, W, Ws); break;
            case 10: fwdnffts<10>()(ip, x, y, W, Ws); break;
            case 11: fwdnfftq<11>()(ip, x, y, W, Ws); break;
            case 12: fwdnffts<12>()(ip, x, y, W, Ws); break;
            case 13: fwdnfftq<13>()(ip, x, y, W, Ws); break;
            case 14: fwdnffts<14>()(ip, x, y, W, Ws); break;
            case 15: fwdnfftq<15>()(ip, x, y, W, Ws); break;
            case 16: fwdnffts<16>()(ip, x, y, W, Ws); break;
            case 17: fwdnfftq<17>()(ip, x, y, W, Ws); break;
            case 18: fwdnffts<18>()(ip, x, y, W, Ws); break;
            case 19: fwdnfftq<19>()(ip, x, y, W, Ws); break;
            case 20: fwdnffts<20>()(ip, x, y, W, Ws); break;
            case 21: fwdnfftq<21>()(ip, x, y, W, Ws); break;
            case 22: fwdnffts<22>()(ip, x, y, W, Ws); break;
            case 23: fwdnfftq<23>()(ip, x, y, W, Ws); break;
            case 24: fwdnffts<24>()(ip, x, y, W, Ws); break;
        }
    }

    inline void fwd0(complex_vector x, complex_vector y) const noexcept
    {
        switch (log_N) {
            case  0: break;
            case  1: OTFFT_AVXDIF16::fwd0fft<(1<<1),1,0>()(x, y, W); break;
            case  2: OTFFT_AVXDIF16::fwd0fft<(1<<2),1,0>()(x, y, W); break;
            case  3: OTFFT_AVXDIF16::fwd0fft<(1<<3),1,0>()(x, y, W); break;
            case  4: fwd0ffts< 4>()(ip, x, y, W, Ws); break;
            case  5: fwd0fftq< 5>()(ip, x, y, W, Ws); break;
            case  6: fwd0ffts< 6>()(ip, x, y, W, Ws); break;
            case  7: fwd0fftq< 7>()(ip, x, y, W, Ws); break;
            case  8: fwd0ffts< 8>()(ip, x, y, W, Ws); break;
            case  9: fwd0fftq< 9>()(ip, x, y, W, Ws); break;
            case 10: fwd0ffts<10>()(ip, x, y, W, Ws); break;
            case 11: fwd0fftq<11>()(ip, x, y, W, Ws); break;
            case 12: fwd0ffts<12>()(ip, x, y, W, Ws); break;
            case 13: fwd0fftq<13>()(ip, x, y, W, Ws); break;
            case 14: fwd0ffts<14>()(ip, x, y, W, Ws); break;
            case 15: fwd0fftq<15>()(ip, x, y, W, Ws); break;
            case 16: fwd0ffts<16>()(ip, x, y, W, Ws); break;
            case 17: fwd0fftq<17>()(ip, x, y, W, Ws); break;
            case 18: fwd0ffts<18>()(ip, x, y, W, Ws); break;
            case 19: fwd0fftq<19>()(ip, x, y, W, Ws); break;
            case 20: fwd0ffts<20>()(ip, x, y, W, Ws); break;
            case 21: fwd0fftq<21>()(ip, x, y, W, Ws); break;
            case 22: fwd0ffts<22>()(ip, x, y, W, Ws); break;
            case 23: fwd0fftq<23>()(ip, x, y, W, Ws); break;
            case 24: fwd0ffts<24>()(ip, x, y, W, Ws); break;
        }
    }

    inline void fwdn(complex_vector x, complex_vector y) const noexcept { fwd(x, y); }

    inline void inv(complex_vector x, complex_vector y) const noexcept
    {
        switch (log_N) {
            case  0: break;
            case  1: OTFFT_AVXDIF16::inv0fft<(1<<1),1,0>()(x, y, W); break;
            case  2: OTFFT_AVXDIF16::inv0fft<(1<<2),1,0>()(x, y, W); break;
            case  3: OTFFT_AVXDIF16::inv0fft<(1<<3),1,0>()(x, y, W); break;
            case  4: inv0ffts< 4>()(ip, x, y, W, Ws); break;
            case  5: inv0fftq< 5>()(ip, x, y, W, Ws); break;
            case  6: inv0ffts< 6>()(ip, x, y, W, Ws); break;
            case  7: inv0fftq< 7>()(ip, x, y, W, Ws); break;
            case  8: inv0ffts< 8>()(ip, x, y, W, Ws); break;
            case  9: inv0fftq< 9>()(ip, x, y, W, Ws); break;
            case 10: inv0ffts<10>()(ip, x, y, W, Ws); break;
            case 11: inv0fftq<11>()(ip, x, y, W, Ws); break;
            case 12: inv0ffts<12>()(ip, x, y, W, Ws); break;
            case 13: inv0fftq<13>()(ip, x, y, W, Ws); break;
            case 14: inv0ffts<14>()(ip, x, y, W, Ws); break;
            case 15: inv0fftq<15>()(ip, x, y, W, Ws); break;
            case 16: inv0ffts<16>()(ip, x, y, W, Ws); break;
            case 17: inv0fftq<17>()(ip, x, y, W, Ws); break;
            case 18: inv0ffts<18>()(ip, x, y, W, Ws); break;
            case 19: inv0fftq<19>()(ip, x, y, W, Ws); break;
            case 20: inv0ffts<20>()(ip, x, y, W, Ws); break;
            case 21: inv0fftq<21>()(ip, x, y, W, Ws); break;
            case 22: inv0ffts<22>()(ip, x, y, W, Ws); break;
            case 23: inv0fftq<23>()(ip, x, y, W, Ws); break;
            case 24: inv0ffts<24>()(ip, x, y, W, Ws); break;
        }
    }

    inline void inv0(complex_vector x, complex_vector y) const noexcept { inv(x, y); }

    inline void invn(complex_vector x, complex_vector y) const noexcept
    {
        switch (log_N) {
            case  0: break;
            case  1: OTFFT_AVXDIF16::invnfft<(1<<1),1,0>()(x, y, W); break;
            case  2: OTFFT_AVXDIF16::invnfft<(1<<2),1,0>()(x, y, W); break;
            case  3: OTFFT_AVXDIF16::invnfft<(1<<3),1,0>()(x, y, W); break;
            case  4: invnffts< 4>()(ip, x, y, W, Ws); break;
            case  5: invnfftq< 5>()(ip, x, y, W, Ws); break;
            case  6: invnffts< 6>()(ip, x, y, W, Ws); break;
            case  7: invnfftq< 7>()(ip, x, y, W, Ws); break;
            case  8: invnffts< 8>()(ip, x, y, W, Ws); break;
            case  9: invnfftq< 9>()(ip, x, y, W, Ws); break;
            case 10: invnffts<10>()(ip, x, y, W, Ws); break;
            case 11: invnfftq<11>()(ip, x, y, W, Ws); break;
            case 12: invnffts<12>()(ip, x, y, W, Ws); break;
            case 13: invnfftq<13>()(ip, x, y, W, Ws); break;
            case 14: invnffts<14>()(ip, x, y, W, Ws); break;
            case 15: invnfftq<15>()(ip, x, y, W, Ws); break;
            case 16: invnffts<16>()(ip, x, y, W, Ws); break;
            case 17: invnfftq<17>()(ip, x, y, W, Ws); break;
            case 18: invnffts<18>()(ip, x, y, W, Ws); break;
            case 19: invnfftq<19>()(ip, x, y, W, Ws); break;
            case 20: invnffts<20>()(ip, x, y, W, Ws); break;
            case 21: invnfftq<21>()(ip, x, y, W, Ws); break;
            case 22: invnffts<22>()(ip, x, y, W, Ws); break;
            case 23: invnfftq<23>()(ip, x, y, W, Ws); break;
            case 24: invnffts<24>()(ip, x, y, W, Ws); break;
        }
    }
};

#if 0
struct FFT1
{
    int N, log_N;
    simd_array<complex_t> weight;
    complex_t* __restrict W;
    OTFFT_AVXDIFx::FFT0 fft1, fft2;

    FFT1() : N(0), log_N(0), W(0) {}
    FFT1(const int n) { setup(n); }

    void setup(int n)
    {
        for (log_N = 0; n > 1; n >>= 1) log_N++;
        setup2(log_N);
    }

    inline void setup2(const int n)
    {
        log_N = n; N = 1 << n;
        weight.setup(N+1); W = &weight;
        if (N < 4) fft1.setup2(n);
        else { fft1.setup2(n/2); fft2.setup2(n - n/2); }
        init_W(N, W);
    }

    inline void fwd(complex_vector x, complex_vector y) const noexcept
    {
        switch (log_N) {
            case  0: break;
            case  1: fft1.fwd(x, y); break;
            case  2: fwdnfftr< 2>()(fft1, fft2, x, y, W); break;
            case  3: fwdnfftr< 3>()(fft1, fft2, x, y, W); break;
            case  4: fwdnfftr< 4>()(fft1, fft2, x, y, W); break;
            case  5: fwdnfftr< 5>()(fft1, fft2, x, y, W); break;
            case  6: fwdnfftr< 6>()(fft1, fft2, x, y, W); break;
            case  7: fwdnfftr< 7>()(fft1, fft2, x, y, W); break;
            case  8: fwdnfftr< 8>()(fft1, fft2, x, y, W); break;
            case  9: fwdnfftr< 9>()(fft1, fft2, x, y, W); break;
            case 10: fwdnfftr<10>()(fft1, fft2, x, y, W); break;
            case 11: fwdnfftr<11>()(fft1, fft2, x, y, W); break;
            case 12: fwdnfftr<12>()(fft1, fft2, x, y, W); break;
            case 13: fwdnfftr<13>()(fft1, fft2, x, y, W); break;
            case 14: fwdnfftr<14>()(fft1, fft2, x, y, W); break;
            case 15: fwdnfftr<15>()(fft1, fft2, x, y, W); break;
            case 16: fwdnfftr<16>()(fft1, fft2, x, y, W); break;
            case 17: fwdnfftr<17>()(fft1, fft2, x, y, W); break;
            case 18: fwdnfftr<18>()(fft1, fft2, x, y, W); break;
            case 19: fwdnfftr<19>()(fft1, fft2, x, y, W); break;
            case 20: fwdnfftr<20>()(fft1, fft2, x, y, W); break;
            case 21: fwdnfftr<21>()(fft1, fft2, x, y, W); break;
            case 22: fwdnfftr<22>()(fft1, fft2, x, y, W); break;
            case 23: fwdnfftr<23>()(fft1, fft2, x, y, W); break;
            case 24: fwdnfftr<24>()(fft1, fft2, x, y, W); break;
        }
    }

    inline void fwd0(complex_vector x, complex_vector y) const noexcept
    {
        switch (log_N) {
            case  0: break;
            case  1: fft1.fwd0(x, y); break;
            case  2: fwd0fftr< 2>()(fft1, fft2, x, y, W); break;
            case  3: fwd0fftr< 3>()(fft1, fft2, x, y, W); break;
            case  4: fwd0fftr< 4>()(fft1, fft2, x, y, W); break;
            case  5: fwd0fftr< 5>()(fft1, fft2, x, y, W); break;
            case  6: fwd0fftr< 6>()(fft1, fft2, x, y, W); break;
            case  7: fwd0fftr< 7>()(fft1, fft2, x, y, W); break;
            case  8: fwd0fftr< 8>()(fft1, fft2, x, y, W); break;
            case  9: fwd0fftr< 9>()(fft1, fft2, x, y, W); break;
            case 10: fwd0fftr<10>()(fft1, fft2, x, y, W); break;
            case 11: fwd0fftr<11>()(fft1, fft2, x, y, W); break;
            case 12: fwd0fftr<12>()(fft1, fft2, x, y, W); break;
            case 13: fwd0fftr<13>()(fft1, fft2, x, y, W); break;
            case 14: fwd0fftr<14>()(fft1, fft2, x, y, W); break;
            case 15: fwd0fftr<15>()(fft1, fft2, x, y, W); break;
            case 16: fwd0fftr<16>()(fft1, fft2, x, y, W); break;
            case 17: fwd0fftr<17>()(fft1, fft2, x, y, W); break;
            case 18: fwd0fftr<18>()(fft1, fft2, x, y, W); break;
            case 19: fwd0fftr<19>()(fft1, fft2, x, y, W); break;
            case 20: fwd0fftr<20>()(fft1, fft2, x, y, W); break;
            case 21: fwd0fftr<21>()(fft1, fft2, x, y, W); break;
            case 22: fwd0fftr<22>()(fft1, fft2, x, y, W); break;
            case 23: fwd0fftr<23>()(fft1, fft2, x, y, W); break;
            case 24: fwd0fftr<24>()(fft1, fft2, x, y, W); break;
        }
    }

    inline void fwdn(complex_vector x, complex_vector y) const noexcept { fwd(x, y); }

    inline void inv(complex_vector x, complex_vector y) const noexcept
    {
        switch (log_N) {
            case  0: break;
            case  1: fft1.inv0(x, y); break;
            case  2: inv0fftr< 2>()(fft1, fft2, x, y, W); break;
            case  3: inv0fftr< 3>()(fft1, fft2, x, y, W); break;
            case  4: inv0fftr< 4>()(fft1, fft2, x, y, W); break;
            case  5: inv0fftr< 5>()(fft1, fft2, x, y, W); break;
            case  6: inv0fftr< 6>()(fft1, fft2, x, y, W); break;
            case  7: inv0fftr< 7>()(fft1, fft2, x, y, W); break;
            case  8: inv0fftr< 8>()(fft1, fft2, x, y, W); break;
            case  9: inv0fftr< 9>()(fft1, fft2, x, y, W); break;
            case 10: inv0fftr<10>()(fft1, fft2, x, y, W); break;
            case 11: inv0fftr<11>()(fft1, fft2, x, y, W); break;
            case 12: inv0fftr<12>()(fft1, fft2, x, y, W); break;
            case 13: inv0fftr<13>()(fft1, fft2, x, y, W); break;
            case 14: inv0fftr<14>()(fft1, fft2, x, y, W); break;
            case 15: inv0fftr<15>()(fft1, fft2, x, y, W); break;
            case 16: inv0fftr<16>()(fft1, fft2, x, y, W); break;
            case 17: inv0fftr<17>()(fft1, fft2, x, y, W); break;
            case 18: inv0fftr<18>()(fft1, fft2, x, y, W); break;
            case 19: inv0fftr<19>()(fft1, fft2, x, y, W); break;
            case 20: inv0fftr<20>()(fft1, fft2, x, y, W); break;
            case 21: inv0fftr<21>()(fft1, fft2, x, y, W); break;
            case 22: inv0fftr<22>()(fft1, fft2, x, y, W); break;
            case 23: inv0fftr<23>()(fft1, fft2, x, y, W); break;
            case 24: inv0fftr<24>()(fft1, fft2, x, y, W); break;
        }
    }

    inline void inv0(complex_vector x, complex_vector y) const noexcept { inv(x, y); }

    inline void invn(complex_vector x, complex_vector y) const noexcept
    {
        switch (log_N) {
            case  0: break;
            case  1: fft1.invn(x, y); break;
            case  2: invnfftr< 2>()(fft1, fft2, x, y, W); break;
            case  3: invnfftr< 3>()(fft1, fft2, x, y, W); break;
            case  4: invnfftr< 4>()(fft1, fft2, x, y, W); break;
            case  5: invnfftr< 5>()(fft1, fft2, x, y, W); break;
            case  6: invnfftr< 6>()(fft1, fft2, x, y, W); break;
            case  7: invnfftr< 7>()(fft1, fft2, x, y, W); break;
            case  8: invnfftr< 8>()(fft1, fft2, x, y, W); break;
            case  9: invnfftr< 9>()(fft1, fft2, x, y, W); break;
            case 10: invnfftr<10>()(fft1, fft2, x, y, W); break;
            case 11: invnfftr<11>()(fft1, fft2, x, y, W); break;
            case 12: invnfftr<12>()(fft1, fft2, x, y, W); break;
            case 13: invnfftr<13>()(fft1, fft2, x, y, W); break;
            case 14: invnfftr<14>()(fft1, fft2, x, y, W); break;
            case 15: invnfftr<15>()(fft1, fft2, x, y, W); break;
            case 16: invnfftr<16>()(fft1, fft2, x, y, W); break;
            case 17: invnfftr<17>()(fft1, fft2, x, y, W); break;
            case 18: invnfftr<18>()(fft1, fft2, x, y, W); break;
            case 19: invnfftr<19>()(fft1, fft2, x, y, W); break;
            case 20: invnfftr<20>()(fft1, fft2, x, y, W); break;
            case 21: invnfftr<21>()(fft1, fft2, x, y, W); break;
            case 22: invnfftr<22>()(fft1, fft2, x, y, W); break;
            case 23: invnfftr<23>()(fft1, fft2, x, y, W); break;
            case 24: invnfftr<24>()(fft1, fft2, x, y, W); break;
        }
    }
};
#endif

#if 0
struct FFT
{
    FFT0 fft;
    simd_array<complex_t> work;
    complex_t* y;

    FFT() : fft(), work(), y(0) {}
    FFT(int n) : fft(n), work(n), y(&work) {}

    inline void setup(const int n) { fft.setup(n); work.setup(n); y = &work; }

    inline void fwd(complex_vector  x) const noexcept { fft.fwd(x, y);  }
    inline void fwd0(complex_vector x) const noexcept { fft.fwd0(x, y); }
    inline void fwdn(complex_vector x) const noexcept { fft.fwdn(x, y); }
    inline void inv(complex_vector  x) const noexcept { fft.inv(x, y);  }
    inline void inv0(complex_vector x) const noexcept { fft.inv0(x, y); }
    inline void invn(complex_vector x) const noexcept { fft.invn(x, y); }
};
#endif

} /////////////////////////////////////////////////////////////////////////////

#endif // otfft_sixstep_h
