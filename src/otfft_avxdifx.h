/******************************************************************************
*  OTFFT AVXDIFx Version 5.4
******************************************************************************/

#ifndef otfft_avxdifx_h
#define otfft_avxdifx_h

#include "otfft/otfft_misc.h"
#include "otfft_avxdif8.h"

namespace OTFFT_AVXDIFx { /////////////////////////////////////////////////////

using namespace OTFFT_MISC;

struct FFT0
{
    int N, log_N;
    simd_array<complex_t> weight;
    complex_t* __restrict W;

    FFT0() : N(0), log_N(0), W(0) {}
    FFT0(const int n) { setup(n); }

    void setup(int n)
    {
        for (log_N = 0; n > 1; n >>= 1) log_N++;
        setup2(log_N);
    }

    inline void setup2(const int n)
    {
        log_N = n; N = 1 << n;
        weight.setup(N+1); W = &weight;
        init_W(N, W);
    }

    ///////////////////////////////////////////////////////////////////////////

    inline void fwd(complex_vector x, complex_vector y) const
    {
        switch (log_N) {
            case  0: break;
            case  1: OTFFT_AVXDIF4::fwdnfft<(1<< 1),1,0>()(x, y, W); break;
            case  2: OTFFT_AVXDIF4::fwdnfft<(1<< 2),1,0>()(x, y, W); break;
            case  3: OTFFT_AVXDIF4::fwdnfft<(1<< 3),1,0>()(x, y, W); break;
            case  4: OTFFT_AVXDIF4::fwdnfft<(1<< 4),1,0>()(x, y, W); break;
            case  5: OTFFT_AVXDIF4::fwdnfft<(1<< 5),1,0>()(x, y, W); break;
            case  6: OTFFT_AVXDIF4::fwdnfft<(1<< 6),1,0>()(x, y, W); break;
            case  7: OTFFT_AVXDIF4::fwdnfft<(1<< 7),1,0>()(x, y, W); break;
            case  8: OTFFT_AVXDIF4::fwdnfft<(1<< 8),1,0>()(x, y, W); break;
            case  9: OTFFT_AVXDIF4::fwdnfft<(1<< 9),1,0>()(x, y, W); break;
            case 10: OTFFT_AVXDIF4::fwdnfft<(1<<10),1,0>()(x, y, W); break;
            case 11: OTFFT_AVXDIF8::fwdnfft<(1<<11),1,0>()(x, y, W); break;
            case 12: OTFFT_AVXDIF8::fwdnfft<(1<<12),1,0>()(x, y, W); break;
            case 13: OTFFT_AVXDIF8::fwdnfft<(1<<13),1,0>()(x, y, W); break;
            case 14: OTFFT_AVXDIF8::fwdnfft<(1<<14),1,0>()(x, y, W); break;
            case 15: OTFFT_AVXDIF8::fwdnfft<(1<<15),1,0>()(x, y, W); break;
            case 16: OTFFT_AVXDIF8::fwdnfft<(1<<16),1,0>()(x, y, W); break;
            case 17: OTFFT_AVXDIF8::fwdnfft<(1<<17),1,0>()(x, y, W); break;
            case 18: OTFFT_AVXDIF8::fwdnfft<(1<<18),1,0>()(x, y, W); break;
            case 19: OTFFT_AVXDIF8::fwdnfft<(1<<19),1,0>()(x, y, W); break;
            case 20: OTFFT_AVXDIF8::fwdnfft<(1<<20),1,0>()(x, y, W); break;
            case 21: OTFFT_AVXDIF8::fwdnfft<(1<<21),1,0>()(x, y, W); break;
            case 22: OTFFT_AVXDIF8::fwdnfft<(1<<22),1,0>()(x, y, W); break;
            case 23: OTFFT_AVXDIF8::fwdnfft<(1<<23),1,0>()(x, y, W); break;
            case 24: OTFFT_AVXDIF8::fwdnfft<(1<<24),1,0>()(x, y, W); break;
        }
    }

    inline void fwd0(complex_vector x, complex_vector y) const
    {
        switch (log_N) {
            case  0: break;
            case  1: OTFFT_AVXDIF4::fwd0fft<(1<< 1),1,0>()(x, y, W); break;
            case  2: OTFFT_AVXDIF4::fwd0fft<(1<< 2),1,0>()(x, y, W); break;
            case  3: OTFFT_AVXDIF4::fwd0fft<(1<< 3),1,0>()(x, y, W); break;
            case  4: OTFFT_AVXDIF4::fwd0fft<(1<< 4),1,0>()(x, y, W); break;
            case  5: OTFFT_AVXDIF4::fwd0fft<(1<< 5),1,0>()(x, y, W); break;
            case  6: OTFFT_AVXDIF4::fwd0fft<(1<< 6),1,0>()(x, y, W); break;
            case  7: OTFFT_AVXDIF4::fwd0fft<(1<< 7),1,0>()(x, y, W); break;
            case  8: OTFFT_AVXDIF4::fwd0fft<(1<< 8),1,0>()(x, y, W); break;
            case  9: OTFFT_AVXDIF4::fwd0fft<(1<< 9),1,0>()(x, y, W); break;
            case 10: OTFFT_AVXDIF4::fwd0fft<(1<<10),1,0>()(x, y, W); break;
            case 11: OTFFT_AVXDIF8::fwd0fft<(1<<11),1,0>()(x, y, W); break;
            case 12: OTFFT_AVXDIF8::fwd0fft<(1<<12),1,0>()(x, y, W); break;
            case 13: OTFFT_AVXDIF8::fwd0fft<(1<<13),1,0>()(x, y, W); break;
            case 14: OTFFT_AVXDIF8::fwd0fft<(1<<14),1,0>()(x, y, W); break;
            case 15: OTFFT_AVXDIF8::fwd0fft<(1<<15),1,0>()(x, y, W); break;
            case 16: OTFFT_AVXDIF8::fwd0fft<(1<<16),1,0>()(x, y, W); break;
            case 17: OTFFT_AVXDIF8::fwd0fft<(1<<17),1,0>()(x, y, W); break;
            case 18: OTFFT_AVXDIF8::fwd0fft<(1<<18),1,0>()(x, y, W); break;
            case 19: OTFFT_AVXDIF8::fwd0fft<(1<<19),1,0>()(x, y, W); break;
            case 20: OTFFT_AVXDIF8::fwd0fft<(1<<20),1,0>()(x, y, W); break;
            case 21: OTFFT_AVXDIF8::fwd0fft<(1<<21),1,0>()(x, y, W); break;
            case 22: OTFFT_AVXDIF8::fwd0fft<(1<<22),1,0>()(x, y, W); break;
            case 23: OTFFT_AVXDIF8::fwd0fft<(1<<23),1,0>()(x, y, W); break;
            case 24: OTFFT_AVXDIF8::fwd0fft<(1<<24),1,0>()(x, y, W); break;
        }
    }

    inline void fwdn(complex_vector x, complex_vector y) const { fwd(x, y); }

    inline void fwd0o(complex_vector x, complex_vector y) const
    {
        switch (log_N) {
            case  0: y[0] = x[0]; break;
            case  1: OTFFT_AVXDIF4::fwd0fft<(1<< 1),1,1>()(x, y, W); break;
            case  2: OTFFT_AVXDIF4::fwd0fft<(1<< 2),1,1>()(x, y, W); break;
            case  3: OTFFT_AVXDIF4::fwd0fft<(1<< 3),1,1>()(x, y, W); break;
            case  4: OTFFT_AVXDIF4::fwd0fft<(1<< 4),1,1>()(x, y, W); break;
            case  5: OTFFT_AVXDIF4::fwd0fft<(1<< 5),1,1>()(x, y, W); break;
            case  6: OTFFT_AVXDIF4::fwd0fft<(1<< 6),1,1>()(x, y, W); break;
            case  7: OTFFT_AVXDIF4::fwd0fft<(1<< 7),1,1>()(x, y, W); break;
            case  8: OTFFT_AVXDIF4::fwd0fft<(1<< 8),1,1>()(x, y, W); break;
            case  9: OTFFT_AVXDIF4::fwd0fft<(1<< 9),1,1>()(x, y, W); break;
            case 10: OTFFT_AVXDIF4::fwd0fft<(1<<10),1,1>()(x, y, W); break;
            case 11: OTFFT_AVXDIF8::fwd0fft<(1<<11),1,1>()(x, y, W); break;
            case 12: OTFFT_AVXDIF8::fwd0fft<(1<<12),1,1>()(x, y, W); break;
            case 13: OTFFT_AVXDIF8::fwd0fft<(1<<13),1,1>()(x, y, W); break;
            case 14: OTFFT_AVXDIF8::fwd0fft<(1<<14),1,1>()(x, y, W); break;
            case 15: OTFFT_AVXDIF8::fwd0fft<(1<<15),1,1>()(x, y, W); break;
            case 16: OTFFT_AVXDIF8::fwd0fft<(1<<16),1,1>()(x, y, W); break;
            case 17: OTFFT_AVXDIF8::fwd0fft<(1<<17),1,1>()(x, y, W); break;
            case 18: OTFFT_AVXDIF8::fwd0fft<(1<<18),1,1>()(x, y, W); break;
            case 19: OTFFT_AVXDIF8::fwd0fft<(1<<19),1,1>()(x, y, W); break;
            case 20: OTFFT_AVXDIF8::fwd0fft<(1<<20),1,1>()(x, y, W); break;
            case 21: OTFFT_AVXDIF8::fwd0fft<(1<<21),1,1>()(x, y, W); break;
            case 22: OTFFT_AVXDIF8::fwd0fft<(1<<22),1,1>()(x, y, W); break;
            case 23: OTFFT_AVXDIF8::fwd0fft<(1<<23),1,1>()(x, y, W); break;
            case 24: OTFFT_AVXDIF8::fwd0fft<(1<<24),1,1>()(x, y, W); break;
        }
    }

    inline void fwdno(complex_vector x, complex_vector y) const
    {
        switch (log_N) {
            case  0: y[0] = x[0]; break;
            case  1: OTFFT_AVXDIF4::fwdnfft<(1<< 1),1,1>()(x, y, W); break;
            case  2: OTFFT_AVXDIF4::fwdnfft<(1<< 2),1,1>()(x, y, W); break;
            case  3: OTFFT_AVXDIF4::fwdnfft<(1<< 3),1,1>()(x, y, W); break;
            case  4: OTFFT_AVXDIF4::fwdnfft<(1<< 4),1,1>()(x, y, W); break;
            case  5: OTFFT_AVXDIF4::fwdnfft<(1<< 5),1,1>()(x, y, W); break;
            case  6: OTFFT_AVXDIF4::fwdnfft<(1<< 6),1,1>()(x, y, W); break;
            case  7: OTFFT_AVXDIF4::fwdnfft<(1<< 7),1,1>()(x, y, W); break;
            case  8: OTFFT_AVXDIF4::fwdnfft<(1<< 8),1,1>()(x, y, W); break;
            case  9: OTFFT_AVXDIF4::fwdnfft<(1<< 9),1,1>()(x, y, W); break;
            case 10: OTFFT_AVXDIF4::fwdnfft<(1<<10),1,1>()(x, y, W); break;
            case 11: OTFFT_AVXDIF8::fwdnfft<(1<<11),1,1>()(x, y, W); break;
            case 12: OTFFT_AVXDIF8::fwdnfft<(1<<12),1,1>()(x, y, W); break;
            case 13: OTFFT_AVXDIF8::fwdnfft<(1<<13),1,1>()(x, y, W); break;
            case 14: OTFFT_AVXDIF8::fwdnfft<(1<<14),1,1>()(x, y, W); break;
            case 15: OTFFT_AVXDIF8::fwdnfft<(1<<15),1,1>()(x, y, W); break;
            case 16: OTFFT_AVXDIF8::fwdnfft<(1<<16),1,1>()(x, y, W); break;
            case 17: OTFFT_AVXDIF8::fwdnfft<(1<<17),1,1>()(x, y, W); break;
            case 18: OTFFT_AVXDIF8::fwdnfft<(1<<18),1,1>()(x, y, W); break;
            case 19: OTFFT_AVXDIF8::fwdnfft<(1<<19),1,1>()(x, y, W); break;
            case 20: OTFFT_AVXDIF8::fwdnfft<(1<<20),1,1>()(x, y, W); break;
            case 21: OTFFT_AVXDIF8::fwdnfft<(1<<21),1,1>()(x, y, W); break;
            case 22: OTFFT_AVXDIF8::fwdnfft<(1<<22),1,1>()(x, y, W); break;
            case 23: OTFFT_AVXDIF8::fwdnfft<(1<<23),1,1>()(x, y, W); break;
            case 24: OTFFT_AVXDIF8::fwdnfft<(1<<24),1,1>()(x, y, W); break;
        }
    }

    ///////////////////////////////////////////////////////////////////////////

    inline void inv(complex_vector x, complex_vector y) const
    {
        switch (log_N) {
            case  0: break;
            case  1: OTFFT_AVXDIF4::inv0fft<(1<< 1),1,0>()(x, y, W); break;
            case  2: OTFFT_AVXDIF4::inv0fft<(1<< 2),1,0>()(x, y, W); break;
            case  3: OTFFT_AVXDIF4::inv0fft<(1<< 3),1,0>()(x, y, W); break;
            case  4: OTFFT_AVXDIF4::inv0fft<(1<< 4),1,0>()(x, y, W); break;
            case  5: OTFFT_AVXDIF4::inv0fft<(1<< 5),1,0>()(x, y, W); break;
            case  6: OTFFT_AVXDIF4::inv0fft<(1<< 6),1,0>()(x, y, W); break;
            case  7: OTFFT_AVXDIF4::inv0fft<(1<< 7),1,0>()(x, y, W); break;
            case  8: OTFFT_AVXDIF4::inv0fft<(1<< 8),1,0>()(x, y, W); break;
            case  9: OTFFT_AVXDIF4::inv0fft<(1<< 9),1,0>()(x, y, W); break;
            case 10: OTFFT_AVXDIF4::inv0fft<(1<<10),1,0>()(x, y, W); break;
            case 11: OTFFT_AVXDIF8::inv0fft<(1<<11),1,0>()(x, y, W); break;
            case 12: OTFFT_AVXDIF8::inv0fft<(1<<12),1,0>()(x, y, W); break;
            case 13: OTFFT_AVXDIF8::inv0fft<(1<<13),1,0>()(x, y, W); break;
            case 14: OTFFT_AVXDIF8::inv0fft<(1<<14),1,0>()(x, y, W); break;
            case 15: OTFFT_AVXDIF8::inv0fft<(1<<15),1,0>()(x, y, W); break;
            case 16: OTFFT_AVXDIF8::inv0fft<(1<<16),1,0>()(x, y, W); break;
            case 17: OTFFT_AVXDIF8::inv0fft<(1<<17),1,0>()(x, y, W); break;
            case 18: OTFFT_AVXDIF8::inv0fft<(1<<18),1,0>()(x, y, W); break;
            case 19: OTFFT_AVXDIF8::inv0fft<(1<<19),1,0>()(x, y, W); break;
            case 20: OTFFT_AVXDIF8::inv0fft<(1<<20),1,0>()(x, y, W); break;
            case 21: OTFFT_AVXDIF8::inv0fft<(1<<21),1,0>()(x, y, W); break;
            case 22: OTFFT_AVXDIF8::inv0fft<(1<<22),1,0>()(x, y, W); break;
            case 23: OTFFT_AVXDIF8::inv0fft<(1<<23),1,0>()(x, y, W); break;
            case 24: OTFFT_AVXDIF8::inv0fft<(1<<24),1,0>()(x, y, W); break;
        }
    }

    inline void inv0(complex_vector x, complex_vector y) const { inv(x, y); }

    inline void invn(complex_vector x, complex_vector y) const
    {
        switch (log_N) {
            case  0: break;
            case  1: OTFFT_AVXDIF4::invnfft<(1<< 1),1,0>()(x, y, W); break;
            case  2: OTFFT_AVXDIF4::invnfft<(1<< 2),1,0>()(x, y, W); break;
            case  3: OTFFT_AVXDIF4::invnfft<(1<< 3),1,0>()(x, y, W); break;
            case  4: OTFFT_AVXDIF4::invnfft<(1<< 4),1,0>()(x, y, W); break;
            case  5: OTFFT_AVXDIF4::invnfft<(1<< 5),1,0>()(x, y, W); break;
            case  6: OTFFT_AVXDIF4::invnfft<(1<< 6),1,0>()(x, y, W); break;
            case  7: OTFFT_AVXDIF4::invnfft<(1<< 7),1,0>()(x, y, W); break;
            case  8: OTFFT_AVXDIF4::invnfft<(1<< 8),1,0>()(x, y, W); break;
            case  9: OTFFT_AVXDIF4::invnfft<(1<< 9),1,0>()(x, y, W); break;
            case 10: OTFFT_AVXDIF4::invnfft<(1<<10),1,0>()(x, y, W); break;
            case 11: OTFFT_AVXDIF8::invnfft<(1<<11),1,0>()(x, y, W); break;
            case 12: OTFFT_AVXDIF8::invnfft<(1<<12),1,0>()(x, y, W); break;
            case 13: OTFFT_AVXDIF8::invnfft<(1<<13),1,0>()(x, y, W); break;
            case 14: OTFFT_AVXDIF8::invnfft<(1<<14),1,0>()(x, y, W); break;
            case 15: OTFFT_AVXDIF8::invnfft<(1<<15),1,0>()(x, y, W); break;
            case 16: OTFFT_AVXDIF8::invnfft<(1<<16),1,0>()(x, y, W); break;
            case 17: OTFFT_AVXDIF8::invnfft<(1<<17),1,0>()(x, y, W); break;
            case 18: OTFFT_AVXDIF8::invnfft<(1<<18),1,0>()(x, y, W); break;
            case 19: OTFFT_AVXDIF8::invnfft<(1<<19),1,0>()(x, y, W); break;
            case 20: OTFFT_AVXDIF8::invnfft<(1<<20),1,0>()(x, y, W); break;
            case 21: OTFFT_AVXDIF8::invnfft<(1<<21),1,0>()(x, y, W); break;
            case 22: OTFFT_AVXDIF8::invnfft<(1<<22),1,0>()(x, y, W); break;
            case 23: OTFFT_AVXDIF8::invnfft<(1<<23),1,0>()(x, y, W); break;
            case 24: OTFFT_AVXDIF8::invnfft<(1<<24),1,0>()(x, y, W); break;
        }
    }

    inline void inv0o(complex_vector x, complex_vector y) const
    {
        switch (log_N) {
            case  0: y[0] = x[0]; break;
            case  1: OTFFT_AVXDIF4::inv0fft<(1<< 1),1,1>()(x, y, W); break;
            case  2: OTFFT_AVXDIF4::inv0fft<(1<< 2),1,1>()(x, y, W); break;
            case  3: OTFFT_AVXDIF4::inv0fft<(1<< 3),1,1>()(x, y, W); break;
            case  4: OTFFT_AVXDIF4::inv0fft<(1<< 4),1,1>()(x, y, W); break;
            case  5: OTFFT_AVXDIF4::inv0fft<(1<< 5),1,1>()(x, y, W); break;
            case  6: OTFFT_AVXDIF4::inv0fft<(1<< 6),1,1>()(x, y, W); break;
            case  7: OTFFT_AVXDIF4::inv0fft<(1<< 7),1,1>()(x, y, W); break;
            case  8: OTFFT_AVXDIF4::inv0fft<(1<< 8),1,1>()(x, y, W); break;
            case  9: OTFFT_AVXDIF4::inv0fft<(1<< 9),1,1>()(x, y, W); break;
            case 10: OTFFT_AVXDIF4::inv0fft<(1<<10),1,1>()(x, y, W); break;
            case 11: OTFFT_AVXDIF8::inv0fft<(1<<11),1,1>()(x, y, W); break;
            case 12: OTFFT_AVXDIF8::inv0fft<(1<<12),1,1>()(x, y, W); break;
            case 13: OTFFT_AVXDIF8::inv0fft<(1<<13),1,1>()(x, y, W); break;
            case 14: OTFFT_AVXDIF8::inv0fft<(1<<14),1,1>()(x, y, W); break;
            case 15: OTFFT_AVXDIF8::inv0fft<(1<<15),1,1>()(x, y, W); break;
            case 16: OTFFT_AVXDIF8::inv0fft<(1<<16),1,1>()(x, y, W); break;
            case 17: OTFFT_AVXDIF8::inv0fft<(1<<17),1,1>()(x, y, W); break;
            case 18: OTFFT_AVXDIF8::inv0fft<(1<<18),1,1>()(x, y, W); break;
            case 19: OTFFT_AVXDIF8::inv0fft<(1<<19),1,1>()(x, y, W); break;
            case 20: OTFFT_AVXDIF8::inv0fft<(1<<20),1,1>()(x, y, W); break;
            case 21: OTFFT_AVXDIF8::inv0fft<(1<<21),1,1>()(x, y, W); break;
            case 22: OTFFT_AVXDIF8::inv0fft<(1<<22),1,1>()(x, y, W); break;
            case 23: OTFFT_AVXDIF8::inv0fft<(1<<23),1,1>()(x, y, W); break;
            case 24: OTFFT_AVXDIF8::inv0fft<(1<<24),1,1>()(x, y, W); break;
        }
    }

    inline void invno(complex_vector x, complex_vector y) const
    {
        switch (log_N) {
            case  0: y[0] = x[0]; break;
            case  1: OTFFT_AVXDIF4::invnfft<(1<< 1),1,1>()(x, y, W); break;
            case  2: OTFFT_AVXDIF4::invnfft<(1<< 2),1,1>()(x, y, W); break;
            case  3: OTFFT_AVXDIF4::invnfft<(1<< 3),1,1>()(x, y, W); break;
            case  4: OTFFT_AVXDIF4::invnfft<(1<< 4),1,1>()(x, y, W); break;
            case  5: OTFFT_AVXDIF4::invnfft<(1<< 5),1,1>()(x, y, W); break;
            case  6: OTFFT_AVXDIF4::invnfft<(1<< 6),1,1>()(x, y, W); break;
            case  7: OTFFT_AVXDIF4::invnfft<(1<< 7),1,1>()(x, y, W); break;
            case  8: OTFFT_AVXDIF4::invnfft<(1<< 8),1,1>()(x, y, W); break;
            case  9: OTFFT_AVXDIF4::invnfft<(1<< 9),1,1>()(x, y, W); break;
            case 10: OTFFT_AVXDIF4::invnfft<(1<<10),1,1>()(x, y, W); break;
            case 11: OTFFT_AVXDIF8::invnfft<(1<<11),1,1>()(x, y, W); break;
            case 12: OTFFT_AVXDIF8::invnfft<(1<<12),1,1>()(x, y, W); break;
            case 13: OTFFT_AVXDIF8::invnfft<(1<<13),1,1>()(x, y, W); break;
            case 14: OTFFT_AVXDIF8::invnfft<(1<<14),1,1>()(x, y, W); break;
            case 15: OTFFT_AVXDIF8::invnfft<(1<<15),1,1>()(x, y, W); break;
            case 16: OTFFT_AVXDIF8::invnfft<(1<<16),1,1>()(x, y, W); break;
            case 17: OTFFT_AVXDIF8::invnfft<(1<<17),1,1>()(x, y, W); break;
            case 18: OTFFT_AVXDIF8::invnfft<(1<<18),1,1>()(x, y, W); break;
            case 19: OTFFT_AVXDIF8::invnfft<(1<<19),1,1>()(x, y, W); break;
            case 20: OTFFT_AVXDIF8::invnfft<(1<<20),1,1>()(x, y, W); break;
            case 21: OTFFT_AVXDIF8::invnfft<(1<<21),1,1>()(x, y, W); break;
            case 22: OTFFT_AVXDIF8::invnfft<(1<<22),1,1>()(x, y, W); break;
            case 23: OTFFT_AVXDIF8::invnfft<(1<<23),1,1>()(x, y, W); break;
            case 24: OTFFT_AVXDIF8::invnfft<(1<<24),1,1>()(x, y, W); break;
        }
    }
};

#if 0
struct FFT {
    FFT0 fft;
    simd_array<complex_t> work;
    complex_t* __restrict y;

    FFT() : y(0) {}
    FFT(const int n) : fft(n), work(n), y(&work) {}

    inline void setup(const int n) { fft.setup(n); work.setup(n); y = &work; }

    inline void fwd(complex_vector x)  const { fft.fwd(x, y);  }
    inline void fwd0(complex_vector x) const { fft.fwd0(x, y); }
    inline void fwdn(complex_vector x) const { fft.fwdn(x, y); }
    inline void inv(complex_vector x)  const { fft.inv(x, y);  }
    inline void inv0(complex_vector x) const { fft.inv0(x, y); }
    inline void invn(complex_vector x) const { fft.invn(x, y); }
};
#endif

} /////////////////////////////////////////////////////////////////////////////

#endif // otfft_avxdifx_h
