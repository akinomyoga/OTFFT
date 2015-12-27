/******************************************************************************
*  C++ Wrapper 2 for OOURA FFT
*
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef ooura2_h
#define ooura2_h

#include <cmath>
#include "otfft/otfft_misc.h"

#ifdef _MSC_VER
#define USE_CDFT_WINTHREADS
#else
#define USE_CDFT_PTHREADS
#endif

#include "fftsg.c"
//#include "fft4g.c"
//#include "fft8g.c"

namespace OOURA { /////////////////////////////////////////////////////////////

using namespace OTFFT_MISC;

class FFT
{
private:
    const int N;
    simd_array<int> iparr;
    simd_array<double> weight;
    int* const ip;
    double* const w;
public:
    FFT(int n) : N(n),
        iparr(2 + ceil(sqrt(n))), weight(ceil(n/2.0)), ip(&iparr), w(&weight)
    {
        ip[0] = 0;
    }

    ///////////////////////////////////////////////////////////////////////////

    void fwd(complex_t* const x) const
    {
        cdft(2*N, -1, &x->Re, ip, w);
        const ymm rN = cmplx2(1.0/N, 1.0/N, 1.0/N, 1.0/N);
        if (N > 1)
            for (int k = 0; k < N; k += 2) setpz2(x+k, mulpd2(rN, getpz2(x+k)));
    }

    void fwd0(complex_t* const x) const
    {
        cdft(2*N, -1, &x->Re, ip, w);
    }

    void fwdn(complex_t* const x) const { fwd(x); }

    ///////////////////////////////////////////////////////////////////////////

    void inv(complex_t* const x) const
    {
        cdft(2*N, 1, &x->Re, ip, w);
    }

    void inv0(complex_t* const x) const { inv(x); }

    void invn(complex_t* const x) const
    {
        cdft(2*N, 1, &x->Re, ip, w);
        const ymm rN = cmplx2(1.0/N, 1.0/N, 1.0/N, 1.0/N);
        if (N > 1)
            for (int k = 0; k < N; k += 2) setpz2(x+k, mulpd2(rN, getpz2(x+k)));
    }
};

} /////////////////////////////////////////////////////////////////////////////

#endif // ooura2_h
