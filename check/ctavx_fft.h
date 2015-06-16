/******************************************************************************
* Cooley Tukey AVX FFT (4 radix)
******************************************************************************/

#ifndef ctavx_fft_h
#define ctavx_fft_h

#include <algorithm>
#include "otfft/otfft_misc.h"
#include "otfft/otfft_ctavx.h"

namespace CTAVX_FFT { /////////////////////////////////////////////////////////

using namespace OTFFT_MISC;

struct FFT
{
    OTFFT_CTAVX::FFT fft;

    FFT(int n) : fft(n) {}

    void fwd0(complex_t* const x) const { fft.fwd0(x); }

    void fwd(complex_t* const x) const { fft.fwd(x); }

    void fwdn(complex_t* const x) const { fft.fwdn(x); }

    void inv0(complex_t* const x) const { fft.inv0(x); }

    void inv(complex_t* const x) const { fft.inv(x); }

    void invn(complex_t* const x) const { fft.invn(x); }
};

} /////////////////////////////////////////////////////////////////////////////

#endif // ctavx_fft_h
