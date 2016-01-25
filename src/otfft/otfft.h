/******************************************************************************
*  OTFFT Header Version 6.5
*
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_h
#define otfft_h

#include <cmath>
#include "otfft_misc.h"

namespace OTFFT_AVXDIF4 { struct FFT0; }
namespace OTFFT_AVXDIT4 { struct FFT0; }
namespace OTFFT_AVXDIF8 { struct FFT0; }
namespace OTFFT_AVXDIT8 { struct FFT0; }
namespace OTFFT_Sixstep { struct FFT0; }
namespace OTFFT_AVXDIF16 { struct FFT0; }
namespace OTFFT_AVXDIT16 { struct FFT0; }

namespace OTFFT { /////////////////////////////////////////////////////////////

using namespace OTFFT_MISC;

/******************************************************************************
*  Complex FFT
******************************************************************************/

struct FFT0
{
    int N, log_N;
    OTFFT_AVXDIF4::FFT0* fft1;
    OTFFT_AVXDIT4::FFT0* fft2;
    OTFFT_AVXDIF8::FFT0* fft3;
    OTFFT_AVXDIT8::FFT0* fft4;
    OTFFT_Sixstep::FFT0* fft5;
    OTFFT_AVXDIF16::FFT0* fft6;
    OTFFT_AVXDIT16::FFT0* fft7;

    FFT0();
    FFT0(int n);
    ~FFT0();

    void setup(int n);
    void setup2(int n);

    void fwd(complex_vector  x, complex_vector y) const noexcept;
    void fwd0(complex_vector x, complex_vector y) const noexcept;
    void fwdn(complex_vector x, complex_vector y) const noexcept;
    void inv(complex_vector  x, complex_vector y) const noexcept;
    void inv0(complex_vector x, complex_vector y) const noexcept;
    void invn(complex_vector x, complex_vector y) const noexcept;
};

struct FFT
{
    FFT0 fft;
    simd_array<complex_t> work;
    complex_t* y;

    FFT() noexcept : fft(), work(), y(0) {}
    FFT(int n) : fft(n), work(n), y(&work) {}

    inline void setup(int n) { fft.setup(n); work.setup(n); y = &work; }

    inline void fwd(complex_vector  x) const noexcept { fft.fwd(x, y);  }
    inline void fwd0(complex_vector x) const noexcept { fft.fwd0(x, y); }
    inline void fwdn(complex_vector x) const noexcept { fft.fwdn(x, y); }
    inline void inv(complex_vector  x) const noexcept { fft.inv(x, y);  }
    inline void inv0(complex_vector x) const noexcept { fft.inv0(x, y); }
    inline void invn(complex_vector x) const noexcept { fft.invn(x, y); }
};

/******************************************************************************
*  Real FFT
******************************************************************************/

struct RFFT
{
#ifdef DO_SINGLE_THREAD
    static const int OMP_THRESHOLD   = 1<<30;
    static const int OMP_THRESHOLD_W = 1<<30;
#else
    static const int OMP_THRESHOLD   = 1<<15;
    static const int OMP_THRESHOLD_W = 1<<16;
#endif

    int N;
    FFT0 fft;
    simd_array<complex_t> weight;
    complex_t* U;

    RFFT();
    RFFT(int n);

    void setup(int n);

    void fwd(const_double_vector  x, complex_vector y) const noexcept;
    void fwd0(const_double_vector x, complex_vector y) const noexcept;
    void fwdn(const_double_vector x, complex_vector y) const noexcept;
    void inv(complex_vector  x, double_vector y) const noexcept;
    void inv0(complex_vector x, double_vector y) const noexcept;
    void invn(complex_vector x, double_vector y) const noexcept;
};

/******************************************************************************
*  DCT
******************************************************************************/

struct DCT0
{
#ifdef DO_SINGLE_THREAD
    static const int OMP_THRESHOLD   = 1<<30;
    static const int OMP_THRESHOLD_W = 1<<30;
#else
    static const int OMP_THRESHOLD   = 1<<15;
    static const int OMP_THRESHOLD_W = 1<<16;
#endif

    int N;
    RFFT rfft;
    simd_array<complex_t> weight;
    complex_t* V;

    DCT0();
    DCT0(int n);

    void setup(int n);

    void fwd(double_vector  x, double_vector y, complex_vector z) const noexcept;
    void fwd0(double_vector x, double_vector y, complex_vector z) const noexcept;
    void fwdn(double_vector x, double_vector y, complex_vector z) const noexcept;
    void inv(double_vector  x, double_vector y, complex_vector z) const noexcept;
    void inv0(double_vector x, double_vector y, complex_vector z) const noexcept;
    void invn(double_vector x, double_vector y, complex_vector z) const noexcept;
};

struct DCT
{
    int N;
    DCT0 dct;
    simd_array<double> work1;
    simd_array<complex_t> work2;
    double* y;
    complex_t* z;

    DCT() : N(0), y(0), z(0) {}
    DCT(int n) { setup(n); }

    void setup(int n)
    {
        N = n;
        dct.setup(N);
        work1.setup(N); y = &work1;
        work2.setup(N); z = &work2;
    }

    void fwd(double_vector  x) const noexcept { dct.fwd(x, y, z);  }
    void fwd0(double_vector x) const noexcept { dct.fwd0(x, y, z); }
    void fwdn(double_vector x) const noexcept { dct.fwdn(x, y, z); }
    void inv(double_vector  x) const noexcept { dct.inv(x, y, z);  }
    void inv0(double_vector x) const noexcept { dct.inv0(x, y, z); }
    void invn(double_vector x) const noexcept { dct.invn(x, y, z); }
};

/******************************************************************************
*  Bluestein's FFT
******************************************************************************/

struct Bluestein
{
#ifdef DO_SINGLE_THREAD
    static const int OMP_THRESHOLD   = 1<<30;
    static const int OMP_THRESHOLD_W = 1<<30;
#else
    static const int OMP_THRESHOLD   = 1<<15;
    static const int OMP_THRESHOLD_W = 1<<16;
#endif

    int N, L;
    FFT fft;
    simd_array<complex_t> work1;
    simd_array<complex_t> work2;
    simd_array<complex_t> weight;
    complex_t* a;
    complex_t* b;
    complex_t* W;

    Bluestein();
    Bluestein(int n);

    void setup(int n);

    void fwd0(complex_vector x) const noexcept;
    void fwd(complex_vector  x) const noexcept;
    void fwdn(complex_vector x) const noexcept;
    void inv0(complex_vector x) const noexcept;
    void inv(complex_vector  x) const noexcept;
    void invn(complex_vector x) const noexcept;
};

} /////////////////////////////////////////////////////////////////////////////

#endif // otfft_h
