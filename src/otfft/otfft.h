/******************************************************************************
*  OTFFT Header Version 5.3
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
//namespace OTFFT_Sixstep { struct FFT1; }

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
    //OTFFT_Sixstep::FFT1* fft6;

    FFT0();
    FFT0(int n);
    ~FFT0();

    void setup(int n);
    void setup2(int n);

    void fwd(complex_vector  x, complex_vector y) const;
    void fwd0(complex_vector x, complex_vector y) const;
    void fwdn(complex_vector x, complex_vector y) const;
    void inv(complex_vector  x, complex_vector y) const;
    void inv0(complex_vector x, complex_vector y) const;
    void invn(complex_vector x, complex_vector y) const;
};

struct FFT
{
    FFT0 fft;
    simd_array<complex_t> work;
    complex_t* y;

    FFT() : fft(), work(), y(0) {}
    FFT(int n) : fft(n), work(n), y(&work) {}

    inline void setup(int n) { fft.setup(n); work.setup(n); y = &work; }

    inline void fwd(complex_vector  x) const { fft.fwd(x, y);  }
    inline void fwd0(complex_vector x) const { fft.fwd0(x, y); }
    inline void fwdn(complex_vector x) const { fft.fwdn(x, y); }
    inline void inv(complex_vector  x) const { fft.inv(x, y);  }
    inline void inv0(complex_vector x) const { fft.inv0(x, y); }
    inline void invn(complex_vector x) const { fft.invn(x, y); }
};

/******************************************************************************
*  Real FFT
******************************************************************************/

struct RFFT
{
    static const int OMP_THRESHOLD   = 1<<15;
    static const int OMP_THRESHOLD_W = 1<<16;

    int N;
    FFT0 fft;
    simd_array<complex_t> weight;
    complex_t* U;

    RFFT();
    RFFT(int n);

    void setup(int n);

    void fwd(const_double_vector  x, complex_vector y) const;
    void fwd0(const_double_vector x, complex_vector y) const;
    void fwdn(const_double_vector x, complex_vector y) const;
    void inv(complex_vector  x, double_vector y) const;
    void inv0(complex_vector x, double_vector y) const;
    void invn(complex_vector x, double_vector y) const;
};

/******************************************************************************
*  DCT
******************************************************************************/

struct DCT0
{
    static const int OMP_THRESHOLD   = 1<<15;
    static const int OMP_THRESHOLD_W = 1<<16;

    int N;
    RFFT rfft;
    simd_array<complex_t> weight;
    complex_t* V;

    DCT0();
    DCT0(int n);

    void setup(int n);

    void fwd(double_vector  x, double_vector y, complex_vector z) const;
    void fwd0(double_vector x, double_vector y, complex_vector z) const;
    void fwdn(double_vector x, double_vector y, complex_vector z) const;
    void inv(double_vector  x, double_vector y, complex_vector z) const;
    void inv0(double_vector x, double_vector y, complex_vector z) const;
    void invn(double_vector x, double_vector y, complex_vector z) const;
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

    void fwd(double_vector  x) const { dct.fwd(x, y, z);  }
    void fwd0(double_vector x) const { dct.fwd0(x, y, z); }
    void fwdn(double_vector x) const { dct.fwdn(x, y, z); }
    void inv(double_vector  x) const { dct.inv(x, y, z);  }
    void inv0(double_vector x) const { dct.inv0(x, y, z); }
    void invn(double_vector x) const { dct.invn(x, y, z); }
};

/******************************************************************************
*  Bluestein's FFT
******************************************************************************/

struct Bluestein
{
    static const int OMP_THRESHOLD   = 1<<15;
    static const int OMP_THRESHOLD_W = 1<<16;

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

    void fwd0(complex_vector x) const;
    void fwd(complex_vector  x) const;
    void fwdn(complex_vector x) const;
    void inv0(complex_vector x) const;
    void inv(complex_vector  x) const;
    void invn(complex_vector x) const;
};

} /////////////////////////////////////////////////////////////////////////////

#endif // otfft_h
