/******************************************************************************
*  FFTW3 using C++
******************************************************************************/

#ifndef cpp_fftw3_h
#define cpp_fftw3_h

#include <new>
#include <fftw3.h>
#include <omp.h>
#include "otfft/otfft_misc.h"

//#define FFTW_FLAG FFTW_ESTIMATE
#define FFTW_FLAG FFTW_MEASURE
//#define FFTW_FLAG FFTW_PATIENT
//#define FFTW_FLAG FFTW_EXHAUSTIVE

namespace FFTW { //////////////////////////////////////////////////////////////

using namespace OTFFT_MISC;

const int FFTW_MT_THRESHOLD = 1<<14;

template <bool b> class FFTPlan
{
private:
    const int N;
    simd_array<complex_t> work, buffer;
    complex_t* const x;
    complex_t* const y;
    complex_t* z;
    mutable bool first;
    fftw_plan p;

public:
    FFTPlan(int N, complex_t* const x) : N(N), work(N), x(x), y(&work), first(true)
    {
#if FFTW_FLAG != FFTW_ESTIMATE
        buffer.setup(N); z = &buffer;
        if (N < 2)
            z[0] = x[0];
        else
            for (int k = 0; k < N; k += 2) setpz2(z+k, getpz2(x+k));
#endif
#ifdef USE_FFTW_THREADS
        if (N >= FFTW_MT_THRESHOLD) fftw_plan_with_nthreads(omp_get_max_threads());
#endif
        p = fftw_plan_dft_1d(N,
            reinterpret_cast<fftw_complex*>(x),
            reinterpret_cast<fftw_complex*>(y),
            FFTW_FORWARD, FFTW_FLAG);
        if (!p) throw std::bad_alloc();
    }

    ~FFTPlan() { fftw_destroy_plan(p); }

    void operator()() const
    {
#if FFTW_FLAG != FFTW_ESTIMATE
        if (b && first) {
            first = false;
            if (N < 2)
                x[0] = z[0];
            else
                for (int k = 0; k < N; k += 2) setpz2(x+k, getpz2(z+k));
        }
#endif
        fftw_execute(p);
        const ymm rN = cmplx2(1.0/N, 1.0/N, 1.0/N, 1.0/N);
        if (N < 2)
            x[0] = y[0];
        else
            for (int k = 0; k < N; k += 2) setpz2(x+k, mulpd2(rN, getpz2(y+k)));
    }
};

template <bool b> class IFFTPlan
{
private:
    int N;
    simd_array<complex_t> work, buffer;
    complex_t* const x;
    complex_t* const y;
    complex_t* z;
    mutable bool first;
    fftw_plan p;

public:
    IFFTPlan(int N, complex_t* const x) : N(N), work(N), x(x), y(&work), first(true)
    {
#if FFTW_FLAG != FFTW_ESTIMATE
        buffer.setup(N); z = &buffer;
        if (N < 2)
            z[0] = x[0];
        else
            for (int k = 0; k < N; k += 2) setpz2(z+k, getpz2(x+k));
#endif
#ifdef USE_FFTW_THREADS
        if (N >= FFTW_MT_THRESHOLD) fftw_plan_with_nthreads(omp_get_max_threads());
#endif
        p = fftw_plan_dft_1d(N,
            reinterpret_cast<fftw_complex*>(x),
            reinterpret_cast<fftw_complex*>(y),
            FFTW_BACKWARD, FFTW_FLAG);
        if (!p) throw std::bad_alloc();
    }

    ~IFFTPlan() { fftw_destroy_plan(p); }

    void operator()() const
    {
#if FFTW_FLAG != FFTW_ESTIMATE
        if (b && first) {
            first = false;
            if (N < 2)
                x[0] = z[0];
            else
                for (int k = 0; k < N; k += 2) setpz2(x+k, getpz2(z+k));
        }
#endif
        fftw_execute(p);
        if (N < 2)
            x[0] = y[0];
        else
            for (int k = 0; k < N; k += 2) setpz2(x+k, getpz2(y+k));
    }
};

} /////////////////////////////////////////////////////////////////////////////

#endif // cpp_fftw3_h
