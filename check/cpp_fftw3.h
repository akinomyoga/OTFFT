/******************************************************************************
*  C++ Wrapper for FFTW3
*
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef cpp_fftw3_h
#define cpp_fftw3_h

#include <new>
#include <fftw3.h>
#include <omp.h>
#include "otfft/otfft_misc.h"

#define FFTW_FLAG FFTW_ESTIMATE
//#define FFTW_FLAG FFTW_MEASURE
//#define FFTW_FLAG FFTW_PATIENT
//#define FFTW_FLAG FFTW_EXHAUSTIVE

namespace CppFFTW3 { //////////////////////////////////////////////////////////

using namespace OTFFT_MISC;

static const int FFTW_MT_THRESHOLD = 1<<14;
static const int FORWARD = FFTW_FORWARD;
static const int INVERSE = FFTW_BACKWARD;

///////////////////////////////////////////////////////////////////////////////

template <int direction, bool normalize, bool backup> class Plan
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
    Plan(int N, complex_t* const x)
        : N(N), work(N), x(x), y(&work), first(true)
    {
#if FFTW_FLAG != FFTW_ESTIMATE
        if (backup) {
            buffer.setup(N); z = &buffer;
            if (N == 1)
                z[0] = x[0];
            else if (N >= 2)
                for (int k = 0; k < N; k += 2) setpz2(z+k, getpz2(x+k));
            else return;
        }
#endif
        if (N >= FFTW_MT_THRESHOLD) {
#ifdef USE_FFTW_THREADS
            fftw_plan_with_nthreads(omp_get_max_threads());
            //fftw_plan_with_nthreads(omp_get_num_procs());
#endif
        }
        p = fftw_plan_dft_1d(N,
                reinterpret_cast<fftw_complex*>(x),
                reinterpret_cast<fftw_complex*>(y),
                direction, FFTW_FLAG);
        if (!p) throw std::bad_alloc();
    }

    ~Plan() { fftw_destroy_plan(p); }

    void operator()() const
    {
#if FFTW_FLAG != FFTW_ESTIMATE
        if (backup && first) {
            first = false;
            if (N == 1)
                x[0] = z[0];
            else if (N >= 2)
                for (int k = 0; k < N; k += 2) setpz2(x+k, getpz2(z+k));
            else return;
        }
#endif
        fftw_execute(p);
        if (N == 1) x[0] = y[0];
        else if (N >= 2) {
            if (normalize) {
                const ymm rN = cmplx2(1.0/N, 1.0/N, 1.0/N, 1.0/N);
                for (int k = 0; k < N; k += 2) {
                    setpz2(x+k, mulpd2(rN, getpz2(y+k)));
                }
            }
            else {
                for (int k = 0; k < N; k += 2) {
                    setpz2(x+k, getpz2(y+k));
                }
            }
        }
    }
};

} /////////////////////////////////////////////////////////////////////////////

#endif // cpp_fftw3_h
