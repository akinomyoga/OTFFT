/******************************************************************************
*  FFT Benchmark 1
******************************************************************************/

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <limits>
#include "cpp_fftw3.h"
#include "ooura1.h"
#include "simple_fft.h"
#include "otfft/otfft.h"
#include "otfft/stopwatch.h"
using OTFFT::complex_t;
using OTFFT::simd_malloc;
using OTFFT::simd_free;

#define TRIES  5
#define DELAY1 1
#define DELAY2 100

template <class FFT, class IFFT>
double laptime1(const FFT& fft, const IFFT& ifft, int LOOPS)
{
    counter_t dt = (std::numeric_limits<counter_t>::max)();
    for (int i = 0; i < TRIES; i++) {
        const counter_t t1 = get_counter();
        for (int j = 0; j < LOOPS; j++) {
            fft();
            ifft();
        }
        const counter_t t2 = get_counter();
        dt = (std::min)(dt, t2 - t1);
        msleep(DELAY1);
    }
    return usec(dt);
}

template <class FFT>
double laptime2(const FFT& fft, int LOOPS, complex_t* x)
{
    counter_t dt = (std::numeric_limits<counter_t>::max)();
    for (int i = 0; i < TRIES; i++) {
        const counter_t t1 = get_counter();
        for (int j = 0; j < LOOPS; j++) {
            fft.fwd(x);
            fft.inv(x);
        }
        const counter_t t2 = get_counter();
        dt = (std::min)(dt, t2 - t1);
        msleep(DELAY1);
    }
    return usec(dt);
}

int main() try
{
    static const int n_max  = 22;
    static const int N_max  = 1 << n_max;
    static const int Nn_max = N_max * n_max;

    fftw_init_threads();
    setbuf(stdout, NULL);
    printf("------+-----------+-----------------+-----------------+-----------------\n");
    printf("length|FFTW3[usec]|   OOURA   [usec]| SimpleFFT [usec]|   OTFFT   [usec]\n");
    printf("------+-----------+-----------------+-----------------+-----------------\n");
    complex_t* x0 = (complex_t*) simd_malloc(N_max*sizeof(complex_t));
    complex_t* x1 = (complex_t*) simd_malloc(N_max*sizeof(complex_t));
    complex_t* x2 = (complex_t*) simd_malloc(N_max*sizeof(complex_t));
    complex_t* x3 = (complex_t*) simd_malloc(N_max*sizeof(complex_t));
    complex_t* x4 = (complex_t*) simd_malloc(N_max*sizeof(complex_t));
    for (int n = 1; n <= n_max; n++) {
        const int N = 1 << n;
        const int LOOPS = static_cast<int>(rint(Nn_max/double(N*n)));
        double lap, lap1;

        //// sample setting ////
        for (int p = 0; p < N; p++) {
            const double t = double(p) / N;
            x0[p].Re = 10 * cos((2*M_PI/N)*t*t);
            x0[p].Im = 10 * sin((2*M_PI/N)*t*t);
            x1[p] = x2[p] = x3[p] = x4[p] = x0[p];
        }
        msleep(DELAY2);

        printf("2^(%2d)|", n);

        //// FFTW3 FFT ////
        FFTW::FFTPlan<1>   fft_plan(N, x1);
        FFTW::IFFTPlan<0> ifft_plan(N, x1);
        lap = laptime1(fft_plan, ifft_plan, LOOPS);
        printf("%11.2f|", lap/LOOPS);
        msleep(DELAY2);
        lap1 = lap;

        //// OOURA FFT ////
        OOURA::FFT ooura_fft(N);
        lap = laptime2(ooura_fft, LOOPS, x2);
        printf("%11.2f(%3.0f%%)|", lap/LOOPS, 100*lap/lap1);
        msleep(DELAY2);

        //// Simple FFT ////
        OTFFT::speedup_magic();
        SimpleFFT::FFT simple_fft(N);
        lap = laptime2(simple_fft, LOOPS, x3);
        printf("%11.2f(%3.0f%%)|", lap/LOOPS, 100*lap/lap1);
        msleep(DELAY2);

        //// OTFFT ////
        OTFFT::speedup_magic();
        OTFFT::FFT ot_fft(N);
        lap = laptime2(ot_fft, LOOPS, x4);
        printf("%11.2f(%3.0f%%)\n", lap/LOOPS, 100*lap/lap1);
        msleep(DELAY2);
    }
    simd_free(x4);
    simd_free(x3);
    simd_free(x2);
    simd_free(x1);
    simd_free(x0);
    printf("------+-----------+-----------------+-----------------+-----------------\n");

    return 0;
}
catch (std::bad_alloc&) { fprintf(stderr, "\n""not enough memory!!\n"); }
