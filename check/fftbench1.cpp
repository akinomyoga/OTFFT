/******************************************************************************
*  FFT Benchmark 1
******************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>
#include <algorithm>
#include "cpp_fftw3.h"
#include "ooura1.h"
#include "simple_fft.h"
#include "otfft/otfft.h"

#if __cplusplus < 201103L
#include "otfft/stopwatch.h"
#else
#include <chrono>
#include "otfft/msleep.h"
typedef std::chrono::microseconds::rep counter_t;
#endif

using OTFFT::complex_t;
using OTFFT::simd_malloc;
using OTFFT::simd_free;
using CppFFTW3::FORWARD;
using CppFFTW3::INVERSE;

#define DELAY1 1
#define DELAY2 100

#ifdef __AVX__
#define FACTOR 4
#else
#define FACTOR 3
#endif

double safe_avr(const std::vector<counter_t>& dt) // 異常値を含まない平均
{
    typedef std::vector<counter_t>::size_type size_type;
    const size_type TRIES = dt.size();
    counter_t sum = 0;
    for (size_type i = 0; i < TRIES; i++) sum += dt[i];
    const double m = double(sum)/TRIES;
    double sum_dd = 0;
    for (size_type i = 0; i < TRIES; i++) {
        const double d = dt[i] - m;
        sum_dd += d*d;
    }
    const double ss = sum_dd/TRIES;
    sum = 0;
    size_type n = 0;
    for (size_type i = 0; i < TRIES; i++) {
        const double d = dt[i] - m;
        if (d*d <= 2*ss) { sum += dt[i]; n++; }
    }
    return double(sum)/n;
}

#if __cplusplus < 201103L
template <typename FFT, typename IFFT>
double laptime1(int LOOPS, int TRIES, const FFT& fft, const IFFT& ifft)
{
    std::vector<counter_t> dt(TRIES);
    for (int i = 0; i < TRIES; i++) {
        const counter_t t1 = get_counter();
        for (int j = 0; j < LOOPS; j++) {
            fft();
            ifft();
        }
        const counter_t t2 = get_counter();
        dt[i] = t2 - t1;
        msleep(DELAY1);
    }
    return usec(lrint(safe_avr(dt)));
}

template <typename FFT>
double laptime2(int LOOPS, int TRIES, const FFT& fft, complex_t *x)
{
    std::vector<counter_t> dt(TRIES);
    for (int i = 0; i < TRIES; i++) {
        const counter_t t1 = get_counter();
        for (int j = 0; j < LOOPS; j++) {
            fft.fwd(x);
            fft.inv(x);
        }
        const counter_t t2 = get_counter();
        dt[i] = t2 - t1;
        msleep(DELAY1);
    }
    return usec(lrint(safe_avr(dt)));
}
#else
template <typename FFT, typename IFFT>
double laptime1(int LOOPS, int TRIES, const FFT& fft, const IFFT& ifft)
{
    using namespace std::chrono;
    std::vector<counter_t> dt(TRIES);
    for (int i = 0; i < TRIES; i++) {
        const system_clock::time_point t1 = system_clock::now();
        for (int j = 0; j < LOOPS; j++) {
            fft();
            ifft();
        }
        const system_clock::time_point t2 = system_clock::now();
        dt[i] = duration_cast<microseconds>(t2 - t1).count();
        msleep(DELAY1);
    }
    return safe_avr(dt);
}

template <typename FFT>
double laptime2(int LOOPS, int TRIES, const FFT& fft, complex_t *x)
{
    using namespace std::chrono;
    std::vector<counter_t> dt(TRIES);
    for (int i = 0; i < TRIES; i++) {
        const system_clock::time_point t1 = system_clock::now();
        for (int j = 0; j < LOOPS; j++) {
            fft.fwd(x);
            fft.inv(x);
        }
        const system_clock::time_point t2 = system_clock::now();
        dt[i] = duration_cast<microseconds>(t2 - t1).count();
        msleep(DELAY1);
    }
    return safe_avr(dt);
}
#endif // __cplusplus < 201103L

int main(int argc, char *argv[]) try
{
    const int n_min  = argc >= 2 ? atoi(argv[1]) : 1;
    const int n_max  = argc >= 3 ? atoi(argv[2]) : (argc == 2 ? n_min : 22);
    const int N_max  = 1 << n_max;
    const int nN_max = (std::max)(22*(1<<22), n_max*N_max);
    if (n_min < 1 || 24 < n_min) throw "argv[1] must be 1..24";
    if (n_max < 1 || 24 < n_max) throw "argv[2] must be 1..24";
    if (n_min > n_max) throw "argv must be argv[1] <= argv[2]";

#ifndef DO_SINGLE_THREAD
    fftw_init_threads();
#endif
    setbuf(stdout, NULL);
    printf("------+-----------+-----------------+-----------------+-----------------+---\n");
    printf("length|FFTW3[usec]|   OOURA   [usec]| SimpleFFT [usec]|   OTFFT   [usec]|err\n");
    printf("------+-----------+-----------------+-----------------+-----------------+---\n");
    complex_t* x0 = (complex_t*) simd_malloc(N_max*sizeof(complex_t));
    complex_t* x1 = (complex_t*) simd_malloc(N_max*sizeof(complex_t));
    complex_t* x2 = (complex_t*) simd_malloc(N_max*sizeof(complex_t));
    complex_t* x3 = (complex_t*) simd_malloc(N_max*sizeof(complex_t));
    complex_t* x4 = (complex_t*) simd_malloc(N_max*sizeof(complex_t));
    double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    for (int n = n_min; n <= n_max; n++) {
        const int N = 1 << n; // FFT する系列の長さ
        const int LOOPS = nN_max/(n*N); // 計測１回分のループ回数
        const int TRIES = (std::min)(70, n*FACTOR); // 計測の試行回数
        double lap, lap1;

        //// サンプル値設定 ////
        for (int p = 0; p < N; p++) {
            const double t = double(p)/N;
            x0[p].Re = 10 * cos(3*2*M_PI*t*t);
            x0[p].Im = 10 * sin(3*2*M_PI*t*t);
            x1[p] = x2[p] = x3[p] = x4[p] = x0[p];
        }
        msleep(DELAY2);

        printf("2^(%2d)|", n);

        //// FFTW3 ////
        CppFFTW3::Plan<FORWARD,1,1> fwd_plan(N, x1);
        CppFFTW3::Plan<INVERSE,0,0> inv_plan(N, x1);
        lap = laptime1(LOOPS, TRIES, fwd_plan, inv_plan);
        sum1 += lap*n/n_max;
        printf("%11.2f|", lap/LOOPS);
        msleep(DELAY2);
        lap1 = lap;

        //// OOURA FFT ////
        OOURA::FFT ooura_fft(N);
        lap = laptime2(LOOPS, TRIES, ooura_fft, x2);
        sum2 += lap*n/n_max;
        printf("%11.2f(%3.0f%%)|", lap/LOOPS, 100*lap/lap1);
        msleep(DELAY2);

        //// Simple FFT ////
        OTFFT::speedup_magic();
        SimpleFFT::FFT simple_fft(N);
        lap = laptime2(LOOPS, TRIES, simple_fft, x3);
        sum3 += lap*n/n_max;
        printf("%11.2f(%3.0f%%)|", lap/LOOPS, 100*lap/lap1);
        msleep(DELAY2);

        //// OTFFT ////
        OTFFT::speedup_magic();
        OTFFT::FFT otfft(N);
        lap = laptime2(LOOPS, TRIES, otfft, x4);
        sum4 += lap*n/n_max;
        printf("%11.2f(%3.0f%%)|", lap/LOOPS, 100*lap/lap1);
        msleep(DELAY2);

        //// エラーチェック(２乗誤差が大きすぎないか) ////
        double err = 0;
        for (int p = 0; p < N; p++) x1[p] = x2[p] = x0[p];
        fwd_plan();
        otfft.fwd(x2);
        for (int k = 0; k < N; k++) {
            const complex_t d = x2[k] - x1[k];
            err += Re(d*conj(d));
        }
        for (int p = 0; p < N; p++) x1[p] = x2[p] = x0[p];
        inv_plan();
        otfft.inv(x2);
        for (int k = 0; k < N; k++) {
            const complex_t d = x2[k] - x1[k];
            err += Re(d*conj(d));
        }
        if (err == 0)
            printf(" -\n");
        else if (err > 0)
            printf("%3.0f\n", log10(err));
        else
            printf("NG\n"); // ありえない
    }
    simd_free(x4);
    simd_free(x3);
    simd_free(x2);
    simd_free(x1);
    simd_free(x0);
    printf("------+-----------+-----------------+-----------------+-----------------+---\n");
    printf(" cost |");
    printf("%11.2f|",            sum1/(n_max-n_min+1));
    printf("%11.2f(%3.0f%%)|",   sum2/(n_max-n_min+1), 100*sum2/sum1);
    printf("%11.2f(%3.0f%%)|",   sum3/(n_max-n_min+1), 100*sum3/sum1);
    printf("%11.2f(%3.0f%%)|\n", sum4/(n_max-n_min+1), 100*sum4/sum1);
    printf("------+-----------+-----------------+-----------------+-----------------+---\n");
#ifndef DO_SINGLE_THREAD
    fftw_cleanup_threads();
#endif

    return 0;
}
catch (std::bad_alloc&) { fprintf(stderr, "\n""not enough memory!!\n"); }
catch (const char *message) { fprintf(stderr, "%s\n", message); }
