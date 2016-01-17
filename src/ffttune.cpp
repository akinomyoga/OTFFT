/******************************************************************************
*  FFT Tuning Command Version 6.4
*
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include "otfft/otfft_misc.h"
#include "otfft_avxdif4.h"
#include "otfft_avxdit4.h"
#include "otfft_avxdif8.h"
#include "otfft_avxdit8.h"
#include "otfft_sixstep.h"
#include "otfft_avxdif16.h"
#include "otfft_avxdit16.h"

#if __cplusplus < 201103L
#include "otfft/stopwatch.h"
#else
#include <chrono>
#include "otfft/msleep.h"
#endif

using namespace OTFFT_MISC;

#ifdef __AVX__
#define N_MAX 21
#define FACTOR 4
#else
#define N_MAX 22
#define FACTOR 3
#endif

#define DELAY1 1
#define DELAY2 100

#if __cplusplus < 201103L
template <class FFT>
double laptime(int LOOPS, int TRIES, const FFT& fft, complex_t *x, complex_t *y)
{
    using namespace std;
    counter_t sum = 0;
    std::vector<counter_t> dt(TRIES);
    for (int i = 0; i < TRIES; i++) {
        speedup_magic();
        const counter_t t1 = get_counter();
        for (int j = 0; j < LOOPS; j++) {
            fft.fwd(x, y);
            fft.inv(x, y);
        }
        const counter_t t2 = get_counter();
        sum += (dt[i] = t2 - t1);
        msleep(DELAY1);
    }
    const double m = double(sum)/TRIES;
    double sum_dd = 0;
    for (int i = 0; i < TRIES; i++) {
        const double d = dt[i] - m;
        sum_dd += d*d;
    }
    const double ss = sum_dd/TRIES;
    sum = 0;
    int n = 0;
    for (int i = 0; i < TRIES; i++) {
        const double d = dt[i] - m;
        if (d*d <= 2*ss) { sum += dt[i]; n++; }
    }
    cout << fixed << setw(9) << setprecision(2) << usec(sum)/n/LOOPS << flush;
    return double(sum)/n;
}
#else
template <class FFT>
double laptime(int LOOPS, int TRIES, const FFT& fft, complex_t *x, complex_t *y)
{
    using namespace std;
    using namespace std::chrono;
    typedef microseconds::rep counter_t;
    counter_t sum = 0;
    std::vector<counter_t> dt(TRIES);
    for (int i = 0; i < TRIES; i++) {
        speedup_magic();
        const system_clock::time_point t1 = system_clock::now();
        for (int j = 0; j < LOOPS; j++) {
            fft.fwd(x, y);
            fft.inv(x, y);
        }
        const system_clock::time_point t2 = system_clock::now();
        sum += (dt[i] = duration_cast<microseconds>(t2 - t1).count());
        msleep(DELAY1);
    }
    const double m = double(sum)/TRIES;
    double sum_dd = 0;
    for (int i = 0; i < TRIES; i++) {
        const double d = dt[i] - m;
        sum_dd += d*d;
    }
    const double ss = sum_dd/TRIES;
    sum = 0;
    int n = 0;
    for (int i = 0; i < TRIES; i++) {
        const double d = dt[i] - m;
        if (d*d <= 2*ss) { sum += dt[i]; n++; }
    }
    cout << fixed << setw(9) << setprecision(2) << double(sum)/n/LOOPS << flush;
    return double(sum)/n;
}
#endif // __cplusplus < 201103L

int main() try
{
    using namespace std;
    static const int n_min  = 1;
    static const int n_max  = N_MAX;
    static const int N_max  = 1 << n_max;
    static const int nN_max = n_max * N_max;

    ofstream fs1("otfft_setup.h");
    ofstream fs2("otfft_fwd.h");
    ofstream fs3("otfft_inv.h");
    ofstream fs4("otfft_fwd0.h");
    ofstream fs5("otfft_invn.h");
    fs1 << "switch (log_N) {\ncase  0: break;\n";
    fs2 << "switch (log_N) {\ncase  0: break;\n";
    fs3 << "switch (log_N) {\ncase  0: break;\n";
    fs4 << "switch (log_N) {\ncase  0: break;\n";
    fs5 << "switch (log_N) {\ncase  0: break;\n";
    complex_vector x = (complex_vector) simd_malloc(N_max*sizeof(complex_t));
    complex_vector y = (complex_vector) simd_malloc(N_max*sizeof(complex_t));
    for (int n = n_min; n <= n_max; n++) {
        const int N = 1 << n;
        const int LOOPS = nN_max/(n*N);
        const int TRIES = (std::min)(70, n*FACTOR);
        double lap, tmp;
        OTFFT_AVXDIF4::FFT0 fft1(N);
        OTFFT_AVXDIT4::FFT0 fft2(N);
        OTFFT_AVXDIF8::FFT0 fft3(N);
        OTFFT_AVXDIT8::FFT0 fft4(N);
        OTFFT_Sixstep::FFT0 fft5(N);
        OTFFT_AVXDIF16::FFT0 fft6(N);
        OTFFT_AVXDIT16::FFT0 fft7(N);
        int fft_num = 1;

        for (int p = 0; p < N; p++) {
            const double t = double(p) / N;
            x[p].Re = 10 * cos((2*M_PI/N)*t*t);
            x[p].Im = 10 * sin((2*M_PI/N)*t*t);
        }

        msleep(DELAY2);
        cout << "2^(" << setw(2) << n << ")" << flush;

        lap = laptime(LOOPS, TRIES, fft1, x, y);

        msleep(DELAY2);

        tmp = laptime(LOOPS, TRIES, fft2, x, y);
        if (tmp < lap) { lap = tmp; fft_num = 2; }

        msleep(DELAY2);

        tmp = laptime(LOOPS, TRIES, fft3, x, y);
        if (tmp < lap) { lap = tmp; fft_num = 3; }

        msleep(DELAY2);

        tmp = laptime(LOOPS, TRIES, fft4, x, y);
        if (tmp < lap) { lap = tmp; fft_num = 4; }

        msleep(DELAY2);

        tmp = laptime(LOOPS, TRIES, fft5, x, y);
        if (tmp < lap) { lap = tmp; fft_num = 5; }

        msleep(DELAY2);

        tmp = laptime(LOOPS, TRIES, fft6, x, y);
        if (tmp < lap) { lap = tmp; fft_num = 6; }

        msleep(DELAY2);

        tmp = laptime(LOOPS, TRIES, fft7, x, y);
        if (tmp < lap) { lap = tmp; fft_num = 7; }

        msleep(DELAY2);

        fs1 << "case " << setw(2) << n << ": fft" << fft_num << "->setup2(log_N); break;\n";
        fs2 << "case " << setw(2) << n << ": fft" << fft_num << "->fwd(x, y); break;\n";
        fs3 << "case " << setw(2) << n << ": fft" << fft_num << "->inv(x, y); break;\n";
        fs4 << "case " << setw(2) << n << ": fft" << fft_num << "->fwd0(x, y); break;\n";
        fs5 << "case " << setw(2) << n << ": fft" << fft_num << "->invn(x, y); break;\n";
        cout << " fft" << fft_num << endl;
    }
    simd_free(y);
    simd_free(x);
#ifndef DO_SINGLE_THREAD
    fs1 << "default: fft5->setup2(log_N); break;\n";
    fs2 << "default: fft5->fwd(x, y); break;\n";
    fs3 << "default: fft5->inv(x, y); break;\n";
    fs4 << "default: fft5->fwd0(x, y); break;\n";
    fs5 << "default: fft5->invn(x, y); break;\n";
#else
    fs1 << "default: fft4->setup2(log_N); break;\n";
    fs2 << "default: fft4->fwd(x, y); break;\n";
    fs3 << "default: fft4->inv(x, y); break;\n";
    fs4 << "default: fft4->fwd0(x, y); break;\n";
    fs5 << "default: fft4->invn(x, y); break;\n";
#endif
    fs1 << "}\n";
    fs2 << "}\n";
    fs3 << "}\n";
    fs4 << "}\n";
    fs5 << "}\n";

    return 0;
}
catch (std::bad_alloc&) { std::cerr << "\n""not enough memory!!\n"; }
