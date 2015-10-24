/******************************************************************************
*  FFT Tuning Command Version 5.3
******************************************************************************/

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include "otfft/otfft_misc.h"
#include "otfft_avxdif4.h"
#include "otfft_avxdit4.h"
#include "otfft_avxdif8.h"
#include "otfft_avxdit8.h"
#include "otfft_sixstep.h"
#include "otfft/stopwatch.h"
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

template <class FFT>
double laptime(int LOOPS, int TRIES, const FFT& fft, complex_t *x, complex_t *y)
{
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
    printf("%10.2f,", usec(sum)/n/LOOPS);
    return double(sum)/n;
}

int main() try
{
    static const int n_min  = 1;
    static const int n_max  = N_MAX;
    static const int N_max  = 1 << n_max;
    static const int Nn_max = N_max * n_max;

    setbuf(stdout, NULL);
    FILE* fp1 = fopen("otfft_setup.h", "w");
    FILE* fp2 = fopen("otfft_fwd.h",   "w");
    FILE* fp3 = fopen("otfft_inv.h",   "w");
    FILE* fp4 = fopen("otfft_fwd0.h",  "w");
    FILE* fp5 = fopen("otfft_invn.h",  "w");
    fprintf(fp1, "switch (log_N) {\ncase  0: break;\n");
    fprintf(fp2, "switch (log_N) {\ncase  0: break;\n");
    fprintf(fp3, "switch (log_N) {\ncase  0: break;\n");
    fprintf(fp4, "switch (log_N) {\ncase  0: break;\n");
    fprintf(fp5, "switch (log_N) {\ncase  0: break;\n");
    complex_vector x = (complex_vector) simd_malloc(N_max*sizeof(complex_t));
    complex_vector y = (complex_vector) simd_malloc(N_max*sizeof(complex_t));
    for (int n = n_min; n <= n_max; n++) {
        const int N = 1 << n;
        const int LOOPS = Nn_max/(N*n);
        const int TRIES = (std::min)(70, n*FACTOR);
        double lap, tmp;
        OTFFT_AVXDIF4::FFT0 fft1(N);
        OTFFT_AVXDIT4::FFT0 fft2(N);
        OTFFT_AVXDIF8::FFT0 fft3(N);
        OTFFT_AVXDIT8::FFT0 fft4(N);
        OTFFT_Sixstep::FFT0 fft5(N);
        //OTFFT_Sixstep::FFT1 fft6(N);
        int fft_num = 1;

        for (int p = 0; p < N; p++) {
            const double t = double(p) / N;
            x[p].Re = 10 * cos((2*M_PI/N)*t*t);
            x[p].Im = 10 * sin((2*M_PI/N)*t*t);
        }

        msleep(DELAY2);
        printf("2^(%2d)", n);

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

        /*
        tmp = laptime(LOOPS, TRIES, fft6, x, y);
        if (tmp < lap) { lap = tmp; fft_num = 6; }

        msleep(DELAY2);
        */

        fprintf(fp1, "case %2d: fft%d->setup2(log_N); break;\n", n, fft_num);
        fprintf(fp2, "case %2d: fft%d->fwd(x, y); break;\n",     n, fft_num);
        fprintf(fp3, "case %2d: fft%d->inv(x, y); break;\n",     n, fft_num);
        fprintf(fp4, "case %2d: fft%d->fwd0(x, y); break;\n",    n, fft_num);
        fprintf(fp5, "case %2d: fft%d->invn(x, y); break;\n",    n, fft_num);
        printf(" fft%d\n", fft_num);
    }
    simd_free(y);
    simd_free(x);
    fprintf(fp1, "default: fft5->setup2(log_N); break;\n");
    fprintf(fp2, "default: fft5->fwd(x, y); break;\n");
    fprintf(fp3, "default: fft5->inv(x, y); break;\n");
    fprintf(fp4, "default: fft5->fwd0(x, y); break;\n");
    fprintf(fp5, "default: fft5->invn(x, y); break;\n");
    fprintf(fp1, "}\n");
    fprintf(fp2, "}\n");
    fprintf(fp3, "}\n");
    fprintf(fp4, "}\n");
    fprintf(fp5, "}\n");
    fclose(fp5);
    fclose(fp4);
    fclose(fp3);
    fclose(fp2);
    fclose(fp1);

    return 0;
}
catch (std::bad_alloc&) { fprintf(stderr, "\n""not enough memory!!\n"); }
