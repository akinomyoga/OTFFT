/******************************************************************************
*  FFT Tuning Command Version 4.0
******************************************************************************/

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <limits>
#include "otfft/otfft_misc.h"
#include "otfft_difavx.h"
#include "otfft_ditavx.h"
#include "otfft_difavx8.h"
#include "otfft_ditavx8.h"
#include "otfft_sixstep.h"
#include "stopwatch.h"
using namespace OTFFT_MISC;

#define TRIES  5
#define DELAY1 1
#define DELAY2 100

template <class FFT>
double laptime(const FFT& fft, int LOOPS, complex_t* x, complex_t* y)
{
    counter_t dt = (std::numeric_limits<counter_t>::max)();
    for (int i = 0; i < TRIES; i++) {
        const counter_t t1 = get_counter();
        for (int j = 0; j < LOOPS; j++) {
            fft.fwd(x, y);
            fft.inv(x, y);
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

    setbuf(stdout, NULL);
    FILE* fp1 = fopen("otfft_setup.h", "w");
    FILE* fp2 = fopen("otfft_fwd.h", "w");
    FILE* fp3 = fopen("otfft_inv.h", "w");
    FILE* fp4 = fopen("otfft_fwd0.h", "w");
    FILE* fp5 = fopen("otfft_invn.h", "w");
    fprintf(fp1, "switch (log_N) {\n");
    fprintf(fp2, "switch (log_N) {\n");
    fprintf(fp3, "switch (log_N) {\n");
    fprintf(fp4, "switch (log_N) {\n");
    fprintf(fp5, "switch (log_N) {\n");
    complex_t* x = (complex_t*) simd_malloc(N_max*sizeof(complex_t));
    complex_t* y = (complex_t*) simd_malloc(N_max*sizeof(complex_t));
    for (int n = 1; n <= n_max; n++) {
        const int N = 1 << n;
        const int LOOPS = static_cast<int>(rint(Nn_max/double(N*n)));
        counter_t cp1, cp2;
        double lap, tmp;
        OTFFT_DIFAVX::FFT0  fft1(N);
        OTFFT_DITAVX::FFT0  fft2(N);
        OTFFT_DIFAVX8::FFT0 fft3(N);
        OTFFT_DITAVX8::FFT0 fft4(N);
        OTFFT_Sixstep::FFT0 fft5(N);
        OTFFT_Sixstep::FFT1 fft6(N);
        int fft_num = 1;

        for (int p = 0; p < N; p++) {
            const double t = double(p) / N;
            x[p].Re = cos(5*(2*M_PI/N)*t*t);
            x[p].Im = sin(5*(2*M_PI/N)*t*t);
        }

        msleep(DELAY2);

        OTFFT_MISC::speedup_magic();
        lap = laptime(fft1, LOOPS, x, y);

        msleep(DELAY2);

        OTFFT_MISC::speedup_magic();
        tmp = laptime(fft2, LOOPS, x, y);
        if (tmp < lap) { lap = tmp; fft_num = 2; }

        msleep(DELAY2);

        OTFFT_MISC::speedup_magic();
        tmp = laptime(fft3, LOOPS, x, y);
        if (tmp < lap) { lap = tmp; fft_num = 3; }

        msleep(DELAY2);

        OTFFT_MISC::speedup_magic();
        tmp = laptime(fft4, LOOPS, x, y);
        if (tmp < lap) { lap = tmp; fft_num = 4; }

        msleep(DELAY2);

        OTFFT_MISC::speedup_magic();
        tmp = laptime(fft5, LOOPS, x, y);
        if (tmp < lap) { lap = tmp; fft_num = 5; }

        msleep(DELAY2);

        OTFFT_MISC::speedup_magic();
        tmp = laptime(fft6, LOOPS, x, y);
        if (tmp < lap) { lap = tmp; fft_num = 6; }

        fprintf(fp1, "case %2d: fft%d->setup2(log_N); break;\n", n, fft_num);
        fprintf(fp2, "case %2d: fft%d->fwd(x, y); break;\n", n, fft_num);
        fprintf(fp3, "case %2d: fft%d->inv(x, y); break;\n", n, fft_num);
        fprintf(fp4, "case %2d: fft%d->fwd0(x, y); break;\n", n, fft_num);
        fprintf(fp5, "case %2d: fft%d->invn(x, y); break;\n", n, fft_num);
        printf("2^(%2d): fft%d\n", n, fft_num);
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
