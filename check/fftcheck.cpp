/******************************************************************************
*  FFT Consistency Check
******************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "simple_fft.h"
#include "otfft/otfft.h"
using OTFFT::complex_t;
using OTFFT::simd_malloc;
using OTFFT::simd_free;

void print_err(const char* name, int N, const complex_t* x, const complex_t* y)
{
    double err = 0;
    for (int k = 0; k < N; k++) {
        const complex_t d = y[k] - x[k];
        err += Re(d*conj(d));
    }
    printf("%-4s: ", name);
    if (err == 0)
        printf("---\n");
    else if (err > 0)
        printf("%3ld\n", lrint(log10(err)));
    else
        printf("ERROR\n");
}

int main() try
{
    srand(time(NULL));
    for (int n = 1; n <= 22; n++) {
        const int N = 1 << n;
        complex_t* x0 = (complex_t*) simd_malloc(N*sizeof(complex_t));
        complex_t* x  = (complex_t*) simd_malloc(N*sizeof(complex_t));
        complex_t* y  = (complex_t*) simd_malloc(N*sizeof(complex_t));
        for (int p = 0; p < N; p++) {
            x0[p].Re = rand() % 100 - 50;
            x0[p].Im = rand() % 100 - 50;
        }
        SimpleFFT::FFT simple_fft(N);
        OTFFT::FFT otfft(N);
        printf("[2^(%02d)]\n", n);

        for (int p = 0; p < N; p++) x[p] = y[p] = x0[p];
        simple_fft.fwd0(x);
        otfft.fwd0(y);
        print_err("fwd0", N, x, y);

        for (int p = 0; p < N; p++) x[p] = y[p] = x0[p];
        simple_fft.fwd(x);
        otfft.fwd(y);
        print_err("fwd", N, x, y);

        for (int p = 0; p < N; p++) x[p] = y[p] = x0[p];
        simple_fft.fwdn(x);
        otfft.fwdn(y);
        print_err("fwdn", N, x, y);

        for (int p = 0; p < N; p++) x[p] = y[p] = x0[p];
        simple_fft.inv0(x);
        otfft.inv0(y);
        print_err("inv0", N, x, y);

        for (int p = 0; p < N; p++) x[p] = y[p] = x0[p];
        simple_fft.inv(x);
        otfft.inv(y);
        print_err("inv", N, x, y);

        for (int p = 0; p < N; p++) x[p] = y[p] = x0[p];
        simple_fft.invn(x);
        otfft.invn(y);
        print_err("invn", N, x, y);

        simd_free(y);
        simd_free(x);
        simd_free(x0);
    }
    return 0;
}
catch (std::bad_alloc&) { fprintf(stderr, "\n""not enough memory!!\n"); }
