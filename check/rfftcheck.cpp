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

void print_err1(const char* name, int N, const complex_t* x, const complex_t* y)
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

void print_err2(const char* name, int N, const double* x, const double* y)
{
    double err = 0;
    for (int k = 0; k < N; k++) {
        const double d = y[k] - x[k];
        err += d*d;
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
        double*    x0 = (double*) simd_malloc(N*sizeof(double));
        double*    x  = (double*) simd_malloc(N*sizeof(double));
        complex_t* y  = (complex_t*) simd_malloc(N*sizeof(complex_t));
        complex_t* z  = (complex_t*) simd_malloc(N*sizeof(complex_t));
        for (int p = 0; p < N; p++) x0[p] = rand() % 100 - 50;
        SimpleFFT::FFT simple_fft(N);
        OTFFT::RFFT otfft(N);
        printf("[2^(%02d)]\n", n);

        for (int p = 0; p < N; p++) y[p] = x0[p];
        simple_fft.fwd0(y);
        otfft.fwd0(x0, z);
        print_err1("fwd0", N, y, z);
        otfft.invn(z, x);
        print_err2("invn", N, x0, x);

        for (int p = 0; p < N; p++) y[p] = x0[p];
        simple_fft.fwd(y);
        otfft.fwd(x0, z);
        print_err1("fwd", N, y, z);
        otfft.inv(z, x);
        print_err2("inv", N, x0, x);

        for (int p = 0; p < N; p++) y[p] = x0[p];
        simple_fft.fwdn(y);
        otfft.fwdn(x0, z);
        print_err1("fwdn", N, y, z);
        otfft.inv0(z, x);
        print_err2("inv0", N, x0, x);

        simd_free(z);
        simd_free(y);
        simd_free(x);
        simd_free(x0);
    }
    return 0;
}
catch (std::bad_alloc&) { fprintf(stderr, "\n""not enough memory!!\n"); }
