/******************************************************************************
*  FFT Consistency Check
******************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "otfft/otfft.h"
using OTFFT::complex_t;
using OTFFT::simd_malloc;
using OTFFT::simd_free;

void print_err1(const char* name, int N, const double* x, const double* y)
{
    if (N < (1<<15)) {
        double* z  = (double*) simd_malloc(N*sizeof(double));
        for (int k = 0; k < N; k++) {
            z[k] = 0;
            for (int p = 0; p < N; p++)
                z[k] += x[p]*cos(M_PI/N*k*(p+1.0/2));
        }
        double err = 0;
        for (int k = 0; k < N; k++) {
            const double d = y[k] - z[k];
            err += d*d;
        }
        printf("%-4s: ", name);
        if (err == 0)
            printf("---\n");
        else if (err > 0)
            printf("%3d\n", static_cast<int>(rint(log10(err))));
        else
            printf("ERROR\n");
        simd_free(z);
    }
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
        printf("%3d\n", static_cast<int>(rint(log10(err))));
    else
        printf("ERROR\n");
}

int main() try
{
    srand(time(NULL));
    for (int n = 1; n <= 22; n++) {
        const int N = 1 << n;
        double* x0 = (double*) simd_malloc(N*sizeof(double));
        double* x  = (double*) simd_malloc(N*sizeof(double));
        for (int p = 0; p < N; p++) x0[p] = rand() % 100 - 50;
        OTFFT::DCT dct(N);
        printf("[2^(%02d)]\n", n);

        for (int p = 0; p < N; p++) x[p] = x0[p];
        dct.fwd0(x);
        print_err1("fwd0", N, x0, x);
        dct.invn(x);
        print_err2("invn", N, x0, x);

        for (int p = 0; p < N; p++) x[p] = x0[p];
        dct.fwdn(x);
        dct.inv0(x);
        print_err2("inv0", N, x0, x);

        simd_free(x);
        simd_free(x0);
    }
    return 0;
}
catch (std::bad_alloc&) { fprintf(stderr, "\n""not enough memory!!\n"); }
