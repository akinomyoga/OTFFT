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

void print_err1(const char* name, int N, const complex_t* x, const complex_t* y)
{
    if (N < (1<<15)) {
        complex_t* z  = (complex_t*) simd_malloc(N*sizeof(complex_t));
        for (int k = 0; k < N; k++) {
            z[k] = 0;
            for (int p = 0; p < N; p++) {
                const double theta = (2*M_PI/N)*k*p;
                z[k] += x[p]*complex_t(cos(theta), -sin(theta));
            }
        }
        double err = 0;
        for (int k = 0; k < N; k++) {
            const complex_t d = y[k] - z[k];
            err += Re(d*conj(d));
        }
        printf("%-4s: ", name);
        if (err == 0)
            printf("---\n");
        else if (err > 0)
            printf("%3ld\n", lrint(log10(err)));
        else
            printf("ERROR\n");
        simd_free(z);
    }
}

void print_err2(const char* name, int N, const complex_t* x, const complex_t* y)
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
        const int N = (1 << n) + 123;
        complex_t* x0 = (complex_t*) simd_malloc(N*sizeof(complex_t));
        complex_t* x  = (complex_t*) simd_malloc(N*sizeof(complex_t));
        for (int p = 0; p < N; p++) {
            x0[p].Re = rand() % 100 - 50;
            x0[p].Im = rand() % 100 - 50;
        }
        OTFFT::Bluestein bst(N);
        printf("[%2d]\n", N);

        for (int p = 0; p < N; p++) x[p] = x0[p];
        bst.fwd0(x);
        print_err1("fwd0", N, x0, x);
        bst.invn(x);
        print_err2("invn", N, x0, x);
        bst.fwd(x);
        bst.inv0(x);
        print_err2("inv0", N, x0, x);

        simd_free(x);
        simd_free(x0);
    }
    return 0;
}
catch (std::bad_alloc&) { fprintf(stderr, "\n""not enough memory!!\n"); }
