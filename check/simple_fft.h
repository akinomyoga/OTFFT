/******************************************************************************
*  Simple FFT (Cooley Tukey Radix-4)
*
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef simple_fft_h
#define simple_fft_h

#include <utility>
#include "otfft/otfft_misc.h"

namespace SimpleFFT { /////////////////////////////////////////////////////////

using namespace OTFFT_MISC;

const int OMP_THRESHOLD = 1<<15;

void fwdbut(int N, complex_t* const x, const complex_t* const W)
{
    int n = N;
    for (int s = 1; n > 2; n /= 4, s *= 4) {
        const int n0 = 0;
        const int n1 = n / 4;
        const int n2 = n / 2;
        const int n3 = n1 + n2;
        if (N < OMP_THRESHOLD) {
            for (int q = 0; q < N; q += n) {
                complex_t* const xq = x + q;
                for (int p = 0; p < n1; p++) {
                    const int sp = s*p;
                    const complex_t w1p = W[1*sp];
                    const complex_t w2p = W[2*sp];
                    const complex_t w3p = W[3*sp];
                    const complex_t a = xq[p+n0];
                    const complex_t b = xq[p+n1];
                    const complex_t c = xq[p+n2];
                    const complex_t d = xq[p+n3];
                    const complex_t  apc =    a + c;
                    const complex_t  amc =    a - c;
                    const complex_t  bpd =    b + d;
                    const complex_t jbmd = jx(b - d);
                    xq[p+n0] =  apc +  bpd;
                    xq[p+n1] = (apc -  bpd) * w2p;
                    xq[p+n2] = (amc - jbmd) * w1p;
                    xq[p+n3] = (amc + jbmd) * w3p;
                }
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int q = 0; q < N; q += n) {
                complex_t* const xq = x + q;
                for (int p = 0; p < n1; p++) {
                    const int sp = s*p;
                    const complex_t w1p = W[1*sp];
                    const complex_t w2p = W[2*sp];
                    const complex_t w3p = W[3*sp];
                    const complex_t a = xq[p+n0];
                    const complex_t b = xq[p+n1];
                    const complex_t c = xq[p+n2];
                    const complex_t d = xq[p+n3];
                    const complex_t  apc =    a + c;
                    const complex_t  amc =    a - c;
                    const complex_t  bpd =    b + d;
                    const complex_t jbmd = jx(b - d);
                    xq[p+n0] =  apc +  bpd;
                    xq[p+n1] = (apc -  bpd) * w2p;
                    xq[p+n2] = (amc - jbmd) * w1p;
                    xq[p+n3] = (amc + jbmd) * w3p;
                }
            }
        }
    }
    if (n == 2) {
        if (N < OMP_THRESHOLD) {
            for (int q = 0; q < N; q += 2) {
                complex_t* const xq = x + q;
                const complex_t a = xq[0];
                const complex_t b = xq[1];
                xq[0] = a + b;
                xq[1] = a - b;
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int q = 0; q < N; q += 2) {
                complex_t* const xq = x + q;
                const complex_t a = xq[0];
                const complex_t b = xq[1];
                xq[0] = a + b;
                xq[1] = a - b;
            }
        }
    }
}

void invbut(int N, complex_t* const x, const complex_t* const W)
{
    int n = N;
    for (int s = 1; n > 2; n /= 4, s *= 4) {
        const int n0 = 0;
        const int n1 = n / 4;
        const int n2 = n / 2;
        const int n3 = n1 + n2;
        if (N < OMP_THRESHOLD) {
            for (int q = 0; q < N; q += n) {
                complex_t* const xq = x + q;
                for (int p = 0; p < n1; p++) {
                    const int sp = s*p;
                    const complex_t w1p = W[N-1*sp];
                    const complex_t w2p = W[N-2*sp];
                    const complex_t w3p = W[N-3*sp];
                    const complex_t a = xq[p+n0];
                    const complex_t b = xq[p+n1];
                    const complex_t c = xq[p+n2];
                    const complex_t d = xq[p+n3];
                    const complex_t  apc =    a + c;
                    const complex_t  amc =    a - c;
                    const complex_t  bpd =    b + d;
                    const complex_t jbmd = jx(b - d);
                    xq[p+n0] =  apc +  bpd;
                    xq[p+n1] = (apc -  bpd) * w2p;
                    xq[p+n2] = (amc + jbmd) * w1p;
                    xq[p+n3] = (amc - jbmd) * w3p;
                }
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int q = 0; q < N; q += n) {
                complex_t* const xq = x + q;
                for (int p = 0; p < n1; p++) {
                    const int sp = s*p;
                    const complex_t w1p = W[N-1*sp];
                    const complex_t w2p = W[N-2*sp];
                    const complex_t w3p = W[N-3*sp];
                    const complex_t a = xq[p+n0];
                    const complex_t b = xq[p+n1];
                    const complex_t c = xq[p+n2];
                    const complex_t d = xq[p+n3];
                    const complex_t  apc =    a + c;
                    const complex_t  amc =    a - c;
                    const complex_t  bpd =    b + d;
                    const complex_t jbmd = jx(b - d);
                    xq[p+n0] =  apc +  bpd;
                    xq[p+n1] = (apc -  bpd) * w2p;
                    xq[p+n2] = (amc + jbmd) * w1p;
                    xq[p+n3] = (amc - jbmd) * w3p;
                }
            }
        }
    }
    if (n == 2) {
        if (N < OMP_THRESHOLD) {
            for (int q = 0; q < N; q += 2) {
                complex_t* const xq = x + q;
                const complex_t a = xq[0];
                const complex_t b = xq[1];
                xq[0] = a + b;
                xq[1] = a - b;
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int q = 0; q < N; q += 2) {
                complex_t* const xq = x + q;
                const complex_t a = xq[0];
                const complex_t b = xq[1];
                xq[0] = a + b;
                xq[1] = a - b;
            }
        }
    }
}

struct FFT
{
    const int N;
    simd_array<complex_t> weight;
    simd_array<int> table;
    complex_t* const W;
    int* bitrev;

    FFT(int n) : N(n), weight(n+1), table(n), W(&weight), bitrev(&table)
    {
        init_W(N, W);
        bitrev[0] = 0; bitrev[N-1] = N-1;
        for (int i = 0, j = 1; j < N-1; j++) {
            for (int k = N >> 1; k > (i ^= k); k >>= 1);
            bitrev[j] = i;
        }
    }

    ///////////////////////////////////////////////////////////////////////////

    void fwd(complex_t* const x) const
    {
        const double rN = 1.0/N;
        fwdbut(N, x, W);
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < N; p++) {
                x[p] *= rN;
                const int q = bitrev[p];
                if (p > q) std::swap(x[p], x[q]);
            }
        }
        else
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 0; p < N; p++) {
                const int q = bitrev[p];
                if (p > q) std::swap(x[p], x[q]);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static) nowait
#endif
            for (int p = 0; p < N; p++) x[p] *= rN;
        }
    }

    void fwd0(complex_t* const x) const
    {
        fwdbut(N, x, W);
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < N; p++) {
                const int q = bitrev[p];
                if (p > q) std::swap(x[p], x[q]);
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int p = 0; p < N; p++) {
                const int q = bitrev[p];
                if (p > q) std::swap(x[p], x[q]);
            }
        }
    }

    void fwdn(complex_t* const x) const { fwd(x); }

    ///////////////////////////////////////////////////////////////////////////

    void inv(complex_t* const x) const
    {
        invbut(N, x, W);
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < N; p++) {
                const int q = bitrev[p];
                if (p > q) std::swap(x[p], x[q]);
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int p = 0; p < N; p++) {
                const int q = bitrev[p];
                if (p > q) std::swap(x[p], x[q]);
            }
        }
    }

    void inv0(complex_t* const x) const { inv(x); }

    void invn(complex_t* const x) const
    {
        const double rN = 1.0/N;
        invbut(N, x, W);
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < N; p++) {
                x[p] *= rN;
                const int q = bitrev[p];
                if (p > q) std::swap(x[p], x[q]);
            }
        }
        else
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 0; p < N; p++) {
                const int q = bitrev[p];
                if (p > q) std::swap(x[p], x[q]);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static) nowait
#endif
            for (int p = 0; p < N; p++) x[p] *= rN;
        }
    }
};

} /////////////////////////////////////////////////////////////////////////////

#endif // simple_fft_h
