/******************************************************************************
*  OTFFT Cooley-Tukey AVX Version 4.0
******************************************************************************/

#ifndef otfft_ctavx_h
#define otfft_ctavx_h

#include <cmath>
#include <algorithm>
#include "otfft_misc.h"

namespace OTFFT_CTAVX { ///////////////////////////////////////////////////////

using namespace OTFFT_MISC;

const int OMP_THRESHOLD = 1<<15;

template <int n, int s> struct fwd0but;
template <int n, int s> struct inv0but;

///////////////////////////////////////////////////////////////////////////////

template <> struct fwd0but<2,1>
{
    inline void operator()(complex_vector x, const_complex_vector W) const
    {
        const xmm a = getpz(x[0]);
        const xmm b = getpz(x[1]);
        setpz(x[0], addpz(a, b));
        setpz(x[1], subpz(a, b));
    }
};

template <> struct inv0but<2,1>
{
    inline void operator()(complex_vector x, const_complex_vector W) const
    {
        fwd0but<2,1>()(x, W);
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int s> struct fwd0but<2,s>
{
    static const int N = 2*s;

    void operator()(complex_vector x, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD) {
            for (int q = 0; q < N; q += 2) {
                complex_vector xq = x + q;
                const xmm a = getpz(xq[0]);
                const xmm b = getpz(xq[1]);
                setpz(xq[0], addpz(a, b));
                setpz(xq[1], subpz(a, b));
            }
        }
        else {
            #pragma omp for schedule(static) nowait
            for (int q = 0; q < N; q += 2) {
                complex_vector xq = x + q;
                const xmm a = getpz(xq[0]);
                const xmm b = getpz(xq[1]);
                setpz(xq[0], addpz(a, b));
                setpz(xq[1], subpz(a, b));
            }
        }
    }
};

template <int s> struct inv0but<2,s>
{
    inline void operator()(complex_vector x, const_complex_vector W) const
    {
        fwd0but<2,s>()(x, W);
    }
};

///////////////////////////////////////////////////////////////////////////////

template <> struct fwd0but<4,1>
{
    inline void operator()(complex_vector x, const_complex_vector W) const
    {
        const xmm a = getpz(x[0]);
        const xmm b = getpz(x[1]);
        const xmm c = getpz(x[2]);
        const xmm d = getpz(x[3]);
        const xmm  apc =      addpz(a, c);
        const xmm  amc =      subpz(a, c);
        const xmm  bpd =      addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(x[0], addpz(apc,  bpd));
        setpz(x[1], subpz(apc,  bpd));
        setpz(x[2], subpz(amc, jbmd));
        setpz(x[3], addpz(amc, jbmd));
    }
};

template <> struct inv0but<4,1>
{
    inline void operator()(complex_vector x, const_complex_vector W) const
    {
        const xmm a = getpz(x[0]);
        const xmm b = getpz(x[1]);
        const xmm c = getpz(x[2]);
        const xmm d = getpz(x[3]);
        const xmm  apc =      addpz(a, c);
        const xmm  amc =      subpz(a, c);
        const xmm  bpd =      addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(x[0], addpz(apc,  bpd));
        setpz(x[1], subpz(apc,  bpd));
        setpz(x[2], addpz(amc, jbmd));
        setpz(x[3], subpz(amc, jbmd));
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int s> struct fwd0but<4,s>
{
    static const int N  = 4*s;

    void operator()(complex_vector x, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD) {
            for (int q = 0; q < N; q += 4) {
                complex_t* const xq = x + q;
                const xmm a = getpz(xq[0]);
                const xmm b = getpz(xq[1]);
                const xmm c = getpz(xq[2]);
                const xmm d = getpz(xq[3]);
                const xmm  apc =      addpz(a, c);
                const xmm  amc =      subpz(a, c);
                const xmm  bpd =      addpz(b, d);
                const xmm jbmd = jxpz(subpz(b, d));
                setpz(xq[0], addpz(apc,  bpd));
                setpz(xq[1], subpz(apc,  bpd));
                setpz(xq[2], subpz(amc, jbmd));
                setpz(xq[3], addpz(amc, jbmd));
            }
        }
        else {
            #pragma omp for schedule(static) nowait
            for (int q = 0; q < N; q += 4) {
                complex_t* const xq = x + q;
                const xmm a = getpz(xq[0]);
                const xmm b = getpz(xq[1]);
                const xmm c = getpz(xq[2]);
                const xmm d = getpz(xq[3]);
                const xmm  apc =      addpz(a, c);
                const xmm  amc =      subpz(a, c);
                const xmm  bpd =      addpz(b, d);
                const xmm jbmd = jxpz(subpz(b, d));
                setpz(xq[0], addpz(apc,  bpd));
                setpz(xq[1], subpz(apc,  bpd));
                setpz(xq[2], subpz(amc, jbmd));
                setpz(xq[3], addpz(amc, jbmd));
            }
        }
    }
};

template <int s> struct inv0but<4,s>
{
    static const int N = 4*s;

    void operator()(complex_vector x, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD) {
            for (int q = 0; q < N; q += 4) {
                complex_t* const xq = x + q;
                const xmm a = getpz(xq[0]);
                const xmm b = getpz(xq[1]);
                const xmm c = getpz(xq[2]);
                const xmm d = getpz(xq[3]);
                const xmm  apc =      addpz(a, c);
                const xmm  amc =      subpz(a, c);
                const xmm  bpd =      addpz(b, d);
                const xmm jbmd = jxpz(subpz(b, d));
                setpz(xq[0], addpz(apc,  bpd));
                setpz(xq[1], subpz(apc,  bpd));
                setpz(xq[2], addpz(amc, jbmd));
                setpz(xq[3], subpz(amc, jbmd));
            }
        }
        else {
            #pragma omp for schedule(static) nowait
            for (int q = 0; q < N; q += 4) {
                complex_t* const xq = x + q;
                const xmm a = getpz(xq[0]);
                const xmm b = getpz(xq[1]);
                const xmm c = getpz(xq[2]);
                const xmm d = getpz(xq[3]);
                const xmm  apc =      addpz(a, c);
                const xmm  amc =      subpz(a, c);
                const xmm  bpd =      addpz(b, d);
                const xmm jbmd = jxpz(subpz(b, d));
                setpz(xq[0], addpz(apc,  bpd));
                setpz(xq[1], subpz(apc,  bpd));
                setpz(xq[2], addpz(amc, jbmd));
                setpz(xq[3], subpz(amc, jbmd));
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int N> struct fwd0but<N,1>
{
    static const int n = N;
    static const int n0 = 0;
    static const int n1 = n/4;
    static const int n2 = n/2;
    static const int n3 = n1 + n2;

    void operator()(complex_vector x, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < n1; p += 2) {
                const ymm w1p = cmplx2(W[1*p], W[1*p+1]);
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm a = getpz2(x+p+n0);
                const ymm b = getpz2(x+p+n1);
                const ymm c = getpz2(x+p+n2);
                const ymm d = getpz2(x+p+n3);
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(x+p+n0,             addpz2(apc,  bpd));
                setpz2(x+p+n1, mulpz2(w2p, subpz2(apc,  bpd)));
                setpz2(x+p+n2, mulpz2(w1p, subpz2(amc, jbmd)));
                setpz2(x+p+n3, mulpz2(w3p, addpz2(amc, jbmd)));
            }
            fwd0but<n/4,4>()(x, W);
        }
        else
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int p = 0; p < n1; p += 2) {
                const ymm w1p = cmplx2(W[1*p], W[1*p+1]);
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm a = getpz2(x+p+n0);
                const ymm b = getpz2(x+p+n1);
                const ymm c = getpz2(x+p+n2);
                const ymm d = getpz2(x+p+n3);
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(x+p+n0,             addpz2(apc,  bpd));
                setpz2(x+p+n1, mulpz2(w2p, subpz2(apc,  bpd)));
                setpz2(x+p+n2, mulpz2(w1p, subpz2(amc, jbmd)));
                setpz2(x+p+n3, mulpz2(w3p, addpz2(amc, jbmd)));
            }
            fwd0but<n/4,4>()(x, W);
        }
    }
};

template <int N> struct inv0but<N,1>
{
    static const int n = N;
    static const int n0 = 0;
    static const int n1 = n/4;
    static const int n2 = n/2;
    static const int n3 = n1 + n2;

    void operator()(complex_vector x, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < n1; p += 2) {
                const ymm w1p = cmplx2(W[N-1*p], W[N-1*p-1]);
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm a = getpz2(x+p+n0);
                const ymm b = getpz2(x+p+n1);
                const ymm c = getpz2(x+p+n2);
                const ymm d = getpz2(x+p+n3);
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(x+p+n0,             addpz2(apc,  bpd));
                setpz2(x+p+n1, mulpz2(w2p, subpz2(apc,  bpd)));
                setpz2(x+p+n2, mulpz2(w1p, addpz2(amc, jbmd)));
                setpz2(x+p+n3, mulpz2(w3p, subpz2(amc, jbmd)));
            }
            inv0but<n/4,4>()(x, W);
        }
        else
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int p = 0; p < n1; p += 2) {
                const ymm w1p = cmplx2(W[N-1*p], W[N-1*p-1]);
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm a = getpz2(x+p+n0);
                const ymm b = getpz2(x+p+n1);
                const ymm c = getpz2(x+p+n2);
                const ymm d = getpz2(x+p+n3);
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(x+p+n0,             addpz2(apc,  bpd));
                setpz2(x+p+n1, mulpz2(w2p, subpz2(apc,  bpd)));
                setpz2(x+p+n2, mulpz2(w1p, addpz2(amc, jbmd)));
                setpz2(x+p+n3, mulpz2(w3p, subpz2(amc, jbmd)));
            }
            inv0but<n/4,4>()(x, W);
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s> struct fwd0but
{
    static const int N = n*s;
    static const int n0 = 0;
    static const int n1 = n/4;
    static const int n2 = n/2;
    static const int n3 = n1 + n2;

    void operator()(complex_vector x, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD) {
            for (int q = 0; q < N; q += n) {
                complex_t* const xq = x + q;
                for (int p = 0; p < n1; p += 2) {
                    const int sp = s*p;
                    const ymm w1p = cmplx2(W[1*sp], W[1*sp+1*s]);
                    const ymm w2p = mulpz2(w1p, w1p);
                    const ymm w3p = mulpz2(w1p, w2p);
                    const ymm a = getpz2(xq+p+n0);
                    const ymm b = getpz2(xq+p+n1);
                    const ymm c = getpz2(xq+p+n2);
                    const ymm d = getpz2(xq+p+n3);
                    const ymm  apc =       addpz2(a, c);
                    const ymm  amc =       subpz2(a, c);
                    const ymm  bpd =       addpz2(b, d);
                    const ymm jbmd = jxpz2(subpz2(b, d));
                    setpz2(xq+p+n0,             addpz2(apc,  bpd));
                    setpz2(xq+p+n1, mulpz2(w2p, subpz2(apc,  bpd)));
                    setpz2(xq+p+n2, mulpz2(w1p, subpz2(amc, jbmd)));
                    setpz2(xq+p+n3, mulpz2(w3p, addpz2(amc, jbmd)));
                }
            }
        }
        else {
            #pragma omp for schedule(static)
            for (int q = 0; q < N; q += n) {
                complex_t* const xq = x + q;
                for (int p = 0; p < n1; p += 2) {
                    const int sp = s*p;
                    const ymm w1p = cmplx2(W[1*sp], W[1*sp+1*s]);
                    const ymm w2p = mulpz2(w1p, w1p);
                    const ymm w3p = mulpz2(w1p, w2p);
                    const ymm a = getpz2(xq+p+n0);
                    const ymm b = getpz2(xq+p+n1);
                    const ymm c = getpz2(xq+p+n2);
                    const ymm d = getpz2(xq+p+n3);
                    const ymm  apc =       addpz2(a, c);
                    const ymm  amc =       subpz2(a, c);
                    const ymm  bpd =       addpz2(b, d);
                    const ymm jbmd = jxpz2(subpz2(b, d));
                    setpz2(xq+p+n0,             addpz2(apc,  bpd));
                    setpz2(xq+p+n1, mulpz2(w2p, subpz2(apc,  bpd)));
                    setpz2(xq+p+n2, mulpz2(w1p, subpz2(amc, jbmd)));
                    setpz2(xq+p+n3, mulpz2(w3p, addpz2(amc, jbmd)));
                }
            }
        }
        fwd0but<n/4,4*s>()(x, W);
    }
};

template <int n, int s> struct inv0but
{
    static const int N = n*s;
    static const int n0 = 0;
    static const int n1 = n/4;
    static const int n2 = n/2;
    static const int n3 = n1 + n2;

    void operator()(complex_vector x, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD) {
            for (int q = 0; q < N; q += n) {
                complex_t* const xq = x + q;
                for (int p = 0; p < n1; p += 2) {
                    const int sp = s*p;
                    const ymm w1p = cmplx2(W[N-1*sp], W[N-1*sp-1*s]);
                    const ymm w2p = mulpz2(w1p, w1p);
                    const ymm w3p = mulpz2(w1p, w2p);
                    const ymm a = getpz2(xq+p+n0);
                    const ymm b = getpz2(xq+p+n1);
                    const ymm c = getpz2(xq+p+n2);
                    const ymm d = getpz2(xq+p+n3);
                    const ymm  apc =       addpz2(a, c);
                    const ymm  amc =       subpz2(a, c);
                    const ymm  bpd =       addpz2(b, d);
                    const ymm jbmd = jxpz2(subpz2(b, d));
                    setpz2(xq+p+n0,             addpz2(apc,  bpd));
                    setpz2(xq+p+n1, mulpz2(w2p, subpz2(apc,  bpd)));
                    setpz2(xq+p+n2, mulpz2(w1p, addpz2(amc, jbmd)));
                    setpz2(xq+p+n3, mulpz2(w3p, subpz2(amc, jbmd)));
                }
            }
        }
        else {
            #pragma omp for schedule(static)
            for (int q = 0; q < N; q += n) {
                complex_t* const xq = x + q;
                for (int p = 0; p < n1; p += 2) {
                    const int sp = s*p;
                    const ymm w1p = cmplx2(W[N-1*sp], W[N-1*sp-1*s]);
                    const ymm w2p = mulpz2(w1p, w1p);
                    const ymm w3p = mulpz2(w1p, w2p);
                    const ymm a = getpz2(xq+p+n0);
                    const ymm b = getpz2(xq+p+n1);
                    const ymm c = getpz2(xq+p+n2);
                    const ymm d = getpz2(xq+p+n3);
                    const ymm  apc =       addpz2(a, c);
                    const ymm  amc =       subpz2(a, c);
                    const ymm  bpd =       addpz2(b, d);
                    const ymm jbmd = jxpz2(subpz2(b, d));
                    setpz2(xq+p+n0,             addpz2(apc,  bpd));
                    setpz2(xq+p+n1, mulpz2(w2p, subpz2(apc,  bpd)));
                    setpz2(xq+p+n2, mulpz2(w1p, addpz2(amc, jbmd)));
                    setpz2(xq+p+n3, mulpz2(w3p, subpz2(amc, jbmd)));
                }
            }
        }
        inv0but<n/4,4*s>()(x, W);
    }
};

///////////////////////////////////////////////////////////////////////////////

#include "otfft_ctavxn.h"

///////////////////////////////////////////////////////////////////////////////

struct FFT
{
    int N, log_N;
    simd_array<complex_t> weight;
    simd_array<int> table;
    complex_t* W;
    int* bitrev;

    FFT() : N(0), log_N(0), W(0), bitrev(0) {}
    FFT(int n) { setup(n); }

    void setup(int n)
    {
        for (log_N = 0; n > 1; n >>= 1) log_N++;
        setup2(log_N);
    }

    void setup2(int n)
    {
        log_N = n; N = 1 << n;
        weight.setup(N+1); W = &weight;
        init_W(N, W);
        table.setup(N); bitrev = &table;
        bitrev[0] = 0; bitrev[N-1] = N-1;
        for (int i = 0, j = 1; j < N-1; j++) {
            for (int k = N >> 1; k > (i ^= k); k >>= 1);
            bitrev[j] = i;
        }
    }

    void bitreverse(complex_vector x) const
    {
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < N; p++) {
                const int q = bitrev[p];
                if (p > q) std::swap(x[p], x[q]);
            }
        }
        else {
            #pragma omp parallel for schedule(static)
            for (int p = 0; p < N; p++) {
                const int q = bitrev[p];
                if (p > q) std::swap(x[p], x[q]);
            }
        }
    }

    inline void fwd0(complex_vector x) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: fwd0but<(1<< 1),1>()(x, W); break;
        case  2: fwd0but<(1<< 2),1>()(x, W); break;
        case  3: fwd0but<(1<< 3),1>()(x, W); break;
        case  4: fwd0but<(1<< 4),1>()(x, W); break;
        case  5: fwd0but<(1<< 5),1>()(x, W); break;
        case  6: fwd0but<(1<< 6),1>()(x, W); break;
        case  7: fwd0but<(1<< 7),1>()(x, W); break;
        case  8: fwd0but<(1<< 8),1>()(x, W); break;
        case  9: fwd0but<(1<< 9),1>()(x, W); break;
        case 10: fwd0but<(1<<10),1>()(x, W); break;
        case 11: fwd0but<(1<<11),1>()(x, W); break;
        case 12: fwd0but<(1<<12),1>()(x, W); break;
        case 13: fwd0but<(1<<13),1>()(x, W); break;
        case 14: fwd0but<(1<<14),1>()(x, W); break;
        case 15: fwd0but<(1<<15),1>()(x, W); break;
        case 16: fwd0but<(1<<16),1>()(x, W); break;
        case 17: fwd0but<(1<<17),1>()(x, W); break;
        case 18: fwd0but<(1<<18),1>()(x, W); break;
        case 19: fwd0but<(1<<19),1>()(x, W); break;
        case 20: fwd0but<(1<<20),1>()(x, W); break;
        case 21: fwd0but<(1<<21),1>()(x, W); break;
        case 22: fwd0but<(1<<22),1>()(x, W); break;
        case 23: fwd0but<(1<<23),1>()(x, W); break;
        case 24: fwd0but<(1<<24),1>()(x, W); break;
        }
        bitreverse(x);
    }

    inline void fwd(complex_vector x) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: fwdnbut<(1<< 1),1>()(x, W); break;
        case  2: fwdnbut<(1<< 2),1>()(x, W); break;
        case  3: fwdnbut<(1<< 3),1>()(x, W); break;
        case  4: fwdnbut<(1<< 4),1>()(x, W); break;
        case  5: fwdnbut<(1<< 5),1>()(x, W); break;
        case  6: fwdnbut<(1<< 6),1>()(x, W); break;
        case  7: fwdnbut<(1<< 7),1>()(x, W); break;
        case  8: fwdnbut<(1<< 8),1>()(x, W); break;
        case  9: fwdnbut<(1<< 9),1>()(x, W); break;
        case 10: fwdnbut<(1<<10),1>()(x, W); break;
        case 11: fwdnbut<(1<<11),1>()(x, W); break;
        case 12: fwdnbut<(1<<12),1>()(x, W); break;
        case 13: fwdnbut<(1<<13),1>()(x, W); break;
        case 14: fwdnbut<(1<<14),1>()(x, W); break;
        case 15: fwdnbut<(1<<15),1>()(x, W); break;
        case 16: fwdnbut<(1<<16),1>()(x, W); break;
        case 17: fwdnbut<(1<<17),1>()(x, W); break;
        case 18: fwdnbut<(1<<18),1>()(x, W); break;
        case 19: fwdnbut<(1<<19),1>()(x, W); break;
        case 20: fwdnbut<(1<<20),1>()(x, W); break;
        case 21: fwdnbut<(1<<21),1>()(x, W); break;
        case 22: fwdnbut<(1<<22),1>()(x, W); break;
        case 23: fwdnbut<(1<<23),1>()(x, W); break;
        case 24: fwdnbut<(1<<24),1>()(x, W); break;
        }
        bitreverse(x);
    }

    inline void fwdn(complex_vector x) const { fwd(x); }

    inline void inv0(complex_vector x) const { inv(x); }

    inline void inv(complex_vector x) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: inv0but<(1<< 1),1>()(x, W); break;
        case  2: inv0but<(1<< 2),1>()(x, W); break;
        case  3: inv0but<(1<< 3),1>()(x, W); break;
        case  4: inv0but<(1<< 4),1>()(x, W); break;
        case  5: inv0but<(1<< 5),1>()(x, W); break;
        case  6: inv0but<(1<< 6),1>()(x, W); break;
        case  7: inv0but<(1<< 7),1>()(x, W); break;
        case  8: inv0but<(1<< 8),1>()(x, W); break;
        case  9: inv0but<(1<< 9),1>()(x, W); break;
        case 10: inv0but<(1<<10),1>()(x, W); break;
        case 11: inv0but<(1<<11),1>()(x, W); break;
        case 12: inv0but<(1<<12),1>()(x, W); break;
        case 13: inv0but<(1<<13),1>()(x, W); break;
        case 14: inv0but<(1<<14),1>()(x, W); break;
        case 15: inv0but<(1<<15),1>()(x, W); break;
        case 16: inv0but<(1<<16),1>()(x, W); break;
        case 17: inv0but<(1<<17),1>()(x, W); break;
        case 18: inv0but<(1<<18),1>()(x, W); break;
        case 19: inv0but<(1<<19),1>()(x, W); break;
        case 20: inv0but<(1<<20),1>()(x, W); break;
        case 21: inv0but<(1<<21),1>()(x, W); break;
        case 22: inv0but<(1<<22),1>()(x, W); break;
        case 23: inv0but<(1<<23),1>()(x, W); break;
        case 24: inv0but<(1<<24),1>()(x, W); break;
        }
        bitreverse(x);
    }

    inline void invn(complex_vector x) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: invnbut<(1<< 1),1>()(x, W); break;
        case  2: invnbut<(1<< 2),1>()(x, W); break;
        case  3: invnbut<(1<< 3),1>()(x, W); break;
        case  4: invnbut<(1<< 4),1>()(x, W); break;
        case  5: invnbut<(1<< 5),1>()(x, W); break;
        case  6: invnbut<(1<< 6),1>()(x, W); break;
        case  7: invnbut<(1<< 7),1>()(x, W); break;
        case  8: invnbut<(1<< 8),1>()(x, W); break;
        case  9: invnbut<(1<< 9),1>()(x, W); break;
        case 10: invnbut<(1<<10),1>()(x, W); break;
        case 11: invnbut<(1<<11),1>()(x, W); break;
        case 12: invnbut<(1<<12),1>()(x, W); break;
        case 13: invnbut<(1<<13),1>()(x, W); break;
        case 14: invnbut<(1<<14),1>()(x, W); break;
        case 15: invnbut<(1<<15),1>()(x, W); break;
        case 16: invnbut<(1<<16),1>()(x, W); break;
        case 17: invnbut<(1<<17),1>()(x, W); break;
        case 18: invnbut<(1<<18),1>()(x, W); break;
        case 19: invnbut<(1<<19),1>()(x, W); break;
        case 20: invnbut<(1<<20),1>()(x, W); break;
        case 21: invnbut<(1<<21),1>()(x, W); break;
        case 22: invnbut<(1<<22),1>()(x, W); break;
        case 23: invnbut<(1<<23),1>()(x, W); break;
        case 24: invnbut<(1<<24),1>()(x, W); break;
        }
        bitreverse(x);
    }
};

void speedup_magic(int N = 1 << 18)
{
    const double theta0 = 2*M_PI/N;
    volatile double sum = 0;
    for (int p = 0; p < N; p++) {
        sum += cos(p * theta0);
    }
}

} /////////////////////////////////////////////////////////////////////////////

#endif // otfft_ctavx_h
