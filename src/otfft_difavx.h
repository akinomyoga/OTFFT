/******************************************************************************
*  OTFFT DIFAVX Version 4.0
******************************************************************************/

#ifndef otfft_difavx_h
#define otfft_difavx_h

#include <cmath>
#include "otfft/otfft_misc.h"

namespace OTFFT_DIFAVX { //////////////////////////////////////////////////////

using namespace OTFFT_MISC;

const int OMP_THRESHOLD = 1<<15;

template <int n, int s, int eo> struct fwd0but;
template <int n, int s, int eo> struct inv0but;

///////////////////////////////////////////////////////////////////////////////

template <> struct fwd0but<2,1,0>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        const xmm a = getpz(x[0]);
        const xmm b = getpz(x[1]);
        setpz(x[0], addpz(a, b));
        setpz(x[1], subpz(a, b));
    }
};

template <> struct fwd0but<2,1,1>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        const xmm a = getpz(x[0]);
        const xmm b = getpz(x[1]);
        setpz(y[0], addpz(a, b));
        setpz(y[1], subpz(a, b));
    }
};

template <> struct inv0but<2,1,0>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        fwd0but<2,1,0>()(x, y, W);
    }
};

template <> struct inv0but<2,1,1>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        fwd0but<2,1,1>()(x, y, W);
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int s> struct fwd0but<2,s,0>
{
    static const int N = 2*s;

    void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD) {
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                const ymm a = getpz2(xq+0);
                const ymm b = getpz2(xq+s);
                setpz2(xq+0, addpz2(a, b));
                setpz2(xq+s, subpz2(a, b));
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp for schedule(static) nowait
#endif
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                const ymm a = getpz2(xq+0);
                const ymm b = getpz2(xq+s);
                setpz2(xq+0, addpz2(a, b));
                setpz2(xq+s, subpz2(a, b));
            }
        }
    }
};

template <int s> struct fwd0but<2,s,1>
{
    static const int N = 2*s;

    void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD) {
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector yq = y + q;
                const ymm a = getpz2(xq+0);
                const ymm b = getpz2(xq+s);
                setpz2(yq+0, addpz2(a, b));
                setpz2(yq+s, subpz2(a, b));
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp for schedule(static) nowait
#endif
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector yq = y + q;
                const ymm a = getpz2(xq+0);
                const ymm b = getpz2(xq+s);
                setpz2(yq+0, addpz2(a, b));
                setpz2(yq+s, subpz2(a, b));
            }
        }
    }
};

template <int s> struct inv0but<2,s,0>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        fwd0but<2,s,0>()(x, y, W);
    }
};

template <int s> struct inv0but<2,s,1>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        fwd0but<2,s,1>()(x, y, W);
    }
};

///////////////////////////////////////////////////////////////////////////////

template <> struct fwd0but<4,1,0>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        const xmm a = getpz(x[0]);
        const xmm b = getpz(x[1]);
        const xmm c = getpz(x[2]);
        const xmm d = getpz(x[3]);
        const xmm  apc = addpz(a, c);
        const xmm  amc = subpz(a, c);
        const xmm  bpd = addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(x[0], addpz(apc,  bpd));
        setpz(x[1], subpz(amc, jbmd));
        setpz(x[2], subpz(apc,  bpd));
        setpz(x[3], addpz(amc, jbmd));
    }
};

template <> struct fwd0but<4,1,1>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        const xmm a = getpz(x[0]);
        const xmm b = getpz(x[1]);
        const xmm c = getpz(x[2]);
        const xmm d = getpz(x[3]);
        const xmm  apc = addpz(a, c);
        const xmm  amc = subpz(a, c);
        const xmm  bpd = addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(y[0], addpz(apc,  bpd));
        setpz(y[1], subpz(amc, jbmd));
        setpz(y[2], subpz(apc,  bpd));
        setpz(y[3], addpz(amc, jbmd));
    }
};

template <> struct inv0but<4,1,0>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        const xmm a = getpz(x[0]);
        const xmm b = getpz(x[1]);
        const xmm c = getpz(x[2]);
        const xmm d = getpz(x[3]);
        const xmm  apc = addpz(a, c);
        const xmm  amc = subpz(a, c);
        const xmm  bpd = addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(x[0], addpz(apc,  bpd));
        setpz(x[1], addpz(amc, jbmd));
        setpz(x[2], subpz(apc,  bpd));
        setpz(x[3], subpz(amc, jbmd));
    }
};

template <> struct inv0but<4,1,1>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        const xmm a = getpz(x[0]);
        const xmm b = getpz(x[1]);
        const xmm c = getpz(x[2]);
        const xmm d = getpz(x[3]);
        const xmm  apc = addpz(a, c);
        const xmm  amc = subpz(a, c);
        const xmm  bpd = addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(y[0], addpz(apc,  bpd));
        setpz(y[1], addpz(amc, jbmd));
        setpz(y[2], subpz(apc,  bpd));
        setpz(y[3], subpz(amc, jbmd));
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int s> struct fwd0but<4,s,0>
{
    static const int N = 4*s;

    void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD) {
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                const ymm a = getpz2(xq+s*0);
                const ymm b = getpz2(xq+s*1);
                const ymm c = getpz2(xq+s*2);
                const ymm d = getpz2(xq+s*3);
                const ymm  apc = addpz2(a, c);
                const ymm  amc = subpz2(a, c);
                const ymm  bpd = addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(xq+s*0, addpz2(apc,  bpd));
                setpz2(xq+s*1, subpz2(amc, jbmd));
                setpz2(xq+s*2, subpz2(apc,  bpd));
                setpz2(xq+s*3, addpz2(amc, jbmd));
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp for schedule(static) nowait
#endif
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                const ymm a = getpz2(xq+s*0);
                const ymm b = getpz2(xq+s*1);
                const ymm c = getpz2(xq+s*2);
                const ymm d = getpz2(xq+s*3);
                const ymm  apc = addpz2(a, c);
                const ymm  amc = subpz2(a, c);
                const ymm  bpd = addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(xq+s*0, addpz2(apc,  bpd));
                setpz2(xq+s*1, subpz2(amc, jbmd));
                setpz2(xq+s*2, subpz2(apc,  bpd));
                setpz2(xq+s*3, addpz2(amc, jbmd));
            }
        }
    }
};

template <int s> struct fwd0but<4,s,1>
{
    static const int N = 4*s;

    void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD) {
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector yq = y + q;
                const ymm a = getpz2(xq+s*0);
                const ymm b = getpz2(xq+s*1);
                const ymm c = getpz2(xq+s*2);
                const ymm d = getpz2(xq+s*3);
                const ymm  apc = addpz2(a, c);
                const ymm  amc = subpz2(a, c);
                const ymm  bpd = addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(yq+s*0, addpz2(apc,  bpd));
                setpz2(yq+s*1, subpz2(amc, jbmd));
                setpz2(yq+s*2, subpz2(apc,  bpd));
                setpz2(yq+s*3, addpz2(amc, jbmd));
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp for schedule(static) nowait
#endif
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector yq = y + q;
                const ymm a = getpz2(xq+s*0);
                const ymm b = getpz2(xq+s*1);
                const ymm c = getpz2(xq+s*2);
                const ymm d = getpz2(xq+s*3);
                const ymm  apc = addpz2(a, c);
                const ymm  amc = subpz2(a, c);
                const ymm  bpd = addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(yq+s*0, addpz2(apc,  bpd));
                setpz2(yq+s*1, subpz2(amc, jbmd));
                setpz2(yq+s*2, subpz2(apc,  bpd));
                setpz2(yq+s*3, addpz2(amc, jbmd));
            }
        }
    }
};

template <int s> struct inv0but<4,s,0>
{
    static const int N = 4*s;

    void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD) {
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                const ymm a = getpz2(xq+s*0);
                const ymm b = getpz2(xq+s*1);
                const ymm c = getpz2(xq+s*2);
                const ymm d = getpz2(xq+s*3);
                const ymm  apc = addpz2(a, c);
                const ymm  amc = subpz2(a, c);
                const ymm  bpd = addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(xq+s*0, addpz2(apc,  bpd));
                setpz2(xq+s*1, addpz2(amc, jbmd));
                setpz2(xq+s*2, subpz2(apc,  bpd));
                setpz2(xq+s*3, subpz2(amc, jbmd));
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp for schedule(static) nowait
#endif
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                const ymm a = getpz2(xq+s*0);
                const ymm b = getpz2(xq+s*1);
                const ymm c = getpz2(xq+s*2);
                const ymm d = getpz2(xq+s*3);
                const ymm  apc = addpz2(a, c);
                const ymm  amc = subpz2(a, c);
                const ymm  bpd = addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(xq+s*0, addpz2(apc,  bpd));
                setpz2(xq+s*1, addpz2(amc, jbmd));
                setpz2(xq+s*2, subpz2(apc,  bpd));
                setpz2(xq+s*3, subpz2(amc, jbmd));
            }
        }
    }
};

template <int s> struct inv0but<4,s,1>
{
    static const int N = 4*s;

    void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD) {
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector yq = y + q;
                const ymm a = getpz2(xq+s*0);
                const ymm b = getpz2(xq+s*1);
                const ymm c = getpz2(xq+s*2);
                const ymm d = getpz2(xq+s*3);
                const ymm  apc = addpz2(a, c);
                const ymm  amc = subpz2(a, c);
                const ymm  bpd = addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(yq+s*0, addpz2(apc,  bpd));
                setpz2(yq+s*1, addpz2(amc, jbmd));
                setpz2(yq+s*2, subpz2(apc,  bpd));
                setpz2(yq+s*3, subpz2(amc, jbmd));
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp for schedule(static) nowait
#endif
            for (int q = 0; q < s; q += 2) {
                complex_vector xq = x + q;
                complex_vector yq = y + q;
                const ymm a = getpz2(xq+s*0);
                const ymm b = getpz2(xq+s*1);
                const ymm c = getpz2(xq+s*2);
                const ymm d = getpz2(xq+s*3);
                const ymm  apc = addpz2(a, c);
                const ymm  amc = subpz2(a, c);
                const ymm  bpd = addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(yq+s*0, addpz2(apc,  bpd));
                setpz2(yq+s*1, addpz2(amc, jbmd));
                setpz2(yq+s*2, subpz2(apc,  bpd));
                setpz2(yq+s*3, subpz2(amc, jbmd));
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int N, int eo> struct fwd0but<N,1,eo>
{
    static const int N0 = 0;
    static const int N1 = N/4;
    static const int N2 = N/2;
    static const int N3 = N1 + N2;

    void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < N/4; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_4p = y + 4*p;
                //const ymm w1p = getwp2<1>(W,p);
                const ymm w1p = getpz2(W+p);
                const ymm w2p = getwp2<2>(W,p);
                const ymm w3p = getwp2<3>(W,p);
                const ymm a = getpz2(x_p+N0);
                const ymm b = getpz2(x_p+N1);
                const ymm c = getpz2(x_p+N2);
                const ymm d = getpz2(x_p+N3);
                const ymm  apc = addpz2(a, c);
                const ymm  amc = subpz2(a, c);
                const ymm  bpd = addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz3<4>(y_4p+0,             addpz2(apc,  bpd));
                setpz3<4>(y_4p+1, mulpz2(w1p, subpz2(amc, jbmd)));
                setpz3<4>(y_4p+2, mulpz2(w2p, subpz2(apc,  bpd)));
                setpz3<4>(y_4p+3, mulpz2(w3p, addpz2(amc, jbmd)));
            }
            fwd0but<N/4,4,!eo>()(y, x, W);
        }
        else
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 0; p < N/4; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_4p = y + 4*p;
                //const ymm w1p = getwp2<1>(W,p);
                const ymm w1p = getpz2(W+p);
                const ymm w2p = getwp2<2>(W,p);
                const ymm w3p = getwp2<3>(W,p);
                const ymm a = getpz2(x_p+N0);
                const ymm b = getpz2(x_p+N1);
                const ymm c = getpz2(x_p+N2);
                const ymm d = getpz2(x_p+N3);
                const ymm  apc = addpz2(a, c);
                const ymm  amc = subpz2(a, c);
                const ymm  bpd = addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz3<4>(y_4p+0,             addpz2(apc,  bpd));
                setpz3<4>(y_4p+1, mulpz2(w1p, subpz2(amc, jbmd)));
                setpz3<4>(y_4p+2, mulpz2(w2p, subpz2(apc,  bpd)));
                setpz3<4>(y_4p+3, mulpz2(w3p, addpz2(amc, jbmd)));
            }
            fwd0but<N/4,4,!eo>()(y, x, W);
        }
    }
};

template <int N, int eo> struct inv0but<N,1,eo>
{
    static const int N0 = 0;
    static const int N1 = N/4;
    static const int N2 = N/2;
    static const int N3 = N1 + N2;

    void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < N/4; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_4p = y + 4*p;
                //const ymm w1p = getwp2<-1>(W+N,p);
                const ymm w1p = cnjpz2(getpz2(W+p));
                const ymm w2p = getwp2<-2>(W+N,p);
                const ymm w3p = getwp2<-3>(W+N,p);
                const ymm a = getpz2(x_p+N0);
                const ymm b = getpz2(x_p+N1);
                const ymm c = getpz2(x_p+N2);
                const ymm d = getpz2(x_p+N3);
                const ymm  apc = addpz2(a, c);
                const ymm  amc = subpz2(a, c);
                const ymm  bpd = addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz3<4>(y_4p+0,             addpz2(apc,  bpd));
                setpz3<4>(y_4p+1, mulpz2(w1p, addpz2(amc, jbmd)));
                setpz3<4>(y_4p+2, mulpz2(w2p, subpz2(apc,  bpd)));
                setpz3<4>(y_4p+3, mulpz2(w3p, subpz2(amc, jbmd)));
            }
            inv0but<N/4,4,!eo>()(y, x, W);
        }
        else
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 0; p < N/4; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_4p = y + 4*p;
                //const ymm w1p = getwp2<-1>(W+N,p);
                const ymm w1p = cnjpz2(getpz2(W+p));
                const ymm w2p = getwp2<-2>(W+N,p);
                const ymm w3p = getwp2<-3>(W+N,p);
                const ymm a = getpz2(x_p+N0);
                const ymm b = getpz2(x_p+N1);
                const ymm c = getpz2(x_p+N2);
                const ymm d = getpz2(x_p+N3);
                const ymm  apc = addpz2(a, c);
                const ymm  amc = subpz2(a, c);
                const ymm  bpd = addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz3<4>(y_4p+0,             addpz2(apc,  bpd));
                setpz3<4>(y_4p+1, mulpz2(w1p, addpz2(amc, jbmd)));
                setpz3<4>(y_4p+2, mulpz2(w2p, subpz2(apc,  bpd)));
                setpz3<4>(y_4p+3, mulpz2(w3p, subpz2(amc, jbmd)));
            }
            inv0but<N/4,4,!eo>()(y, x, W);
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, int eo> struct fwd0but
{
    static const int N = n*s;
    static const int N0 = 0;
    static const int N1 = N/4;
    static const int N2 = N/2;
    static const int N3 = N1 + N2;

    void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < n/4; p++) {
                const int sp = s*p;
                const int s4p = 4*sp;
                //const ymm w1p = duppz2(getpz(W[sp]));
                const ymm w1p = duppz3(W[sp]);
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                for (int q = 0; q < s; q += 2) {
                    complex_vector xq_sp  = x + q + sp;
                    complex_vector yq_s4p = y + q + s4p;
                    const ymm a = getpz2(xq_sp+N0);
                    const ymm b = getpz2(xq_sp+N1);
                    const ymm c = getpz2(xq_sp+N2);
                    const ymm d = getpz2(xq_sp+N3);
                    const ymm  apc = addpz2(a, c);
                    const ymm  amc = subpz2(a, c);
                    const ymm  bpd = addpz2(b, d);
                    const ymm jbmd = jxpz2(subpz2(b, d));
                    setpz2(yq_s4p+s*0,             addpz2(apc,  bpd));
                    setpz2(yq_s4p+s*1, mulpz2(w1p, subpz2(amc, jbmd)));
                    setpz2(yq_s4p+s*2, mulpz2(w2p, subpz2(apc,  bpd)));
                    setpz2(yq_s4p+s*3, mulpz2(w3p, addpz2(amc, jbmd)));
                }
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 0; p < n/4; p++) {
                const int sp = s*p;
                const int s4p = 4*sp;
                //const ymm w1p = duppz2(getpz(W[sp]));
                const ymm w1p = duppz3(W[sp]);
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                for (int q = 0; q < s; q += 2) {
                    complex_vector xq_sp  = x + q + sp;
                    complex_vector yq_s4p = y + q + s4p;
                    const ymm a = getpz2(xq_sp+N0);
                    const ymm b = getpz2(xq_sp+N1);
                    const ymm c = getpz2(xq_sp+N2);
                    const ymm d = getpz2(xq_sp+N3);
                    const ymm  apc = addpz2(a, c);
                    const ymm  amc = subpz2(a, c);
                    const ymm  bpd = addpz2(b, d);
                    const ymm jbmd = jxpz2(subpz2(b, d));
                    setpz2(yq_s4p+s*0,             addpz2(apc,  bpd));
                    setpz2(yq_s4p+s*1, mulpz2(w1p, subpz2(amc, jbmd)));
                    setpz2(yq_s4p+s*2, mulpz2(w2p, subpz2(apc,  bpd)));
                    setpz2(yq_s4p+s*3, mulpz2(w3p, addpz2(amc, jbmd)));
                }
            }
        }
        fwd0but<n/4,4*s,!eo>()(y, x, W);
    }
};

template <int n, int s, int eo> struct inv0but
{
    static const int N = n*s;
    static const int N0 = 0;
    static const int N1 = N/4;
    static const int N2 = N/2;
    static const int N3 = N1 + N2;

    void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < n/4; p++) {
                const int sp = s*p;
                const int s4p = 4*sp;
                //const ymm w1p = duppz2(getpz(W[N-sp]));
                const ymm w1p = duppz3(W[N-sp]);
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                for (int q = 0; q < s; q += 2) {
                    complex_vector xq_sp  = x + q + sp;
                    complex_vector yq_s4p = y + q + s4p;
                    const ymm a = getpz2(xq_sp+N0);
                    const ymm b = getpz2(xq_sp+N1);
                    const ymm c = getpz2(xq_sp+N2);
                    const ymm d = getpz2(xq_sp+N3);
                    const ymm  apc = addpz2(a, c);
                    const ymm  amc = subpz2(a, c);
                    const ymm  bpd = addpz2(b, d);
                    const ymm jbmd = jxpz2(subpz2(b, d));
                    setpz2(yq_s4p+s*0,             addpz2(apc,  bpd));
                    setpz2(yq_s4p+s*1, mulpz2(w1p, addpz2(amc, jbmd)));
                    setpz2(yq_s4p+s*2, mulpz2(w2p, subpz2(apc,  bpd)));
                    setpz2(yq_s4p+s*3, mulpz2(w3p, subpz2(amc, jbmd)));
                }
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 0; p < n/4; p++) {
                const int sp = s*p;
                const int s4p = 4*sp;
                //const ymm w1p = duppz2(getpz(W[N-sp]));
                const ymm w1p = duppz3(W[N-sp]);
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                for (int q = 0; q < s; q += 2) {
                    complex_vector xq_sp  = x + q + sp;
                    complex_vector yq_s4p = y + q + s4p;
                    const ymm a = getpz2(xq_sp+N0);
                    const ymm b = getpz2(xq_sp+N1);
                    const ymm c = getpz2(xq_sp+N2);
                    const ymm d = getpz2(xq_sp+N3);
                    const ymm  apc = addpz2(a, c);
                    const ymm  amc = subpz2(a, c);
                    const ymm  bpd = addpz2(b, d);
                    const ymm jbmd = jxpz2(subpz2(b, d));
                    setpz2(yq_s4p+s*0,             addpz2(apc,  bpd));
                    setpz2(yq_s4p+s*1, mulpz2(w1p, addpz2(amc, jbmd)));
                    setpz2(yq_s4p+s*2, mulpz2(w2p, subpz2(apc,  bpd)));
                    setpz2(yq_s4p+s*3, mulpz2(w3p, subpz2(amc, jbmd)));
                }
            }
        }
        inv0but<n/4,4*s,!eo>()(y, x, W);
    }
};

///////////////////////////////////////////////////////////////////////////////

#include "otfft_difavxn.h"

///////////////////////////////////////////////////////////////////////////////

struct FFT0
{
    int N, log_N;
    simd_array<complex_t> weight;
    complex_t* W;

    FFT0() : N(0), log_N(0), W(0) {}
    FFT0(int n) { setup(n); }

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
    }

    inline void fwd0(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: fwd0but<(1<< 1),1,0>()(x, y, W); break;
        case  2: fwd0but<(1<< 2),1,0>()(x, y, W); break;
        case  3: fwd0but<(1<< 3),1,0>()(x, y, W); break;
        case  4: fwd0but<(1<< 4),1,0>()(x, y, W); break;
        case  5: fwd0but<(1<< 5),1,0>()(x, y, W); break;
        case  6: fwd0but<(1<< 6),1,0>()(x, y, W); break;
        case  7: fwd0but<(1<< 7),1,0>()(x, y, W); break;
        case  8: fwd0but<(1<< 8),1,0>()(x, y, W); break;
        case  9: fwd0but<(1<< 9),1,0>()(x, y, W); break;
        case 10: fwd0but<(1<<10),1,0>()(x, y, W); break;
        case 11: fwd0but<(1<<11),1,0>()(x, y, W); break;
        case 12: fwd0but<(1<<12),1,0>()(x, y, W); break;
        case 13: fwd0but<(1<<13),1,0>()(x, y, W); break;
        case 14: fwd0but<(1<<14),1,0>()(x, y, W); break;
        case 15: fwd0but<(1<<15),1,0>()(x, y, W); break;
        case 16: fwd0but<(1<<16),1,0>()(x, y, W); break;
        case 17: fwd0but<(1<<17),1,0>()(x, y, W); break;
        case 18: fwd0but<(1<<18),1,0>()(x, y, W); break;
        case 19: fwd0but<(1<<19),1,0>()(x, y, W); break;
        case 20: fwd0but<(1<<20),1,0>()(x, y, W); break;
        case 21: fwd0but<(1<<21),1,0>()(x, y, W); break;
        case 22: fwd0but<(1<<22),1,0>()(x, y, W); break;
        case 23: fwd0but<(1<<23),1,0>()(x, y, W); break;
        case 24: fwd0but<(1<<24),1,0>()(x, y, W); break;
        }
    }

    inline void fwd(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: fwdnbut<(1<< 1),1,0>()(x, y, W); break;
        case  2: fwdnbut<(1<< 2),1,0>()(x, y, W); break;
        case  3: fwdnbut<(1<< 3),1,0>()(x, y, W); break;
        case  4: fwdnbut<(1<< 4),1,0>()(x, y, W); break;
        case  5: fwdnbut<(1<< 5),1,0>()(x, y, W); break;
        case  6: fwdnbut<(1<< 6),1,0>()(x, y, W); break;
        case  7: fwdnbut<(1<< 7),1,0>()(x, y, W); break;
        case  8: fwdnbut<(1<< 8),1,0>()(x, y, W); break;
        case  9: fwdnbut<(1<< 9),1,0>()(x, y, W); break;
        case 10: fwdnbut<(1<<10),1,0>()(x, y, W); break;
        case 11: fwdnbut<(1<<11),1,0>()(x, y, W); break;
        case 12: fwdnbut<(1<<12),1,0>()(x, y, W); break;
        case 13: fwdnbut<(1<<13),1,0>()(x, y, W); break;
        case 14: fwdnbut<(1<<14),1,0>()(x, y, W); break;
        case 15: fwdnbut<(1<<15),1,0>()(x, y, W); break;
        case 16: fwdnbut<(1<<16),1,0>()(x, y, W); break;
        case 17: fwdnbut<(1<<17),1,0>()(x, y, W); break;
        case 18: fwdnbut<(1<<18),1,0>()(x, y, W); break;
        case 19: fwdnbut<(1<<19),1,0>()(x, y, W); break;
        case 20: fwdnbut<(1<<20),1,0>()(x, y, W); break;
        case 21: fwdnbut<(1<<21),1,0>()(x, y, W); break;
        case 22: fwdnbut<(1<<22),1,0>()(x, y, W); break;
        case 23: fwdnbut<(1<<23),1,0>()(x, y, W); break;
        case 24: fwdnbut<(1<<24),1,0>()(x, y, W); break;
        }
    }

    inline void fwdn(complex_vector x, complex_vector y) const { fwd(x, y); }

    inline void fwd0o(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: fwd0but<(1<< 1),1,1>()(x, y, W); break;
        case  2: fwd0but<(1<< 2),1,1>()(x, y, W); break;
        case  3: fwd0but<(1<< 3),1,1>()(x, y, W); break;
        case  4: fwd0but<(1<< 4),1,1>()(x, y, W); break;
        case  5: fwd0but<(1<< 5),1,1>()(x, y, W); break;
        case  6: fwd0but<(1<< 6),1,1>()(x, y, W); break;
        case  7: fwd0but<(1<< 7),1,1>()(x, y, W); break;
        case  8: fwd0but<(1<< 8),1,1>()(x, y, W); break;
        case  9: fwd0but<(1<< 9),1,1>()(x, y, W); break;
        case 10: fwd0but<(1<<10),1,1>()(x, y, W); break;
        case 11: fwd0but<(1<<11),1,1>()(x, y, W); break;
        case 12: fwd0but<(1<<12),1,1>()(x, y, W); break;
        case 13: fwd0but<(1<<13),1,1>()(x, y, W); break;
        case 14: fwd0but<(1<<14),1,1>()(x, y, W); break;
        case 15: fwd0but<(1<<15),1,1>()(x, y, W); break;
        case 16: fwd0but<(1<<16),1,1>()(x, y, W); break;
        case 17: fwd0but<(1<<17),1,1>()(x, y, W); break;
        case 18: fwd0but<(1<<18),1,1>()(x, y, W); break;
        case 19: fwd0but<(1<<19),1,1>()(x, y, W); break;
        case 20: fwd0but<(1<<20),1,1>()(x, y, W); break;
        case 21: fwd0but<(1<<21),1,1>()(x, y, W); break;
        case 22: fwd0but<(1<<22),1,1>()(x, y, W); break;
        case 23: fwd0but<(1<<23),1,1>()(x, y, W); break;
        case 24: fwd0but<(1<<24),1,1>()(x, y, W); break;
        }
    }

    inline void fwdno(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: fwdnbut<(1<< 1),1,1>()(x, y, W); break;
        case  2: fwdnbut<(1<< 2),1,1>()(x, y, W); break;
        case  3: fwdnbut<(1<< 3),1,1>()(x, y, W); break;
        case  4: fwdnbut<(1<< 4),1,1>()(x, y, W); break;
        case  5: fwdnbut<(1<< 5),1,1>()(x, y, W); break;
        case  6: fwdnbut<(1<< 6),1,1>()(x, y, W); break;
        case  7: fwdnbut<(1<< 7),1,1>()(x, y, W); break;
        case  8: fwdnbut<(1<< 8),1,1>()(x, y, W); break;
        case  9: fwdnbut<(1<< 9),1,1>()(x, y, W); break;
        case 10: fwdnbut<(1<<10),1,1>()(x, y, W); break;
        case 11: fwdnbut<(1<<11),1,1>()(x, y, W); break;
        case 12: fwdnbut<(1<<12),1,1>()(x, y, W); break;
        case 13: fwdnbut<(1<<13),1,1>()(x, y, W); break;
        case 14: fwdnbut<(1<<14),1,1>()(x, y, W); break;
        case 15: fwdnbut<(1<<15),1,1>()(x, y, W); break;
        case 16: fwdnbut<(1<<16),1,1>()(x, y, W); break;
        case 17: fwdnbut<(1<<17),1,1>()(x, y, W); break;
        case 18: fwdnbut<(1<<18),1,1>()(x, y, W); break;
        case 19: fwdnbut<(1<<19),1,1>()(x, y, W); break;
        case 20: fwdnbut<(1<<20),1,1>()(x, y, W); break;
        case 21: fwdnbut<(1<<21),1,1>()(x, y, W); break;
        case 22: fwdnbut<(1<<22),1,1>()(x, y, W); break;
        case 23: fwdnbut<(1<<23),1,1>()(x, y, W); break;
        case 24: fwdnbut<(1<<24),1,1>()(x, y, W); break;
        }
    }

    inline void inv0(complex_vector x, complex_vector y) const { inv(x, y); }

    inline void inv(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: inv0but<(1<< 1),1,0>()(x, y, W); break;
        case  2: inv0but<(1<< 2),1,0>()(x, y, W); break;
        case  3: inv0but<(1<< 3),1,0>()(x, y, W); break;
        case  4: inv0but<(1<< 4),1,0>()(x, y, W); break;
        case  5: inv0but<(1<< 5),1,0>()(x, y, W); break;
        case  6: inv0but<(1<< 6),1,0>()(x, y, W); break;
        case  7: inv0but<(1<< 7),1,0>()(x, y, W); break;
        case  8: inv0but<(1<< 8),1,0>()(x, y, W); break;
        case  9: inv0but<(1<< 9),1,0>()(x, y, W); break;
        case 10: inv0but<(1<<10),1,0>()(x, y, W); break;
        case 11: inv0but<(1<<11),1,0>()(x, y, W); break;
        case 12: inv0but<(1<<12),1,0>()(x, y, W); break;
        case 13: inv0but<(1<<13),1,0>()(x, y, W); break;
        case 14: inv0but<(1<<14),1,0>()(x, y, W); break;
        case 15: inv0but<(1<<15),1,0>()(x, y, W); break;
        case 16: inv0but<(1<<16),1,0>()(x, y, W); break;
        case 17: inv0but<(1<<17),1,0>()(x, y, W); break;
        case 18: inv0but<(1<<18),1,0>()(x, y, W); break;
        case 19: inv0but<(1<<19),1,0>()(x, y, W); break;
        case 20: inv0but<(1<<20),1,0>()(x, y, W); break;
        case 21: inv0but<(1<<21),1,0>()(x, y, W); break;
        case 22: inv0but<(1<<22),1,0>()(x, y, W); break;
        case 23: inv0but<(1<<23),1,0>()(x, y, W); break;
        case 24: inv0but<(1<<24),1,0>()(x, y, W); break;
        }
    }

    inline void invn(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: invnbut<(1<< 1),1,0>()(x, y, W); break;
        case  2: invnbut<(1<< 2),1,0>()(x, y, W); break;
        case  3: invnbut<(1<< 3),1,0>()(x, y, W); break;
        case  4: invnbut<(1<< 4),1,0>()(x, y, W); break;
        case  5: invnbut<(1<< 5),1,0>()(x, y, W); break;
        case  6: invnbut<(1<< 6),1,0>()(x, y, W); break;
        case  7: invnbut<(1<< 7),1,0>()(x, y, W); break;
        case  8: invnbut<(1<< 8),1,0>()(x, y, W); break;
        case  9: invnbut<(1<< 9),1,0>()(x, y, W); break;
        case 10: invnbut<(1<<10),1,0>()(x, y, W); break;
        case 11: invnbut<(1<<11),1,0>()(x, y, W); break;
        case 12: invnbut<(1<<12),1,0>()(x, y, W); break;
        case 13: invnbut<(1<<13),1,0>()(x, y, W); break;
        case 14: invnbut<(1<<14),1,0>()(x, y, W); break;
        case 15: invnbut<(1<<15),1,0>()(x, y, W); break;
        case 16: invnbut<(1<<16),1,0>()(x, y, W); break;
        case 17: invnbut<(1<<17),1,0>()(x, y, W); break;
        case 18: invnbut<(1<<18),1,0>()(x, y, W); break;
        case 19: invnbut<(1<<19),1,0>()(x, y, W); break;
        case 20: invnbut<(1<<20),1,0>()(x, y, W); break;
        case 21: invnbut<(1<<21),1,0>()(x, y, W); break;
        case 22: invnbut<(1<<22),1,0>()(x, y, W); break;
        case 23: invnbut<(1<<23),1,0>()(x, y, W); break;
        case 24: invnbut<(1<<24),1,0>()(x, y, W); break;
        }
    }

    inline void inv0o(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: inv0but<(1<< 1),1,1>()(x, y, W); break;
        case  2: inv0but<(1<< 2),1,1>()(x, y, W); break;
        case  3: inv0but<(1<< 3),1,1>()(x, y, W); break;
        case  4: inv0but<(1<< 4),1,1>()(x, y, W); break;
        case  5: inv0but<(1<< 5),1,1>()(x, y, W); break;
        case  6: inv0but<(1<< 6),1,1>()(x, y, W); break;
        case  7: inv0but<(1<< 7),1,1>()(x, y, W); break;
        case  8: inv0but<(1<< 8),1,1>()(x, y, W); break;
        case  9: inv0but<(1<< 9),1,1>()(x, y, W); break;
        case 10: inv0but<(1<<10),1,1>()(x, y, W); break;
        case 11: inv0but<(1<<11),1,1>()(x, y, W); break;
        case 12: inv0but<(1<<12),1,1>()(x, y, W); break;
        case 13: inv0but<(1<<13),1,1>()(x, y, W); break;
        case 14: inv0but<(1<<14),1,1>()(x, y, W); break;
        case 15: inv0but<(1<<15),1,1>()(x, y, W); break;
        case 16: inv0but<(1<<16),1,1>()(x, y, W); break;
        case 17: inv0but<(1<<17),1,1>()(x, y, W); break;
        case 18: inv0but<(1<<18),1,1>()(x, y, W); break;
        case 19: inv0but<(1<<19),1,1>()(x, y, W); break;
        case 20: inv0but<(1<<20),1,1>()(x, y, W); break;
        case 21: inv0but<(1<<21),1,1>()(x, y, W); break;
        case 22: inv0but<(1<<22),1,1>()(x, y, W); break;
        case 23: inv0but<(1<<23),1,1>()(x, y, W); break;
        case 24: inv0but<(1<<24),1,1>()(x, y, W); break;
        }
    }

    inline void invno(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: invnbut<(1<< 1),1,1>()(x, y, W); break;
        case  2: invnbut<(1<< 2),1,1>()(x, y, W); break;
        case  3: invnbut<(1<< 3),1,1>()(x, y, W); break;
        case  4: invnbut<(1<< 4),1,1>()(x, y, W); break;
        case  5: invnbut<(1<< 5),1,1>()(x, y, W); break;
        case  6: invnbut<(1<< 6),1,1>()(x, y, W); break;
        case  7: invnbut<(1<< 7),1,1>()(x, y, W); break;
        case  8: invnbut<(1<< 8),1,1>()(x, y, W); break;
        case  9: invnbut<(1<< 9),1,1>()(x, y, W); break;
        case 10: invnbut<(1<<10),1,1>()(x, y, W); break;
        case 11: invnbut<(1<<11),1,1>()(x, y, W); break;
        case 12: invnbut<(1<<12),1,1>()(x, y, W); break;
        case 13: invnbut<(1<<13),1,1>()(x, y, W); break;
        case 14: invnbut<(1<<14),1,1>()(x, y, W); break;
        case 15: invnbut<(1<<15),1,1>()(x, y, W); break;
        case 16: invnbut<(1<<16),1,1>()(x, y, W); break;
        case 17: invnbut<(1<<17),1,1>()(x, y, W); break;
        case 18: invnbut<(1<<18),1,1>()(x, y, W); break;
        case 19: invnbut<(1<<19),1,1>()(x, y, W); break;
        case 20: invnbut<(1<<20),1,1>()(x, y, W); break;
        case 21: invnbut<(1<<21),1,1>()(x, y, W); break;
        case 22: invnbut<(1<<22),1,1>()(x, y, W); break;
        case 23: invnbut<(1<<23),1,1>()(x, y, W); break;
        case 24: invnbut<(1<<24),1,1>()(x, y, W); break;
        }
    }
};

#if 0
struct FFT
{
    FFT0 fft;
    simd_array<complex_t> work;
    complex_t* y;

    FFT() : y(0) {}
    FFT(int n) : fft(n), work(n), y(&work) {}

    inline void setup(int n) { fft.setup(n); work.setup(n); y = &work; }

    inline void fwd(complex_vector x)  const { fft.fwd(x, y);  }
    inline void fwd0(complex_vector x) const { fft.fwd0(x, y); }
    inline void inv(complex_vector x)  const { fft.inv(x, y);  }
    inline void invn(complex_vector x) const { fft.invn(x, y);  }
    inline void inv0(complex_vector x) const { fft.inv0(x, y); }
};
#endif

} /////////////////////////////////////////////////////////////////////////////

#endif // otfft_difavx_h
