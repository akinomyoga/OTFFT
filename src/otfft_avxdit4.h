/******************************************************************************
*  OTFFT AVXDIT(Radix-4) Version 6.4
*
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_avxdit4_h
#define otfft_avxdit4_h

#include "otfft/otfft_misc.h"
#include "otfft_avxdit4omp.h"

namespace OTFFT_AVXDIT4 { /////////////////////////////////////////////////////

using namespace OTFFT_MISC;

#ifdef DO_SINGLE_THREAD
static const int OMP_THRESHOLD = 1<<30;
#else
static const int OMP_THRESHOLD = 1<<15;
#endif

///////////////////////////////////////////////////////////////////////////////
// Forward butterfly operation
///////////////////////////////////////////////////////////////////////////////

template <int n, int s> struct fwdcore
{
    static const int n1 = n/4;
    static const int N  = n*s;
    static const int N0 = 0;
    static const int N1 = N/4;
    static const int N2 = N1*2;
    static const int N3 = N1*3;

    void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        for (int p = 0; p < n1; p++) {
            const int sp = s*p;
            const int s4p = 4*sp;
            //const ymm w1p = duppz2(getpz(W[sp]));
            const ymm w1p = duppz3(W[1*sp]);
            const ymm w2p = duppz3(W[2*sp]);
            const ymm w3p = duppz3(W[3*sp]);
            for (int q = 0; q < s; q += 2) {
                complex_vector xq_sp  = x + q + sp;
                complex_vector yq_s4p = y + q + s4p;
                const ymm a =             getpz2(yq_s4p+s*0);
                const ymm b = mulpz2(w1p, getpz2(yq_s4p+s*1));
                const ymm c = mulpz2(w2p, getpz2(yq_s4p+s*2));
                const ymm d = mulpz2(w3p, getpz2(yq_s4p+s*3));
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(xq_sp+N0, addpz2(apc,  bpd));
                setpz2(xq_sp+N1, subpz2(amc, jbmd));
                setpz2(xq_sp+N2, subpz2(apc,  bpd));
                setpz2(xq_sp+N3, addpz2(amc, jbmd));
            }
        }
    }
};

template <int N> struct fwdcore<N,1>
{
    static const int N0 = 0;
    static const int N1 = N/4;
    static const int N2 = N1*2;
    static const int N3 = N1*3;

    void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        for (int p = 0; p < N1; p += 2) {
            complex_vector x_p  = x + p;
            complex_vector y_4p = y + 4*p;
            const ymm w1p = getpz2(W+p);
            const ymm w2p = getwp2<2>(W,p);
            const ymm w3p = getwp2<3>(W,p);
#if 0
            const ymm a =             getpz3<4>(y_4p+0);
            const ymm b = mulpz2(w1p, getpz3<4>(y_4p+1));
            const ymm c = mulpz2(w2p, getpz3<4>(y_4p+2));
            const ymm d = mulpz2(w3p, getpz3<4>(y_4p+3));
#else
            const ymm ab = getpz2(y_4p+0);
            const ymm cd = getpz2(y_4p+2);
            const ymm ef = getpz2(y_4p+4);
            const ymm gh = getpz2(y_4p+6);
            const ymm a =             catlo(ab, ef);
            const ymm b = mulpz2(w1p, cathi(ab, ef));
            const ymm c = mulpz2(w2p, catlo(cd, gh));
            const ymm d = mulpz2(w3p, cathi(cd, gh));
#endif
            const ymm  apc =       addpz2(a, c);
            const ymm  amc =       subpz2(a, c);
            const ymm  bpd =       addpz2(b, d);
            const ymm jbmd = jxpz2(subpz2(b, d));
            setpz2(x_p+N0, addpz2(apc,  bpd));
            setpz2(x_p+N1, subpz2(amc, jbmd));
            setpz2(x_p+N2, subpz2(apc,  bpd));
            setpz2(x_p+N3, addpz2(amc, jbmd));
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct fwd0end;

//-----------------------------------------------------------------------------

template <int s> struct fwd0end<4,s,1>
{
    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
            const ymm a = getpz2(yq+s*0);
            const ymm b = getpz2(yq+s*1);
            const ymm c = getpz2(yq+s*2);
            const ymm d = getpz2(yq+s*3);
            const ymm  apc =       addpz2(a, c);
            const ymm  amc =       subpz2(a, c);
            const ymm  bpd =       addpz2(b, d);
            const ymm jbmd = jxpz2(subpz2(b, d));
            setpz2(xq+s*0, addpz2(apc,  bpd));
            setpz2(xq+s*1, subpz2(amc, jbmd));
            setpz2(xq+s*2, subpz2(apc,  bpd));
            setpz2(xq+s*3, addpz2(amc, jbmd));
        }
    }
};

template <> struct fwd0end<4,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        zeroupper();
        const xmm a = getpz(y[0]);
        const xmm b = getpz(y[1]);
        const xmm c = getpz(y[2]);
        const xmm d = getpz(y[3]);
        const xmm  apc =      addpz(a, c);
        const xmm  amc =      subpz(a, c);
        const xmm  bpd =      addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(x[0], addpz(apc,  bpd));
        setpz(x[1], subpz(amc, jbmd));
        setpz(x[2], subpz(apc,  bpd));
        setpz(x[3], addpz(amc, jbmd));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct fwd0end<4,s,0>
{
    void operator()(complex_vector x, complex_vector) const noexcept
    {
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            const ymm a = getpz2(xq+s*0);
            const ymm b = getpz2(xq+s*1);
            const ymm c = getpz2(xq+s*2);
            const ymm d = getpz2(xq+s*3);
            const ymm  apc =       addpz2(a, c);
            const ymm  amc =       subpz2(a, c);
            const ymm  bpd =       addpz2(b, d);
            const ymm jbmd = jxpz2(subpz2(b, d));
            setpz2(xq+s*0, addpz2(apc,  bpd));
            setpz2(xq+s*1, subpz2(amc, jbmd));
            setpz2(xq+s*2, subpz2(apc,  bpd));
            setpz2(xq+s*3, addpz2(amc, jbmd));
        }
    }
};

template <> struct fwd0end<4,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const noexcept
    {
        zeroupper();
        const xmm a = getpz(x[0]);
        const xmm b = getpz(x[1]);
        const xmm c = getpz(x[2]);
        const xmm d = getpz(x[3]);
        const xmm  apc =      addpz(a, c);
        const xmm  amc =      subpz(a, c);
        const xmm  bpd =      addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(x[0], addpz(apc,  bpd));
        setpz(x[1], subpz(amc, jbmd));
        setpz(x[2], subpz(apc,  bpd));
        setpz(x[3], addpz(amc, jbmd));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct fwd0end<2,s,1>
{
    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
            const ymm a = getpz2(yq+0);
            const ymm b = getpz2(yq+s);
            setpz2(xq+0, addpz2(a, b));
            setpz2(xq+s, subpz2(a, b));
        }
    }
};

template <> struct fwd0end<2,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        zeroupper();
        const xmm a = getpz(y[0]);
        const xmm b = getpz(y[1]);
        setpz(x[0], addpz(a, b));
        setpz(x[1], subpz(a, b));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct fwd0end<2,s,0>
{
    void operator()(complex_vector x, complex_vector) const noexcept
    {
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            const ymm a = getpz2(xq+0);
            const ymm b = getpz2(xq+s);
            setpz2(xq+0, addpz2(a, b));
            setpz2(xq+s, subpz2(a, b));
        }
    }
};

template <> struct fwd0end<2,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const noexcept
    {
        zeroupper();
        const xmm a = getpz(x[0]);
        const xmm b = getpz(x[1]);
        setpz(x[0], addpz(a, b));
        setpz(x[1], subpz(a, b));
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct fwdnend;

//-----------------------------------------------------------------------------

template <int s> struct fwdnend<4,s,1>
{
    static const int N = 4*s;

    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
            const ymm a = mulpd2(rN, getpz2(yq+s*0));
            const ymm b = mulpd2(rN, getpz2(yq+s*1));
            const ymm c = mulpd2(rN, getpz2(yq+s*2));
            const ymm d = mulpd2(rN, getpz2(yq+s*3));
            const ymm  apc =       addpz2(a, c);
            const ymm  amc =       subpz2(a, c);
            const ymm  bpd =       addpz2(b, d);
            const ymm jbmd = jxpz2(subpz2(b, d));
            setpz2(xq+s*0, addpz2(apc,  bpd));
            setpz2(xq+s*1, subpz2(amc, jbmd));
            setpz2(xq+s*2, subpz2(apc,  bpd));
            setpz2(xq+s*3, addpz2(amc, jbmd));
        }
    }
};

template <> struct fwdnend<4,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        zeroupper();
        static const xmm rN = { 1.0/4, 1.0/4 };
        const xmm a = mulpd(rN, getpz(y[0]));
        const xmm b = mulpd(rN, getpz(y[1]));
        const xmm c = mulpd(rN, getpz(y[2]));
        const xmm d = mulpd(rN, getpz(y[3]));
        const xmm  apc =      addpz(a, c);
        const xmm  amc =      subpz(a, c);
        const xmm  bpd =      addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(x[0], addpz(apc,  bpd));
        setpz(x[1], subpz(amc, jbmd));
        setpz(x[2], subpz(apc,  bpd));
        setpz(x[3], addpz(amc, jbmd));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct fwdnend<4,s,0>
{
    static const int N = 4*s;

    void operator()(complex_vector x, complex_vector) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            const ymm a = mulpd2(rN, getpz2(xq+s*0));
            const ymm b = mulpd2(rN, getpz2(xq+s*1));
            const ymm c = mulpd2(rN, getpz2(xq+s*2));
            const ymm d = mulpd2(rN, getpz2(xq+s*3));
            const ymm  apc =       addpz2(a, c);
            const ymm  amc =       subpz2(a, c);
            const ymm  bpd =       addpz2(b, d);
            const ymm jbmd = jxpz2(subpz2(b, d));
            setpz2(xq+s*0, addpz2(apc,  bpd));
            setpz2(xq+s*1, subpz2(amc, jbmd));
            setpz2(xq+s*2, subpz2(apc,  bpd));
            setpz2(xq+s*3, addpz2(amc, jbmd));
        }
    }
};

template <> struct fwdnend<4,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const noexcept
    {
        zeroupper();
        static const xmm rN = { 1.0/4, 1.0/4 };
        const xmm a = mulpd(rN, getpz(x[0]));
        const xmm b = mulpd(rN, getpz(x[1]));
        const xmm c = mulpd(rN, getpz(x[2]));
        const xmm d = mulpd(rN, getpz(x[3]));
        const xmm  apc =      addpz(a, c);
        const xmm  amc =      subpz(a, c);
        const xmm  bpd =      addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(x[0], addpz(apc,  bpd));
        setpz(x[1], subpz(amc, jbmd));
        setpz(x[2], subpz(apc,  bpd));
        setpz(x[3], addpz(amc, jbmd));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct fwdnend<2,s,1>
{
    static const int N = 2*s;

    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
            const ymm a = mulpd2(rN, getpz2(yq+0));
            const ymm b = mulpd2(rN, getpz2(yq+s));
            setpz2(xq+0, addpz2(a, b));
            setpz2(xq+s, subpz2(a, b));
        }
    }
};

template <> struct fwdnend<2,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        zeroupper();
        static const xmm rN = { 1.0/2, 1.0/2 };
        const xmm a = mulpd(rN, getpz(y[0]));
        const xmm b = mulpd(rN, getpz(y[1]));
        setpz(x[0], addpz(a, b));
        setpz(x[1], subpz(a, b));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct fwdnend<2,s,0>
{
    static const int N = 2*s;

    void operator()(complex_vector x, complex_vector) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            const ymm a = mulpd2(rN, getpz2(xq+0));
            const ymm b = mulpd2(rN, getpz2(xq+s));
            setpz2(xq+0, addpz2(a, b));
            setpz2(xq+s, subpz2(a, b));
        }
    }
};

template <> struct fwdnend<2,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const noexcept
    {
        zeroupper();
        static const xmm rN = { 1.0/2, 1.0/2 };
        const xmm a = mulpd(rN, getpz(x[0]));
        const xmm b = mulpd(rN, getpz(x[1]));
        setpz(x[0], addpz(a, b));
        setpz(x[1], subpz(a, b));
    }
};

///////////////////////////////////////////////////////////////////////////////
// Forward FFT
///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct fwd0fft
{
    inline void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        fwd0fft<n/4,4*s,!eo>()(y, x, W);
        fwdcore<n,s>()(x, y, W);
    }
};

template <int s, bool eo> struct fwd0fft<4,s,eo>
{
    inline void operator()(
            complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        fwd0end<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwd0fft<2,s,eo>
{
    inline void operator()(
            complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        fwd0end<2,s,eo>()(x, y);
    }
};

//-----------------------------------------------------------------------------

template <int n, int s, bool eo> struct fwdnfft
{
    inline void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        fwdnfft<n/4,4*s,!eo>()(y, x, W);
        fwdcore<n, s>()(x, y, W);
    }
};

template <int s, bool eo> struct fwdnfft<4,s,eo>
{
    inline void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        fwdnend<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwdnfft<2,s,eo>
{
    inline void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        fwdnend<2,s,eo>()(x, y);
    }
};

///////////////////////////////////////////////////////////////////////////////
// Inverse butterfly operation
///////////////////////////////////////////////////////////////////////////////

template <int n, int s> struct invcore
{
    static const int n1 = n/4;
    static const int N  = n*s;
    static const int N0 = 0;
    static const int N1 = N/4;
    static const int N2 = N1*2;
    static const int N3 = N1*3;

    void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        for (int p = 0; p < n1; p++) {
            const int sp = s*p;
            const int s4p = 4*sp;
            //const ymm w1p = duppz2(getpz(W[N-sp]));
            const ymm w1p = duppz3(W[N-1*sp]);
            const ymm w2p = duppz3(W[N-2*sp]);
            const ymm w3p = duppz3(W[N-3*sp]);
            for (int q = 0; q < s; q += 2) {
                complex_vector xq_sp  = x + q + sp;
                complex_vector yq_s4p = y + q + s4p;
                const ymm a =             getpz2(yq_s4p+s*0);
                const ymm b = mulpz2(w1p, getpz2(yq_s4p+s*1));
                const ymm c = mulpz2(w2p, getpz2(yq_s4p+s*2));
                const ymm d = mulpz2(w3p, getpz2(yq_s4p+s*3));
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(xq_sp+N0, addpz2(apc,  bpd));
                setpz2(xq_sp+N1, addpz2(amc, jbmd));
                setpz2(xq_sp+N2, subpz2(apc,  bpd));
                setpz2(xq_sp+N3, subpz2(amc, jbmd));
            }
        }
    }
};

template <int N> struct invcore<N,1>
{
    static const int N0 = 0;
    static const int N1 = N/4;
    static const int N2 = N1*2;
    static const int N3 = N1*3;

    void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        const_complex_vector WN = W + N;
        for (int p = 0; p < N1; p += 2) {
            complex_vector x_p  = x + p;
            complex_vector y_4p = y + 4*p;
            //const ymm w1p = getwp2<-1>(WN,p);
            const ymm w1p = cnjpz2(getpz2(W+p));
            const ymm w2p = getwp2<-2>(WN,p);
            const ymm w3p = getwp2<-3>(WN,p);
#if 0
            const ymm a =             getpz3<4>(y_4p+0);
            const ymm b = mulpz2(w1p, getpz3<4>(y_4p+1));
            const ymm c = mulpz2(w2p, getpz3<4>(y_4p+2));
            const ymm d = mulpz2(w3p, getpz3<4>(y_4p+3));
#else
            const ymm ab = getpz2(y_4p+0);
            const ymm cd = getpz2(y_4p+2);
            const ymm ef = getpz2(y_4p+4);
            const ymm gh = getpz2(y_4p+6);
            const ymm a =             catlo(ab, ef);
            const ymm b = mulpz2(w1p, cathi(ab, ef));
            const ymm c = mulpz2(w2p, catlo(cd, gh));
            const ymm d = mulpz2(w3p, cathi(cd, gh));
#endif
            const ymm  apc =       addpz2(a, c);
            const ymm  amc =       subpz2(a, c);
            const ymm  bpd =       addpz2(b, d);
            const ymm jbmd = jxpz2(subpz2(b, d));
            setpz2(x_p+N0, addpz2(apc,  bpd));
            setpz2(x_p+N1, addpz2(amc, jbmd));
            setpz2(x_p+N2, subpz2(apc,  bpd));
            setpz2(x_p+N3, subpz2(amc, jbmd));
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct inv0end;

//-----------------------------------------------------------------------------

template <int s> struct inv0end<4,s,1>
{
    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
            const ymm a = getpz2(yq+s*0);
            const ymm b = getpz2(yq+s*1);
            const ymm c = getpz2(yq+s*2);
            const ymm d = getpz2(yq+s*3);
            const ymm  apc =       addpz2(a, c);
            const ymm  amc =       subpz2(a, c);
            const ymm  bpd =       addpz2(b, d);
            const ymm jbmd = jxpz2(subpz2(b, d));
            setpz2(xq+s*0, addpz2(apc,  bpd));
            setpz2(xq+s*1, addpz2(amc, jbmd));
            setpz2(xq+s*2, subpz2(apc,  bpd));
            setpz2(xq+s*3, subpz2(amc, jbmd));
        }
    }
};

template <> struct inv0end<4,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        zeroupper();
        const xmm a = getpz(y[0]);
        const xmm b = getpz(y[1]);
        const xmm c = getpz(y[2]);
        const xmm d = getpz(y[3]);
        const xmm  apc =      addpz(a, c);
        const xmm  amc =      subpz(a, c);
        const xmm  bpd =      addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(x[0], addpz(apc,  bpd));
        setpz(x[1], addpz(amc, jbmd));
        setpz(x[2], subpz(apc,  bpd));
        setpz(x[3], subpz(amc, jbmd));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct inv0end<4,s,0>
{
    void operator()(complex_vector x, complex_vector) const noexcept
    {
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            const ymm a = getpz2(xq+s*0);
            const ymm b = getpz2(xq+s*1);
            const ymm c = getpz2(xq+s*2);
            const ymm d = getpz2(xq+s*3);
            const ymm  apc =       addpz2(a, c);
            const ymm  amc =       subpz2(a, c);
            const ymm  bpd =       addpz2(b, d);
            const ymm jbmd = jxpz2(subpz2(b, d));
            setpz2(xq+s*0, addpz2(apc,  bpd));
            setpz2(xq+s*1, addpz2(amc, jbmd));
            setpz2(xq+s*2, subpz2(apc,  bpd));
            setpz2(xq+s*3, subpz2(amc, jbmd));
        }
    }
};

template <> struct inv0end<4,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const noexcept
    {
        zeroupper();
        const xmm a = getpz(x[0]);
        const xmm b = getpz(x[1]);
        const xmm c = getpz(x[2]);
        const xmm d = getpz(x[3]);
        const xmm  apc =      addpz(a, c);
        const xmm  amc =      subpz(a, c);
        const xmm  bpd =      addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(x[0], addpz(apc,  bpd));
        setpz(x[1], addpz(amc, jbmd));
        setpz(x[2], subpz(apc,  bpd));
        setpz(x[3], subpz(amc, jbmd));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct inv0end<2,s,1>
{
    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
            const ymm a = getpz2(yq+0);
            const ymm b = getpz2(yq+s);
            setpz2(xq+0, addpz2(a, b));
            setpz2(xq+s, subpz2(a, b));
        }
    }
};

template <> struct inv0end<2,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        zeroupper();
        const xmm a = getpz(y[0]);
        const xmm b = getpz(y[1]);
        setpz(x[0], addpz(a, b));
        setpz(x[1], subpz(a, b));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct inv0end<2,s,0>
{
    void operator()(complex_vector x, complex_vector) const noexcept
    {
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            const ymm a = getpz2(xq+0);
            const ymm b = getpz2(xq+s);
            setpz2(xq+0, addpz2(a, b));
            setpz2(xq+s, subpz2(a, b));
        }
    }
};

template <> struct inv0end<2,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const noexcept
    {
        zeroupper();
        const xmm a = getpz(x[0]);
        const xmm b = getpz(x[1]);
        setpz(x[0], addpz(a, b));
        setpz(x[1], subpz(a, b));
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct invnend;

//-----------------------------------------------------------------------------

template <int s> struct invnend<4,s,1>
{
    static const int N  = 4*s;

    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
            const ymm a = mulpd2(rN, getpz2(yq+s*0));
            const ymm b = mulpd2(rN, getpz2(yq+s*1));
            const ymm c = mulpd2(rN, getpz2(yq+s*2));
            const ymm d = mulpd2(rN, getpz2(yq+s*3));
            const ymm  apc =       addpz2(a, c);
            const ymm  amc =       subpz2(a, c);
            const ymm  bpd =       addpz2(b, d);
            const ymm jbmd = jxpz2(subpz2(b, d));
            setpz2(xq+s*0, addpz2(apc,  bpd));
            setpz2(xq+s*1, addpz2(amc, jbmd));
            setpz2(xq+s*2, subpz2(apc,  bpd));
            setpz2(xq+s*3, subpz2(amc, jbmd));
        }
    }
};

template <> struct invnend<4,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        zeroupper();
        static const xmm rN = { 1.0/4, 1.0/4 };
        const xmm a = mulpd(rN, getpz(y[0]));
        const xmm b = mulpd(rN, getpz(y[1]));
        const xmm c = mulpd(rN, getpz(y[2]));
        const xmm d = mulpd(rN, getpz(y[3]));
        const xmm  apc =      addpz(a, c);
        const xmm  amc =      subpz(a, c);
        const xmm  bpd =      addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(x[0], addpz(apc,  bpd));
        setpz(x[1], addpz(amc, jbmd));
        setpz(x[2], subpz(apc,  bpd));
        setpz(x[3], subpz(amc, jbmd));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct invnend<4,s,0>
{
    static const int N  = 4*s;

    void operator()(complex_vector x, complex_vector) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            const ymm a = mulpd2(rN, getpz2(xq+s*0));
            const ymm b = mulpd2(rN, getpz2(xq+s*1));
            const ymm c = mulpd2(rN, getpz2(xq+s*2));
            const ymm d = mulpd2(rN, getpz2(xq+s*3));
            const ymm  apc =       addpz2(a, c);
            const ymm  amc =       subpz2(a, c);
            const ymm  bpd =       addpz2(b, d);
            const ymm jbmd = jxpz2(subpz2(b, d));
            setpz2(xq+s*0, addpz2(apc,  bpd));
            setpz2(xq+s*1, addpz2(amc, jbmd));
            setpz2(xq+s*2, subpz2(apc,  bpd));
            setpz2(xq+s*3, subpz2(amc, jbmd));
        }
    }
};

template <> struct invnend<4,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const noexcept
    {
        zeroupper();
        static const xmm rN = { 1.0/4, 1.0/4 };
        const xmm a = mulpd(rN, getpz(x[0]));
        const xmm b = mulpd(rN, getpz(x[1]));
        const xmm c = mulpd(rN, getpz(x[2]));
        const xmm d = mulpd(rN, getpz(x[3]));
        const xmm  apc =      addpz(a, c);
        const xmm  amc =      subpz(a, c);
        const xmm  bpd =      addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(x[0], addpz(apc,  bpd));
        setpz(x[1], addpz(amc, jbmd));
        setpz(x[2], subpz(apc,  bpd));
        setpz(x[3], subpz(amc, jbmd));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct invnend<2,s,1>
{
    static const int N  = 2*s;

    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
            const ymm a = mulpd2(rN, getpz2(yq+0));
            const ymm b = mulpd2(rN, getpz2(yq+s));
            setpz2(xq+0, addpz2(a, b));
            setpz2(xq+s, subpz2(a, b));
        }
    }
};

template <> struct invnend<2,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        zeroupper();
        static const xmm rN = { 1.0/2, 1.0/2 };
        const xmm a = mulpd(rN, getpz(y[0]));
        const xmm b = mulpd(rN, getpz(y[1]));
        setpz(x[0], addpz(a, b));
        setpz(x[1], subpz(a, b));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct invnend<2,s,0>
{
    static const int N  = 2*s;

    void operator()(complex_vector x, complex_vector) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            const ymm a = mulpd2(rN, getpz2(xq+0));
            const ymm b = mulpd2(rN, getpz2(xq+s));
            setpz2(xq+0, addpz2(a, b));
            setpz2(xq+s, subpz2(a, b));
        }
    }
};

template <> struct invnend<2,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const noexcept
    {
        zeroupper();
        static const xmm rN = { 1.0/2, 1.0/2 };
        const xmm a = mulpd(rN, getpz(x[0]));
        const xmm b = mulpd(rN, getpz(x[1]));
        setpz(x[0], addpz(a, b));
        setpz(x[1], subpz(a, b));
    }
};

///////////////////////////////////////////////////////////////////////////////
// Inverse FFT
///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct inv0fft
{
    inline void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        inv0fft<n/4,4*s,!eo>()(y, x, W);
        invcore<n,s>()(x, y, W);
    }
};

template <int s, bool eo> struct inv0fft<4,s,eo>
{
    inline void operator()(
            complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        inv0end<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct inv0fft<2,s,eo>
{
    inline void operator()(
            complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        inv0end<2,s,eo>()(x, y);
    }
};

//-----------------------------------------------------------------------------

template <int n, int s, bool eo> struct invnfft
{
    inline void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        invnfft<n/4,4*s,!eo>()(y, x, W);
        invcore<n,s>()(x, y, W);
    }
};

template <int s, bool eo> struct invnfft<4,s,eo>
{
    inline void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        invnend<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct invnfft<2,s,eo>
{
    inline void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        invnend<2,s,eo>()(x, y);
    }
};

///////////////////////////////////////////////////////////////////////////////
// FFT object
///////////////////////////////////////////////////////////////////////////////

struct FFT0
{
    int N, log_N;
    simd_array<complex_t> weight;
    complex_t* __restrict W;

    FFT0() noexcept : N(0), log_N(0), W(0) {}
    FFT0(const int n) { setup(n); }

    void setup(int n)
    {
        for (log_N = 0; n > 1; n >>= 1) log_N++;
        setup2(log_N);
    }

    inline void setup2(const int n)
    {
        log_N = n; N = 1 << n;
        weight.setup(N+1); W = &weight;
        init_W(N, W);
    }

    ///////////////////////////////////////////////////////////////////////////

    inline void fwd(complex_vector x, complex_vector y) const noexcept
    {
        if (N < OMP_THRESHOLD) {
            switch (log_N) {
                case  0: break;
                case  1: fwdnfft<(1<< 1),1,0>()(x, y, W); break;
                case  2: fwdnfft<(1<< 2),1,0>()(x, y, W); break;
                case  3: fwdnfft<(1<< 3),1,0>()(x, y, W); break;
                case  4: fwdnfft<(1<< 4),1,0>()(x, y, W); break;
                case  5: fwdnfft<(1<< 5),1,0>()(x, y, W); break;
                case  6: fwdnfft<(1<< 6),1,0>()(x, y, W); break;
                case  7: fwdnfft<(1<< 7),1,0>()(x, y, W); break;
                case  8: fwdnfft<(1<< 8),1,0>()(x, y, W); break;
                case  9: fwdnfft<(1<< 9),1,0>()(x, y, W); break;
                case 10: fwdnfft<(1<<10),1,0>()(x, y, W); break;
                case 11: fwdnfft<(1<<11),1,0>()(x, y, W); break;
                case 12: fwdnfft<(1<<12),1,0>()(x, y, W); break;
                case 13: fwdnfft<(1<<13),1,0>()(x, y, W); break;
                case 14: fwdnfft<(1<<14),1,0>()(x, y, W); break;
                case 15: fwdnfft<(1<<15),1,0>()(x, y, W); break;
                case 16: fwdnfft<(1<<16),1,0>()(x, y, W); break;
                case 17: fwdnfft<(1<<17),1,0>()(x, y, W); break;
                case 18: fwdnfft<(1<<18),1,0>()(x, y, W); break;
                case 19: fwdnfft<(1<<19),1,0>()(x, y, W); break;
                case 20: fwdnfft<(1<<20),1,0>()(x, y, W); break;
                case 21: fwdnfft<(1<<21),1,0>()(x, y, W); break;
                case 22: fwdnfft<(1<<22),1,0>()(x, y, W); break;
                case 23: fwdnfft<(1<<23),1,0>()(x, y, W); break;
                case 24: fwdnfft<(1<<24),1,0>()(x, y, W); break;
            }
        }
        else OTFFT_AVXDIT4omp::fwd(log_N, x, y, W);
    }

    inline void fwd0(complex_vector x, complex_vector y) const noexcept
    {
        if (N < OMP_THRESHOLD) {
            switch (log_N) {
                case  0: break;
                case  1: fwd0fft<(1<< 1),1,0>()(x, y, W); break;
                case  2: fwd0fft<(1<< 2),1,0>()(x, y, W); break;
                case  3: fwd0fft<(1<< 3),1,0>()(x, y, W); break;
                case  4: fwd0fft<(1<< 4),1,0>()(x, y, W); break;
                case  5: fwd0fft<(1<< 5),1,0>()(x, y, W); break;
                case  6: fwd0fft<(1<< 6),1,0>()(x, y, W); break;
                case  7: fwd0fft<(1<< 7),1,0>()(x, y, W); break;
                case  8: fwd0fft<(1<< 8),1,0>()(x, y, W); break;
                case  9: fwd0fft<(1<< 9),1,0>()(x, y, W); break;
                case 10: fwd0fft<(1<<10),1,0>()(x, y, W); break;
                case 11: fwd0fft<(1<<11),1,0>()(x, y, W); break;
                case 12: fwd0fft<(1<<12),1,0>()(x, y, W); break;
                case 13: fwd0fft<(1<<13),1,0>()(x, y, W); break;
                case 14: fwd0fft<(1<<14),1,0>()(x, y, W); break;
                case 15: fwd0fft<(1<<15),1,0>()(x, y, W); break;
                case 16: fwd0fft<(1<<16),1,0>()(x, y, W); break;
                case 17: fwd0fft<(1<<17),1,0>()(x, y, W); break;
                case 18: fwd0fft<(1<<18),1,0>()(x, y, W); break;
                case 19: fwd0fft<(1<<19),1,0>()(x, y, W); break;
                case 20: fwd0fft<(1<<20),1,0>()(x, y, W); break;
                case 21: fwd0fft<(1<<21),1,0>()(x, y, W); break;
                case 22: fwd0fft<(1<<22),1,0>()(x, y, W); break;
                case 23: fwd0fft<(1<<23),1,0>()(x, y, W); break;
                case 24: fwd0fft<(1<<24),1,0>()(x, y, W); break;
            }
        }
        else OTFFT_AVXDIT4omp::fwd0(log_N, x, y, W);
    }

    inline void fwdn(complex_vector x, complex_vector y) const noexcept { fwd(x, y); }

    inline void fwd0o(complex_vector x, complex_vector y) const noexcept
    {
        if (N < OMP_THRESHOLD) {
            switch (log_N) {
                case  0: y[0] = x[0]; break;
                case  1: fwd0fft<(1<< 1),1,1>()(y, x, W); break;
                case  2: fwd0fft<(1<< 2),1,1>()(y, x, W); break;
                case  3: fwd0fft<(1<< 3),1,1>()(y, x, W); break;
                case  4: fwd0fft<(1<< 4),1,1>()(y, x, W); break;
                case  5: fwd0fft<(1<< 5),1,1>()(y, x, W); break;
                case  6: fwd0fft<(1<< 6),1,1>()(y, x, W); break;
                case  7: fwd0fft<(1<< 7),1,1>()(y, x, W); break;
                case  8: fwd0fft<(1<< 8),1,1>()(y, x, W); break;
                case  9: fwd0fft<(1<< 9),1,1>()(y, x, W); break;
                case 10: fwd0fft<(1<<10),1,1>()(y, x, W); break;
                case 11: fwd0fft<(1<<11),1,1>()(y, x, W); break;
                case 12: fwd0fft<(1<<12),1,1>()(y, x, W); break;
                case 13: fwd0fft<(1<<13),1,1>()(y, x, W); break;
                case 14: fwd0fft<(1<<14),1,1>()(y, x, W); break;
                case 15: fwd0fft<(1<<15),1,1>()(y, x, W); break;
                case 16: fwd0fft<(1<<16),1,1>()(y, x, W); break;
                case 17: fwd0fft<(1<<17),1,1>()(y, x, W); break;
                case 18: fwd0fft<(1<<18),1,1>()(y, x, W); break;
                case 19: fwd0fft<(1<<19),1,1>()(y, x, W); break;
                case 20: fwd0fft<(1<<20),1,1>()(y, x, W); break;
                case 21: fwd0fft<(1<<21),1,1>()(y, x, W); break;
                case 22: fwd0fft<(1<<22),1,1>()(y, x, W); break;
                case 23: fwd0fft<(1<<23),1,1>()(y, x, W); break;
                case 24: fwd0fft<(1<<24),1,1>()(y, x, W); break;
            }
        }
        else OTFFT_AVXDIT4omp::fwd0o(log_N, x, y, W);
    }

    inline void fwdno(complex_vector x, complex_vector y) const noexcept
    {
        if (N < OMP_THRESHOLD) {
            switch (log_N) {
                case  0: y[0] = x[0]; break;
                case  1: fwdnfft<(1<< 1),1,1>()(y, x, W); break;
                case  2: fwdnfft<(1<< 2),1,1>()(y, x, W); break;
                case  3: fwdnfft<(1<< 3),1,1>()(y, x, W); break;
                case  4: fwdnfft<(1<< 4),1,1>()(y, x, W); break;
                case  5: fwdnfft<(1<< 5),1,1>()(y, x, W); break;
                case  6: fwdnfft<(1<< 6),1,1>()(y, x, W); break;
                case  7: fwdnfft<(1<< 7),1,1>()(y, x, W); break;
                case  8: fwdnfft<(1<< 8),1,1>()(y, x, W); break;
                case  9: fwdnfft<(1<< 9),1,1>()(y, x, W); break;
                case 10: fwdnfft<(1<<10),1,1>()(y, x, W); break;
                case 11: fwdnfft<(1<<11),1,1>()(y, x, W); break;
                case 12: fwdnfft<(1<<12),1,1>()(y, x, W); break;
                case 13: fwdnfft<(1<<13),1,1>()(y, x, W); break;
                case 14: fwdnfft<(1<<14),1,1>()(y, x, W); break;
                case 15: fwdnfft<(1<<15),1,1>()(y, x, W); break;
                case 16: fwdnfft<(1<<16),1,1>()(y, x, W); break;
                case 17: fwdnfft<(1<<17),1,1>()(y, x, W); break;
                case 18: fwdnfft<(1<<18),1,1>()(y, x, W); break;
                case 19: fwdnfft<(1<<19),1,1>()(y, x, W); break;
                case 20: fwdnfft<(1<<20),1,1>()(y, x, W); break;
                case 21: fwdnfft<(1<<21),1,1>()(y, x, W); break;
                case 22: fwdnfft<(1<<22),1,1>()(y, x, W); break;
                case 23: fwdnfft<(1<<23),1,1>()(y, x, W); break;
                case 24: fwdnfft<(1<<24),1,1>()(y, x, W); break;
            }
        }
        else OTFFT_AVXDIT4omp::fwdno(log_N, x, y, W);
    }

    ///////////////////////////////////////////////////////////////////////////

    inline void inv(complex_vector x, complex_vector y) const noexcept
    {
        if (N < OMP_THRESHOLD) {
            switch (log_N) {
                case  0: break;
                case  1: inv0fft<(1<< 1),1,0>()(x, y, W); break;
                case  2: inv0fft<(1<< 2),1,0>()(x, y, W); break;
                case  3: inv0fft<(1<< 3),1,0>()(x, y, W); break;
                case  4: inv0fft<(1<< 4),1,0>()(x, y, W); break;
                case  5: inv0fft<(1<< 5),1,0>()(x, y, W); break;
                case  6: inv0fft<(1<< 6),1,0>()(x, y, W); break;
                case  7: inv0fft<(1<< 7),1,0>()(x, y, W); break;
                case  8: inv0fft<(1<< 8),1,0>()(x, y, W); break;
                case  9: inv0fft<(1<< 9),1,0>()(x, y, W); break;
                case 10: inv0fft<(1<<10),1,0>()(x, y, W); break;
                case 11: inv0fft<(1<<11),1,0>()(x, y, W); break;
                case 12: inv0fft<(1<<12),1,0>()(x, y, W); break;
                case 13: inv0fft<(1<<13),1,0>()(x, y, W); break;
                case 14: inv0fft<(1<<14),1,0>()(x, y, W); break;
                case 15: inv0fft<(1<<15),1,0>()(x, y, W); break;
                case 16: inv0fft<(1<<16),1,0>()(x, y, W); break;
                case 17: inv0fft<(1<<17),1,0>()(x, y, W); break;
                case 18: inv0fft<(1<<18),1,0>()(x, y, W); break;
                case 19: inv0fft<(1<<19),1,0>()(x, y, W); break;
                case 20: inv0fft<(1<<20),1,0>()(x, y, W); break;
                case 21: inv0fft<(1<<21),1,0>()(x, y, W); break;
                case 22: inv0fft<(1<<22),1,0>()(x, y, W); break;
                case 23: inv0fft<(1<<23),1,0>()(x, y, W); break;
                case 24: inv0fft<(1<<24),1,0>()(x, y, W); break;
            }
        }
        else OTFFT_AVXDIT4omp::inv(log_N, x, y, W);
    }

    inline void inv0(complex_vector x, complex_vector y) const noexcept { inv(x, y); }

    inline void invn(complex_vector x, complex_vector y) const noexcept
    {
        if (N < OMP_THRESHOLD) {
            switch (log_N) {
                case  0: break;
                case  1: invnfft<(1<< 1),1,0>()(x, y, W); break;
                case  2: invnfft<(1<< 2),1,0>()(x, y, W); break;
                case  3: invnfft<(1<< 3),1,0>()(x, y, W); break;
                case  4: invnfft<(1<< 4),1,0>()(x, y, W); break;
                case  5: invnfft<(1<< 5),1,0>()(x, y, W); break;
                case  6: invnfft<(1<< 6),1,0>()(x, y, W); break;
                case  7: invnfft<(1<< 7),1,0>()(x, y, W); break;
                case  8: invnfft<(1<< 8),1,0>()(x, y, W); break;
                case  9: invnfft<(1<< 9),1,0>()(x, y, W); break;
                case 10: invnfft<(1<<10),1,0>()(x, y, W); break;
                case 11: invnfft<(1<<11),1,0>()(x, y, W); break;
                case 12: invnfft<(1<<12),1,0>()(x, y, W); break;
                case 13: invnfft<(1<<13),1,0>()(x, y, W); break;
                case 14: invnfft<(1<<14),1,0>()(x, y, W); break;
                case 15: invnfft<(1<<15),1,0>()(x, y, W); break;
                case 16: invnfft<(1<<16),1,0>()(x, y, W); break;
                case 17: invnfft<(1<<17),1,0>()(x, y, W); break;
                case 18: invnfft<(1<<18),1,0>()(x, y, W); break;
                case 19: invnfft<(1<<19),1,0>()(x, y, W); break;
                case 20: invnfft<(1<<20),1,0>()(x, y, W); break;
                case 21: invnfft<(1<<21),1,0>()(x, y, W); break;
                case 22: invnfft<(1<<22),1,0>()(x, y, W); break;
                case 23: invnfft<(1<<23),1,0>()(x, y, W); break;
                case 24: invnfft<(1<<24),1,0>()(x, y, W); break;
            }
        }
        else OTFFT_AVXDIT4omp::invn(log_N, x, y, W);
    }

    inline void inv0o(complex_vector x, complex_vector y) const noexcept
    {
        if (N < OMP_THRESHOLD) {
            switch (log_N) {
                case  0: y[0] = x[0]; break;
                case  1: inv0fft<(1<< 1),1,1>()(y, x, W); break;
                case  2: inv0fft<(1<< 2),1,1>()(y, x, W); break;
                case  3: inv0fft<(1<< 3),1,1>()(y, x, W); break;
                case  4: inv0fft<(1<< 4),1,1>()(y, x, W); break;
                case  5: inv0fft<(1<< 5),1,1>()(y, x, W); break;
                case  6: inv0fft<(1<< 6),1,1>()(y, x, W); break;
                case  7: inv0fft<(1<< 7),1,1>()(y, x, W); break;
                case  8: inv0fft<(1<< 8),1,1>()(y, x, W); break;
                case  9: inv0fft<(1<< 9),1,1>()(y, x, W); break;
                case 10: inv0fft<(1<<10),1,1>()(y, x, W); break;
                case 11: inv0fft<(1<<11),1,1>()(y, x, W); break;
                case 12: inv0fft<(1<<12),1,1>()(y, x, W); break;
                case 13: inv0fft<(1<<13),1,1>()(y, x, W); break;
                case 14: inv0fft<(1<<14),1,1>()(y, x, W); break;
                case 15: inv0fft<(1<<15),1,1>()(y, x, W); break;
                case 16: inv0fft<(1<<16),1,1>()(y, x, W); break;
                case 17: inv0fft<(1<<17),1,1>()(y, x, W); break;
                case 18: inv0fft<(1<<18),1,1>()(y, x, W); break;
                case 19: inv0fft<(1<<19),1,1>()(y, x, W); break;
                case 20: inv0fft<(1<<20),1,1>()(y, x, W); break;
                case 21: inv0fft<(1<<21),1,1>()(y, x, W); break;
                case 22: inv0fft<(1<<22),1,1>()(y, x, W); break;
                case 23: inv0fft<(1<<23),1,1>()(y, x, W); break;
                case 24: inv0fft<(1<<24),1,1>()(y, x, W); break;
            }
        }
        else OTFFT_AVXDIT4omp::inv0o(log_N, x, y, W);
    }

    inline void invno(complex_vector x, complex_vector y) const noexcept
    {
        if (N < OMP_THRESHOLD) {
            switch (log_N) {
                case  0: y[0] = x[0]; break;
                case  1: invnfft<(1<< 1),1,1>()(y, x, W); break;
                case  2: invnfft<(1<< 2),1,1>()(y, x, W); break;
                case  3: invnfft<(1<< 3),1,1>()(y, x, W); break;
                case  4: invnfft<(1<< 4),1,1>()(y, x, W); break;
                case  5: invnfft<(1<< 5),1,1>()(y, x, W); break;
                case  6: invnfft<(1<< 6),1,1>()(y, x, W); break;
                case  7: invnfft<(1<< 7),1,1>()(y, x, W); break;
                case  8: invnfft<(1<< 8),1,1>()(y, x, W); break;
                case  9: invnfft<(1<< 9),1,1>()(y, x, W); break;
                case 10: invnfft<(1<<10),1,1>()(y, x, W); break;
                case 11: invnfft<(1<<11),1,1>()(y, x, W); break;
                case 12: invnfft<(1<<12),1,1>()(y, x, W); break;
                case 13: invnfft<(1<<13),1,1>()(y, x, W); break;
                case 14: invnfft<(1<<14),1,1>()(y, x, W); break;
                case 15: invnfft<(1<<15),1,1>()(y, x, W); break;
                case 16: invnfft<(1<<16),1,1>()(y, x, W); break;
                case 17: invnfft<(1<<17),1,1>()(y, x, W); break;
                case 18: invnfft<(1<<18),1,1>()(y, x, W); break;
                case 19: invnfft<(1<<19),1,1>()(y, x, W); break;
                case 20: invnfft<(1<<20),1,1>()(y, x, W); break;
                case 21: invnfft<(1<<21),1,1>()(y, x, W); break;
                case 22: invnfft<(1<<22),1,1>()(y, x, W); break;
                case 23: invnfft<(1<<23),1,1>()(y, x, W); break;
                case 24: invnfft<(1<<24),1,1>()(y, x, W); break;
            }
        }
        else OTFFT_AVXDIT4omp::invno(log_N, x, y, W);
    }
};

#if 0
struct FFT {
    FFT0 fft;
    simd_array<complex_t> work;
    complex_t* __restrict y;

    FFT() : y(0) {}
    FFT(const int n) : fft(n), work(n), y(&work) {}

    inline void setup(const int n) { fft.setup(n); work.setup(n); y = &work; }

    inline void fwd(complex_vector x)  const noexcept { fft.fwd(x, y);  }
    inline void fwd0(complex_vector x) const noexcept { fft.fwd0(x, y); }
    inline void fwdn(complex_vector x) const noexcept { fft.fwdn(x, y); }
    inline void inv(complex_vector x)  const noexcept { fft.inv(x, y);  }
    inline void inv0(complex_vector x) const noexcept { fft.inv0(x, y); }
    inline void invn(complex_vector x) const noexcept { fft.invn(x, y); }
};
#endif

} /////////////////////////////////////////////////////////////////////////////

#endif // otfft_avxdit4_h
