/******************************************************************************
*  OTFFT AVXDIT(Radix-8) Version 6.4
*
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_avxdit8_h
#define otfft_avxdit8_h

#include "otfft/otfft_misc.h"
#include "otfft_avxdit4.h"
#include "otfft_avxdit8omp.h"

namespace OTFFT_AVXDIT8 { /////////////////////////////////////////////////////

using namespace OTFFT_MISC;

#ifdef DO_SINGLE_THREAD
static const int OMP_THRESHOLD = 1<<30;
#else
static const int OMP_THRESHOLD = 1<<15;
#endif

///////////////////////////////////////////////////////////////////////////////
// Forward buffterfly operation
///////////////////////////////////////////////////////////////////////////////

template <int n, int s> struct fwdcore
{
    static const int n1 = n/8;
    static const int N  = n*s;
    static const int N0 = 0;
    static const int N1 = N/8;
    static const int N2 = N1*2;
    static const int N3 = N1*3;
    static const int N4 = N1*4;
    static const int N5 = N1*5;
    static const int N6 = N1*6;
    static const int N7 = N1*7;

    void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        for (int p = 0; p < n1; p++) {
            const int sp = s*p;
            const int s8p = 8*sp;
            //const ymm w1p = duppz2(getpz(W[sp]));
            const ymm w1p = duppz3(W[1*sp]);
            const ymm w2p = duppz3(W[2*sp]);
            const ymm w3p = duppz3(W[3*sp]);
            const ymm w4p = mulpz2(w2p, w2p);
            const ymm w5p = mulpz2(w2p, w3p);
            const ymm w6p = mulpz2(w3p, w3p);
            const ymm w7p = mulpz2(w3p, w4p);
            for (int q = 0; q < s; q += 2) {
                complex_vector xq_sp  = x + q + sp;
                complex_vector yq_s8p = y + q + s8p;
                const ymm y0 =             getpz2(yq_s8p+s*0);
                const ymm y1 = mulpz2(w1p, getpz2(yq_s8p+s*1));
                const ymm y2 = mulpz2(w2p, getpz2(yq_s8p+s*2));
                const ymm y3 = mulpz2(w3p, getpz2(yq_s8p+s*3));
                const ymm y4 = mulpz2(w4p, getpz2(yq_s8p+s*4));
                const ymm y5 = mulpz2(w5p, getpz2(yq_s8p+s*5));
                const ymm y6 = mulpz2(w6p, getpz2(yq_s8p+s*6));
                const ymm y7 = mulpz2(w7p, getpz2(yq_s8p+s*7));
                const ymm  a04 =       addpz2(y0, y4);
                const ymm  s04 =       subpz2(y0, y4);
                const ymm  a26 =       addpz2(y2, y6);
                const ymm js26 = jxpz2(subpz2(y2, y6));
                const ymm  a15 =       addpz2(y1, y5);
                const ymm  s15 =       subpz2(y1, y5);
                const ymm  a37 =       addpz2(y3, y7);
                const ymm js37 = jxpz2(subpz2(y3, y7));
                const ymm    a04_p1_a26 =        addpz2(a04,  a26);
                const ymm    s04_mj_s26 =        subpz2(s04, js26);
                const ymm    a04_m1_a26 =        subpz2(a04,  a26);
                const ymm    s04_pj_s26 =        addpz2(s04, js26);
                const ymm    a15_p1_a37 =        addpz2(a15,  a37);
                const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
                const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
                const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
                setpz2(xq_sp+N0, addpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(xq_sp+N1, addpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(xq_sp+N2, subpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(xq_sp+N3, subpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(xq_sp+N4, subpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(xq_sp+N5, subpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(xq_sp+N6, addpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(xq_sp+N7, addpz2(s04_pj_s26, v8_s15_pj_s37));
            }
        }
    }
};

template <int N> struct fwdcore<N,1>
{
    static const int N0 = 0;
    static const int N1 = N/8;
    static const int N2 = N1*2;
    static const int N3 = N1*3;
    static const int N4 = N1*4;
    static const int N5 = N1*5;
    static const int N6 = N1*6;
    static const int N7 = N1*7;

    void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        for (int p = 0; p < N1; p += 2) {
            complex_vector x_p  = x + p;
            complex_vector y_8p = y + 8*p;
            const ymm w1p = getpz2(W+p);
            const ymm w2p = mulpz2(w1p, w1p);
            const ymm w3p = mulpz2(w1p, w2p);
            const ymm w4p = mulpz2(w2p, w2p);
            const ymm w5p = mulpz2(w2p, w3p);
            const ymm w6p = mulpz2(w3p, w3p);
            const ymm w7p = mulpz2(w3p, w4p);
#if 0
            const ymm y0 =             getpz3<8>(y_8p+0);
            const ymm y1 = mulpz2(w1p, getpz3<8>(y_8p+1));
            const ymm y2 = mulpz2(w2p, getpz3<8>(y_8p+2));
            const ymm y3 = mulpz2(w3p, getpz3<8>(y_8p+3));
            const ymm y4 = mulpz2(w4p, getpz3<8>(y_8p+4));
            const ymm y5 = mulpz2(w5p, getpz3<8>(y_8p+5));
            const ymm y6 = mulpz2(w6p, getpz3<8>(y_8p+6));
            const ymm y7 = mulpz2(w7p, getpz3<8>(y_8p+7));
#else
            const ymm ab = getpz2(y_8p+ 0);
            const ymm cd = getpz2(y_8p+ 2);
            const ymm ef = getpz2(y_8p+ 4);
            const ymm gh = getpz2(y_8p+ 6);
            const ymm AB = getpz2(y_8p+ 8);
            const ymm CD = getpz2(y_8p+10);
            const ymm EF = getpz2(y_8p+12);
            const ymm GH = getpz2(y_8p+14);
            const ymm y0 =             catlo(ab, AB);
            const ymm y1 = mulpz2(w1p, cathi(ab, AB));
            const ymm y2 = mulpz2(w2p, catlo(cd, CD));
            const ymm y3 = mulpz2(w3p, cathi(cd, CD));
            const ymm y4 = mulpz2(w4p, catlo(ef, EF));
            const ymm y5 = mulpz2(w5p, cathi(ef, EF));
            const ymm y6 = mulpz2(w6p, catlo(gh, GH));
            const ymm y7 = mulpz2(w7p, cathi(gh, GH));
#endif
            const ymm  a04 =       addpz2(y0, y4);
            const ymm  s04 =       subpz2(y0, y4);
            const ymm  a26 =       addpz2(y2, y6);
            const ymm js26 = jxpz2(subpz2(y2, y6));
            const ymm  a15 =       addpz2(y1, y5);
            const ymm  s15 =       subpz2(y1, y5);
            const ymm  a37 =       addpz2(y3, y7);
            const ymm js37 = jxpz2(subpz2(y3, y7));
            const ymm    a04_p1_a26 =        addpz2(a04,  a26);
            const ymm    s04_mj_s26 =        subpz2(s04, js26);
            const ymm    a04_m1_a26 =        subpz2(a04,  a26);
            const ymm    s04_pj_s26 =        addpz2(s04, js26);
            const ymm    a15_p1_a37 =        addpz2(a15,  a37);
            const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
            const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
            const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
            setpz2(x_p+N0, addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(x_p+N1, addpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(x_p+N2, subpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(x_p+N3, subpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(x_p+N4, subpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(x_p+N5, subpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(x_p+N6, addpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(x_p+N7, addpz2(s04_pj_s26, v8_s15_pj_s37));
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct fwd0end;

//-----------------------------------------------------------------------------

template <int s> struct fwd0end<8,s,1>
{
    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
            const ymm y0 = getpz2(yq+s*0);
            const ymm y1 = getpz2(yq+s*1);
            const ymm y2 = getpz2(yq+s*2);
            const ymm y3 = getpz2(yq+s*3);
            const ymm y4 = getpz2(yq+s*4);
            const ymm y5 = getpz2(yq+s*5);
            const ymm y6 = getpz2(yq+s*6);
            const ymm y7 = getpz2(yq+s*7);
            const ymm  a04 =       addpz2(y0, y4);
            const ymm  s04 =       subpz2(y0, y4);
            const ymm  a26 =       addpz2(y2, y6);
            const ymm js26 = jxpz2(subpz2(y2, y6));
            const ymm  a15 =       addpz2(y1, y5);
            const ymm  s15 =       subpz2(y1, y5);
            const ymm  a37 =       addpz2(y3, y7);
            const ymm js37 = jxpz2(subpz2(y3, y7));
            const ymm    a04_p1_a26 =        addpz2(a04,  a26);
            const ymm    s04_mj_s26 =        subpz2(s04, js26);
            const ymm    a04_m1_a26 =        subpz2(a04,  a26);
            const ymm    s04_pj_s26 =        addpz2(s04, js26);
            const ymm    a15_p1_a37 =        addpz2(a15,  a37);
            const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
            const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
            const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
            setpz2(xq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(xq+s*1, addpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(xq+s*2, subpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(xq+s*3, subpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(xq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(xq+s*5, subpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(xq+s*6, addpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(xq+s*7, addpz2(s04_pj_s26, v8_s15_pj_s37));
        }
    }
};

template <> struct fwd0end<8,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        zeroupper();
        const xmm y0 = getpz(y[0]);
        const xmm y1 = getpz(y[1]);
        const xmm y2 = getpz(y[2]);
        const xmm y3 = getpz(y[3]);
        const xmm y4 = getpz(y[4]);
        const xmm y5 = getpz(y[5]);
        const xmm y6 = getpz(y[6]);
        const xmm y7 = getpz(y[7]);
        const xmm  a04 =      addpz(y0, y4);
        const xmm  s04 =      subpz(y0, y4);
        const xmm  a26 =      addpz(y2, y6);
        const xmm js26 = jxpz(subpz(y2, y6));
        const xmm  a15 =      addpz(y1, y5);
        const xmm  s15 =      subpz(y1, y5);
        const xmm  a37 =      addpz(y3, y7);
        const xmm js37 = jxpz(subpz(y3, y7));
        const xmm    a04_p1_a26 =       addpz(a04,  a26);
        const xmm    s04_mj_s26 =       subpz(s04, js26);
        const xmm    a04_m1_a26 =       subpz(a04,  a26);
        const xmm    s04_pj_s26 =       addpz(s04, js26);
        const xmm    a15_p1_a37 =       addpz(a15,  a37);
        const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
        const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
        const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
        setpz(x[0], addpz(a04_p1_a26,    a15_p1_a37));
        setpz(x[1], addpz(s04_mj_s26, w8_s15_mj_s37));
        setpz(x[2], subpz(a04_m1_a26,  j_a15_m1_a37));
        setpz(x[3], subpz(s04_pj_s26, v8_s15_pj_s37));
        setpz(x[4], subpz(a04_p1_a26,    a15_p1_a37));
        setpz(x[5], subpz(s04_mj_s26, w8_s15_mj_s37));
        setpz(x[6], addpz(a04_m1_a26,  j_a15_m1_a37));
        setpz(x[7], addpz(s04_pj_s26, v8_s15_pj_s37));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct fwd0end<8,s,0>
{
    void operator()(complex_vector x, complex_vector) const noexcept
    {
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            const ymm x0 = getpz2(xq+s*0);
            const ymm x1 = getpz2(xq+s*1);
            const ymm x2 = getpz2(xq+s*2);
            const ymm x3 = getpz2(xq+s*3);
            const ymm x4 = getpz2(xq+s*4);
            const ymm x5 = getpz2(xq+s*5);
            const ymm x6 = getpz2(xq+s*6);
            const ymm x7 = getpz2(xq+s*7);
            const ymm  a04 =       addpz2(x0, x4);
            const ymm  s04 =       subpz2(x0, x4);
            const ymm  a26 =       addpz2(x2, x6);
            const ymm js26 = jxpz2(subpz2(x2, x6));
            const ymm  a15 =       addpz2(x1, x5);
            const ymm  s15 =       subpz2(x1, x5);
            const ymm  a37 =       addpz2(x3, x7);
            const ymm js37 = jxpz2(subpz2(x3, x7));
            const ymm    a04_p1_a26 =        addpz2(a04,  a26);
            const ymm    s04_mj_s26 =        subpz2(s04, js26);
            const ymm    a04_m1_a26 =        subpz2(a04,  a26);
            const ymm    s04_pj_s26 =        addpz2(s04, js26);
            const ymm    a15_p1_a37 =        addpz2(a15,  a37);
            const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
            const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
            const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
            setpz2(xq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(xq+s*1, addpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(xq+s*2, subpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(xq+s*3, subpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(xq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(xq+s*5, subpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(xq+s*6, addpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(xq+s*7, addpz2(s04_pj_s26, v8_s15_pj_s37));
        }
    }
};

template <> struct fwd0end<8,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const noexcept
    {
        zeroupper();
        const xmm x0 = getpz(x[0]);
        const xmm x1 = getpz(x[1]);
        const xmm x2 = getpz(x[2]);
        const xmm x3 = getpz(x[3]);
        const xmm x4 = getpz(x[4]);
        const xmm x5 = getpz(x[5]);
        const xmm x6 = getpz(x[6]);
        const xmm x7 = getpz(x[7]);
        const xmm  a04 =      addpz(x0, x4);
        const xmm  s04 =      subpz(x0, x4);
        const xmm  a26 =      addpz(x2, x6);
        const xmm js26 = jxpz(subpz(x2, x6));
        const xmm  a15 =      addpz(x1, x5);
        const xmm  s15 =      subpz(x1, x5);
        const xmm  a37 =      addpz(x3, x7);
        const xmm js37 = jxpz(subpz(x3, x7));
        const xmm    a04_p1_a26 =       addpz(a04,  a26);
        const xmm    s04_mj_s26 =       subpz(s04, js26);
        const xmm    a04_m1_a26 =       subpz(a04,  a26);
        const xmm    s04_pj_s26 =       addpz(s04, js26);
        const xmm    a15_p1_a37 =       addpz(a15,  a37);
        const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
        const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
        const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
        setpz(x[0], addpz(a04_p1_a26,    a15_p1_a37));
        setpz(x[1], addpz(s04_mj_s26, w8_s15_mj_s37));
        setpz(x[2], subpz(a04_m1_a26,  j_a15_m1_a37));
        setpz(x[3], subpz(s04_pj_s26, v8_s15_pj_s37));
        setpz(x[4], subpz(a04_p1_a26,    a15_p1_a37));
        setpz(x[5], subpz(s04_mj_s26, w8_s15_mj_s37));
        setpz(x[6], addpz(a04_m1_a26,  j_a15_m1_a37));
        setpz(x[7], addpz(s04_pj_s26, v8_s15_pj_s37));
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct fwdnend;

//-----------------------------------------------------------------------------

template <int s> struct fwdnend<8,s,1>
{
    static const int N = 8*s;

    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
            const ymm y0 = mulpd2(rN, getpz2(yq+s*0));
            const ymm y1 = mulpd2(rN, getpz2(yq+s*1));
            const ymm y2 = mulpd2(rN, getpz2(yq+s*2));
            const ymm y3 = mulpd2(rN, getpz2(yq+s*3));
            const ymm y4 = mulpd2(rN, getpz2(yq+s*4));
            const ymm y5 = mulpd2(rN, getpz2(yq+s*5));
            const ymm y6 = mulpd2(rN, getpz2(yq+s*6));
            const ymm y7 = mulpd2(rN, getpz2(yq+s*7));
            const ymm  a04 =       addpz2(y0, y4);
            const ymm  s04 =       subpz2(y0, y4);
            const ymm  a26 =       addpz2(y2, y6);
            const ymm js26 = jxpz2(subpz2(y2, y6));
            const ymm  a15 =       addpz2(y1, y5);
            const ymm  s15 =       subpz2(y1, y5);
            const ymm  a37 =       addpz2(y3, y7);
            const ymm js37 = jxpz2(subpz2(y3, y7));
            const ymm    a04_p1_a26 =        addpz2(a04,  a26);
            const ymm    s04_mj_s26 =        subpz2(s04, js26);
            const ymm    a04_m1_a26 =        subpz2(a04,  a26);
            const ymm    s04_pj_s26 =        addpz2(s04, js26);
            const ymm    a15_p1_a37 =        addpz2(a15,  a37);
            const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
            const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
            const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
            setpz2(xq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(xq+s*1, addpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(xq+s*2, subpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(xq+s*3, subpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(xq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(xq+s*5, subpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(xq+s*6, addpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(xq+s*7, addpz2(s04_pj_s26, v8_s15_pj_s37));
        }
    }
};

template <> struct fwdnend<8,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        zeroupper();
        static const xmm rN = { 1.0/8, 1.0/8 };
        const xmm y0 = mulpd(rN, getpz(y[0]));
        const xmm y1 = mulpd(rN, getpz(y[1]));
        const xmm y2 = mulpd(rN, getpz(y[2]));
        const xmm y3 = mulpd(rN, getpz(y[3]));
        const xmm y4 = mulpd(rN, getpz(y[4]));
        const xmm y5 = mulpd(rN, getpz(y[5]));
        const xmm y6 = mulpd(rN, getpz(y[6]));
        const xmm y7 = mulpd(rN, getpz(y[7]));
        const xmm  a04 =      addpz(y0, y4);
        const xmm  s04 =      subpz(y0, y4);
        const xmm  a26 =      addpz(y2, y6);
        const xmm js26 = jxpz(subpz(y2, y6));
        const xmm  a15 =      addpz(y1, y5);
        const xmm  s15 =      subpz(y1, y5);
        const xmm  a37 =      addpz(y3, y7);
        const xmm js37 = jxpz(subpz(y3, y7));
        const xmm    a04_p1_a26 =       addpz(a04,  a26);
        const xmm    s04_mj_s26 =       subpz(s04, js26);
        const xmm    a04_m1_a26 =       subpz(a04,  a26);
        const xmm    s04_pj_s26 =       addpz(s04, js26);
        const xmm    a15_p1_a37 =       addpz(a15,  a37);
        const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
        const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
        const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
        setpz(x[0], addpz(a04_p1_a26,    a15_p1_a37));
        setpz(x[1], addpz(s04_mj_s26, w8_s15_mj_s37));
        setpz(x[2], subpz(a04_m1_a26,  j_a15_m1_a37));
        setpz(x[3], subpz(s04_pj_s26, v8_s15_pj_s37));
        setpz(x[4], subpz(a04_p1_a26,    a15_p1_a37));
        setpz(x[5], subpz(s04_mj_s26, w8_s15_mj_s37));
        setpz(x[6], addpz(a04_m1_a26,  j_a15_m1_a37));
        setpz(x[7], addpz(s04_pj_s26, v8_s15_pj_s37));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct fwdnend<8,s,0>
{
    static const int N = 8*s;

    void operator()(complex_vector x, complex_vector) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            const ymm x0 = mulpd2(rN, getpz2(xq+s*0));
            const ymm x1 = mulpd2(rN, getpz2(xq+s*1));
            const ymm x2 = mulpd2(rN, getpz2(xq+s*2));
            const ymm x3 = mulpd2(rN, getpz2(xq+s*3));
            const ymm x4 = mulpd2(rN, getpz2(xq+s*4));
            const ymm x5 = mulpd2(rN, getpz2(xq+s*5));
            const ymm x6 = mulpd2(rN, getpz2(xq+s*6));
            const ymm x7 = mulpd2(rN, getpz2(xq+s*7));
            const ymm  a04 =       addpz2(x0, x4);
            const ymm  s04 =       subpz2(x0, x4);
            const ymm  a26 =       addpz2(x2, x6);
            const ymm js26 = jxpz2(subpz2(x2, x6));
            const ymm  a15 =       addpz2(x1, x5);
            const ymm  s15 =       subpz2(x1, x5);
            const ymm  a37 =       addpz2(x3, x7);
            const ymm js37 = jxpz2(subpz2(x3, x7));
            const ymm    a04_p1_a26 =        addpz2(a04,  a26);
            const ymm    s04_mj_s26 =        subpz2(s04, js26);
            const ymm    a04_m1_a26 =        subpz2(a04,  a26);
            const ymm    s04_pj_s26 =        addpz2(s04, js26);
            const ymm    a15_p1_a37 =        addpz2(a15,  a37);
            const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
            const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
            const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
            setpz2(xq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(xq+s*1, addpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(xq+s*2, subpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(xq+s*3, subpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(xq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(xq+s*5, subpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(xq+s*6, addpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(xq+s*7, addpz2(s04_pj_s26, v8_s15_pj_s37));
        }
    }
};

template <> struct fwdnend<8,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const noexcept
    {
        zeroupper();
        static const xmm rN = { 1.0/8, 1.0/8 };
        const xmm x0 = mulpd(rN, getpz(x[0]));
        const xmm x1 = mulpd(rN, getpz(x[1]));
        const xmm x2 = mulpd(rN, getpz(x[2]));
        const xmm x3 = mulpd(rN, getpz(x[3]));
        const xmm x4 = mulpd(rN, getpz(x[4]));
        const xmm x5 = mulpd(rN, getpz(x[5]));
        const xmm x6 = mulpd(rN, getpz(x[6]));
        const xmm x7 = mulpd(rN, getpz(x[7]));
        const xmm  a04 =      addpz(x0, x4);
        const xmm  s04 =      subpz(x0, x4);
        const xmm  a26 =      addpz(x2, x6);
        const xmm js26 = jxpz(subpz(x2, x6));
        const xmm  a15 =      addpz(x1, x5);
        const xmm  s15 =      subpz(x1, x5);
        const xmm  a37 =      addpz(x3, x7);
        const xmm js37 = jxpz(subpz(x3, x7));
        const xmm    a04_p1_a26 =       addpz(a04,  a26);
        const xmm    s04_mj_s26 =       subpz(s04, js26);
        const xmm    a04_m1_a26 =       subpz(a04,  a26);
        const xmm    s04_pj_s26 =       addpz(s04, js26);
        const xmm    a15_p1_a37 =       addpz(a15,  a37);
        const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
        const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
        const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
        setpz(x[0], addpz(a04_p1_a26,    a15_p1_a37));
        setpz(x[1], addpz(s04_mj_s26, w8_s15_mj_s37));
        setpz(x[2], subpz(a04_m1_a26,  j_a15_m1_a37));
        setpz(x[3], subpz(s04_pj_s26, v8_s15_pj_s37));
        setpz(x[4], subpz(a04_p1_a26,    a15_p1_a37));
        setpz(x[5], subpz(s04_mj_s26, w8_s15_mj_s37));
        setpz(x[6], addpz(a04_m1_a26,  j_a15_m1_a37));
        setpz(x[7], addpz(s04_pj_s26, v8_s15_pj_s37));
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
        fwd0fft<n/8,8*s,!eo>()(y, x, W);
        fwdcore<n,s>()(x, y, W);
    }
};

template <int s, bool eo> struct fwd0fft<8,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        fwd0end<8,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwd0fft<4,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT4::fwd0end<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwd0fft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT4::fwd0end<2,s,eo>()(x, y);
    }
};

//-----------------------------------------------------------------------------

template <int n, int s, bool eo> struct fwdnfft
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        fwdnfft<n/8,8*s,!eo>()(y, x, W);
        fwdcore<n,s>()(x, y, W);
    }
};

template <int s, bool eo> struct fwdnfft<8,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        fwdnend<8,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwdnfft<4,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT4::fwdnend<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwdnfft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT4::fwdnend<2,s,eo>()(x, y);
    }
};

///////////////////////////////////////////////////////////////////////////////
// Inverse butterfly operation
///////////////////////////////////////////////////////////////////////////////

template <int n, int s> struct invcore
{
    static const int n1 = n/8;
    static const int N  = n*s;
    static const int N0 = 0;
    static const int N1 = N/8;
    static const int N2 = N1*2;
    static const int N3 = N1*3;
    static const int N4 = N1*4;
    static const int N5 = N1*5;
    static const int N6 = N1*6;
    static const int N7 = N1*7;

    void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        for (int p = 0; p < n1; p++) {
            const int sp = s*p;
            const int s8p = 8*sp;
            //const ymm w1p = duppz2(getpz(W[N-sp]));
            const ymm w1p = duppz3(W[N-1*sp]);
            const ymm w2p = duppz3(W[N-2*sp]);
            const ymm w3p = duppz3(W[N-3*sp]);
            const ymm w4p = mulpz2(w2p, w2p);
            const ymm w5p = mulpz2(w2p, w3p);
            const ymm w6p = mulpz2(w3p, w3p);
            const ymm w7p = mulpz2(w3p, w4p);
            for (int q = 0; q < s; q += 2) {
                complex_vector xq_sp  = x + q + sp;
                complex_vector yq_s8p = y + q + s8p;
                const ymm y0 =             getpz2(yq_s8p+s*0);
                const ymm y1 = mulpz2(w1p, getpz2(yq_s8p+s*1));
                const ymm y2 = mulpz2(w2p, getpz2(yq_s8p+s*2));
                const ymm y3 = mulpz2(w3p, getpz2(yq_s8p+s*3));
                const ymm y4 = mulpz2(w4p, getpz2(yq_s8p+s*4));
                const ymm y5 = mulpz2(w5p, getpz2(yq_s8p+s*5));
                const ymm y6 = mulpz2(w6p, getpz2(yq_s8p+s*6));
                const ymm y7 = mulpz2(w7p, getpz2(yq_s8p+s*7));
                const ymm  a04 =       addpz2(y0, y4);
                const ymm  s04 =       subpz2(y0, y4);
                const ymm  a26 =       addpz2(y2, y6);
                const ymm js26 = jxpz2(subpz2(y2, y6));
                const ymm  a15 =       addpz2(y1, y5);
                const ymm  s15 =       subpz2(y1, y5);
                const ymm  a37 =       addpz2(y3, y7);
                const ymm js37 = jxpz2(subpz2(y3, y7));
                const ymm    a04_p1_a26 =        addpz2(a04,  a26);
                const ymm    s04_pj_s26 =        addpz2(s04, js26);
                const ymm    a04_m1_a26 =        subpz2(a04,  a26);
                const ymm    s04_mj_s26 =        subpz2(s04, js26);
                const ymm    a15_p1_a37 =        addpz2(a15,  a37);
                const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
                const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
                const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
                setpz2(xq_sp+N0, addpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(xq_sp+N1, addpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(xq_sp+N2, addpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(xq_sp+N3, subpz2(s04_mj_s26, w8_s15_mj_s37));
                setpz2(xq_sp+N4, subpz2(a04_p1_a26,    a15_p1_a37));
                setpz2(xq_sp+N5, subpz2(s04_pj_s26, v8_s15_pj_s37));
                setpz2(xq_sp+N6, subpz2(a04_m1_a26,  j_a15_m1_a37));
                setpz2(xq_sp+N7, addpz2(s04_mj_s26, w8_s15_mj_s37));
            }
        }
    }
};

template <int N> struct invcore<N,1>
{
    static const int N0 = 0;
    static const int N1 = N/8;
    static const int N2 = N1*2;
    static const int N3 = N1*3;
    static const int N4 = N1*4;
    static const int N5 = N1*5;
    static const int N6 = N1*6;
    static const int N7 = N1*7;

    void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        //const_complex_vector WN = W + N;
        for (int p = 0; p < N/8; p += 2) {
            complex_vector x_p  = x + p;
            complex_vector y_8p = y + 8*p;
            //const ymm w1p = getwp2<-1>(WN,p);
            const ymm w1p = cnjpz2(getpz2(W+p));
            const ymm w2p = mulpz2(w1p, w1p);
            const ymm w3p = mulpz2(w1p, w2p);
            const ymm w4p = mulpz2(w2p, w2p);
            const ymm w5p = mulpz2(w2p, w3p);
            const ymm w6p = mulpz2(w3p, w3p);
            const ymm w7p = mulpz2(w3p, w4p);
#if 0
            const ymm y0 =             getpz3<8>(y_8p+0);
            const ymm y1 = mulpz2(w1p, getpz3<8>(y_8p+1));
            const ymm y2 = mulpz2(w2p, getpz3<8>(y_8p+2));
            const ymm y3 = mulpz2(w3p, getpz3<8>(y_8p+3));
            const ymm y4 = mulpz2(w4p, getpz3<8>(y_8p+4));
            const ymm y5 = mulpz2(w5p, getpz3<8>(y_8p+5));
            const ymm y6 = mulpz2(w6p, getpz3<8>(y_8p+6));
            const ymm y7 = mulpz2(w7p, getpz3<8>(y_8p+7));
#else
            const ymm ab = getpz2(y_8p+ 0);
            const ymm cd = getpz2(y_8p+ 2);
            const ymm ef = getpz2(y_8p+ 4);
            const ymm gh = getpz2(y_8p+ 6);
            const ymm AB = getpz2(y_8p+ 8);
            const ymm CD = getpz2(y_8p+10);
            const ymm EF = getpz2(y_8p+12);
            const ymm GH = getpz2(y_8p+14);
            const ymm y0 =             catlo(ab, AB);
            const ymm y1 = mulpz2(w1p, cathi(ab, AB));
            const ymm y2 = mulpz2(w2p, catlo(cd, CD));
            const ymm y3 = mulpz2(w3p, cathi(cd, CD));
            const ymm y4 = mulpz2(w4p, catlo(ef, EF));
            const ymm y5 = mulpz2(w5p, cathi(ef, EF));
            const ymm y6 = mulpz2(w6p, catlo(gh, GH));
            const ymm y7 = mulpz2(w7p, cathi(gh, GH));
#endif
            const ymm  a04 =       addpz2(y0, y4);
            const ymm  s04 =       subpz2(y0, y4);
            const ymm  a26 =       addpz2(y2, y6);
            const ymm js26 = jxpz2(subpz2(y2, y6));
            const ymm  a15 =       addpz2(y1, y5);
            const ymm  s15 =       subpz2(y1, y5);
            const ymm  a37 =       addpz2(y3, y7);
            const ymm js37 = jxpz2(subpz2(y3, y7));
            const ymm    a04_p1_a26 =        addpz2(a04,  a26);
            const ymm    s04_pj_s26 =        addpz2(s04, js26);
            const ymm    a04_m1_a26 =        subpz2(a04,  a26);
            const ymm    s04_mj_s26 =        subpz2(s04, js26);
            const ymm    a15_p1_a37 =        addpz2(a15,  a37);
            const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
            const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
            const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
            setpz2(x_p+N0, addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(x_p+N1, addpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(x_p+N2, addpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(x_p+N3, subpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(x_p+N4, subpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(x_p+N5, subpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(x_p+N6, subpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(x_p+N7, addpz2(s04_mj_s26, w8_s15_mj_s37));
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct inv0end;

//-----------------------------------------------------------------------------

template <int s> struct inv0end<8,s,1>
{
    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
            const ymm y0 = getpz2(yq+s*0);
            const ymm y1 = getpz2(yq+s*1);
            const ymm y2 = getpz2(yq+s*2);
            const ymm y3 = getpz2(yq+s*3);
            const ymm y4 = getpz2(yq+s*4);
            const ymm y5 = getpz2(yq+s*5);
            const ymm y6 = getpz2(yq+s*6);
            const ymm y7 = getpz2(yq+s*7);
            const ymm  a04 =       addpz2(y0, y4);
            const ymm  s04 =       subpz2(y0, y4);
            const ymm  a26 =       addpz2(y2, y6);
            const ymm js26 = jxpz2(subpz2(y2, y6));
            const ymm  a15 =       addpz2(y1, y5);
            const ymm  s15 =       subpz2(y1, y5);
            const ymm  a37 =       addpz2(y3, y7);
            const ymm js37 = jxpz2(subpz2(y3, y7));
            const ymm    a04_p1_a26 =        addpz2(a04,  a26);
            const ymm    s04_pj_s26 =        addpz2(s04, js26);
            const ymm    a04_m1_a26 =        subpz2(a04,  a26);
            const ymm    s04_mj_s26 =        subpz2(s04, js26);
            const ymm    a15_p1_a37 =        addpz2(a15,  a37);
            const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
            const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
            const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
            setpz2(xq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(xq+s*1, addpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(xq+s*2, addpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(xq+s*3, subpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(xq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(xq+s*5, subpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(xq+s*6, subpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(xq+s*7, addpz2(s04_mj_s26, w8_s15_mj_s37));
        }
    }
};

template <> struct inv0end<8,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        zeroupper();
        const xmm y0 = getpz(y[0]);
        const xmm y1 = getpz(y[1]);
        const xmm y2 = getpz(y[2]);
        const xmm y3 = getpz(y[3]);
        const xmm y4 = getpz(y[4]);
        const xmm y5 = getpz(y[5]);
        const xmm y6 = getpz(y[6]);
        const xmm y7 = getpz(y[7]);
        const xmm  a04 =      addpz(y0, y4);
        const xmm  s04 =      subpz(y0, y4);
        const xmm  a26 =      addpz(y2, y6);
        const xmm js26 = jxpz(subpz(y2, y6));
        const xmm  a15 =      addpz(y1, y5);
        const xmm  s15 =      subpz(y1, y5);
        const xmm  a37 =      addpz(y3, y7);
        const xmm js37 = jxpz(subpz(y3, y7));
        const xmm    a04_p1_a26 =       addpz(a04,  a26);
        const xmm    s04_pj_s26 =       addpz(s04, js26);
        const xmm    a04_m1_a26 =       subpz(a04,  a26);
        const xmm    s04_mj_s26 =       subpz(s04, js26);
        const xmm    a15_p1_a37 =       addpz(a15,  a37);
        const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
        const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
        const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
        setpz(x[0], addpz(a04_p1_a26,    a15_p1_a37));
        setpz(x[1], addpz(s04_pj_s26, v8_s15_pj_s37));
        setpz(x[2], addpz(a04_m1_a26,  j_a15_m1_a37));
        setpz(x[3], subpz(s04_mj_s26, w8_s15_mj_s37));
        setpz(x[4], subpz(a04_p1_a26,    a15_p1_a37));
        setpz(x[5], subpz(s04_pj_s26, v8_s15_pj_s37));
        setpz(x[6], subpz(a04_m1_a26,  j_a15_m1_a37));
        setpz(x[7], addpz(s04_mj_s26, w8_s15_mj_s37));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct inv0end<8,s,0>
{
    void operator()(complex_vector x, complex_vector) const noexcept
    {
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            const ymm x0 = getpz2(xq+s*0);
            const ymm x1 = getpz2(xq+s*1);
            const ymm x2 = getpz2(xq+s*2);
            const ymm x3 = getpz2(xq+s*3);
            const ymm x4 = getpz2(xq+s*4);
            const ymm x5 = getpz2(xq+s*5);
            const ymm x6 = getpz2(xq+s*6);
            const ymm x7 = getpz2(xq+s*7);
            const ymm  a04 =       addpz2(x0, x4);
            const ymm  s04 =       subpz2(x0, x4);
            const ymm  a26 =       addpz2(x2, x6);
            const ymm js26 = jxpz2(subpz2(x2, x6));
            const ymm  a15 =       addpz2(x1, x5);
            const ymm  s15 =       subpz2(x1, x5);
            const ymm  a37 =       addpz2(x3, x7);
            const ymm js37 = jxpz2(subpz2(x3, x7));
            const ymm    a04_p1_a26 =        addpz2(a04,  a26);
            const ymm    s04_pj_s26 =        addpz2(s04, js26);
            const ymm    a04_m1_a26 =        subpz2(a04,  a26);
            const ymm    s04_mj_s26 =        subpz2(s04, js26);
            const ymm    a15_p1_a37 =        addpz2(a15,  a37);
            const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
            const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
            const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
            setpz2(xq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(xq+s*1, addpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(xq+s*2, addpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(xq+s*3, subpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(xq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(xq+s*5, subpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(xq+s*6, subpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(xq+s*7, addpz2(s04_mj_s26, w8_s15_mj_s37));
        }
    }
};

template <> struct inv0end<8,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const noexcept
    {
        zeroupper();
        const xmm x0 = getpz(x[0]);
        const xmm x1 = getpz(x[1]);
        const xmm x2 = getpz(x[2]);
        const xmm x3 = getpz(x[3]);
        const xmm x4 = getpz(x[4]);
        const xmm x5 = getpz(x[5]);
        const xmm x6 = getpz(x[6]);
        const xmm x7 = getpz(x[7]);
        const xmm  a04 =      addpz(x0, x4);
        const xmm  s04 =      subpz(x0, x4);
        const xmm  a26 =      addpz(x2, x6);
        const xmm js26 = jxpz(subpz(x2, x6));
        const xmm  a15 =      addpz(x1, x5);
        const xmm  s15 =      subpz(x1, x5);
        const xmm  a37 =      addpz(x3, x7);
        const xmm js37 = jxpz(subpz(x3, x7));
        const xmm    a04_p1_a26 =       addpz(a04,  a26);
        const xmm    s04_pj_s26 =       addpz(s04, js26);
        const xmm    a04_m1_a26 =       subpz(a04,  a26);
        const xmm    s04_mj_s26 =       subpz(s04, js26);
        const xmm    a15_p1_a37 =       addpz(a15,  a37);
        const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
        const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
        const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
        setpz(x[0], addpz(a04_p1_a26,    a15_p1_a37));
        setpz(x[1], addpz(s04_pj_s26, v8_s15_pj_s37));
        setpz(x[2], addpz(a04_m1_a26,  j_a15_m1_a37));
        setpz(x[3], subpz(s04_mj_s26, w8_s15_mj_s37));
        setpz(x[4], subpz(a04_p1_a26,    a15_p1_a37));
        setpz(x[5], subpz(s04_pj_s26, v8_s15_pj_s37));
        setpz(x[6], subpz(a04_m1_a26,  j_a15_m1_a37));
        setpz(x[7], addpz(s04_mj_s26, w8_s15_mj_s37));
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct invnend;

//-----------------------------------------------------------------------------

template <int s> struct invnend<8,s,1>
{
    static const int N  = 8*s;

    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
            const ymm y0 = mulpd2(rN, getpz2(yq+s*0));
            const ymm y1 = mulpd2(rN, getpz2(yq+s*1));
            const ymm y2 = mulpd2(rN, getpz2(yq+s*2));
            const ymm y3 = mulpd2(rN, getpz2(yq+s*3));
            const ymm y4 = mulpd2(rN, getpz2(yq+s*4));
            const ymm y5 = mulpd2(rN, getpz2(yq+s*5));
            const ymm y6 = mulpd2(rN, getpz2(yq+s*6));
            const ymm y7 = mulpd2(rN, getpz2(yq+s*7));
            const ymm  a04 =       addpz2(y0, y4);
            const ymm  s04 =       subpz2(y0, y4);
            const ymm  a26 =       addpz2(y2, y6);
            const ymm js26 = jxpz2(subpz2(y2, y6));
            const ymm  a15 =       addpz2(y1, y5);
            const ymm  s15 =       subpz2(y1, y5);
            const ymm  a37 =       addpz2(y3, y7);
            const ymm js37 = jxpz2(subpz2(y3, y7));
            const ymm    a04_p1_a26 =        addpz2(a04,  a26);
            const ymm    s04_pj_s26 =        addpz2(s04, js26);
            const ymm    a04_m1_a26 =        subpz2(a04,  a26);
            const ymm    s04_mj_s26 =        subpz2(s04, js26);
            const ymm    a15_p1_a37 =        addpz2(a15,  a37);
            const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
            const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
            const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
            setpz2(xq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(xq+s*1, addpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(xq+s*2, addpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(xq+s*3, subpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(xq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(xq+s*5, subpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(xq+s*6, subpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(xq+s*7, addpz2(s04_mj_s26, w8_s15_mj_s37));
        }
    }
};

template <> struct invnend<8,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        zeroupper();
        static const xmm rN = { 1.0/8, 1.0/8 };
        const xmm y0 = mulpd(rN, getpz(y[0]));
        const xmm y1 = mulpd(rN, getpz(y[1]));
        const xmm y2 = mulpd(rN, getpz(y[2]));
        const xmm y3 = mulpd(rN, getpz(y[3]));
        const xmm y4 = mulpd(rN, getpz(y[4]));
        const xmm y5 = mulpd(rN, getpz(y[5]));
        const xmm y6 = mulpd(rN, getpz(y[6]));
        const xmm y7 = mulpd(rN, getpz(y[7]));
        const xmm  a04 =      addpz(y0, y4);
        const xmm  s04 =      subpz(y0, y4);
        const xmm  a26 =      addpz(y2, y6);
        const xmm js26 = jxpz(subpz(y2, y6));
        const xmm  a15 =      addpz(y1, y5);
        const xmm  s15 =      subpz(y1, y5);
        const xmm  a37 =      addpz(y3, y7);
        const xmm js37 = jxpz(subpz(y3, y7));
        const xmm    a04_p1_a26 =       addpz(a04,  a26);
        const xmm    s04_pj_s26 =       addpz(s04, js26);
        const xmm    a04_m1_a26 =       subpz(a04,  a26);
        const xmm    s04_mj_s26 =       subpz(s04, js26);
        const xmm    a15_p1_a37 =       addpz(a15,  a37);
        const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
        const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
        const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
        setpz(x[0], addpz(a04_p1_a26,    a15_p1_a37));
        setpz(x[1], addpz(s04_pj_s26, v8_s15_pj_s37));
        setpz(x[2], addpz(a04_m1_a26,  j_a15_m1_a37));
        setpz(x[3], subpz(s04_mj_s26, w8_s15_mj_s37));
        setpz(x[4], subpz(a04_p1_a26,    a15_p1_a37));
        setpz(x[5], subpz(s04_pj_s26, v8_s15_pj_s37));
        setpz(x[6], subpz(a04_m1_a26,  j_a15_m1_a37));
        setpz(x[7], addpz(s04_mj_s26, w8_s15_mj_s37));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct invnend<8,s,0>
{
    static const int N = 8*s;

    void operator()(complex_vector x, complex_vector) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            const ymm x0 = mulpd2(rN, getpz2(xq+s*0));
            const ymm x1 = mulpd2(rN, getpz2(xq+s*1));
            const ymm x2 = mulpd2(rN, getpz2(xq+s*2));
            const ymm x3 = mulpd2(rN, getpz2(xq+s*3));
            const ymm x4 = mulpd2(rN, getpz2(xq+s*4));
            const ymm x5 = mulpd2(rN, getpz2(xq+s*5));
            const ymm x6 = mulpd2(rN, getpz2(xq+s*6));
            const ymm x7 = mulpd2(rN, getpz2(xq+s*7));
            const ymm  a04 =       addpz2(x0, x4);
            const ymm  s04 =       subpz2(x0, x4);
            const ymm  a26 =       addpz2(x2, x6);
            const ymm js26 = jxpz2(subpz2(x2, x6));
            const ymm  a15 =       addpz2(x1, x5);
            const ymm  s15 =       subpz2(x1, x5);
            const ymm  a37 =       addpz2(x3, x7);
            const ymm js37 = jxpz2(subpz2(x3, x7));
            const ymm    a04_p1_a26 =        addpz2(a04,  a26);
            const ymm    s04_pj_s26 =        addpz2(s04, js26);
            const ymm    a04_m1_a26 =        subpz2(a04,  a26);
            const ymm    s04_mj_s26 =        subpz2(s04, js26);
            const ymm    a15_p1_a37 =        addpz2(a15,  a37);
            const ymm v8_s15_pj_s37 = v8xpz2(addpz2(s15, js37));
            const ymm  j_a15_m1_a37 =  jxpz2(subpz2(a15,  a37));
            const ymm w8_s15_mj_s37 = w8xpz2(subpz2(s15, js37));
            setpz2(xq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(xq+s*1, addpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(xq+s*2, addpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(xq+s*3, subpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(xq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(xq+s*5, subpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(xq+s*6, subpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(xq+s*7, addpz2(s04_mj_s26, w8_s15_mj_s37));
        }
    }
};

template <> struct invnend<8,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const noexcept
    {
        zeroupper();
        static const xmm rN = { 1.0/8, 1.0/8 };
        const xmm x0 = mulpd(rN, getpz(x[0]));
        const xmm x1 = mulpd(rN, getpz(x[1]));
        const xmm x2 = mulpd(rN, getpz(x[2]));
        const xmm x3 = mulpd(rN, getpz(x[3]));
        const xmm x4 = mulpd(rN, getpz(x[4]));
        const xmm x5 = mulpd(rN, getpz(x[5]));
        const xmm x6 = mulpd(rN, getpz(x[6]));
        const xmm x7 = mulpd(rN, getpz(x[7]));
        const xmm  a04 =      addpz(x0, x4);
        const xmm  s04 =      subpz(x0, x4);
        const xmm  a26 =      addpz(x2, x6);
        const xmm js26 = jxpz(subpz(x2, x6));
        const xmm  a15 =      addpz(x1, x5);
        const xmm  s15 =      subpz(x1, x5);
        const xmm  a37 =      addpz(x3, x7);
        const xmm js37 = jxpz(subpz(x3, x7));
        const xmm    a04_p1_a26 =       addpz(a04,  a26);
        const xmm    s04_pj_s26 =       addpz(s04, js26);
        const xmm    a04_m1_a26 =       subpz(a04,  a26);
        const xmm    s04_mj_s26 =       subpz(s04, js26);
        const xmm    a15_p1_a37 =       addpz(a15,  a37);
        const xmm v8_s15_pj_s37 = v8xpz(addpz(s15, js37));
        const xmm  j_a15_m1_a37 =  jxpz(subpz(a15,  a37));
        const xmm w8_s15_mj_s37 = w8xpz(subpz(s15, js37));
        setpz(x[0], addpz(a04_p1_a26,    a15_p1_a37));
        setpz(x[1], addpz(s04_pj_s26, v8_s15_pj_s37));
        setpz(x[2], addpz(a04_m1_a26,  j_a15_m1_a37));
        setpz(x[3], subpz(s04_mj_s26, w8_s15_mj_s37));
        setpz(x[4], subpz(a04_p1_a26,    a15_p1_a37));
        setpz(x[5], subpz(s04_pj_s26, v8_s15_pj_s37));
        setpz(x[6], subpz(a04_m1_a26,  j_a15_m1_a37));
        setpz(x[7], addpz(s04_mj_s26, w8_s15_mj_s37));
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
        inv0fft<n/8,8*s,!eo>()(y, x, W);
        invcore<n,s>()(x, y, W);
    }
};

template <int s, bool eo> struct inv0fft<8,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        inv0end<8,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct inv0fft<4,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT4::inv0end<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct inv0fft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT4::inv0end<2,s,eo>()(x, y);
    }
};

//-----------------------------------------------------------------------------

template <int n, int s, bool eo> struct invnfft
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        invnfft<n/8,8*s,!eo>()(y, x, W);
        invcore<n,s>()(x, y, W);
    }
};

template <int s, bool eo> struct invnfft<8,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        invnend<8,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct invnfft<4,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT4::invnend<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct invnfft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT4::invnend<2,s,eo>()(x, y);
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
        else OTFFT_AVXDIT8omp::fwd(log_N, x, y, W);
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
        else OTFFT_AVXDIT8omp::fwd0(log_N, x, y, W);
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
        else OTFFT_AVXDIT8omp::fwd0o(log_N, x, y, W);
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
        else OTFFT_AVXDIT8omp::fwdno(log_N, x, y, W);
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
        else OTFFT_AVXDIT8omp::inv(log_N, x, y, W);
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
        else OTFFT_AVXDIT8omp::invn(log_N, x, y, W);
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
        else OTFFT_AVXDIT8omp::inv0o(log_N, x, y, W);
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
        else OTFFT_AVXDIT8omp::invno(log_N, x, y, W);
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

#endif // otfft_avxdit8_h
