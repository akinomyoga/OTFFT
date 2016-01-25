/******************************************************************************
*  OTFFT AVXDIF(Radix-8) of OpenMP Version 6.5
*
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_avxdif8omp_h
#define otfft_avxdif8omp_h

//#include "otfft/otfft_misc.h"
//#include "otfft_avxdif4.h"

namespace OTFFT_AVXDIF8omp { //////////////////////////////////////////////////

using namespace OTFFT_MISC;

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
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int i = 0; i < N/16; i++) {
            const int p = i / (s/2);
            const int q = i % (s/2) * 2;
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
            complex_vector xq_sp  = x + q + sp;
            complex_vector yq_s8p = y + q + s8p;
            const ymm x0 = getpz2(xq_sp+N0);
            const ymm x1 = getpz2(xq_sp+N1);
            const ymm x2 = getpz2(xq_sp+N2);
            const ymm x3 = getpz2(xq_sp+N3);
            const ymm x4 = getpz2(xq_sp+N4);
            const ymm x5 = getpz2(xq_sp+N5);
            const ymm x6 = getpz2(xq_sp+N6);
            const ymm x7 = getpz2(xq_sp+N7);
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
            setpz2(yq_s8p+s*0,             addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(yq_s8p+s*1, mulpz2(w1p, addpz2(s04_mj_s26, w8_s15_mj_s37)));
            setpz2(yq_s8p+s*2, mulpz2(w2p, subpz2(a04_m1_a26,  j_a15_m1_a37)));
            setpz2(yq_s8p+s*3, mulpz2(w3p, subpz2(s04_pj_s26, v8_s15_pj_s37)));
            setpz2(yq_s8p+s*4, mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37)));
            setpz2(yq_s8p+s*5, mulpz2(w5p, subpz2(s04_mj_s26, w8_s15_mj_s37)));
            setpz2(yq_s8p+s*6, mulpz2(w6p, addpz2(a04_m1_a26,  j_a15_m1_a37)));
            setpz2(yq_s8p+s*7, mulpz2(w7p, addpz2(s04_pj_s26, v8_s15_pj_s37)));
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
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
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
            const ymm x0 = getpz2(x_p+N0);
            const ymm x1 = getpz2(x_p+N1);
            const ymm x2 = getpz2(x_p+N2);
            const ymm x3 = getpz2(x_p+N3);
            const ymm x4 = getpz2(x_p+N4);
            const ymm x5 = getpz2(x_p+N5);
            const ymm x6 = getpz2(x_p+N6);
            const ymm x7 = getpz2(x_p+N7);
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
#if 0
            setpz3<8>(y_8p+0,             addpz2(a04_p1_a26,    a15_p1_a37));
            setpz3<8>(y_8p+1, mulpz2(w1p, addpz2(s04_mj_s26, w8_s15_mj_s37)));
            setpz3<8>(y_8p+2, mulpz2(w2p, subpz2(a04_m1_a26,  j_a15_m1_a37)));
            setpz3<8>(y_8p+3, mulpz2(w3p, subpz2(s04_pj_s26, v8_s15_pj_s37)));
            setpz3<8>(y_8p+4, mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37)));
            setpz3<8>(y_8p+5, mulpz2(w5p, subpz2(s04_mj_s26, w8_s15_mj_s37)));
            setpz3<8>(y_8p+6, mulpz2(w6p, addpz2(a04_m1_a26,  j_a15_m1_a37)));
            setpz3<8>(y_8p+7, mulpz2(w7p, addpz2(s04_pj_s26, v8_s15_pj_s37)));
#else
            const ymm aA =             addpz2(a04_p1_a26,    a15_p1_a37);
            const ymm bB = mulpz2(w1p, addpz2(s04_mj_s26, w8_s15_mj_s37));
            const ymm cC = mulpz2(w2p, subpz2(a04_m1_a26,  j_a15_m1_a37));
            const ymm dD = mulpz2(w3p, subpz2(s04_pj_s26, v8_s15_pj_s37));
            const ymm eE = mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37));
            const ymm fF = mulpz2(w5p, subpz2(s04_mj_s26, w8_s15_mj_s37));
            const ymm gG = mulpz2(w6p, addpz2(a04_m1_a26,  j_a15_m1_a37));
            const ymm hH = mulpz2(w7p, addpz2(s04_pj_s26, v8_s15_pj_s37));
            const ymm ab = catlo(aA, bB);
            const ymm AB = cathi(aA, bB);
            const ymm cd = catlo(cC, dD);
            const ymm CD = cathi(cC, dD);
            const ymm ef = catlo(eE, fF);
            const ymm EF = cathi(eE, fF);
            const ymm gh = catlo(gG, hH);
            const ymm GH = cathi(gG, hH);
            setpz2(y_8p+ 0, ab);
            setpz2(y_8p+ 2, cd);
            setpz2(y_8p+ 4, ef);
            setpz2(y_8p+ 6, gh);
            setpz2(y_8p+ 8, AB);
            setpz2(y_8p+10, CD);
            setpz2(y_8p+12, EF);
            setpz2(y_8p+14, GH);
#endif
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
#ifdef _OPENMP
        #pragma omp for schedule(static) nowait
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
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
            setpz2(yq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(yq+s*1, addpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(yq+s*2, subpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(yq+s*3, subpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(yq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(yq+s*5, subpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(yq+s*6, addpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(yq+s*7, addpz2(s04_pj_s26, v8_s15_pj_s37));
        }
    }
};

template <> struct fwd0end<8,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
#ifdef _OPENMP
        #pragma omp single
#endif
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
            setpz(y[0], addpz(a04_p1_a26,    a15_p1_a37));
            setpz(y[1], addpz(s04_mj_s26, w8_s15_mj_s37));
            setpz(y[2], subpz(a04_m1_a26,  j_a15_m1_a37));
            setpz(y[3], subpz(s04_pj_s26, v8_s15_pj_s37));
            setpz(y[4], subpz(a04_p1_a26,    a15_p1_a37));
            setpz(y[5], subpz(s04_mj_s26, w8_s15_mj_s37));
            setpz(y[6], addpz(a04_m1_a26,  j_a15_m1_a37));
            setpz(y[7], addpz(s04_pj_s26, v8_s15_pj_s37));
        }
    }
};

//-----------------------------------------------------------------------------

template <int s> struct fwd0end<8,s,0>
{
    void operator()(complex_vector x, complex_vector) const noexcept
    {
#ifdef _OPENMP
        #pragma omp for schedule(static) nowait
#endif
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
#ifdef _OPENMP
        #pragma omp single
#endif
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
#ifdef _OPENMP
        #pragma omp for schedule(static) nowait
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
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
            setpz2(yq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(yq+s*1, addpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(yq+s*2, subpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(yq+s*3, subpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(yq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(yq+s*5, subpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(yq+s*6, addpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(yq+s*7, addpz2(s04_pj_s26, v8_s15_pj_s37));
        }
    }
};

template <> struct fwdnend<8,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        static const xmm rN = { 1.0/8, 1.0/8 };
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
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
            setpz(y[0], addpz(a04_p1_a26,    a15_p1_a37));
            setpz(y[1], addpz(s04_mj_s26, w8_s15_mj_s37));
            setpz(y[2], subpz(a04_m1_a26,  j_a15_m1_a37));
            setpz(y[3], subpz(s04_pj_s26, v8_s15_pj_s37));
            setpz(y[4], subpz(a04_p1_a26,    a15_p1_a37));
            setpz(y[5], subpz(s04_mj_s26, w8_s15_mj_s37));
            setpz(y[6], addpz(a04_m1_a26,  j_a15_m1_a37));
            setpz(y[7], addpz(s04_pj_s26, v8_s15_pj_s37));
        }
    }
};

//-----------------------------------------------------------------------------

template <int s> struct fwdnend<8,s,0>
{
    static const int N = 8*s;

    void operator()(complex_vector x, complex_vector) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
#ifdef _OPENMP
        #pragma omp for schedule(static) nowait
#endif
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
        static const xmm rN = { 1.0/8, 1.0/8 };
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
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
        fwdcore<n,s>()(x, y, W);
        fwd0fft<n/8,8*s,!eo>()(y, x, W);
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
        OTFFT_AVXDIF4omp::fwd0end<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwd0fft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIF4omp::fwd0end<2,s,eo>()(x, y);
    }
};

//-----------------------------------------------------------------------------

template <int n, int s, bool eo> struct fwdnfft
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        fwdcore<n,s>()(x, y, W);
        fwdnfft<n/8,8*s,!eo>()(y, x, W);
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
        OTFFT_AVXDIF4omp::fwdnend<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwdnfft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIF4omp::fwdnend<2,s,eo>()(x, y);
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
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int i = 0; i < N/16; i++) {
            const int p = i / (s/2);
            const int q = i % (s/2) * 2;
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
            complex_vector xq_sp  = x + q + sp;
            complex_vector yq_s8p = y + q + s8p;
            const ymm x0 = getpz2(xq_sp+N0);
            const ymm x1 = getpz2(xq_sp+N1);
            const ymm x2 = getpz2(xq_sp+N2);
            const ymm x3 = getpz2(xq_sp+N3);
            const ymm x4 = getpz2(xq_sp+N4);
            const ymm x5 = getpz2(xq_sp+N5);
            const ymm x6 = getpz2(xq_sp+N6);
            const ymm x7 = getpz2(xq_sp+N7);
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
            setpz2(yq_s8p+s*0,             addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(yq_s8p+s*1, mulpz2(w1p, addpz2(s04_pj_s26, v8_s15_pj_s37)));
            setpz2(yq_s8p+s*2, mulpz2(w2p, addpz2(a04_m1_a26,  j_a15_m1_a37)));
            setpz2(yq_s8p+s*3, mulpz2(w3p, subpz2(s04_mj_s26, w8_s15_mj_s37)));
            setpz2(yq_s8p+s*4, mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37)));
            setpz2(yq_s8p+s*5, mulpz2(w5p, subpz2(s04_pj_s26, v8_s15_pj_s37)));
            setpz2(yq_s8p+s*6, mulpz2(w6p, subpz2(a04_m1_a26,  j_a15_m1_a37)));
            setpz2(yq_s8p+s*7, mulpz2(w7p, addpz2(s04_mj_s26, w8_s15_mj_s37)));
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
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int p = 0; p < N1; p += 2) {
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
            const ymm x0 = getpz2(x_p+N0);
            const ymm x1 = getpz2(x_p+N1);
            const ymm x2 = getpz2(x_p+N2);
            const ymm x3 = getpz2(x_p+N3);
            const ymm x4 = getpz2(x_p+N4);
            const ymm x5 = getpz2(x_p+N5);
            const ymm x6 = getpz2(x_p+N6);
            const ymm x7 = getpz2(x_p+N7);
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
#if 0
            setpz3<8>(y_8p+0,             addpz2(a04_p1_a26,    a15_p1_a37));
            setpz3<8>(y_8p+1, mulpz2(w1p, addpz2(s04_pj_s26, v8_s15_pj_s37)));
            setpz3<8>(y_8p+2, mulpz2(w2p, addpz2(a04_m1_a26,  j_a15_m1_a37)));
            setpz3<8>(y_8p+3, mulpz2(w3p, subpz2(s04_mj_s26, w8_s15_mj_s37)));
            setpz3<8>(y_8p+4, mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37)));
            setpz3<8>(y_8p+5, mulpz2(w5p, subpz2(s04_pj_s26, v8_s15_pj_s37)));
            setpz3<8>(y_8p+6, mulpz2(w6p, subpz2(a04_m1_a26,  j_a15_m1_a37)));
            setpz3<8>(y_8p+7, mulpz2(w7p, addpz2(s04_mj_s26, w8_s15_mj_s37)));
#else
            const ymm aA =             addpz2(a04_p1_a26,    a15_p1_a37);
            const ymm bB = mulpz2(w1p, addpz2(s04_pj_s26, v8_s15_pj_s37));
            const ymm cC = mulpz2(w2p, addpz2(a04_m1_a26,  j_a15_m1_a37));
            const ymm dD = mulpz2(w3p, subpz2(s04_mj_s26, w8_s15_mj_s37));
            const ymm eE = mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37));
            const ymm fF = mulpz2(w5p, subpz2(s04_pj_s26, v8_s15_pj_s37));
            const ymm gG = mulpz2(w6p, subpz2(a04_m1_a26,  j_a15_m1_a37));
            const ymm hH = mulpz2(w7p, addpz2(s04_mj_s26, w8_s15_mj_s37));
            const ymm ab = catlo(aA, bB);
            const ymm AB = cathi(aA, bB);
            const ymm cd = catlo(cC, dD);
            const ymm CD = cathi(cC, dD);
            const ymm ef = catlo(eE, fF);
            const ymm EF = cathi(eE, fF);
            const ymm gh = catlo(gG, hH);
            const ymm GH = cathi(gG, hH);
            setpz2(y_8p+ 0, ab);
            setpz2(y_8p+ 2, cd);
            setpz2(y_8p+ 4, ef);
            setpz2(y_8p+ 6, gh);
            setpz2(y_8p+ 8, AB);
            setpz2(y_8p+10, CD);
            setpz2(y_8p+12, EF);
            setpz2(y_8p+14, GH);
#endif
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
#ifdef _OPENMP
        #pragma omp for schedule(static) nowait
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
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
            setpz2(yq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(yq+s*1, addpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(yq+s*2, addpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(yq+s*3, subpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(yq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(yq+s*5, subpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(yq+s*6, subpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(yq+s*7, addpz2(s04_mj_s26, w8_s15_mj_s37));
        }
    }
};

template <> struct inv0end<8,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
#ifdef _OPENMP
        #pragma omp single
#endif
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
            setpz(y[0], addpz(a04_p1_a26,    a15_p1_a37));
            setpz(y[1], addpz(s04_pj_s26, v8_s15_pj_s37));
            setpz(y[2], addpz(a04_m1_a26,  j_a15_m1_a37));
            setpz(y[3], subpz(s04_mj_s26, w8_s15_mj_s37));
            setpz(y[4], subpz(a04_p1_a26,    a15_p1_a37));
            setpz(y[5], subpz(s04_pj_s26, v8_s15_pj_s37));
            setpz(y[6], subpz(a04_m1_a26,  j_a15_m1_a37));
            setpz(y[7], addpz(s04_mj_s26, w8_s15_mj_s37));
        }
    }
};

//-----------------------------------------------------------------------------

template <int s> struct inv0end<8,s,0>
{
    void operator()(complex_vector x, complex_vector) const noexcept
    {
#ifdef _OPENMP
        #pragma omp for schedule(static) nowait
#endif
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
#ifdef _OPENMP
        #pragma omp single
#endif
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
#ifdef _OPENMP
        #pragma omp for schedule(static) nowait
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
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
            setpz2(yq+s*0, addpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(yq+s*1, addpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(yq+s*2, addpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(yq+s*3, subpz2(s04_mj_s26, w8_s15_mj_s37));
            setpz2(yq+s*4, subpz2(a04_p1_a26,    a15_p1_a37));
            setpz2(yq+s*5, subpz2(s04_pj_s26, v8_s15_pj_s37));
            setpz2(yq+s*6, subpz2(a04_m1_a26,  j_a15_m1_a37));
            setpz2(yq+s*7, addpz2(s04_mj_s26, w8_s15_mj_s37));
        }
    }
};

template <> struct invnend<8,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        static const xmm rN = { 1.0/8, 1.0/8 };
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
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
            setpz(y[0], addpz(a04_p1_a26,    a15_p1_a37));
            setpz(y[1], addpz(s04_pj_s26, v8_s15_pj_s37));
            setpz(y[2], addpz(a04_m1_a26,  j_a15_m1_a37));
            setpz(y[3], subpz(s04_mj_s26, w8_s15_mj_s37));
            setpz(y[4], subpz(a04_p1_a26,    a15_p1_a37));
            setpz(y[5], subpz(s04_pj_s26, v8_s15_pj_s37));
            setpz(y[6], subpz(a04_m1_a26,  j_a15_m1_a37));
            setpz(y[7], addpz(s04_mj_s26, w8_s15_mj_s37));
        }
    }
};

//-----------------------------------------------------------------------------

template <int s> struct invnend<8,s,0>
{
    static const int N = 8*s;

    void operator()(complex_vector x, complex_vector) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
#ifdef _OPENMP
        #pragma omp for schedule(static) nowait
#endif
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
        static const xmm rN = { 1.0/8, 1.0/8 };
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
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
        invcore<n,s>()(x, y, W);
        inv0fft<n/8,8*s,!eo>()(y, x, W);
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
        OTFFT_AVXDIF4omp::inv0end<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct inv0fft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIF4omp::inv0end<2,s,eo>()(x, y);
    }
};

//-----------------------------------------------------------------------------

template <int n, int s, bool eo> struct invnfft
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        invcore<n,s>()(x, y, W);
        invnfft<n/8,8*s,!eo>()(y, x, W);
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
        OTFFT_AVXDIF4omp::invnend<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct invnfft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIF4omp::invnend<2,s,eo>()(x, y);
    }
};

///////////////////////////////////////////////////////////////////////////////
// 2 powered FFT routine
///////////////////////////////////////////////////////////////////////////////

inline void fwd(const int log_N,
        complex_vector x, complex_vector y, const_complex_vector W) noexcept
{
#ifdef _OPENMP
    #pragma omp parallel firstprivate(x,y,W)
#endif
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

inline void fwd0(const int log_N,
        complex_vector x, complex_vector y, const_complex_vector W) noexcept
{
#ifdef _OPENMP
    #pragma omp parallel firstprivate(x,y,W)
#endif
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

inline void fwdn(const int log_N,
        complex_vector x, complex_vector y, const_complex_vector W) noexcept
{
    fwd(log_N, x, y, W);
}

inline void fwd0o(const int log_N,
        complex_vector x, complex_vector y, const_complex_vector W) noexcept
{
#ifdef _OPENMP
    #pragma omp parallel firstprivate(x,y,W)
#endif
    switch (log_N) {
        case  0: break;
        case  1: fwd0fft<(1<< 1),1,1>()(x, y, W); break;
        case  2: fwd0fft<(1<< 2),1,1>()(x, y, W); break;
        case  3: fwd0fft<(1<< 3),1,1>()(x, y, W); break;
        case  4: fwd0fft<(1<< 4),1,1>()(x, y, W); break;
        case  5: fwd0fft<(1<< 5),1,1>()(x, y, W); break;
        case  6: fwd0fft<(1<< 6),1,1>()(x, y, W); break;
        case  7: fwd0fft<(1<< 7),1,1>()(x, y, W); break;
        case  8: fwd0fft<(1<< 8),1,1>()(x, y, W); break;
        case  9: fwd0fft<(1<< 9),1,1>()(x, y, W); break;
        case 10: fwd0fft<(1<<10),1,1>()(x, y, W); break;
        case 11: fwd0fft<(1<<11),1,1>()(x, y, W); break;
        case 12: fwd0fft<(1<<12),1,1>()(x, y, W); break;
        case 13: fwd0fft<(1<<13),1,1>()(x, y, W); break;
        case 14: fwd0fft<(1<<14),1,1>()(x, y, W); break;
        case 15: fwd0fft<(1<<15),1,1>()(x, y, W); break;
        case 16: fwd0fft<(1<<16),1,1>()(x, y, W); break;
        case 17: fwd0fft<(1<<17),1,1>()(x, y, W); break;
        case 18: fwd0fft<(1<<18),1,1>()(x, y, W); break;
        case 19: fwd0fft<(1<<19),1,1>()(x, y, W); break;
        case 20: fwd0fft<(1<<20),1,1>()(x, y, W); break;
        case 21: fwd0fft<(1<<21),1,1>()(x, y, W); break;
        case 22: fwd0fft<(1<<22),1,1>()(x, y, W); break;
        case 23: fwd0fft<(1<<23),1,1>()(x, y, W); break;
        case 24: fwd0fft<(1<<24),1,1>()(x, y, W); break;
    }
}

inline void fwdno(const int log_N,
        complex_vector x, complex_vector y, const_complex_vector W) noexcept
{
#ifdef _OPENMP
    #pragma omp parallel firstprivate(x,y,W)
#endif
    switch (log_N) {
        case  0: break;
        case  1: fwdnfft<(1<< 1),1,1>()(x, y, W); break;
        case  2: fwdnfft<(1<< 2),1,1>()(x, y, W); break;
        case  3: fwdnfft<(1<< 3),1,1>()(x, y, W); break;
        case  4: fwdnfft<(1<< 4),1,1>()(x, y, W); break;
        case  5: fwdnfft<(1<< 5),1,1>()(x, y, W); break;
        case  6: fwdnfft<(1<< 6),1,1>()(x, y, W); break;
        case  7: fwdnfft<(1<< 7),1,1>()(x, y, W); break;
        case  8: fwdnfft<(1<< 8),1,1>()(x, y, W); break;
        case  9: fwdnfft<(1<< 9),1,1>()(x, y, W); break;
        case 10: fwdnfft<(1<<10),1,1>()(x, y, W); break;
        case 11: fwdnfft<(1<<11),1,1>()(x, y, W); break;
        case 12: fwdnfft<(1<<12),1,1>()(x, y, W); break;
        case 13: fwdnfft<(1<<13),1,1>()(x, y, W); break;
        case 14: fwdnfft<(1<<14),1,1>()(x, y, W); break;
        case 15: fwdnfft<(1<<15),1,1>()(x, y, W); break;
        case 16: fwdnfft<(1<<16),1,1>()(x, y, W); break;
        case 17: fwdnfft<(1<<17),1,1>()(x, y, W); break;
        case 18: fwdnfft<(1<<18),1,1>()(x, y, W); break;
        case 19: fwdnfft<(1<<19),1,1>()(x, y, W); break;
        case 20: fwdnfft<(1<<20),1,1>()(x, y, W); break;
        case 21: fwdnfft<(1<<21),1,1>()(x, y, W); break;
        case 22: fwdnfft<(1<<22),1,1>()(x, y, W); break;
        case 23: fwdnfft<(1<<23),1,1>()(x, y, W); break;
        case 24: fwdnfft<(1<<24),1,1>()(x, y, W); break;
    }
}

///////////////////////////////////////////////////////////////////////////////

inline void inv(const int log_N,
        complex_vector x, complex_vector y, const_complex_vector W) noexcept
{
#ifdef _OPENMP
    #pragma omp parallel firstprivate(x,y,W)
#endif
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

inline void inv0(const int log_N,
        complex_vector x, complex_vector y, const_complex_vector W) noexcept
{
    inv(log_N, x, y, W);
}

inline void invn(const int log_N,
        complex_vector x, complex_vector y, const_complex_vector W) noexcept
{
#ifdef _OPENMP
    #pragma omp parallel firstprivate(x,y,W)
#endif
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

inline void inv0o(const int log_N,
        complex_vector x, complex_vector y, const_complex_vector W) noexcept
{
#ifdef _OPENMP
    #pragma omp parallel firstprivate(x,y,W)
#endif
    switch (log_N) {
        case  0: break;
        case  1: inv0fft<(1<< 1),1,1>()(x, y, W); break;
        case  2: inv0fft<(1<< 2),1,1>()(x, y, W); break;
        case  3: inv0fft<(1<< 3),1,1>()(x, y, W); break;
        case  4: inv0fft<(1<< 4),1,1>()(x, y, W); break;
        case  5: inv0fft<(1<< 5),1,1>()(x, y, W); break;
        case  6: inv0fft<(1<< 6),1,1>()(x, y, W); break;
        case  7: inv0fft<(1<< 7),1,1>()(x, y, W); break;
        case  8: inv0fft<(1<< 8),1,1>()(x, y, W); break;
        case  9: inv0fft<(1<< 9),1,1>()(x, y, W); break;
        case 10: inv0fft<(1<<10),1,1>()(x, y, W); break;
        case 11: inv0fft<(1<<11),1,1>()(x, y, W); break;
        case 12: inv0fft<(1<<12),1,1>()(x, y, W); break;
        case 13: inv0fft<(1<<13),1,1>()(x, y, W); break;
        case 14: inv0fft<(1<<14),1,1>()(x, y, W); break;
        case 15: inv0fft<(1<<15),1,1>()(x, y, W); break;
        case 16: inv0fft<(1<<16),1,1>()(x, y, W); break;
        case 17: inv0fft<(1<<17),1,1>()(x, y, W); break;
        case 18: inv0fft<(1<<18),1,1>()(x, y, W); break;
        case 19: inv0fft<(1<<19),1,1>()(x, y, W); break;
        case 20: inv0fft<(1<<20),1,1>()(x, y, W); break;
        case 21: inv0fft<(1<<21),1,1>()(x, y, W); break;
        case 22: inv0fft<(1<<22),1,1>()(x, y, W); break;
        case 23: inv0fft<(1<<23),1,1>()(x, y, W); break;
        case 24: inv0fft<(1<<24),1,1>()(x, y, W); break;
    }
}

inline void invno(const int log_N,
        complex_vector x, complex_vector y, const_complex_vector W) noexcept
{
#ifdef _OPENMP
    #pragma omp parallel firstprivate(x,y,W)
#endif
    switch (log_N) {
        case  0: break;
        case  1: invnfft<(1<< 1),1,1>()(x, y, W); break;
        case  2: invnfft<(1<< 2),1,1>()(x, y, W); break;
        case  3: invnfft<(1<< 3),1,1>()(x, y, W); break;
        case  4: invnfft<(1<< 4),1,1>()(x, y, W); break;
        case  5: invnfft<(1<< 5),1,1>()(x, y, W); break;
        case  6: invnfft<(1<< 6),1,1>()(x, y, W); break;
        case  7: invnfft<(1<< 7),1,1>()(x, y, W); break;
        case  8: invnfft<(1<< 8),1,1>()(x, y, W); break;
        case  9: invnfft<(1<< 9),1,1>()(x, y, W); break;
        case 10: invnfft<(1<<10),1,1>()(x, y, W); break;
        case 11: invnfft<(1<<11),1,1>()(x, y, W); break;
        case 12: invnfft<(1<<12),1,1>()(x, y, W); break;
        case 13: invnfft<(1<<13),1,1>()(x, y, W); break;
        case 14: invnfft<(1<<14),1,1>()(x, y, W); break;
        case 15: invnfft<(1<<15),1,1>()(x, y, W); break;
        case 16: invnfft<(1<<16),1,1>()(x, y, W); break;
        case 17: invnfft<(1<<17),1,1>()(x, y, W); break;
        case 18: invnfft<(1<<18),1,1>()(x, y, W); break;
        case 19: invnfft<(1<<19),1,1>()(x, y, W); break;
        case 20: invnfft<(1<<20),1,1>()(x, y, W); break;
        case 21: invnfft<(1<<21),1,1>()(x, y, W); break;
        case 22: invnfft<(1<<22),1,1>()(x, y, W); break;
        case 23: invnfft<(1<<23),1,1>()(x, y, W); break;
        case 24: invnfft<(1<<24),1,1>()(x, y, W); break;
    }
}

} /////////////////////////////////////////////////////////////////////////////

#endif // otfft_avxdif8omp_h
