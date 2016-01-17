/******************************************************************************
*  OTFFT AVXDIT(Radix-16) of OpenMP Version 6.4
*
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_avxdit16omp_h
#define otfft_avxdit16omp_h

//#include "otfft/otfft_misc.h"
//#include "otfft_avxdit8.h"

namespace OTFFT_AVXDIT16omp { /////////////////////////////////////////////////

using namespace OTFFT_MISC;

///////////////////////////////////////////////////////////////////////////////
// Forward buffterfly operation
///////////////////////////////////////////////////////////////////////////////

template <int n, int s> struct fwdcore
{
    static const int n1 = n/16;
    static const int N  = n*s;
    static const int N0 = 0;
    static const int N1 = N/16;
    static const int N2 = N1*2;
    static const int N3 = N1*3;
    static const int N4 = N1*4;
    static const int N5 = N1*5;
    static const int N6 = N1*6;
    static const int N7 = N1*7;
    static const int N8 = N1*8;
    static const int N9 = N1*9;
    static const int Na = N1*10;
    static const int Nb = N1*11;
    static const int Nc = N1*12;
    static const int Nd = N1*13;
    static const int Ne = N1*14;
    static const int Nf = N1*15;

    void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int i = 0; i < N/32; i++) {
            const int p = i / (s/2);
            const int q = i % (s/2) * 2;
            const int sp   = s*p;
            const int s16p = 16*sp;

            const ymm w1p = duppz3(W[1*sp]);
            const ymm w2p = duppz3(W[2*sp]);
            const ymm w3p = duppz3(W[3*sp]);
            const ymm w4p = mulpz2(w2p, w2p);
            const ymm w5p = mulpz2(w2p, w3p);
            const ymm w6p = mulpz2(w3p, w3p);
            const ymm w7p = mulpz2(w3p, w4p);
            const ymm w8p = mulpz2(w4p, w4p);
            const ymm w9p = mulpz2(w4p, w5p);
            const ymm wap = mulpz2(w5p, w5p);
            const ymm wbp = mulpz2(w5p, w6p);
            const ymm wcp = mulpz2(w6p, w6p);
            const ymm wdp = mulpz2(w6p, w7p);
            const ymm wep = mulpz2(w7p, w7p);
            const ymm wfp = mulpz2(w7p, w8p);

            complex_vector xq_sp   = x + q + sp;
            complex_vector yq_s16p = y + q + s16p;

            const ymm y0 =             getpz2(yq_s16p+s*0x0);
            const ymm y1 = mulpz2(w1p, getpz2(yq_s16p+s*0x1));
            const ymm y2 = mulpz2(w2p, getpz2(yq_s16p+s*0x2));
            const ymm y3 = mulpz2(w3p, getpz2(yq_s16p+s*0x3));
            const ymm y4 = mulpz2(w4p, getpz2(yq_s16p+s*0x4));
            const ymm y5 = mulpz2(w5p, getpz2(yq_s16p+s*0x5));
            const ymm y6 = mulpz2(w6p, getpz2(yq_s16p+s*0x6));
            const ymm y7 = mulpz2(w7p, getpz2(yq_s16p+s*0x7));
            const ymm y8 = mulpz2(w8p, getpz2(yq_s16p+s*0x8));
            const ymm y9 = mulpz2(w9p, getpz2(yq_s16p+s*0x9));
            const ymm ya = mulpz2(wap, getpz2(yq_s16p+s*0xa));
            const ymm yb = mulpz2(wbp, getpz2(yq_s16p+s*0xb));
            const ymm yc = mulpz2(wcp, getpz2(yq_s16p+s*0xc));
            const ymm yd = mulpz2(wdp, getpz2(yq_s16p+s*0xd));
            const ymm ye = mulpz2(wep, getpz2(yq_s16p+s*0xe));
            const ymm yf = mulpz2(wfp, getpz2(yq_s16p+s*0xf));

            const ymm a08 = addpz2(y0, y8); const ymm s08 = subpz2(y0, y8);
            const ymm a4c = addpz2(y4, yc); const ymm s4c = subpz2(y4, yc);
            const ymm a2a = addpz2(y2, ya); const ymm s2a = subpz2(y2, ya);
            const ymm a6e = addpz2(y6, ye); const ymm s6e = subpz2(y6, ye);
            const ymm a19 = addpz2(y1, y9); const ymm s19 = subpz2(y1, y9);
            const ymm a5d = addpz2(y5, yd); const ymm s5d = subpz2(y5, yd);
            const ymm a3b = addpz2(y3, yb); const ymm s3b = subpz2(y3, yb);
            const ymm a7f = addpz2(y7, yf); const ymm s7f = subpz2(y7, yf);

            const ymm js4c = jxpz2(s4c);
            const ymm js6e = jxpz2(s6e);
            const ymm js5d = jxpz2(s5d);
            const ymm js7f = jxpz2(s7f);

            const ymm a08p1a4c = addpz2(a08, a4c); const ymm s08mjs4c = subpz2(s08, js4c);
            const ymm a08m1a4c = subpz2(a08, a4c); const ymm s08pjs4c = addpz2(s08, js4c);
            const ymm a2ap1a6e = addpz2(a2a, a6e); const ymm s2amjs6e = subpz2(s2a, js6e);
            const ymm a2am1a6e = subpz2(a2a, a6e); const ymm s2apjs6e = addpz2(s2a, js6e);
            const ymm a19p1a5d = addpz2(a19, a5d); const ymm s19mjs5d = subpz2(s19, js5d);
            const ymm a19m1a5d = subpz2(a19, a5d); const ymm s19pjs5d = addpz2(s19, js5d);
            const ymm a3bp1a7f = addpz2(a3b, a7f); const ymm s3bmjs7f = subpz2(s3b, js7f);
            const ymm a3bm1a7f = subpz2(a3b, a7f); const ymm s3bpjs7f = addpz2(s3b, js7f);

            const ymm w8_s2amjs6e = w8xpz2(s2amjs6e);
            const ymm  j_a2am1a6e =  jxpz2(a2am1a6e);
            const ymm v8_s2apjs6e = v8xpz2(s2apjs6e);

            const ymm a08p1a4c_p1_a2ap1a6e = addpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_pw_s2amjs6e = addpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_mj_a2am1a6e = subpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_mv_s2apjs6e = subpz2(s08pjs4c, v8_s2apjs6e);
            const ymm a08p1a4c_m1_a2ap1a6e = subpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_mw_s2amjs6e = subpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_pj_a2am1a6e = addpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_pv_s2apjs6e = addpz2(s08pjs4c, v8_s2apjs6e);

            const ymm w8_s3bmjs7f = w8xpz2(s3bmjs7f);
            const ymm  j_a3bm1a7f =  jxpz2(a3bm1a7f);
            const ymm v8_s3bpjs7f = v8xpz2(s3bpjs7f);

            const ymm a19p1a5d_p1_a3bp1a7f = addpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_pw_s3bmjs7f = addpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_mj_a3bm1a7f = subpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_mv_s3bpjs7f = subpz2(s19pjs5d, v8_s3bpjs7f);
            const ymm a19p1a5d_m1_a3bp1a7f = subpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_mw_s3bmjs7f = subpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_pj_a3bm1a7f = addpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_pv_s3bpjs7f = addpz2(s19pjs5d, v8_s3bpjs7f);

            const ymm h1_s19mjs5d_pw_s3bmjs7f = h1xpz2(s19mjs5d_pw_s3bmjs7f);
            const ymm w8_a19m1a5d_mj_a3bm1a7f = w8xpz2(a19m1a5d_mj_a3bm1a7f);
            const ymm h3_s19pjs5d_mv_s3bpjs7f = h3xpz2(s19pjs5d_mv_s3bpjs7f);
            const ymm  j_a19p1a5d_m1_a3bp1a7f =  jxpz2(a19p1a5d_m1_a3bp1a7f);
            const ymm hd_s19mjs5d_mw_s3bmjs7f = hdxpz2(s19mjs5d_mw_s3bmjs7f);
            const ymm v8_a19m1a5d_pj_a3bm1a7f = v8xpz2(a19m1a5d_pj_a3bm1a7f);
            const ymm hf_s19pjs5d_pv_s3bpjs7f = hfxpz2(s19pjs5d_pv_s3bpjs7f);

            setpz2(xq_sp+N0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq_sp+N1, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz2(xq_sp+N2, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq_sp+N3, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq_sp+N4, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq_sp+N5, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq_sp+N6, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq_sp+N7, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

            setpz2(xq_sp+N8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq_sp+N9, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz2(xq_sp+Na, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq_sp+Nb, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq_sp+Nc, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq_sp+Nd, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq_sp+Ne, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq_sp+Nf, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
        }
    }
};

template <int N> struct fwdcore<N,1>
{
    static const int N0 = 0;
    static const int N1 = N/16;
    static const int N2 = N1*2;
    static const int N3 = N1*3;
    static const int N4 = N1*4;
    static const int N5 = N1*5;
    static const int N6 = N1*6;
    static const int N7 = N1*7;
    static const int N8 = N1*8;
    static const int N9 = N1*9;
    static const int Na = N1*10;
    static const int Nb = N1*11;
    static const int Nc = N1*12;
    static const int Nd = N1*13;
    static const int Ne = N1*14;
    static const int Nf = N1*15;

    void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
#ifdef _OPENMP
        #pragma omp for schedule(static) nowait
#endif
        for (int p = 0; p < N1; p += 2) {
            complex_vector x_p   = x + p;
            complex_vector y_16p = y + 16*p;

            const ymm w1p = getpz2(W+p);
            const ymm w2p = mulpz2(w1p, w1p);
            const ymm w3p = mulpz2(w1p, w2p);
            const ymm w4p = mulpz2(w2p, w2p);
            const ymm w5p = mulpz2(w2p, w3p);
            const ymm w6p = mulpz2(w3p, w3p);
            const ymm w7p = mulpz2(w3p, w4p);
            const ymm w8p = mulpz2(w4p, w4p);
            const ymm w9p = mulpz2(w4p, w5p);
            const ymm wap = mulpz2(w5p, w5p);
            const ymm wbp = mulpz2(w5p, w6p);
            const ymm wcp = mulpz2(w6p, w6p);
            const ymm wdp = mulpz2(w6p, w7p);
            const ymm wep = mulpz2(w7p, w7p);
            const ymm wfp = mulpz2(w7p, w8p);
#if 0
            const ymm y0 =             getpz3<16>(y_16p+0x0);
            const ymm y1 = mulpz2(w1p, getpz3<16>(y_16p+0x1));
            const ymm y2 = mulpz2(w2p, getpz3<16>(y_16p+0x2));
            const ymm y3 = mulpz2(w3p, getpz3<16>(y_16p+0x3));
            const ymm y4 = mulpz2(w4p, getpz3<16>(y_16p+0x4));
            const ymm y5 = mulpz2(w5p, getpz3<16>(y_16p+0x5));
            const ymm y6 = mulpz2(w6p, getpz3<16>(y_16p+0x6));
            const ymm y7 = mulpz2(w7p, getpz3<16>(y_16p+0x7));
            const ymm y8 = mulpz2(w8p, getpz3<16>(y_16p+0x8));
            const ymm y9 = mulpz2(w9p, getpz3<16>(y_16p+0x9));
            const ymm ya = mulpz2(wap, getpz3<16>(y_16p+0xa));
            const ymm yb = mulpz2(wbp, getpz3<16>(y_16p+0xb));
            const ymm yc = mulpz2(wcp, getpz3<16>(y_16p+0xc));
            const ymm yd = mulpz2(wdp, getpz3<16>(y_16p+0xd));
            const ymm ye = mulpz2(wep, getpz3<16>(y_16p+0xe));
            const ymm yf = mulpz2(wfp, getpz3<16>(y_16p+0xf));
#else
            const ymm ab = getpz2(y_16p+0x00);
            const ymm cd = getpz2(y_16p+0x02);
            const ymm ef = getpz2(y_16p+0x04);
            const ymm gh = getpz2(y_16p+0x06);
            const ymm ij = getpz2(y_16p+0x08);
            const ymm kl = getpz2(y_16p+0x0a);
            const ymm mn = getpz2(y_16p+0x0c);
            const ymm op = getpz2(y_16p+0x0e);

            const ymm AB = getpz2(y_16p+0x10);
            const ymm CD = getpz2(y_16p+0x12);
            const ymm EF = getpz2(y_16p+0x14);
            const ymm GH = getpz2(y_16p+0x16);
            const ymm IJ = getpz2(y_16p+0x18);
            const ymm KL = getpz2(y_16p+0x1a);
            const ymm MN = getpz2(y_16p+0x1c);
            const ymm OP = getpz2(y_16p+0x1e);

            const ymm y0 =             catlo(ab, AB);
            const ymm y1 = mulpz2(w1p, cathi(ab, AB));
            const ymm y2 = mulpz2(w2p, catlo(cd, CD));
            const ymm y3 = mulpz2(w3p, cathi(cd, CD));
            const ymm y4 = mulpz2(w4p, catlo(ef, EF));
            const ymm y5 = mulpz2(w5p, cathi(ef, EF));
            const ymm y6 = mulpz2(w6p, catlo(gh, GH));
            const ymm y7 = mulpz2(w7p, cathi(gh, GH));

            const ymm y8 = mulpz2(w8p, catlo(ij, IJ));
            const ymm y9 = mulpz2(w9p, cathi(ij, IJ));
            const ymm ya = mulpz2(wap, catlo(kl, KL));
            const ymm yb = mulpz2(wbp, cathi(kl, KL));
            const ymm yc = mulpz2(wcp, catlo(mn, MN));
            const ymm yd = mulpz2(wdp, cathi(mn, MN));
            const ymm ye = mulpz2(wep, catlo(op, OP));
            const ymm yf = mulpz2(wfp, cathi(op, OP));
#endif
            const ymm a08 = addpz2(y0, y8); const ymm s08 = subpz2(y0, y8);
            const ymm a4c = addpz2(y4, yc); const ymm s4c = subpz2(y4, yc);
            const ymm a2a = addpz2(y2, ya); const ymm s2a = subpz2(y2, ya);
            const ymm a6e = addpz2(y6, ye); const ymm s6e = subpz2(y6, ye);
            const ymm a19 = addpz2(y1, y9); const ymm s19 = subpz2(y1, y9);
            const ymm a5d = addpz2(y5, yd); const ymm s5d = subpz2(y5, yd);
            const ymm a3b = addpz2(y3, yb); const ymm s3b = subpz2(y3, yb);
            const ymm a7f = addpz2(y7, yf); const ymm s7f = subpz2(y7, yf);

            const ymm js4c = jxpz2(s4c);
            const ymm js6e = jxpz2(s6e);
            const ymm js5d = jxpz2(s5d);
            const ymm js7f = jxpz2(s7f);

            const ymm a08p1a4c = addpz2(a08, a4c); const ymm s08mjs4c = subpz2(s08, js4c);
            const ymm a08m1a4c = subpz2(a08, a4c); const ymm s08pjs4c = addpz2(s08, js4c);
            const ymm a2ap1a6e = addpz2(a2a, a6e); const ymm s2amjs6e = subpz2(s2a, js6e);
            const ymm a2am1a6e = subpz2(a2a, a6e); const ymm s2apjs6e = addpz2(s2a, js6e);
            const ymm a19p1a5d = addpz2(a19, a5d); const ymm s19mjs5d = subpz2(s19, js5d);
            const ymm a19m1a5d = subpz2(a19, a5d); const ymm s19pjs5d = addpz2(s19, js5d);
            const ymm a3bp1a7f = addpz2(a3b, a7f); const ymm s3bmjs7f = subpz2(s3b, js7f);
            const ymm a3bm1a7f = subpz2(a3b, a7f); const ymm s3bpjs7f = addpz2(s3b, js7f);

            const ymm w8_s2amjs6e = w8xpz2(s2amjs6e);
            const ymm  j_a2am1a6e =  jxpz2(a2am1a6e);
            const ymm v8_s2apjs6e = v8xpz2(s2apjs6e);

            const ymm a08p1a4c_p1_a2ap1a6e = addpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_pw_s2amjs6e = addpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_mj_a2am1a6e = subpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_mv_s2apjs6e = subpz2(s08pjs4c, v8_s2apjs6e);
            const ymm a08p1a4c_m1_a2ap1a6e = subpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_mw_s2amjs6e = subpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_pj_a2am1a6e = addpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_pv_s2apjs6e = addpz2(s08pjs4c, v8_s2apjs6e);

            const ymm w8_s3bmjs7f = w8xpz2(s3bmjs7f);
            const ymm  j_a3bm1a7f =  jxpz2(a3bm1a7f);
            const ymm v8_s3bpjs7f = v8xpz2(s3bpjs7f);

            const ymm a19p1a5d_p1_a3bp1a7f = addpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_pw_s3bmjs7f = addpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_mj_a3bm1a7f = subpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_mv_s3bpjs7f = subpz2(s19pjs5d, v8_s3bpjs7f);
            const ymm a19p1a5d_m1_a3bp1a7f = subpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_mw_s3bmjs7f = subpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_pj_a3bm1a7f = addpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_pv_s3bpjs7f = addpz2(s19pjs5d, v8_s3bpjs7f);

            const ymm h1_s19mjs5d_pw_s3bmjs7f = h1xpz2(s19mjs5d_pw_s3bmjs7f);
            const ymm w8_a19m1a5d_mj_a3bm1a7f = w8xpz2(a19m1a5d_mj_a3bm1a7f);
            const ymm h3_s19pjs5d_mv_s3bpjs7f = h3xpz2(s19pjs5d_mv_s3bpjs7f);
            const ymm  j_a19p1a5d_m1_a3bp1a7f =  jxpz2(a19p1a5d_m1_a3bp1a7f);
            const ymm hd_s19mjs5d_mw_s3bmjs7f = hdxpz2(s19mjs5d_mw_s3bmjs7f);
            const ymm v8_a19m1a5d_pj_a3bm1a7f = v8xpz2(a19m1a5d_pj_a3bm1a7f);
            const ymm hf_s19pjs5d_pv_s3bpjs7f = hfxpz2(s19pjs5d_pv_s3bpjs7f);

            setpz2(x_p+N0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(x_p+N1, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz2(x_p+N2, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(x_p+N3, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(x_p+N4, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(x_p+N5, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(x_p+N6, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(x_p+N7, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

            setpz2(x_p+N8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(x_p+N9, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz2(x_p+Na, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(x_p+Nb, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(x_p+Nc, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(x_p+Nd, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(x_p+Ne, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(x_p+Nf, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct fwd0end;

//-----------------------------------------------------------------------------

template <int s> struct fwd0end<16,s,1>
{
    void operator()(complex_vector x, complex_vector y) const noexcept
    {
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;

            const ymm y0 = getpz2(yq+s*0x0);
            const ymm y1 = getpz2(yq+s*0x1);
            const ymm y2 = getpz2(yq+s*0x2);
            const ymm y3 = getpz2(yq+s*0x3);
            const ymm y4 = getpz2(yq+s*0x4);
            const ymm y5 = getpz2(yq+s*0x5);
            const ymm y6 = getpz2(yq+s*0x6);
            const ymm y7 = getpz2(yq+s*0x7);
            const ymm y8 = getpz2(yq+s*0x8);
            const ymm y9 = getpz2(yq+s*0x9);
            const ymm ya = getpz2(yq+s*0xa);
            const ymm yb = getpz2(yq+s*0xb);
            const ymm yc = getpz2(yq+s*0xc);
            const ymm yd = getpz2(yq+s*0xd);
            const ymm ye = getpz2(yq+s*0xe);
            const ymm yf = getpz2(yq+s*0xf);

            const ymm a08 = addpz2(y0, y8); const ymm s08 = subpz2(y0, y8);
            const ymm a4c = addpz2(y4, yc); const ymm s4c = subpz2(y4, yc);
            const ymm a2a = addpz2(y2, ya); const ymm s2a = subpz2(y2, ya);
            const ymm a6e = addpz2(y6, ye); const ymm s6e = subpz2(y6, ye);
            const ymm a19 = addpz2(y1, y9); const ymm s19 = subpz2(y1, y9);
            const ymm a5d = addpz2(y5, yd); const ymm s5d = subpz2(y5, yd);
            const ymm a3b = addpz2(y3, yb); const ymm s3b = subpz2(y3, yb);
            const ymm a7f = addpz2(y7, yf); const ymm s7f = subpz2(y7, yf);

            const ymm js4c = jxpz2(s4c);
            const ymm js6e = jxpz2(s6e);
            const ymm js5d = jxpz2(s5d);
            const ymm js7f = jxpz2(s7f);

            const ymm a08p1a4c = addpz2(a08, a4c); const ymm s08mjs4c = subpz2(s08, js4c);
            const ymm a08m1a4c = subpz2(a08, a4c); const ymm s08pjs4c = addpz2(s08, js4c);
            const ymm a2ap1a6e = addpz2(a2a, a6e); const ymm s2amjs6e = subpz2(s2a, js6e);
            const ymm a2am1a6e = subpz2(a2a, a6e); const ymm s2apjs6e = addpz2(s2a, js6e);
            const ymm a19p1a5d = addpz2(a19, a5d); const ymm s19mjs5d = subpz2(s19, js5d);
            const ymm a19m1a5d = subpz2(a19, a5d); const ymm s19pjs5d = addpz2(s19, js5d);
            const ymm a3bp1a7f = addpz2(a3b, a7f); const ymm s3bmjs7f = subpz2(s3b, js7f);
            const ymm a3bm1a7f = subpz2(a3b, a7f); const ymm s3bpjs7f = addpz2(s3b, js7f);

            const ymm w8_s2amjs6e = w8xpz2(s2amjs6e);
            const ymm  j_a2am1a6e =  jxpz2(a2am1a6e);
            const ymm v8_s2apjs6e = v8xpz2(s2apjs6e);

            const ymm a08p1a4c_p1_a2ap1a6e = addpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_pw_s2amjs6e = addpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_mj_a2am1a6e = subpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_mv_s2apjs6e = subpz2(s08pjs4c, v8_s2apjs6e);
            const ymm a08p1a4c_m1_a2ap1a6e = subpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_mw_s2amjs6e = subpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_pj_a2am1a6e = addpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_pv_s2apjs6e = addpz2(s08pjs4c, v8_s2apjs6e);

            const ymm w8_s3bmjs7f = w8xpz2(s3bmjs7f);
            const ymm  j_a3bm1a7f =  jxpz2(a3bm1a7f);
            const ymm v8_s3bpjs7f = v8xpz2(s3bpjs7f);

            const ymm a19p1a5d_p1_a3bp1a7f = addpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_pw_s3bmjs7f = addpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_mj_a3bm1a7f = subpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_mv_s3bpjs7f = subpz2(s19pjs5d, v8_s3bpjs7f);
            const ymm a19p1a5d_m1_a3bp1a7f = subpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_mw_s3bmjs7f = subpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_pj_a3bm1a7f = addpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_pv_s3bpjs7f = addpz2(s19pjs5d, v8_s3bpjs7f);

            const ymm h1_s19mjs5d_pw_s3bmjs7f = h1xpz2(s19mjs5d_pw_s3bmjs7f);
            const ymm w8_a19m1a5d_mj_a3bm1a7f = w8xpz2(a19m1a5d_mj_a3bm1a7f);
            const ymm h3_s19pjs5d_mv_s3bpjs7f = h3xpz2(s19pjs5d_mv_s3bpjs7f);
            const ymm  j_a19p1a5d_m1_a3bp1a7f =  jxpz2(a19p1a5d_m1_a3bp1a7f);
            const ymm hd_s19mjs5d_mw_s3bmjs7f = hdxpz2(s19mjs5d_mw_s3bmjs7f);
            const ymm v8_a19m1a5d_pj_a3bm1a7f = v8xpz2(a19m1a5d_pj_a3bm1a7f);
            const ymm hf_s19pjs5d_pv_s3bpjs7f = hfxpz2(s19pjs5d_pv_s3bpjs7f);

            setpz2(xq+s*0x0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq+s*0x1, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz2(xq+s*0x2, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq+s*0x3, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq+s*0x4, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq+s*0x5, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq+s*0x6, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq+s*0x7, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

            setpz2(xq+s*0x8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq+s*0x9, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz2(xq+s*0xa, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq+s*0xb, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq+s*0xc, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq+s*0xd, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq+s*0xe, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq+s*0xf, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
        }
    }
};

template <> struct fwd0end<16,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
            const xmm y0 = getpz(y[0x0]);
            const xmm y1 = getpz(y[0x1]);
            const xmm y2 = getpz(y[0x2]);
            const xmm y3 = getpz(y[0x3]);
            const xmm y4 = getpz(y[0x4]);
            const xmm y5 = getpz(y[0x5]);
            const xmm y6 = getpz(y[0x6]);
            const xmm y7 = getpz(y[0x7]);
            const xmm y8 = getpz(y[0x8]);
            const xmm y9 = getpz(y[0x9]);
            const xmm ya = getpz(y[0xa]);
            const xmm yb = getpz(y[0xb]);
            const xmm yc = getpz(y[0xc]);
            const xmm yd = getpz(y[0xd]);
            const xmm ye = getpz(y[0xe]);
            const xmm yf = getpz(y[0xf]);

            const xmm a08 = addpz(y0, y8); const xmm s08 = subpz(y0, y8);
            const xmm a4c = addpz(y4, yc); const xmm s4c = subpz(y4, yc);
            const xmm a2a = addpz(y2, ya); const xmm s2a = subpz(y2, ya);
            const xmm a6e = addpz(y6, ye); const xmm s6e = subpz(y6, ye);
            const xmm a19 = addpz(y1, y9); const xmm s19 = subpz(y1, y9);
            const xmm a5d = addpz(y5, yd); const xmm s5d = subpz(y5, yd);
            const xmm a3b = addpz(y3, yb); const xmm s3b = subpz(y3, yb);
            const xmm a7f = addpz(y7, yf); const xmm s7f = subpz(y7, yf);

            const xmm js4c = jxpz(s4c);
            const xmm js6e = jxpz(s6e);
            const xmm js5d = jxpz(s5d);
            const xmm js7f = jxpz(s7f);

            const xmm a08p1a4c = addpz(a08, a4c); const xmm s08mjs4c = subpz(s08, js4c);
            const xmm a08m1a4c = subpz(a08, a4c); const xmm s08pjs4c = addpz(s08, js4c);
            const xmm a2ap1a6e = addpz(a2a, a6e); const xmm s2amjs6e = subpz(s2a, js6e);
            const xmm a2am1a6e = subpz(a2a, a6e); const xmm s2apjs6e = addpz(s2a, js6e);
            const xmm a19p1a5d = addpz(a19, a5d); const xmm s19mjs5d = subpz(s19, js5d);
            const xmm a19m1a5d = subpz(a19, a5d); const xmm s19pjs5d = addpz(s19, js5d);
            const xmm a3bp1a7f = addpz(a3b, a7f); const xmm s3bmjs7f = subpz(s3b, js7f);
            const xmm a3bm1a7f = subpz(a3b, a7f); const xmm s3bpjs7f = addpz(s3b, js7f);

            const xmm w8_s2amjs6e = w8xpz(s2amjs6e);
            const xmm  j_a2am1a6e =  jxpz(a2am1a6e);
            const xmm v8_s2apjs6e = v8xpz(s2apjs6e);

            const xmm a08p1a4c_p1_a2ap1a6e = addpz(a08p1a4c,    a2ap1a6e);
            const xmm s08mjs4c_pw_s2amjs6e = addpz(s08mjs4c, w8_s2amjs6e);
            const xmm a08m1a4c_mj_a2am1a6e = subpz(a08m1a4c,  j_a2am1a6e);
            const xmm s08pjs4c_mv_s2apjs6e = subpz(s08pjs4c, v8_s2apjs6e);
            const xmm a08p1a4c_m1_a2ap1a6e = subpz(a08p1a4c,    a2ap1a6e);
            const xmm s08mjs4c_mw_s2amjs6e = subpz(s08mjs4c, w8_s2amjs6e);
            const xmm a08m1a4c_pj_a2am1a6e = addpz(a08m1a4c,  j_a2am1a6e);
            const xmm s08pjs4c_pv_s2apjs6e = addpz(s08pjs4c, v8_s2apjs6e);

            const xmm w8_s3bmjs7f = w8xpz(s3bmjs7f);
            const xmm  j_a3bm1a7f =  jxpz(a3bm1a7f);
            const xmm v8_s3bpjs7f = v8xpz(s3bpjs7f);

            const xmm a19p1a5d_p1_a3bp1a7f = addpz(a19p1a5d,    a3bp1a7f);
            const xmm s19mjs5d_pw_s3bmjs7f = addpz(s19mjs5d, w8_s3bmjs7f);
            const xmm a19m1a5d_mj_a3bm1a7f = subpz(a19m1a5d,  j_a3bm1a7f);
            const xmm s19pjs5d_mv_s3bpjs7f = subpz(s19pjs5d, v8_s3bpjs7f);
            const xmm a19p1a5d_m1_a3bp1a7f = subpz(a19p1a5d,    a3bp1a7f);
            const xmm s19mjs5d_mw_s3bmjs7f = subpz(s19mjs5d, w8_s3bmjs7f);
            const xmm a19m1a5d_pj_a3bm1a7f = addpz(a19m1a5d,  j_a3bm1a7f);
            const xmm s19pjs5d_pv_s3bpjs7f = addpz(s19pjs5d, v8_s3bpjs7f);

            const xmm h1_s19mjs5d_pw_s3bmjs7f = h1xpz(s19mjs5d_pw_s3bmjs7f);
            const xmm w8_a19m1a5d_mj_a3bm1a7f = w8xpz(a19m1a5d_mj_a3bm1a7f);
            const xmm h3_s19pjs5d_mv_s3bpjs7f = h3xpz(s19pjs5d_mv_s3bpjs7f);
            const xmm  j_a19p1a5d_m1_a3bp1a7f =  jxpz(a19p1a5d_m1_a3bp1a7f);
            const xmm hd_s19mjs5d_mw_s3bmjs7f = hdxpz(s19mjs5d_mw_s3bmjs7f);
            const xmm v8_a19m1a5d_pj_a3bm1a7f = v8xpz(a19m1a5d_pj_a3bm1a7f);
            const xmm hf_s19pjs5d_pv_s3bpjs7f = hfxpz(s19pjs5d_pv_s3bpjs7f);

            setpz(x[0x0], addpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz(x[0x1], addpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz(x[0x2], addpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz(x[0x3], addpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz(x[0x4], subpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz(x[0x5], subpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz(x[0x6], subpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz(x[0x7], subpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

            setpz(x[0x8], subpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz(x[0x9], subpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz(x[0xa], subpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz(x[0xb], subpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz(x[0xc], addpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz(x[0xd], addpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz(x[0xe], addpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz(x[0xf], addpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
        }
    }
};

//-----------------------------------------------------------------------------

template <int s> struct fwd0end<16,s,0>
{
    void operator()(complex_vector x, complex_vector) const noexcept
    {
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;

            const ymm x0 = getpz2(xq+s*0x0);
            const ymm x1 = getpz2(xq+s*0x1);
            const ymm x2 = getpz2(xq+s*0x2);
            const ymm x3 = getpz2(xq+s*0x3);
            const ymm x4 = getpz2(xq+s*0x4);
            const ymm x5 = getpz2(xq+s*0x5);
            const ymm x6 = getpz2(xq+s*0x6);
            const ymm x7 = getpz2(xq+s*0x7);
            const ymm x8 = getpz2(xq+s*0x8);
            const ymm x9 = getpz2(xq+s*0x9);
            const ymm xa = getpz2(xq+s*0xa);
            const ymm xb = getpz2(xq+s*0xb);
            const ymm xc = getpz2(xq+s*0xc);
            const ymm xd = getpz2(xq+s*0xd);
            const ymm xe = getpz2(xq+s*0xe);
            const ymm xf = getpz2(xq+s*0xf);

            const ymm a08 = addpz2(x0, x8); const ymm s08 = subpz2(x0, x8);
            const ymm a4c = addpz2(x4, xc); const ymm s4c = subpz2(x4, xc);
            const ymm a2a = addpz2(x2, xa); const ymm s2a = subpz2(x2, xa);
            const ymm a6e = addpz2(x6, xe); const ymm s6e = subpz2(x6, xe);
            const ymm a19 = addpz2(x1, x9); const ymm s19 = subpz2(x1, x9);
            const ymm a5d = addpz2(x5, xd); const ymm s5d = subpz2(x5, xd);
            const ymm a3b = addpz2(x3, xb); const ymm s3b = subpz2(x3, xb);
            const ymm a7f = addpz2(x7, xf); const ymm s7f = subpz2(x7, xf);

            const ymm js4c = jxpz2(s4c);
            const ymm js6e = jxpz2(s6e);
            const ymm js5d = jxpz2(s5d);
            const ymm js7f = jxpz2(s7f);

            const ymm a08p1a4c = addpz2(a08, a4c); const ymm s08mjs4c = subpz2(s08, js4c);
            const ymm a08m1a4c = subpz2(a08, a4c); const ymm s08pjs4c = addpz2(s08, js4c);
            const ymm a2ap1a6e = addpz2(a2a, a6e); const ymm s2amjs6e = subpz2(s2a, js6e);
            const ymm a2am1a6e = subpz2(a2a, a6e); const ymm s2apjs6e = addpz2(s2a, js6e);
            const ymm a19p1a5d = addpz2(a19, a5d); const ymm s19mjs5d = subpz2(s19, js5d);
            const ymm a19m1a5d = subpz2(a19, a5d); const ymm s19pjs5d = addpz2(s19, js5d);
            const ymm a3bp1a7f = addpz2(a3b, a7f); const ymm s3bmjs7f = subpz2(s3b, js7f);
            const ymm a3bm1a7f = subpz2(a3b, a7f); const ymm s3bpjs7f = addpz2(s3b, js7f);

            const ymm w8_s2amjs6e = w8xpz2(s2amjs6e);
            const ymm  j_a2am1a6e =  jxpz2(a2am1a6e);
            const ymm v8_s2apjs6e = v8xpz2(s2apjs6e);

            const ymm a08p1a4c_p1_a2ap1a6e = addpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_pw_s2amjs6e = addpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_mj_a2am1a6e = subpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_mv_s2apjs6e = subpz2(s08pjs4c, v8_s2apjs6e);
            const ymm a08p1a4c_m1_a2ap1a6e = subpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_mw_s2amjs6e = subpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_pj_a2am1a6e = addpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_pv_s2apjs6e = addpz2(s08pjs4c, v8_s2apjs6e);

            const ymm w8_s3bmjs7f = w8xpz2(s3bmjs7f);
            const ymm  j_a3bm1a7f =  jxpz2(a3bm1a7f);
            const ymm v8_s3bpjs7f = v8xpz2(s3bpjs7f);

            const ymm a19p1a5d_p1_a3bp1a7f = addpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_pw_s3bmjs7f = addpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_mj_a3bm1a7f = subpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_mv_s3bpjs7f = subpz2(s19pjs5d, v8_s3bpjs7f);
            const ymm a19p1a5d_m1_a3bp1a7f = subpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_mw_s3bmjs7f = subpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_pj_a3bm1a7f = addpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_pv_s3bpjs7f = addpz2(s19pjs5d, v8_s3bpjs7f);

            const ymm h1_s19mjs5d_pw_s3bmjs7f = h1xpz2(s19mjs5d_pw_s3bmjs7f);
            const ymm w8_a19m1a5d_mj_a3bm1a7f = w8xpz2(a19m1a5d_mj_a3bm1a7f);
            const ymm h3_s19pjs5d_mv_s3bpjs7f = h3xpz2(s19pjs5d_mv_s3bpjs7f);
            const ymm  j_a19p1a5d_m1_a3bp1a7f =  jxpz2(a19p1a5d_m1_a3bp1a7f);
            const ymm hd_s19mjs5d_mw_s3bmjs7f = hdxpz2(s19mjs5d_mw_s3bmjs7f);
            const ymm v8_a19m1a5d_pj_a3bm1a7f = v8xpz2(a19m1a5d_pj_a3bm1a7f);
            const ymm hf_s19pjs5d_pv_s3bpjs7f = hfxpz2(s19pjs5d_pv_s3bpjs7f);

            setpz2(xq+s*0x0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq+s*0x1, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz2(xq+s*0x2, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq+s*0x3, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq+s*0x4, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq+s*0x5, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq+s*0x6, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq+s*0x7, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

            setpz2(xq+s*0x8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq+s*0x9, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz2(xq+s*0xa, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq+s*0xb, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq+s*0xc, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq+s*0xd, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq+s*0xe, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq+s*0xf, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
        }
    }
};

template <> struct fwd0end<16,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const noexcept
    {
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
            const xmm x0 = getpz(x[0x0]);
            const xmm x1 = getpz(x[0x1]);
            const xmm x2 = getpz(x[0x2]);
            const xmm x3 = getpz(x[0x3]);
            const xmm x4 = getpz(x[0x4]);
            const xmm x5 = getpz(x[0x5]);
            const xmm x6 = getpz(x[0x6]);
            const xmm x7 = getpz(x[0x7]);
            const xmm x8 = getpz(x[0x8]);
            const xmm x9 = getpz(x[0x9]);
            const xmm xa = getpz(x[0xa]);
            const xmm xb = getpz(x[0xb]);
            const xmm xc = getpz(x[0xc]);
            const xmm xd = getpz(x[0xd]);
            const xmm xe = getpz(x[0xe]);
            const xmm xf = getpz(x[0xf]);

            const xmm a08 = addpz(x0, x8); const xmm s08 = subpz(x0, x8);
            const xmm a4c = addpz(x4, xc); const xmm s4c = subpz(x4, xc);
            const xmm a2a = addpz(x2, xa); const xmm s2a = subpz(x2, xa);
            const xmm a6e = addpz(x6, xe); const xmm s6e = subpz(x6, xe);
            const xmm a19 = addpz(x1, x9); const xmm s19 = subpz(x1, x9);
            const xmm a5d = addpz(x5, xd); const xmm s5d = subpz(x5, xd);
            const xmm a3b = addpz(x3, xb); const xmm s3b = subpz(x3, xb);
            const xmm a7f = addpz(x7, xf); const xmm s7f = subpz(x7, xf);

            const xmm js4c = jxpz(s4c);
            const xmm js6e = jxpz(s6e);
            const xmm js5d = jxpz(s5d);
            const xmm js7f = jxpz(s7f);

            const xmm a08p1a4c = addpz(a08, a4c); const xmm s08mjs4c = subpz(s08, js4c);
            const xmm a08m1a4c = subpz(a08, a4c); const xmm s08pjs4c = addpz(s08, js4c);
            const xmm a2ap1a6e = addpz(a2a, a6e); const xmm s2amjs6e = subpz(s2a, js6e);
            const xmm a2am1a6e = subpz(a2a, a6e); const xmm s2apjs6e = addpz(s2a, js6e);
            const xmm a19p1a5d = addpz(a19, a5d); const xmm s19mjs5d = subpz(s19, js5d);
            const xmm a19m1a5d = subpz(a19, a5d); const xmm s19pjs5d = addpz(s19, js5d);
            const xmm a3bp1a7f = addpz(a3b, a7f); const xmm s3bmjs7f = subpz(s3b, js7f);
            const xmm a3bm1a7f = subpz(a3b, a7f); const xmm s3bpjs7f = addpz(s3b, js7f);

            const xmm w8_s2amjs6e = w8xpz(s2amjs6e);
            const xmm  j_a2am1a6e =  jxpz(a2am1a6e);
            const xmm v8_s2apjs6e = v8xpz(s2apjs6e);

            const xmm a08p1a4c_p1_a2ap1a6e = addpz(a08p1a4c,    a2ap1a6e);
            const xmm s08mjs4c_pw_s2amjs6e = addpz(s08mjs4c, w8_s2amjs6e);
            const xmm a08m1a4c_mj_a2am1a6e = subpz(a08m1a4c,  j_a2am1a6e);
            const xmm s08pjs4c_mv_s2apjs6e = subpz(s08pjs4c, v8_s2apjs6e);
            const xmm a08p1a4c_m1_a2ap1a6e = subpz(a08p1a4c,    a2ap1a6e);
            const xmm s08mjs4c_mw_s2amjs6e = subpz(s08mjs4c, w8_s2amjs6e);
            const xmm a08m1a4c_pj_a2am1a6e = addpz(a08m1a4c,  j_a2am1a6e);
            const xmm s08pjs4c_pv_s2apjs6e = addpz(s08pjs4c, v8_s2apjs6e);

            const xmm w8_s3bmjs7f = w8xpz(s3bmjs7f);
            const xmm  j_a3bm1a7f =  jxpz(a3bm1a7f);
            const xmm v8_s3bpjs7f = v8xpz(s3bpjs7f);

            const xmm a19p1a5d_p1_a3bp1a7f = addpz(a19p1a5d,    a3bp1a7f);
            const xmm s19mjs5d_pw_s3bmjs7f = addpz(s19mjs5d, w8_s3bmjs7f);
            const xmm a19m1a5d_mj_a3bm1a7f = subpz(a19m1a5d,  j_a3bm1a7f);
            const xmm s19pjs5d_mv_s3bpjs7f = subpz(s19pjs5d, v8_s3bpjs7f);
            const xmm a19p1a5d_m1_a3bp1a7f = subpz(a19p1a5d,    a3bp1a7f);
            const xmm s19mjs5d_mw_s3bmjs7f = subpz(s19mjs5d, w8_s3bmjs7f);
            const xmm a19m1a5d_pj_a3bm1a7f = addpz(a19m1a5d,  j_a3bm1a7f);
            const xmm s19pjs5d_pv_s3bpjs7f = addpz(s19pjs5d, v8_s3bpjs7f);

            const xmm h1_s19mjs5d_pw_s3bmjs7f = h1xpz(s19mjs5d_pw_s3bmjs7f);
            const xmm w8_a19m1a5d_mj_a3bm1a7f = w8xpz(a19m1a5d_mj_a3bm1a7f);
            const xmm h3_s19pjs5d_mv_s3bpjs7f = h3xpz(s19pjs5d_mv_s3bpjs7f);
            const xmm  j_a19p1a5d_m1_a3bp1a7f =  jxpz(a19p1a5d_m1_a3bp1a7f);
            const xmm hd_s19mjs5d_mw_s3bmjs7f = hdxpz(s19mjs5d_mw_s3bmjs7f);
            const xmm v8_a19m1a5d_pj_a3bm1a7f = v8xpz(a19m1a5d_pj_a3bm1a7f);
            const xmm hf_s19pjs5d_pv_s3bpjs7f = hfxpz(s19pjs5d_pv_s3bpjs7f);

            setpz(x[0x0], addpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz(x[0x1], addpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz(x[0x2], addpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz(x[0x3], addpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz(x[0x4], subpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz(x[0x5], subpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz(x[0x6], subpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz(x[0x7], subpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

            setpz(x[0x8], subpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz(x[0x9], subpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz(x[0xa], subpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz(x[0xb], subpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz(x[0xc], addpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz(x[0xd], addpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz(x[0xe], addpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz(x[0xf], addpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct fwdnend;

//-----------------------------------------------------------------------------

template <int s> struct fwdnend<16,s,1>
{
    static const int N = 16*s;

    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;

            const ymm y0 = mulpd2(rN, getpz2(yq+s*0x0));
            const ymm y1 = mulpd2(rN, getpz2(yq+s*0x1));
            const ymm y2 = mulpd2(rN, getpz2(yq+s*0x2));
            const ymm y3 = mulpd2(rN, getpz2(yq+s*0x3));
            const ymm y4 = mulpd2(rN, getpz2(yq+s*0x4));
            const ymm y5 = mulpd2(rN, getpz2(yq+s*0x5));
            const ymm y6 = mulpd2(rN, getpz2(yq+s*0x6));
            const ymm y7 = mulpd2(rN, getpz2(yq+s*0x7));
            const ymm y8 = mulpd2(rN, getpz2(yq+s*0x8));
            const ymm y9 = mulpd2(rN, getpz2(yq+s*0x9));
            const ymm ya = mulpd2(rN, getpz2(yq+s*0xa));
            const ymm yb = mulpd2(rN, getpz2(yq+s*0xb));
            const ymm yc = mulpd2(rN, getpz2(yq+s*0xc));
            const ymm yd = mulpd2(rN, getpz2(yq+s*0xd));
            const ymm ye = mulpd2(rN, getpz2(yq+s*0xe));
            const ymm yf = mulpd2(rN, getpz2(yq+s*0xf));

            const ymm a08 = addpz2(y0, y8); const ymm s08 = subpz2(y0, y8);
            const ymm a4c = addpz2(y4, yc); const ymm s4c = subpz2(y4, yc);
            const ymm a2a = addpz2(y2, ya); const ymm s2a = subpz2(y2, ya);
            const ymm a6e = addpz2(y6, ye); const ymm s6e = subpz2(y6, ye);
            const ymm a19 = addpz2(y1, y9); const ymm s19 = subpz2(y1, y9);
            const ymm a5d = addpz2(y5, yd); const ymm s5d = subpz2(y5, yd);
            const ymm a3b = addpz2(y3, yb); const ymm s3b = subpz2(y3, yb);
            const ymm a7f = addpz2(y7, yf); const ymm s7f = subpz2(y7, yf);

            const ymm js4c = jxpz2(s4c);
            const ymm js6e = jxpz2(s6e);
            const ymm js5d = jxpz2(s5d);
            const ymm js7f = jxpz2(s7f);

            const ymm a08p1a4c = addpz2(a08, a4c); const ymm s08mjs4c = subpz2(s08, js4c);
            const ymm a08m1a4c = subpz2(a08, a4c); const ymm s08pjs4c = addpz2(s08, js4c);
            const ymm a2ap1a6e = addpz2(a2a, a6e); const ymm s2amjs6e = subpz2(s2a, js6e);
            const ymm a2am1a6e = subpz2(a2a, a6e); const ymm s2apjs6e = addpz2(s2a, js6e);
            const ymm a19p1a5d = addpz2(a19, a5d); const ymm s19mjs5d = subpz2(s19, js5d);
            const ymm a19m1a5d = subpz2(a19, a5d); const ymm s19pjs5d = addpz2(s19, js5d);
            const ymm a3bp1a7f = addpz2(a3b, a7f); const ymm s3bmjs7f = subpz2(s3b, js7f);
            const ymm a3bm1a7f = subpz2(a3b, a7f); const ymm s3bpjs7f = addpz2(s3b, js7f);

            const ymm w8_s2amjs6e = w8xpz2(s2amjs6e);
            const ymm  j_a2am1a6e =  jxpz2(a2am1a6e);
            const ymm v8_s2apjs6e = v8xpz2(s2apjs6e);

            const ymm a08p1a4c_p1_a2ap1a6e = addpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_pw_s2amjs6e = addpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_mj_a2am1a6e = subpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_mv_s2apjs6e = subpz2(s08pjs4c, v8_s2apjs6e);
            const ymm a08p1a4c_m1_a2ap1a6e = subpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_mw_s2amjs6e = subpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_pj_a2am1a6e = addpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_pv_s2apjs6e = addpz2(s08pjs4c, v8_s2apjs6e);

            const ymm w8_s3bmjs7f = w8xpz2(s3bmjs7f);
            const ymm  j_a3bm1a7f =  jxpz2(a3bm1a7f);
            const ymm v8_s3bpjs7f = v8xpz2(s3bpjs7f);

            const ymm a19p1a5d_p1_a3bp1a7f = addpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_pw_s3bmjs7f = addpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_mj_a3bm1a7f = subpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_mv_s3bpjs7f = subpz2(s19pjs5d, v8_s3bpjs7f);
            const ymm a19p1a5d_m1_a3bp1a7f = subpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_mw_s3bmjs7f = subpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_pj_a3bm1a7f = addpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_pv_s3bpjs7f = addpz2(s19pjs5d, v8_s3bpjs7f);

            const ymm h1_s19mjs5d_pw_s3bmjs7f = h1xpz2(s19mjs5d_pw_s3bmjs7f);
            const ymm w8_a19m1a5d_mj_a3bm1a7f = w8xpz2(a19m1a5d_mj_a3bm1a7f);
            const ymm h3_s19pjs5d_mv_s3bpjs7f = h3xpz2(s19pjs5d_mv_s3bpjs7f);
            const ymm  j_a19p1a5d_m1_a3bp1a7f =  jxpz2(a19p1a5d_m1_a3bp1a7f);
            const ymm hd_s19mjs5d_mw_s3bmjs7f = hdxpz2(s19mjs5d_mw_s3bmjs7f);
            const ymm v8_a19m1a5d_pj_a3bm1a7f = v8xpz2(a19m1a5d_pj_a3bm1a7f);
            const ymm hf_s19pjs5d_pv_s3bpjs7f = hfxpz2(s19pjs5d_pv_s3bpjs7f);

            setpz2(xq+s*0x0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq+s*0x1, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz2(xq+s*0x2, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq+s*0x3, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq+s*0x4, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq+s*0x5, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq+s*0x6, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq+s*0x7, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

            setpz2(xq+s*0x8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq+s*0x9, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz2(xq+s*0xa, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq+s*0xb, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq+s*0xc, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq+s*0xd, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq+s*0xe, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq+s*0xf, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
        }
    }
};

template <> struct fwdnend<16,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        static const xmm rN = { 1.0/16, 1.0/16 };
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
            const xmm y0 = mulpd(rN, getpz(y[0x0]));
            const xmm y1 = mulpd(rN, getpz(y[0x1]));
            const xmm y2 = mulpd(rN, getpz(y[0x2]));
            const xmm y3 = mulpd(rN, getpz(y[0x3]));
            const xmm y4 = mulpd(rN, getpz(y[0x4]));
            const xmm y5 = mulpd(rN, getpz(y[0x5]));
            const xmm y6 = mulpd(rN, getpz(y[0x6]));
            const xmm y7 = mulpd(rN, getpz(y[0x7]));
            const xmm y8 = mulpd(rN, getpz(y[0x8]));
            const xmm y9 = mulpd(rN, getpz(y[0x9]));
            const xmm ya = mulpd(rN, getpz(y[0xa]));
            const xmm yb = mulpd(rN, getpz(y[0xb]));
            const xmm yc = mulpd(rN, getpz(y[0xc]));
            const xmm yd = mulpd(rN, getpz(y[0xd]));
            const xmm ye = mulpd(rN, getpz(y[0xe]));
            const xmm yf = mulpd(rN, getpz(y[0xf]));

            const xmm a08 = addpz(y0, y8); const xmm s08 = subpz(y0, y8);
            const xmm a4c = addpz(y4, yc); const xmm s4c = subpz(y4, yc);
            const xmm a2a = addpz(y2, ya); const xmm s2a = subpz(y2, ya);
            const xmm a6e = addpz(y6, ye); const xmm s6e = subpz(y6, ye);
            const xmm a19 = addpz(y1, y9); const xmm s19 = subpz(y1, y9);
            const xmm a5d = addpz(y5, yd); const xmm s5d = subpz(y5, yd);
            const xmm a3b = addpz(y3, yb); const xmm s3b = subpz(y3, yb);
            const xmm a7f = addpz(y7, yf); const xmm s7f = subpz(y7, yf);

            const xmm js4c = jxpz(s4c);
            const xmm js6e = jxpz(s6e);
            const xmm js5d = jxpz(s5d);
            const xmm js7f = jxpz(s7f);

            const xmm a08p1a4c = addpz(a08, a4c); const xmm s08mjs4c = subpz(s08, js4c);
            const xmm a08m1a4c = subpz(a08, a4c); const xmm s08pjs4c = addpz(s08, js4c);
            const xmm a2ap1a6e = addpz(a2a, a6e); const xmm s2amjs6e = subpz(s2a, js6e);
            const xmm a2am1a6e = subpz(a2a, a6e); const xmm s2apjs6e = addpz(s2a, js6e);
            const xmm a19p1a5d = addpz(a19, a5d); const xmm s19mjs5d = subpz(s19, js5d);
            const xmm a19m1a5d = subpz(a19, a5d); const xmm s19pjs5d = addpz(s19, js5d);
            const xmm a3bp1a7f = addpz(a3b, a7f); const xmm s3bmjs7f = subpz(s3b, js7f);
            const xmm a3bm1a7f = subpz(a3b, a7f); const xmm s3bpjs7f = addpz(s3b, js7f);

            const xmm w8_s2amjs6e = w8xpz(s2amjs6e);
            const xmm  j_a2am1a6e =  jxpz(a2am1a6e);
            const xmm v8_s2apjs6e = v8xpz(s2apjs6e);

            const xmm a08p1a4c_p1_a2ap1a6e = addpz(a08p1a4c,    a2ap1a6e);
            const xmm s08mjs4c_pw_s2amjs6e = addpz(s08mjs4c, w8_s2amjs6e);
            const xmm a08m1a4c_mj_a2am1a6e = subpz(a08m1a4c,  j_a2am1a6e);
            const xmm s08pjs4c_mv_s2apjs6e = subpz(s08pjs4c, v8_s2apjs6e);
            const xmm a08p1a4c_m1_a2ap1a6e = subpz(a08p1a4c,    a2ap1a6e);
            const xmm s08mjs4c_mw_s2amjs6e = subpz(s08mjs4c, w8_s2amjs6e);
            const xmm a08m1a4c_pj_a2am1a6e = addpz(a08m1a4c,  j_a2am1a6e);
            const xmm s08pjs4c_pv_s2apjs6e = addpz(s08pjs4c, v8_s2apjs6e);

            const xmm w8_s3bmjs7f = w8xpz(s3bmjs7f);
            const xmm  j_a3bm1a7f =  jxpz(a3bm1a7f);
            const xmm v8_s3bpjs7f = v8xpz(s3bpjs7f);

            const xmm a19p1a5d_p1_a3bp1a7f = addpz(a19p1a5d,    a3bp1a7f);
            const xmm s19mjs5d_pw_s3bmjs7f = addpz(s19mjs5d, w8_s3bmjs7f);
            const xmm a19m1a5d_mj_a3bm1a7f = subpz(a19m1a5d,  j_a3bm1a7f);
            const xmm s19pjs5d_mv_s3bpjs7f = subpz(s19pjs5d, v8_s3bpjs7f);
            const xmm a19p1a5d_m1_a3bp1a7f = subpz(a19p1a5d,    a3bp1a7f);
            const xmm s19mjs5d_mw_s3bmjs7f = subpz(s19mjs5d, w8_s3bmjs7f);
            const xmm a19m1a5d_pj_a3bm1a7f = addpz(a19m1a5d,  j_a3bm1a7f);
            const xmm s19pjs5d_pv_s3bpjs7f = addpz(s19pjs5d, v8_s3bpjs7f);

            const xmm h1_s19mjs5d_pw_s3bmjs7f = h1xpz(s19mjs5d_pw_s3bmjs7f);
            const xmm w8_a19m1a5d_mj_a3bm1a7f = w8xpz(a19m1a5d_mj_a3bm1a7f);
            const xmm h3_s19pjs5d_mv_s3bpjs7f = h3xpz(s19pjs5d_mv_s3bpjs7f);
            const xmm  j_a19p1a5d_m1_a3bp1a7f =  jxpz(a19p1a5d_m1_a3bp1a7f);
            const xmm hd_s19mjs5d_mw_s3bmjs7f = hdxpz(s19mjs5d_mw_s3bmjs7f);
            const xmm v8_a19m1a5d_pj_a3bm1a7f = v8xpz(a19m1a5d_pj_a3bm1a7f);
            const xmm hf_s19pjs5d_pv_s3bpjs7f = hfxpz(s19pjs5d_pv_s3bpjs7f);

            setpz(x[0x0], addpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz(x[0x1], addpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz(x[0x2], addpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz(x[0x3], addpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz(x[0x4], subpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz(x[0x5], subpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz(x[0x6], subpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz(x[0x7], subpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

            setpz(x[0x8], subpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz(x[0x9], subpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz(x[0xa], subpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz(x[0xb], subpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz(x[0xc], addpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz(x[0xd], addpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz(x[0xe], addpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz(x[0xf], addpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
        }
    }
};

//-----------------------------------------------------------------------------

template <int s> struct fwdnend<16,s,0>
{
    static const int N = 16*s;

    void operator()(complex_vector x, complex_vector) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;

            const ymm x0 = mulpd2(rN, getpz2(xq+s*0x0));
            const ymm x1 = mulpd2(rN, getpz2(xq+s*0x1));
            const ymm x2 = mulpd2(rN, getpz2(xq+s*0x2));
            const ymm x3 = mulpd2(rN, getpz2(xq+s*0x3));
            const ymm x4 = mulpd2(rN, getpz2(xq+s*0x4));
            const ymm x5 = mulpd2(rN, getpz2(xq+s*0x5));
            const ymm x6 = mulpd2(rN, getpz2(xq+s*0x6));
            const ymm x7 = mulpd2(rN, getpz2(xq+s*0x7));
            const ymm x8 = mulpd2(rN, getpz2(xq+s*0x8));
            const ymm x9 = mulpd2(rN, getpz2(xq+s*0x9));
            const ymm xa = mulpd2(rN, getpz2(xq+s*0xa));
            const ymm xb = mulpd2(rN, getpz2(xq+s*0xb));
            const ymm xc = mulpd2(rN, getpz2(xq+s*0xc));
            const ymm xd = mulpd2(rN, getpz2(xq+s*0xd));
            const ymm xe = mulpd2(rN, getpz2(xq+s*0xe));
            const ymm xf = mulpd2(rN, getpz2(xq+s*0xf));

            const ymm a08 = addpz2(x0, x8); const ymm s08 = subpz2(x0, x8);
            const ymm a4c = addpz2(x4, xc); const ymm s4c = subpz2(x4, xc);
            const ymm a2a = addpz2(x2, xa); const ymm s2a = subpz2(x2, xa);
            const ymm a6e = addpz2(x6, xe); const ymm s6e = subpz2(x6, xe);
            const ymm a19 = addpz2(x1, x9); const ymm s19 = subpz2(x1, x9);
            const ymm a5d = addpz2(x5, xd); const ymm s5d = subpz2(x5, xd);
            const ymm a3b = addpz2(x3, xb); const ymm s3b = subpz2(x3, xb);
            const ymm a7f = addpz2(x7, xf); const ymm s7f = subpz2(x7, xf);

            const ymm js4c = jxpz2(s4c);
            const ymm js6e = jxpz2(s6e);
            const ymm js5d = jxpz2(s5d);
            const ymm js7f = jxpz2(s7f);

            const ymm a08p1a4c = addpz2(a08, a4c); const ymm s08mjs4c = subpz2(s08, js4c);
            const ymm a08m1a4c = subpz2(a08, a4c); const ymm s08pjs4c = addpz2(s08, js4c);
            const ymm a2ap1a6e = addpz2(a2a, a6e); const ymm s2amjs6e = subpz2(s2a, js6e);
            const ymm a2am1a6e = subpz2(a2a, a6e); const ymm s2apjs6e = addpz2(s2a, js6e);
            const ymm a19p1a5d = addpz2(a19, a5d); const ymm s19mjs5d = subpz2(s19, js5d);
            const ymm a19m1a5d = subpz2(a19, a5d); const ymm s19pjs5d = addpz2(s19, js5d);
            const ymm a3bp1a7f = addpz2(a3b, a7f); const ymm s3bmjs7f = subpz2(s3b, js7f);
            const ymm a3bm1a7f = subpz2(a3b, a7f); const ymm s3bpjs7f = addpz2(s3b, js7f);

            const ymm w8_s2amjs6e = w8xpz2(s2amjs6e);
            const ymm  j_a2am1a6e =  jxpz2(a2am1a6e);
            const ymm v8_s2apjs6e = v8xpz2(s2apjs6e);

            const ymm a08p1a4c_p1_a2ap1a6e = addpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_pw_s2amjs6e = addpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_mj_a2am1a6e = subpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_mv_s2apjs6e = subpz2(s08pjs4c, v8_s2apjs6e);
            const ymm a08p1a4c_m1_a2ap1a6e = subpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_mw_s2amjs6e = subpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_pj_a2am1a6e = addpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_pv_s2apjs6e = addpz2(s08pjs4c, v8_s2apjs6e);

            const ymm w8_s3bmjs7f = w8xpz2(s3bmjs7f);
            const ymm  j_a3bm1a7f =  jxpz2(a3bm1a7f);
            const ymm v8_s3bpjs7f = v8xpz2(s3bpjs7f);

            const ymm a19p1a5d_p1_a3bp1a7f = addpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_pw_s3bmjs7f = addpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_mj_a3bm1a7f = subpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_mv_s3bpjs7f = subpz2(s19pjs5d, v8_s3bpjs7f);
            const ymm a19p1a5d_m1_a3bp1a7f = subpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_mw_s3bmjs7f = subpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_pj_a3bm1a7f = addpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_pv_s3bpjs7f = addpz2(s19pjs5d, v8_s3bpjs7f);

            const ymm h1_s19mjs5d_pw_s3bmjs7f = h1xpz2(s19mjs5d_pw_s3bmjs7f);
            const ymm w8_a19m1a5d_mj_a3bm1a7f = w8xpz2(a19m1a5d_mj_a3bm1a7f);
            const ymm h3_s19pjs5d_mv_s3bpjs7f = h3xpz2(s19pjs5d_mv_s3bpjs7f);
            const ymm  j_a19p1a5d_m1_a3bp1a7f =  jxpz2(a19p1a5d_m1_a3bp1a7f);
            const ymm hd_s19mjs5d_mw_s3bmjs7f = hdxpz2(s19mjs5d_mw_s3bmjs7f);
            const ymm v8_a19m1a5d_pj_a3bm1a7f = v8xpz2(a19m1a5d_pj_a3bm1a7f);
            const ymm hf_s19pjs5d_pv_s3bpjs7f = hfxpz2(s19pjs5d_pv_s3bpjs7f);

            setpz2(xq+s*0x0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq+s*0x1, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz2(xq+s*0x2, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq+s*0x3, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq+s*0x4, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq+s*0x5, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq+s*0x6, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq+s*0x7, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

            setpz2(xq+s*0x8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq+s*0x9, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz2(xq+s*0xa, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq+s*0xb, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq+s*0xc, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq+s*0xd, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq+s*0xe, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq+s*0xf, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
        }
    }
};

template <> struct fwdnend<16,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const noexcept
    {
        static const xmm rN = { 1.0/16, 1.0/16 };
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
            const xmm x0 = mulpd(rN, getpz(x[0x0]));
            const xmm x1 = mulpd(rN, getpz(x[0x1]));
            const xmm x2 = mulpd(rN, getpz(x[0x2]));
            const xmm x3 = mulpd(rN, getpz(x[0x3]));
            const xmm x4 = mulpd(rN, getpz(x[0x4]));
            const xmm x5 = mulpd(rN, getpz(x[0x5]));
            const xmm x6 = mulpd(rN, getpz(x[0x6]));
            const xmm x7 = mulpd(rN, getpz(x[0x7]));
            const xmm x8 = mulpd(rN, getpz(x[0x8]));
            const xmm x9 = mulpd(rN, getpz(x[0x9]));
            const xmm xa = mulpd(rN, getpz(x[0xa]));
            const xmm xb = mulpd(rN, getpz(x[0xb]));
            const xmm xc = mulpd(rN, getpz(x[0xc]));
            const xmm xd = mulpd(rN, getpz(x[0xd]));
            const xmm xe = mulpd(rN, getpz(x[0xe]));
            const xmm xf = mulpd(rN, getpz(x[0xf]));

            const xmm a08 = addpz(x0, x8); const xmm s08 = subpz(x0, x8);
            const xmm a4c = addpz(x4, xc); const xmm s4c = subpz(x4, xc);
            const xmm a2a = addpz(x2, xa); const xmm s2a = subpz(x2, xa);
            const xmm a6e = addpz(x6, xe); const xmm s6e = subpz(x6, xe);
            const xmm a19 = addpz(x1, x9); const xmm s19 = subpz(x1, x9);
            const xmm a5d = addpz(x5, xd); const xmm s5d = subpz(x5, xd);
            const xmm a3b = addpz(x3, xb); const xmm s3b = subpz(x3, xb);
            const xmm a7f = addpz(x7, xf); const xmm s7f = subpz(x7, xf);

            const xmm js4c = jxpz(s4c);
            const xmm js6e = jxpz(s6e);
            const xmm js5d = jxpz(s5d);
            const xmm js7f = jxpz(s7f);

            const xmm a08p1a4c = addpz(a08, a4c); const xmm s08mjs4c = subpz(s08, js4c);
            const xmm a08m1a4c = subpz(a08, a4c); const xmm s08pjs4c = addpz(s08, js4c);
            const xmm a2ap1a6e = addpz(a2a, a6e); const xmm s2amjs6e = subpz(s2a, js6e);
            const xmm a2am1a6e = subpz(a2a, a6e); const xmm s2apjs6e = addpz(s2a, js6e);
            const xmm a19p1a5d = addpz(a19, a5d); const xmm s19mjs5d = subpz(s19, js5d);
            const xmm a19m1a5d = subpz(a19, a5d); const xmm s19pjs5d = addpz(s19, js5d);
            const xmm a3bp1a7f = addpz(a3b, a7f); const xmm s3bmjs7f = subpz(s3b, js7f);
            const xmm a3bm1a7f = subpz(a3b, a7f); const xmm s3bpjs7f = addpz(s3b, js7f);

            const xmm w8_s2amjs6e = w8xpz(s2amjs6e);
            const xmm  j_a2am1a6e =  jxpz(a2am1a6e);
            const xmm v8_s2apjs6e = v8xpz(s2apjs6e);

            const xmm a08p1a4c_p1_a2ap1a6e = addpz(a08p1a4c,    a2ap1a6e);
            const xmm s08mjs4c_pw_s2amjs6e = addpz(s08mjs4c, w8_s2amjs6e);
            const xmm a08m1a4c_mj_a2am1a6e = subpz(a08m1a4c,  j_a2am1a6e);
            const xmm s08pjs4c_mv_s2apjs6e = subpz(s08pjs4c, v8_s2apjs6e);
            const xmm a08p1a4c_m1_a2ap1a6e = subpz(a08p1a4c,    a2ap1a6e);
            const xmm s08mjs4c_mw_s2amjs6e = subpz(s08mjs4c, w8_s2amjs6e);
            const xmm a08m1a4c_pj_a2am1a6e = addpz(a08m1a4c,  j_a2am1a6e);
            const xmm s08pjs4c_pv_s2apjs6e = addpz(s08pjs4c, v8_s2apjs6e);

            const xmm w8_s3bmjs7f = w8xpz(s3bmjs7f);
            const xmm  j_a3bm1a7f =  jxpz(a3bm1a7f);
            const xmm v8_s3bpjs7f = v8xpz(s3bpjs7f);

            const xmm a19p1a5d_p1_a3bp1a7f = addpz(a19p1a5d,    a3bp1a7f);
            const xmm s19mjs5d_pw_s3bmjs7f = addpz(s19mjs5d, w8_s3bmjs7f);
            const xmm a19m1a5d_mj_a3bm1a7f = subpz(a19m1a5d,  j_a3bm1a7f);
            const xmm s19pjs5d_mv_s3bpjs7f = subpz(s19pjs5d, v8_s3bpjs7f);
            const xmm a19p1a5d_m1_a3bp1a7f = subpz(a19p1a5d,    a3bp1a7f);
            const xmm s19mjs5d_mw_s3bmjs7f = subpz(s19mjs5d, w8_s3bmjs7f);
            const xmm a19m1a5d_pj_a3bm1a7f = addpz(a19m1a5d,  j_a3bm1a7f);
            const xmm s19pjs5d_pv_s3bpjs7f = addpz(s19pjs5d, v8_s3bpjs7f);

            const xmm h1_s19mjs5d_pw_s3bmjs7f = h1xpz(s19mjs5d_pw_s3bmjs7f);
            const xmm w8_a19m1a5d_mj_a3bm1a7f = w8xpz(a19m1a5d_mj_a3bm1a7f);
            const xmm h3_s19pjs5d_mv_s3bpjs7f = h3xpz(s19pjs5d_mv_s3bpjs7f);
            const xmm  j_a19p1a5d_m1_a3bp1a7f =  jxpz(a19p1a5d_m1_a3bp1a7f);
            const xmm hd_s19mjs5d_mw_s3bmjs7f = hdxpz(s19mjs5d_mw_s3bmjs7f);
            const xmm v8_a19m1a5d_pj_a3bm1a7f = v8xpz(a19m1a5d_pj_a3bm1a7f);
            const xmm hf_s19pjs5d_pv_s3bpjs7f = hfxpz(s19pjs5d_pv_s3bpjs7f);

            setpz(x[0x0], addpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz(x[0x1], addpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz(x[0x2], addpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz(x[0x3], addpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz(x[0x4], subpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz(x[0x5], subpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz(x[0x6], subpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz(x[0x7], subpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

            setpz(x[0x8], subpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz(x[0x9], subpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz(x[0xa], subpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz(x[0xb], subpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz(x[0xc], addpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz(x[0xd], addpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz(x[0xe], addpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz(x[0xf], addpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
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
        fwd0fft<n/16,16*s,!eo>()(y, x, W);
        fwdcore<n,s>()(x, y, W);
    }
};

template <int s, bool eo> struct fwd0fft<16,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        fwd0end<16,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwd0fft<8,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT8omp::fwd0end<8,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwd0fft<4,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT4omp::fwd0end<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwd0fft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT4omp::fwd0end<2,s,eo>()(x, y);
    }
};

//-----------------------------------------------------------------------------

template <int n, int s, bool eo> struct fwdnfft
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        fwdnfft<n/16,16*s,!eo>()(y, x, W);
        fwdcore<n,s>()(x, y, W);
    }
};

template <int s, bool eo> struct fwdnfft<16,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        fwdnend<16,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwdnfft<8,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT8omp::fwdnend<8,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwdnfft<4,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT4omp::fwdnend<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwdnfft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT4omp::fwdnend<2,s,eo>()(x, y);
    }
};

///////////////////////////////////////////////////////////////////////////////
// Inverse butterfly operation
///////////////////////////////////////////////////////////////////////////////

template <int n, int s> struct invcore
{
    static const int n1 = n/16;
    static const int N  = n*s;
    static const int N0 = 0;
    static const int N1 = N/16;
    static const int N2 = N1*2;
    static const int N3 = N1*3;
    static const int N4 = N1*4;
    static const int N5 = N1*5;
    static const int N6 = N1*6;
    static const int N7 = N1*7;
    static const int N8 = N1*8;
    static const int N9 = N1*9;
    static const int Na = N1*10;
    static const int Nb = N1*11;
    static const int Nc = N1*12;
    static const int Nd = N1*13;
    static const int Ne = N1*14;
    static const int Nf = N1*15;

    void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int i = 0; i < N/32; i++) {
            const int p = i / (s/2);
            const int q = i % (s/2) * 2;
            const int sp   = s*p;
            const int s16p = 16*sp;

            const ymm w1p = duppz3(W[N-1*sp]);
            const ymm w2p = duppz3(W[N-2*sp]);
            const ymm w3p = duppz3(W[N-3*sp]);
            const ymm w4p = mulpz2(w2p, w2p);
            const ymm w5p = mulpz2(w2p, w3p);
            const ymm w6p = mulpz2(w3p, w3p);
            const ymm w7p = mulpz2(w3p, w4p);
            const ymm w8p = mulpz2(w4p, w4p);
            const ymm w9p = mulpz2(w4p, w5p);
            const ymm wap = mulpz2(w5p, w5p);
            const ymm wbp = mulpz2(w5p, w6p);
            const ymm wcp = mulpz2(w6p, w6p);
            const ymm wdp = mulpz2(w6p, w7p);
            const ymm wep = mulpz2(w7p, w7p);
            const ymm wfp = mulpz2(w7p, w8p);

            complex_vector xq_sp   = x + q + sp;
            complex_vector yq_s16p = y + q + s16p;

            const ymm y0 =             getpz2(yq_s16p+s*0x0);
            const ymm y1 = mulpz2(w1p, getpz2(yq_s16p+s*0x1));
            const ymm y2 = mulpz2(w2p, getpz2(yq_s16p+s*0x2));
            const ymm y3 = mulpz2(w3p, getpz2(yq_s16p+s*0x3));
            const ymm y4 = mulpz2(w4p, getpz2(yq_s16p+s*0x4));
            const ymm y5 = mulpz2(w5p, getpz2(yq_s16p+s*0x5));
            const ymm y6 = mulpz2(w6p, getpz2(yq_s16p+s*0x6));
            const ymm y7 = mulpz2(w7p, getpz2(yq_s16p+s*0x7));
            const ymm y8 = mulpz2(w8p, getpz2(yq_s16p+s*0x8));
            const ymm y9 = mulpz2(w9p, getpz2(yq_s16p+s*0x9));
            const ymm ya = mulpz2(wap, getpz2(yq_s16p+s*0xa));
            const ymm yb = mulpz2(wbp, getpz2(yq_s16p+s*0xb));
            const ymm yc = mulpz2(wcp, getpz2(yq_s16p+s*0xc));
            const ymm yd = mulpz2(wdp, getpz2(yq_s16p+s*0xd));
            const ymm ye = mulpz2(wep, getpz2(yq_s16p+s*0xe));
            const ymm yf = mulpz2(wfp, getpz2(yq_s16p+s*0xf));

            const ymm a08 = addpz2(y0, y8); const ymm s08 = subpz2(y0, y8);
            const ymm a4c = addpz2(y4, yc); const ymm s4c = subpz2(y4, yc);
            const ymm a2a = addpz2(y2, ya); const ymm s2a = subpz2(y2, ya);
            const ymm a6e = addpz2(y6, ye); const ymm s6e = subpz2(y6, ye);
            const ymm a19 = addpz2(y1, y9); const ymm s19 = subpz2(y1, y9);
            const ymm a5d = addpz2(y5, yd); const ymm s5d = subpz2(y5, yd);
            const ymm a3b = addpz2(y3, yb); const ymm s3b = subpz2(y3, yb);
            const ymm a7f = addpz2(y7, yf); const ymm s7f = subpz2(y7, yf);

            const ymm js4c = jxpz2(s4c);
            const ymm js6e = jxpz2(s6e);
            const ymm js5d = jxpz2(s5d);
            const ymm js7f = jxpz2(s7f);

            const ymm a08p1a4c = addpz2(a08, a4c); const ymm s08mjs4c = subpz2(s08, js4c);
            const ymm a08m1a4c = subpz2(a08, a4c); const ymm s08pjs4c = addpz2(s08, js4c);
            const ymm a2ap1a6e = addpz2(a2a, a6e); const ymm s2amjs6e = subpz2(s2a, js6e);
            const ymm a2am1a6e = subpz2(a2a, a6e); const ymm s2apjs6e = addpz2(s2a, js6e);
            const ymm a19p1a5d = addpz2(a19, a5d); const ymm s19mjs5d = subpz2(s19, js5d);
            const ymm a19m1a5d = subpz2(a19, a5d); const ymm s19pjs5d = addpz2(s19, js5d);
            const ymm a3bp1a7f = addpz2(a3b, a7f); const ymm s3bmjs7f = subpz2(s3b, js7f);
            const ymm a3bm1a7f = subpz2(a3b, a7f); const ymm s3bpjs7f = addpz2(s3b, js7f);

            const ymm w8_s2amjs6e = w8xpz2(s2amjs6e);
            const ymm  j_a2am1a6e =  jxpz2(a2am1a6e);
            const ymm v8_s2apjs6e = v8xpz2(s2apjs6e);

            const ymm a08p1a4c_p1_a2ap1a6e = addpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_pw_s2amjs6e = addpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_mj_a2am1a6e = subpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_mv_s2apjs6e = subpz2(s08pjs4c, v8_s2apjs6e);
            const ymm a08p1a4c_m1_a2ap1a6e = subpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_mw_s2amjs6e = subpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_pj_a2am1a6e = addpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_pv_s2apjs6e = addpz2(s08pjs4c, v8_s2apjs6e);

            const ymm w8_s3bmjs7f = w8xpz2(s3bmjs7f);
            const ymm  j_a3bm1a7f =  jxpz2(a3bm1a7f);
            const ymm v8_s3bpjs7f = v8xpz2(s3bpjs7f);

            const ymm a19p1a5d_p1_a3bp1a7f = addpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_pw_s3bmjs7f = addpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_mj_a3bm1a7f = subpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_mv_s3bpjs7f = subpz2(s19pjs5d, v8_s3bpjs7f);
            const ymm a19p1a5d_m1_a3bp1a7f = subpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_mw_s3bmjs7f = subpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_pj_a3bm1a7f = addpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_pv_s3bpjs7f = addpz2(s19pjs5d, v8_s3bpjs7f);

            const ymm h1_s19mjs5d_pw_s3bmjs7f = h1xpz2(s19mjs5d_pw_s3bmjs7f);
            const ymm w8_a19m1a5d_mj_a3bm1a7f = w8xpz2(a19m1a5d_mj_a3bm1a7f);
            const ymm h3_s19pjs5d_mv_s3bpjs7f = h3xpz2(s19pjs5d_mv_s3bpjs7f);
            const ymm  j_a19p1a5d_m1_a3bp1a7f =  jxpz2(a19p1a5d_m1_a3bp1a7f);
            const ymm hd_s19mjs5d_mw_s3bmjs7f = hdxpz2(s19mjs5d_mw_s3bmjs7f);
            const ymm v8_a19m1a5d_pj_a3bm1a7f = v8xpz2(a19m1a5d_pj_a3bm1a7f);
            const ymm hf_s19pjs5d_pv_s3bpjs7f = hfxpz2(s19pjs5d_pv_s3bpjs7f);

            setpz2(xq_sp+N0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq_sp+N1, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz2(xq_sp+N2, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq_sp+N3, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq_sp+N4, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq_sp+N5, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq_sp+N6, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq_sp+N7, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

            setpz2(xq_sp+N8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq_sp+N9, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz2(xq_sp+Na, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq_sp+Nb, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq_sp+Nc, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq_sp+Nd, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq_sp+Ne, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq_sp+Nf, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        }
    }
};

template <int N> struct invcore<N,1>
{
    static const int N0 = 0;
    static const int N1 = N/16;
    static const int N2 = N1*2;
    static const int N3 = N1*3;
    static const int N4 = N1*4;
    static const int N5 = N1*5;
    static const int N6 = N1*6;
    static const int N7 = N1*7;
    static const int N8 = N1*8;
    static const int N9 = N1*9;
    static const int Na = N1*10;
    static const int Nb = N1*11;
    static const int Nc = N1*12;
    static const int Nd = N1*13;
    static const int Ne = N1*14;
    static const int Nf = N1*15;

    void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
#ifdef _OPENMP
        #pragma omp for schedule(static) nowait
#endif
        for (int p = 0; p < N1; p += 2) {
            complex_vector x_p   = x + p;
            complex_vector y_16p = y + 16*p;

            const ymm w1p = cnjpz2(getpz2(W+p));
            const ymm w2p = mulpz2(w1p, w1p);
            const ymm w3p = mulpz2(w1p, w2p);
            const ymm w4p = mulpz2(w2p, w2p);
            const ymm w5p = mulpz2(w2p, w3p);
            const ymm w6p = mulpz2(w3p, w3p);
            const ymm w7p = mulpz2(w3p, w4p);
            const ymm w8p = mulpz2(w4p, w4p);
            const ymm w9p = mulpz2(w4p, w5p);
            const ymm wap = mulpz2(w5p, w5p);
            const ymm wbp = mulpz2(w5p, w6p);
            const ymm wcp = mulpz2(w6p, w6p);
            const ymm wdp = mulpz2(w6p, w7p);
            const ymm wep = mulpz2(w7p, w7p);
            const ymm wfp = mulpz2(w7p, w8p);
#if 0
            const ymm y0 =             getpz3<16>(y_16p+0x0);
            const ymm y1 = mulpz2(w1p, getpz3<16>(y_16p+0x1));
            const ymm y2 = mulpz2(w2p, getpz3<16>(y_16p+0x2));
            const ymm y3 = mulpz2(w3p, getpz3<16>(y_16p+0x3));
            const ymm y4 = mulpz2(w4p, getpz3<16>(y_16p+0x4));
            const ymm y5 = mulpz2(w5p, getpz3<16>(y_16p+0x5));
            const ymm y6 = mulpz2(w6p, getpz3<16>(y_16p+0x6));
            const ymm y7 = mulpz2(w7p, getpz3<16>(y_16p+0x7));
            const ymm y8 = mulpz2(w8p, getpz3<16>(y_16p+0x8));
            const ymm y9 = mulpz2(w9p, getpz3<16>(y_16p+0x9));
            const ymm ya = mulpz2(wap, getpz3<16>(y_16p+0xa));
            const ymm yb = mulpz2(wbp, getpz3<16>(y_16p+0xb));
            const ymm yc = mulpz2(wcp, getpz3<16>(y_16p+0xc));
            const ymm yd = mulpz2(wdp, getpz3<16>(y_16p+0xd));
            const ymm ye = mulpz2(wep, getpz3<16>(y_16p+0xe));
            const ymm yf = mulpz2(wfp, getpz3<16>(y_16p+0xf));
#else
            const ymm ab = getpz2(y_16p+0x00);
            const ymm cd = getpz2(y_16p+0x02);
            const ymm ef = getpz2(y_16p+0x04);
            const ymm gh = getpz2(y_16p+0x06);
            const ymm ij = getpz2(y_16p+0x08);
            const ymm kl = getpz2(y_16p+0x0a);
            const ymm mn = getpz2(y_16p+0x0c);
            const ymm op = getpz2(y_16p+0x0e);

            const ymm AB = getpz2(y_16p+0x10);
            const ymm CD = getpz2(y_16p+0x12);
            const ymm EF = getpz2(y_16p+0x14);
            const ymm GH = getpz2(y_16p+0x16);
            const ymm IJ = getpz2(y_16p+0x18);
            const ymm KL = getpz2(y_16p+0x1a);
            const ymm MN = getpz2(y_16p+0x1c);
            const ymm OP = getpz2(y_16p+0x1e);

            const ymm y0 =             catlo(ab, AB);
            const ymm y1 = mulpz2(w1p, cathi(ab, AB));
            const ymm y2 = mulpz2(w2p, catlo(cd, CD));
            const ymm y3 = mulpz2(w3p, cathi(cd, CD));
            const ymm y4 = mulpz2(w4p, catlo(ef, EF));
            const ymm y5 = mulpz2(w5p, cathi(ef, EF));
            const ymm y6 = mulpz2(w6p, catlo(gh, GH));
            const ymm y7 = mulpz2(w7p, cathi(gh, GH));

            const ymm y8 = mulpz2(w8p, catlo(ij, IJ));
            const ymm y9 = mulpz2(w9p, cathi(ij, IJ));
            const ymm ya = mulpz2(wap, catlo(kl, KL));
            const ymm yb = mulpz2(wbp, cathi(kl, KL));
            const ymm yc = mulpz2(wcp, catlo(mn, MN));
            const ymm yd = mulpz2(wdp, cathi(mn, MN));
            const ymm ye = mulpz2(wep, catlo(op, OP));
            const ymm yf = mulpz2(wfp, cathi(op, OP));
#endif
            const ymm a08 = addpz2(y0, y8); const ymm s08 = subpz2(y0, y8);
            const ymm a4c = addpz2(y4, yc); const ymm s4c = subpz2(y4, yc);
            const ymm a2a = addpz2(y2, ya); const ymm s2a = subpz2(y2, ya);
            const ymm a6e = addpz2(y6, ye); const ymm s6e = subpz2(y6, ye);
            const ymm a19 = addpz2(y1, y9); const ymm s19 = subpz2(y1, y9);
            const ymm a5d = addpz2(y5, yd); const ymm s5d = subpz2(y5, yd);
            const ymm a3b = addpz2(y3, yb); const ymm s3b = subpz2(y3, yb);
            const ymm a7f = addpz2(y7, yf); const ymm s7f = subpz2(y7, yf);

            const ymm js4c = jxpz2(s4c);
            const ymm js6e = jxpz2(s6e);
            const ymm js5d = jxpz2(s5d);
            const ymm js7f = jxpz2(s7f);

            const ymm a08p1a4c = addpz2(a08, a4c); const ymm s08mjs4c = subpz2(s08, js4c);
            const ymm a08m1a4c = subpz2(a08, a4c); const ymm s08pjs4c = addpz2(s08, js4c);
            const ymm a2ap1a6e = addpz2(a2a, a6e); const ymm s2amjs6e = subpz2(s2a, js6e);
            const ymm a2am1a6e = subpz2(a2a, a6e); const ymm s2apjs6e = addpz2(s2a, js6e);
            const ymm a19p1a5d = addpz2(a19, a5d); const ymm s19mjs5d = subpz2(s19, js5d);
            const ymm a19m1a5d = subpz2(a19, a5d); const ymm s19pjs5d = addpz2(s19, js5d);
            const ymm a3bp1a7f = addpz2(a3b, a7f); const ymm s3bmjs7f = subpz2(s3b, js7f);
            const ymm a3bm1a7f = subpz2(a3b, a7f); const ymm s3bpjs7f = addpz2(s3b, js7f);

            const ymm w8_s2amjs6e = w8xpz2(s2amjs6e);
            const ymm  j_a2am1a6e =  jxpz2(a2am1a6e);
            const ymm v8_s2apjs6e = v8xpz2(s2apjs6e);

            const ymm a08p1a4c_p1_a2ap1a6e = addpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_pw_s2amjs6e = addpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_mj_a2am1a6e = subpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_mv_s2apjs6e = subpz2(s08pjs4c, v8_s2apjs6e);
            const ymm a08p1a4c_m1_a2ap1a6e = subpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_mw_s2amjs6e = subpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_pj_a2am1a6e = addpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_pv_s2apjs6e = addpz2(s08pjs4c, v8_s2apjs6e);

            const ymm w8_s3bmjs7f = w8xpz2(s3bmjs7f);
            const ymm  j_a3bm1a7f =  jxpz2(a3bm1a7f);
            const ymm v8_s3bpjs7f = v8xpz2(s3bpjs7f);

            const ymm a19p1a5d_p1_a3bp1a7f = addpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_pw_s3bmjs7f = addpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_mj_a3bm1a7f = subpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_mv_s3bpjs7f = subpz2(s19pjs5d, v8_s3bpjs7f);
            const ymm a19p1a5d_m1_a3bp1a7f = subpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_mw_s3bmjs7f = subpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_pj_a3bm1a7f = addpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_pv_s3bpjs7f = addpz2(s19pjs5d, v8_s3bpjs7f);

            const ymm h1_s19mjs5d_pw_s3bmjs7f = h1xpz2(s19mjs5d_pw_s3bmjs7f);
            const ymm w8_a19m1a5d_mj_a3bm1a7f = w8xpz2(a19m1a5d_mj_a3bm1a7f);
            const ymm h3_s19pjs5d_mv_s3bpjs7f = h3xpz2(s19pjs5d_mv_s3bpjs7f);
            const ymm  j_a19p1a5d_m1_a3bp1a7f =  jxpz2(a19p1a5d_m1_a3bp1a7f);
            const ymm hd_s19mjs5d_mw_s3bmjs7f = hdxpz2(s19mjs5d_mw_s3bmjs7f);
            const ymm v8_a19m1a5d_pj_a3bm1a7f = v8xpz2(a19m1a5d_pj_a3bm1a7f);
            const ymm hf_s19pjs5d_pv_s3bpjs7f = hfxpz2(s19pjs5d_pv_s3bpjs7f);

            setpz2(x_p+N0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(x_p+N1, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz2(x_p+N2, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(x_p+N3, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(x_p+N4, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(x_p+N5, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(x_p+N6, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(x_p+N7, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

            setpz2(x_p+N8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(x_p+N9, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz2(x_p+Na, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(x_p+Nb, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(x_p+Nc, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(x_p+Nd, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(x_p+Ne, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(x_p+Nf, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct inv0end;

//-----------------------------------------------------------------------------

template <int s> struct inv0end<16,s,1>
{
    void operator()(complex_vector x, complex_vector y) const noexcept
    {
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;

            const ymm y0 = getpz2(yq+s*0x0);
            const ymm y1 = getpz2(yq+s*0x1);
            const ymm y2 = getpz2(yq+s*0x2);
            const ymm y3 = getpz2(yq+s*0x3);
            const ymm y4 = getpz2(yq+s*0x4);
            const ymm y5 = getpz2(yq+s*0x5);
            const ymm y6 = getpz2(yq+s*0x6);
            const ymm y7 = getpz2(yq+s*0x7);
            const ymm y8 = getpz2(yq+s*0x8);
            const ymm y9 = getpz2(yq+s*0x9);
            const ymm ya = getpz2(yq+s*0xa);
            const ymm yb = getpz2(yq+s*0xb);
            const ymm yc = getpz2(yq+s*0xc);
            const ymm yd = getpz2(yq+s*0xd);
            const ymm ye = getpz2(yq+s*0xe);
            const ymm yf = getpz2(yq+s*0xf);

            const ymm a08 = addpz2(y0, y8); const ymm s08 = subpz2(y0, y8);
            const ymm a4c = addpz2(y4, yc); const ymm s4c = subpz2(y4, yc);
            const ymm a2a = addpz2(y2, ya); const ymm s2a = subpz2(y2, ya);
            const ymm a6e = addpz2(y6, ye); const ymm s6e = subpz2(y6, ye);
            const ymm a19 = addpz2(y1, y9); const ymm s19 = subpz2(y1, y9);
            const ymm a5d = addpz2(y5, yd); const ymm s5d = subpz2(y5, yd);
            const ymm a3b = addpz2(y3, yb); const ymm s3b = subpz2(y3, yb);
            const ymm a7f = addpz2(y7, yf); const ymm s7f = subpz2(y7, yf);

            const ymm js4c = jxpz2(s4c);
            const ymm js6e = jxpz2(s6e);
            const ymm js5d = jxpz2(s5d);
            const ymm js7f = jxpz2(s7f);

            const ymm a08p1a4c = addpz2(a08, a4c); const ymm s08mjs4c = subpz2(s08, js4c);
            const ymm a08m1a4c = subpz2(a08, a4c); const ymm s08pjs4c = addpz2(s08, js4c);
            const ymm a2ap1a6e = addpz2(a2a, a6e); const ymm s2amjs6e = subpz2(s2a, js6e);
            const ymm a2am1a6e = subpz2(a2a, a6e); const ymm s2apjs6e = addpz2(s2a, js6e);
            const ymm a19p1a5d = addpz2(a19, a5d); const ymm s19mjs5d = subpz2(s19, js5d);
            const ymm a19m1a5d = subpz2(a19, a5d); const ymm s19pjs5d = addpz2(s19, js5d);
            const ymm a3bp1a7f = addpz2(a3b, a7f); const ymm s3bmjs7f = subpz2(s3b, js7f);
            const ymm a3bm1a7f = subpz2(a3b, a7f); const ymm s3bpjs7f = addpz2(s3b, js7f);

            const ymm w8_s2amjs6e = w8xpz2(s2amjs6e);
            const ymm  j_a2am1a6e =  jxpz2(a2am1a6e);
            const ymm v8_s2apjs6e = v8xpz2(s2apjs6e);

            const ymm a08p1a4c_p1_a2ap1a6e = addpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_pw_s2amjs6e = addpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_mj_a2am1a6e = subpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_mv_s2apjs6e = subpz2(s08pjs4c, v8_s2apjs6e);
            const ymm a08p1a4c_m1_a2ap1a6e = subpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_mw_s2amjs6e = subpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_pj_a2am1a6e = addpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_pv_s2apjs6e = addpz2(s08pjs4c, v8_s2apjs6e);

            const ymm w8_s3bmjs7f = w8xpz2(s3bmjs7f);
            const ymm  j_a3bm1a7f =  jxpz2(a3bm1a7f);
            const ymm v8_s3bpjs7f = v8xpz2(s3bpjs7f);

            const ymm a19p1a5d_p1_a3bp1a7f = addpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_pw_s3bmjs7f = addpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_mj_a3bm1a7f = subpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_mv_s3bpjs7f = subpz2(s19pjs5d, v8_s3bpjs7f);
            const ymm a19p1a5d_m1_a3bp1a7f = subpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_mw_s3bmjs7f = subpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_pj_a3bm1a7f = addpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_pv_s3bpjs7f = addpz2(s19pjs5d, v8_s3bpjs7f);

            const ymm h1_s19mjs5d_pw_s3bmjs7f = h1xpz2(s19mjs5d_pw_s3bmjs7f);
            const ymm w8_a19m1a5d_mj_a3bm1a7f = w8xpz2(a19m1a5d_mj_a3bm1a7f);
            const ymm h3_s19pjs5d_mv_s3bpjs7f = h3xpz2(s19pjs5d_mv_s3bpjs7f);
            const ymm  j_a19p1a5d_m1_a3bp1a7f =  jxpz2(a19p1a5d_m1_a3bp1a7f);
            const ymm hd_s19mjs5d_mw_s3bmjs7f = hdxpz2(s19mjs5d_mw_s3bmjs7f);
            const ymm v8_a19m1a5d_pj_a3bm1a7f = v8xpz2(a19m1a5d_pj_a3bm1a7f);
            const ymm hf_s19pjs5d_pv_s3bpjs7f = hfxpz2(s19pjs5d_pv_s3bpjs7f);

            setpz2(xq+s*0x0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq+s*0x1, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz2(xq+s*0x2, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq+s*0x3, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq+s*0x4, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq+s*0x5, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq+s*0x6, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq+s*0x7, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

            setpz2(xq+s*0x8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq+s*0x9, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz2(xq+s*0xa, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq+s*0xb, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq+s*0xc, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq+s*0xd, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq+s*0xe, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq+s*0xf, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        }
    }
};

template <> struct inv0end<16,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
            const xmm y0 = getpz(y[0x0]);
            const xmm y1 = getpz(y[0x1]);
            const xmm y2 = getpz(y[0x2]);
            const xmm y3 = getpz(y[0x3]);
            const xmm y4 = getpz(y[0x4]);
            const xmm y5 = getpz(y[0x5]);
            const xmm y6 = getpz(y[0x6]);
            const xmm y7 = getpz(y[0x7]);
            const xmm y8 = getpz(y[0x8]);
            const xmm y9 = getpz(y[0x9]);
            const xmm ya = getpz(y[0xa]);
            const xmm yb = getpz(y[0xb]);
            const xmm yc = getpz(y[0xc]);
            const xmm yd = getpz(y[0xd]);
            const xmm ye = getpz(y[0xe]);
            const xmm yf = getpz(y[0xf]);

            const xmm a08 = addpz(y0, y8); const xmm s08 = subpz(y0, y8);
            const xmm a4c = addpz(y4, yc); const xmm s4c = subpz(y4, yc);
            const xmm a2a = addpz(y2, ya); const xmm s2a = subpz(y2, ya);
            const xmm a6e = addpz(y6, ye); const xmm s6e = subpz(y6, ye);
            const xmm a19 = addpz(y1, y9); const xmm s19 = subpz(y1, y9);
            const xmm a5d = addpz(y5, yd); const xmm s5d = subpz(y5, yd);
            const xmm a3b = addpz(y3, yb); const xmm s3b = subpz(y3, yb);
            const xmm a7f = addpz(y7, yf); const xmm s7f = subpz(y7, yf);

            const xmm js4c = jxpz(s4c);
            const xmm js6e = jxpz(s6e);
            const xmm js5d = jxpz(s5d);
            const xmm js7f = jxpz(s7f);

            const xmm a08p1a4c = addpz(a08, a4c); const xmm s08mjs4c = subpz(s08, js4c);
            const xmm a08m1a4c = subpz(a08, a4c); const xmm s08pjs4c = addpz(s08, js4c);
            const xmm a2ap1a6e = addpz(a2a, a6e); const xmm s2amjs6e = subpz(s2a, js6e);
            const xmm a2am1a6e = subpz(a2a, a6e); const xmm s2apjs6e = addpz(s2a, js6e);
            const xmm a19p1a5d = addpz(a19, a5d); const xmm s19mjs5d = subpz(s19, js5d);
            const xmm a19m1a5d = subpz(a19, a5d); const xmm s19pjs5d = addpz(s19, js5d);
            const xmm a3bp1a7f = addpz(a3b, a7f); const xmm s3bmjs7f = subpz(s3b, js7f);
            const xmm a3bm1a7f = subpz(a3b, a7f); const xmm s3bpjs7f = addpz(s3b, js7f);

            const xmm w8_s2amjs6e = w8xpz(s2amjs6e);
            const xmm  j_a2am1a6e =  jxpz(a2am1a6e);
            const xmm v8_s2apjs6e = v8xpz(s2apjs6e);

            const xmm a08p1a4c_p1_a2ap1a6e = addpz(a08p1a4c,    a2ap1a6e);
            const xmm s08mjs4c_pw_s2amjs6e = addpz(s08mjs4c, w8_s2amjs6e);
            const xmm a08m1a4c_mj_a2am1a6e = subpz(a08m1a4c,  j_a2am1a6e);
            const xmm s08pjs4c_mv_s2apjs6e = subpz(s08pjs4c, v8_s2apjs6e);
            const xmm a08p1a4c_m1_a2ap1a6e = subpz(a08p1a4c,    a2ap1a6e);
            const xmm s08mjs4c_mw_s2amjs6e = subpz(s08mjs4c, w8_s2amjs6e);
            const xmm a08m1a4c_pj_a2am1a6e = addpz(a08m1a4c,  j_a2am1a6e);
            const xmm s08pjs4c_pv_s2apjs6e = addpz(s08pjs4c, v8_s2apjs6e);

            const xmm w8_s3bmjs7f = w8xpz(s3bmjs7f);
            const xmm  j_a3bm1a7f =  jxpz(a3bm1a7f);
            const xmm v8_s3bpjs7f = v8xpz(s3bpjs7f);

            const xmm a19p1a5d_p1_a3bp1a7f = addpz(a19p1a5d,    a3bp1a7f);
            const xmm s19mjs5d_pw_s3bmjs7f = addpz(s19mjs5d, w8_s3bmjs7f);
            const xmm a19m1a5d_mj_a3bm1a7f = subpz(a19m1a5d,  j_a3bm1a7f);
            const xmm s19pjs5d_mv_s3bpjs7f = subpz(s19pjs5d, v8_s3bpjs7f);
            const xmm a19p1a5d_m1_a3bp1a7f = subpz(a19p1a5d,    a3bp1a7f);
            const xmm s19mjs5d_mw_s3bmjs7f = subpz(s19mjs5d, w8_s3bmjs7f);
            const xmm a19m1a5d_pj_a3bm1a7f = addpz(a19m1a5d,  j_a3bm1a7f);
            const xmm s19pjs5d_pv_s3bpjs7f = addpz(s19pjs5d, v8_s3bpjs7f);

            const xmm h1_s19mjs5d_pw_s3bmjs7f = h1xpz(s19mjs5d_pw_s3bmjs7f);
            const xmm w8_a19m1a5d_mj_a3bm1a7f = w8xpz(a19m1a5d_mj_a3bm1a7f);
            const xmm h3_s19pjs5d_mv_s3bpjs7f = h3xpz(s19pjs5d_mv_s3bpjs7f);
            const xmm  j_a19p1a5d_m1_a3bp1a7f =  jxpz(a19p1a5d_m1_a3bp1a7f);
            const xmm hd_s19mjs5d_mw_s3bmjs7f = hdxpz(s19mjs5d_mw_s3bmjs7f);
            const xmm v8_a19m1a5d_pj_a3bm1a7f = v8xpz(a19m1a5d_pj_a3bm1a7f);
            const xmm hf_s19pjs5d_pv_s3bpjs7f = hfxpz(s19pjs5d_pv_s3bpjs7f);

            setpz(x[0x0], addpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz(x[0x1], addpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz(x[0x2], addpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz(x[0x3], addpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz(x[0x4], addpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz(x[0x5], subpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz(x[0x6], subpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz(x[0x7], subpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

            setpz(x[0x8], subpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz(x[0x9], subpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz(x[0xa], subpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz(x[0xb], subpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz(x[0xc], subpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz(x[0xd], addpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz(x[0xe], addpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz(x[0xf], addpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        }
    }
};

//-----------------------------------------------------------------------------

template <int s> struct inv0end<16,s,0>
{
    void operator()(complex_vector x, complex_vector) const noexcept
    {
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;

            const ymm x0 = getpz2(xq+s*0x0);
            const ymm x1 = getpz2(xq+s*0x1);
            const ymm x2 = getpz2(xq+s*0x2);
            const ymm x3 = getpz2(xq+s*0x3);
            const ymm x4 = getpz2(xq+s*0x4);
            const ymm x5 = getpz2(xq+s*0x5);
            const ymm x6 = getpz2(xq+s*0x6);
            const ymm x7 = getpz2(xq+s*0x7);
            const ymm x8 = getpz2(xq+s*0x8);
            const ymm x9 = getpz2(xq+s*0x9);
            const ymm xa = getpz2(xq+s*0xa);
            const ymm xb = getpz2(xq+s*0xb);
            const ymm xc = getpz2(xq+s*0xc);
            const ymm xd = getpz2(xq+s*0xd);
            const ymm xe = getpz2(xq+s*0xe);
            const ymm xf = getpz2(xq+s*0xf);

            const ymm a08 = addpz2(x0, x8); const ymm s08 = subpz2(x0, x8);
            const ymm a4c = addpz2(x4, xc); const ymm s4c = subpz2(x4, xc);
            const ymm a2a = addpz2(x2, xa); const ymm s2a = subpz2(x2, xa);
            const ymm a6e = addpz2(x6, xe); const ymm s6e = subpz2(x6, xe);
            const ymm a19 = addpz2(x1, x9); const ymm s19 = subpz2(x1, x9);
            const ymm a5d = addpz2(x5, xd); const ymm s5d = subpz2(x5, xd);
            const ymm a3b = addpz2(x3, xb); const ymm s3b = subpz2(x3, xb);
            const ymm a7f = addpz2(x7, xf); const ymm s7f = subpz2(x7, xf);

            const ymm js4c = jxpz2(s4c);
            const ymm js6e = jxpz2(s6e);
            const ymm js5d = jxpz2(s5d);
            const ymm js7f = jxpz2(s7f);

            const ymm a08p1a4c = addpz2(a08, a4c); const ymm s08mjs4c = subpz2(s08, js4c);
            const ymm a08m1a4c = subpz2(a08, a4c); const ymm s08pjs4c = addpz2(s08, js4c);
            const ymm a2ap1a6e = addpz2(a2a, a6e); const ymm s2amjs6e = subpz2(s2a, js6e);
            const ymm a2am1a6e = subpz2(a2a, a6e); const ymm s2apjs6e = addpz2(s2a, js6e);
            const ymm a19p1a5d = addpz2(a19, a5d); const ymm s19mjs5d = subpz2(s19, js5d);
            const ymm a19m1a5d = subpz2(a19, a5d); const ymm s19pjs5d = addpz2(s19, js5d);
            const ymm a3bp1a7f = addpz2(a3b, a7f); const ymm s3bmjs7f = subpz2(s3b, js7f);
            const ymm a3bm1a7f = subpz2(a3b, a7f); const ymm s3bpjs7f = addpz2(s3b, js7f);

            const ymm w8_s2amjs6e = w8xpz2(s2amjs6e);
            const ymm  j_a2am1a6e =  jxpz2(a2am1a6e);
            const ymm v8_s2apjs6e = v8xpz2(s2apjs6e);

            const ymm a08p1a4c_p1_a2ap1a6e = addpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_pw_s2amjs6e = addpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_mj_a2am1a6e = subpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_mv_s2apjs6e = subpz2(s08pjs4c, v8_s2apjs6e);
            const ymm a08p1a4c_m1_a2ap1a6e = subpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_mw_s2amjs6e = subpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_pj_a2am1a6e = addpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_pv_s2apjs6e = addpz2(s08pjs4c, v8_s2apjs6e);

            const ymm w8_s3bmjs7f = w8xpz2(s3bmjs7f);
            const ymm  j_a3bm1a7f =  jxpz2(a3bm1a7f);
            const ymm v8_s3bpjs7f = v8xpz2(s3bpjs7f);

            const ymm a19p1a5d_p1_a3bp1a7f = addpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_pw_s3bmjs7f = addpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_mj_a3bm1a7f = subpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_mv_s3bpjs7f = subpz2(s19pjs5d, v8_s3bpjs7f);
            const ymm a19p1a5d_m1_a3bp1a7f = subpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_mw_s3bmjs7f = subpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_pj_a3bm1a7f = addpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_pv_s3bpjs7f = addpz2(s19pjs5d, v8_s3bpjs7f);

            const ymm h1_s19mjs5d_pw_s3bmjs7f = h1xpz2(s19mjs5d_pw_s3bmjs7f);
            const ymm w8_a19m1a5d_mj_a3bm1a7f = w8xpz2(a19m1a5d_mj_a3bm1a7f);
            const ymm h3_s19pjs5d_mv_s3bpjs7f = h3xpz2(s19pjs5d_mv_s3bpjs7f);
            const ymm  j_a19p1a5d_m1_a3bp1a7f =  jxpz2(a19p1a5d_m1_a3bp1a7f);
            const ymm hd_s19mjs5d_mw_s3bmjs7f = hdxpz2(s19mjs5d_mw_s3bmjs7f);
            const ymm v8_a19m1a5d_pj_a3bm1a7f = v8xpz2(a19m1a5d_pj_a3bm1a7f);
            const ymm hf_s19pjs5d_pv_s3bpjs7f = hfxpz2(s19pjs5d_pv_s3bpjs7f);

            setpz2(xq+s*0x0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq+s*0x1, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz2(xq+s*0x2, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq+s*0x3, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq+s*0x4, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq+s*0x5, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq+s*0x6, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq+s*0x7, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

            setpz2(xq+s*0x8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq+s*0x9, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz2(xq+s*0xa, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq+s*0xb, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq+s*0xc, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq+s*0xd, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq+s*0xe, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq+s*0xf, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        }
    }
};

template <> struct inv0end<16,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const noexcept
    {
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
            const xmm x0 = getpz(x[0x0]);
            const xmm x1 = getpz(x[0x1]);
            const xmm x2 = getpz(x[0x2]);
            const xmm x3 = getpz(x[0x3]);
            const xmm x4 = getpz(x[0x4]);
            const xmm x5 = getpz(x[0x5]);
            const xmm x6 = getpz(x[0x6]);
            const xmm x7 = getpz(x[0x7]);
            const xmm x8 = getpz(x[0x8]);
            const xmm x9 = getpz(x[0x9]);
            const xmm xa = getpz(x[0xa]);
            const xmm xb = getpz(x[0xb]);
            const xmm xc = getpz(x[0xc]);
            const xmm xd = getpz(x[0xd]);
            const xmm xe = getpz(x[0xe]);
            const xmm xf = getpz(x[0xf]);

            const xmm a08 = addpz(x0, x8); const xmm s08 = subpz(x0, x8);
            const xmm a4c = addpz(x4, xc); const xmm s4c = subpz(x4, xc);
            const xmm a2a = addpz(x2, xa); const xmm s2a = subpz(x2, xa);
            const xmm a6e = addpz(x6, xe); const xmm s6e = subpz(x6, xe);
            const xmm a19 = addpz(x1, x9); const xmm s19 = subpz(x1, x9);
            const xmm a5d = addpz(x5, xd); const xmm s5d = subpz(x5, xd);
            const xmm a3b = addpz(x3, xb); const xmm s3b = subpz(x3, xb);
            const xmm a7f = addpz(x7, xf); const xmm s7f = subpz(x7, xf);

            const xmm js4c = jxpz(s4c);
            const xmm js6e = jxpz(s6e);
            const xmm js5d = jxpz(s5d);
            const xmm js7f = jxpz(s7f);

            const xmm a08p1a4c = addpz(a08, a4c); const xmm s08mjs4c = subpz(s08, js4c);
            const xmm a08m1a4c = subpz(a08, a4c); const xmm s08pjs4c = addpz(s08, js4c);
            const xmm a2ap1a6e = addpz(a2a, a6e); const xmm s2amjs6e = subpz(s2a, js6e);
            const xmm a2am1a6e = subpz(a2a, a6e); const xmm s2apjs6e = addpz(s2a, js6e);
            const xmm a19p1a5d = addpz(a19, a5d); const xmm s19mjs5d = subpz(s19, js5d);
            const xmm a19m1a5d = subpz(a19, a5d); const xmm s19pjs5d = addpz(s19, js5d);
            const xmm a3bp1a7f = addpz(a3b, a7f); const xmm s3bmjs7f = subpz(s3b, js7f);
            const xmm a3bm1a7f = subpz(a3b, a7f); const xmm s3bpjs7f = addpz(s3b, js7f);

            const xmm w8_s2amjs6e = w8xpz(s2amjs6e);
            const xmm  j_a2am1a6e =  jxpz(a2am1a6e);
            const xmm v8_s2apjs6e = v8xpz(s2apjs6e);

            const xmm a08p1a4c_p1_a2ap1a6e = addpz(a08p1a4c,    a2ap1a6e);
            const xmm s08mjs4c_pw_s2amjs6e = addpz(s08mjs4c, w8_s2amjs6e);
            const xmm a08m1a4c_mj_a2am1a6e = subpz(a08m1a4c,  j_a2am1a6e);
            const xmm s08pjs4c_mv_s2apjs6e = subpz(s08pjs4c, v8_s2apjs6e);
            const xmm a08p1a4c_m1_a2ap1a6e = subpz(a08p1a4c,    a2ap1a6e);
            const xmm s08mjs4c_mw_s2amjs6e = subpz(s08mjs4c, w8_s2amjs6e);
            const xmm a08m1a4c_pj_a2am1a6e = addpz(a08m1a4c,  j_a2am1a6e);
            const xmm s08pjs4c_pv_s2apjs6e = addpz(s08pjs4c, v8_s2apjs6e);

            const xmm w8_s3bmjs7f = w8xpz(s3bmjs7f);
            const xmm  j_a3bm1a7f =  jxpz(a3bm1a7f);
            const xmm v8_s3bpjs7f = v8xpz(s3bpjs7f);

            const xmm a19p1a5d_p1_a3bp1a7f = addpz(a19p1a5d,    a3bp1a7f);
            const xmm s19mjs5d_pw_s3bmjs7f = addpz(s19mjs5d, w8_s3bmjs7f);
            const xmm a19m1a5d_mj_a3bm1a7f = subpz(a19m1a5d,  j_a3bm1a7f);
            const xmm s19pjs5d_mv_s3bpjs7f = subpz(s19pjs5d, v8_s3bpjs7f);
            const xmm a19p1a5d_m1_a3bp1a7f = subpz(a19p1a5d,    a3bp1a7f);
            const xmm s19mjs5d_mw_s3bmjs7f = subpz(s19mjs5d, w8_s3bmjs7f);
            const xmm a19m1a5d_pj_a3bm1a7f = addpz(a19m1a5d,  j_a3bm1a7f);
            const xmm s19pjs5d_pv_s3bpjs7f = addpz(s19pjs5d, v8_s3bpjs7f);

            const xmm h1_s19mjs5d_pw_s3bmjs7f = h1xpz(s19mjs5d_pw_s3bmjs7f);
            const xmm w8_a19m1a5d_mj_a3bm1a7f = w8xpz(a19m1a5d_mj_a3bm1a7f);
            const xmm h3_s19pjs5d_mv_s3bpjs7f = h3xpz(s19pjs5d_mv_s3bpjs7f);
            const xmm  j_a19p1a5d_m1_a3bp1a7f =  jxpz(a19p1a5d_m1_a3bp1a7f);
            const xmm hd_s19mjs5d_mw_s3bmjs7f = hdxpz(s19mjs5d_mw_s3bmjs7f);
            const xmm v8_a19m1a5d_pj_a3bm1a7f = v8xpz(a19m1a5d_pj_a3bm1a7f);
            const xmm hf_s19pjs5d_pv_s3bpjs7f = hfxpz(s19pjs5d_pv_s3bpjs7f);

            setpz(x[0x0], addpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz(x[0x1], addpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz(x[0x2], addpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz(x[0x3], addpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz(x[0x4], addpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz(x[0x5], subpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz(x[0x6], subpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz(x[0x7], subpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

            setpz(x[0x8], subpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz(x[0x9], subpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz(x[0xa], subpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz(x[0xb], subpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz(x[0xc], subpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz(x[0xd], addpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz(x[0xe], addpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz(x[0xf], addpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct invnend;

//-----------------------------------------------------------------------------

template <int s> struct invnend<16,s,1>
{
    static const int N = 16*s;

    void operator()(complex_vector x, complex_vector y) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;

            const ymm y0 = mulpd2(rN, getpz2(yq+s*0x0));
            const ymm y1 = mulpd2(rN, getpz2(yq+s*0x1));
            const ymm y2 = mulpd2(rN, getpz2(yq+s*0x2));
            const ymm y3 = mulpd2(rN, getpz2(yq+s*0x3));
            const ymm y4 = mulpd2(rN, getpz2(yq+s*0x4));
            const ymm y5 = mulpd2(rN, getpz2(yq+s*0x5));
            const ymm y6 = mulpd2(rN, getpz2(yq+s*0x6));
            const ymm y7 = mulpd2(rN, getpz2(yq+s*0x7));
            const ymm y8 = mulpd2(rN, getpz2(yq+s*0x8));
            const ymm y9 = mulpd2(rN, getpz2(yq+s*0x9));
            const ymm ya = mulpd2(rN, getpz2(yq+s*0xa));
            const ymm yb = mulpd2(rN, getpz2(yq+s*0xb));
            const ymm yc = mulpd2(rN, getpz2(yq+s*0xc));
            const ymm yd = mulpd2(rN, getpz2(yq+s*0xd));
            const ymm ye = mulpd2(rN, getpz2(yq+s*0xe));
            const ymm yf = mulpd2(rN, getpz2(yq+s*0xf));

            const ymm a08 = addpz2(y0, y8); const ymm s08 = subpz2(y0, y8);
            const ymm a4c = addpz2(y4, yc); const ymm s4c = subpz2(y4, yc);
            const ymm a2a = addpz2(y2, ya); const ymm s2a = subpz2(y2, ya);
            const ymm a6e = addpz2(y6, ye); const ymm s6e = subpz2(y6, ye);
            const ymm a19 = addpz2(y1, y9); const ymm s19 = subpz2(y1, y9);
            const ymm a5d = addpz2(y5, yd); const ymm s5d = subpz2(y5, yd);
            const ymm a3b = addpz2(y3, yb); const ymm s3b = subpz2(y3, yb);
            const ymm a7f = addpz2(y7, yf); const ymm s7f = subpz2(y7, yf);

            const ymm js4c = jxpz2(s4c);
            const ymm js6e = jxpz2(s6e);
            const ymm js5d = jxpz2(s5d);
            const ymm js7f = jxpz2(s7f);

            const ymm a08p1a4c = addpz2(a08, a4c); const ymm s08mjs4c = subpz2(s08, js4c);
            const ymm a08m1a4c = subpz2(a08, a4c); const ymm s08pjs4c = addpz2(s08, js4c);
            const ymm a2ap1a6e = addpz2(a2a, a6e); const ymm s2amjs6e = subpz2(s2a, js6e);
            const ymm a2am1a6e = subpz2(a2a, a6e); const ymm s2apjs6e = addpz2(s2a, js6e);
            const ymm a19p1a5d = addpz2(a19, a5d); const ymm s19mjs5d = subpz2(s19, js5d);
            const ymm a19m1a5d = subpz2(a19, a5d); const ymm s19pjs5d = addpz2(s19, js5d);
            const ymm a3bp1a7f = addpz2(a3b, a7f); const ymm s3bmjs7f = subpz2(s3b, js7f);
            const ymm a3bm1a7f = subpz2(a3b, a7f); const ymm s3bpjs7f = addpz2(s3b, js7f);

            const ymm w8_s2amjs6e = w8xpz2(s2amjs6e);
            const ymm  j_a2am1a6e =  jxpz2(a2am1a6e);
            const ymm v8_s2apjs6e = v8xpz2(s2apjs6e);

            const ymm a08p1a4c_p1_a2ap1a6e = addpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_pw_s2amjs6e = addpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_mj_a2am1a6e = subpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_mv_s2apjs6e = subpz2(s08pjs4c, v8_s2apjs6e);
            const ymm a08p1a4c_m1_a2ap1a6e = subpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_mw_s2amjs6e = subpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_pj_a2am1a6e = addpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_pv_s2apjs6e = addpz2(s08pjs4c, v8_s2apjs6e);

            const ymm w8_s3bmjs7f = w8xpz2(s3bmjs7f);
            const ymm  j_a3bm1a7f =  jxpz2(a3bm1a7f);
            const ymm v8_s3bpjs7f = v8xpz2(s3bpjs7f);

            const ymm a19p1a5d_p1_a3bp1a7f = addpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_pw_s3bmjs7f = addpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_mj_a3bm1a7f = subpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_mv_s3bpjs7f = subpz2(s19pjs5d, v8_s3bpjs7f);
            const ymm a19p1a5d_m1_a3bp1a7f = subpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_mw_s3bmjs7f = subpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_pj_a3bm1a7f = addpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_pv_s3bpjs7f = addpz2(s19pjs5d, v8_s3bpjs7f);

            const ymm h1_s19mjs5d_pw_s3bmjs7f = h1xpz2(s19mjs5d_pw_s3bmjs7f);
            const ymm w8_a19m1a5d_mj_a3bm1a7f = w8xpz2(a19m1a5d_mj_a3bm1a7f);
            const ymm h3_s19pjs5d_mv_s3bpjs7f = h3xpz2(s19pjs5d_mv_s3bpjs7f);
            const ymm  j_a19p1a5d_m1_a3bp1a7f =  jxpz2(a19p1a5d_m1_a3bp1a7f);
            const ymm hd_s19mjs5d_mw_s3bmjs7f = hdxpz2(s19mjs5d_mw_s3bmjs7f);
            const ymm v8_a19m1a5d_pj_a3bm1a7f = v8xpz2(a19m1a5d_pj_a3bm1a7f);
            const ymm hf_s19pjs5d_pv_s3bpjs7f = hfxpz2(s19pjs5d_pv_s3bpjs7f);

            setpz2(xq+s*0x0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq+s*0x1, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz2(xq+s*0x2, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq+s*0x3, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq+s*0x4, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq+s*0x5, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq+s*0x6, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq+s*0x7, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

            setpz2(xq+s*0x8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq+s*0x9, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz2(xq+s*0xa, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq+s*0xb, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq+s*0xc, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq+s*0xd, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq+s*0xe, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq+s*0xf, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        }
    }
};

template <> struct invnend<16,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        static const xmm rN = { 1.0/16, 1.0/16 };
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
            const xmm y0 = mulpd(rN, getpz(y[0x0]));
            const xmm y1 = mulpd(rN, getpz(y[0x1]));
            const xmm y2 = mulpd(rN, getpz(y[0x2]));
            const xmm y3 = mulpd(rN, getpz(y[0x3]));
            const xmm y4 = mulpd(rN, getpz(y[0x4]));
            const xmm y5 = mulpd(rN, getpz(y[0x5]));
            const xmm y6 = mulpd(rN, getpz(y[0x6]));
            const xmm y7 = mulpd(rN, getpz(y[0x7]));
            const xmm y8 = mulpd(rN, getpz(y[0x8]));
            const xmm y9 = mulpd(rN, getpz(y[0x9]));
            const xmm ya = mulpd(rN, getpz(y[0xa]));
            const xmm yb = mulpd(rN, getpz(y[0xb]));
            const xmm yc = mulpd(rN, getpz(y[0xc]));
            const xmm yd = mulpd(rN, getpz(y[0xd]));
            const xmm ye = mulpd(rN, getpz(y[0xe]));
            const xmm yf = mulpd(rN, getpz(y[0xf]));

            const xmm a08 = addpz(y0, y8); const xmm s08 = subpz(y0, y8);
            const xmm a4c = addpz(y4, yc); const xmm s4c = subpz(y4, yc);
            const xmm a2a = addpz(y2, ya); const xmm s2a = subpz(y2, ya);
            const xmm a6e = addpz(y6, ye); const xmm s6e = subpz(y6, ye);
            const xmm a19 = addpz(y1, y9); const xmm s19 = subpz(y1, y9);
            const xmm a5d = addpz(y5, yd); const xmm s5d = subpz(y5, yd);
            const xmm a3b = addpz(y3, yb); const xmm s3b = subpz(y3, yb);
            const xmm a7f = addpz(y7, yf); const xmm s7f = subpz(y7, yf);

            const xmm js4c = jxpz(s4c);
            const xmm js6e = jxpz(s6e);
            const xmm js5d = jxpz(s5d);
            const xmm js7f = jxpz(s7f);

            const xmm a08p1a4c = addpz(a08, a4c); const xmm s08mjs4c = subpz(s08, js4c);
            const xmm a08m1a4c = subpz(a08, a4c); const xmm s08pjs4c = addpz(s08, js4c);
            const xmm a2ap1a6e = addpz(a2a, a6e); const xmm s2amjs6e = subpz(s2a, js6e);
            const xmm a2am1a6e = subpz(a2a, a6e); const xmm s2apjs6e = addpz(s2a, js6e);
            const xmm a19p1a5d = addpz(a19, a5d); const xmm s19mjs5d = subpz(s19, js5d);
            const xmm a19m1a5d = subpz(a19, a5d); const xmm s19pjs5d = addpz(s19, js5d);
            const xmm a3bp1a7f = addpz(a3b, a7f); const xmm s3bmjs7f = subpz(s3b, js7f);
            const xmm a3bm1a7f = subpz(a3b, a7f); const xmm s3bpjs7f = addpz(s3b, js7f);

            const xmm w8_s2amjs6e = w8xpz(s2amjs6e);
            const xmm  j_a2am1a6e =  jxpz(a2am1a6e);
            const xmm v8_s2apjs6e = v8xpz(s2apjs6e);

            const xmm a08p1a4c_p1_a2ap1a6e = addpz(a08p1a4c,    a2ap1a6e);
            const xmm s08mjs4c_pw_s2amjs6e = addpz(s08mjs4c, w8_s2amjs6e);
            const xmm a08m1a4c_mj_a2am1a6e = subpz(a08m1a4c,  j_a2am1a6e);
            const xmm s08pjs4c_mv_s2apjs6e = subpz(s08pjs4c, v8_s2apjs6e);
            const xmm a08p1a4c_m1_a2ap1a6e = subpz(a08p1a4c,    a2ap1a6e);
            const xmm s08mjs4c_mw_s2amjs6e = subpz(s08mjs4c, w8_s2amjs6e);
            const xmm a08m1a4c_pj_a2am1a6e = addpz(a08m1a4c,  j_a2am1a6e);
            const xmm s08pjs4c_pv_s2apjs6e = addpz(s08pjs4c, v8_s2apjs6e);

            const xmm w8_s3bmjs7f = w8xpz(s3bmjs7f);
            const xmm  j_a3bm1a7f =  jxpz(a3bm1a7f);
            const xmm v8_s3bpjs7f = v8xpz(s3bpjs7f);

            const xmm a19p1a5d_p1_a3bp1a7f = addpz(a19p1a5d,    a3bp1a7f);
            const xmm s19mjs5d_pw_s3bmjs7f = addpz(s19mjs5d, w8_s3bmjs7f);
            const xmm a19m1a5d_mj_a3bm1a7f = subpz(a19m1a5d,  j_a3bm1a7f);
            const xmm s19pjs5d_mv_s3bpjs7f = subpz(s19pjs5d, v8_s3bpjs7f);
            const xmm a19p1a5d_m1_a3bp1a7f = subpz(a19p1a5d,    a3bp1a7f);
            const xmm s19mjs5d_mw_s3bmjs7f = subpz(s19mjs5d, w8_s3bmjs7f);
            const xmm a19m1a5d_pj_a3bm1a7f = addpz(a19m1a5d,  j_a3bm1a7f);
            const xmm s19pjs5d_pv_s3bpjs7f = addpz(s19pjs5d, v8_s3bpjs7f);

            const xmm h1_s19mjs5d_pw_s3bmjs7f = h1xpz(s19mjs5d_pw_s3bmjs7f);
            const xmm w8_a19m1a5d_mj_a3bm1a7f = w8xpz(a19m1a5d_mj_a3bm1a7f);
            const xmm h3_s19pjs5d_mv_s3bpjs7f = h3xpz(s19pjs5d_mv_s3bpjs7f);
            const xmm  j_a19p1a5d_m1_a3bp1a7f =  jxpz(a19p1a5d_m1_a3bp1a7f);
            const xmm hd_s19mjs5d_mw_s3bmjs7f = hdxpz(s19mjs5d_mw_s3bmjs7f);
            const xmm v8_a19m1a5d_pj_a3bm1a7f = v8xpz(a19m1a5d_pj_a3bm1a7f);
            const xmm hf_s19pjs5d_pv_s3bpjs7f = hfxpz(s19pjs5d_pv_s3bpjs7f);

            setpz(x[0x0], addpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz(x[0x1], addpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz(x[0x2], addpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz(x[0x3], addpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz(x[0x4], addpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz(x[0x5], subpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz(x[0x6], subpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz(x[0x7], subpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

            setpz(x[0x8], subpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz(x[0x9], subpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz(x[0xa], subpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz(x[0xb], subpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz(x[0xc], subpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz(x[0xd], addpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz(x[0xe], addpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz(x[0xf], addpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        }
    }
};

//-----------------------------------------------------------------------------

template <int s> struct invnend<16,s,0>
{
    static const int N = 16*s;

    void operator()(complex_vector x, complex_vector) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;

            const ymm x0 = mulpd2(rN, getpz2(xq+s*0x0));
            const ymm x1 = mulpd2(rN, getpz2(xq+s*0x1));
            const ymm x2 = mulpd2(rN, getpz2(xq+s*0x2));
            const ymm x3 = mulpd2(rN, getpz2(xq+s*0x3));
            const ymm x4 = mulpd2(rN, getpz2(xq+s*0x4));
            const ymm x5 = mulpd2(rN, getpz2(xq+s*0x5));
            const ymm x6 = mulpd2(rN, getpz2(xq+s*0x6));
            const ymm x7 = mulpd2(rN, getpz2(xq+s*0x7));
            const ymm x8 = mulpd2(rN, getpz2(xq+s*0x8));
            const ymm x9 = mulpd2(rN, getpz2(xq+s*0x9));
            const ymm xa = mulpd2(rN, getpz2(xq+s*0xa));
            const ymm xb = mulpd2(rN, getpz2(xq+s*0xb));
            const ymm xc = mulpd2(rN, getpz2(xq+s*0xc));
            const ymm xd = mulpd2(rN, getpz2(xq+s*0xd));
            const ymm xe = mulpd2(rN, getpz2(xq+s*0xe));
            const ymm xf = mulpd2(rN, getpz2(xq+s*0xf));

            const ymm a08 = addpz2(x0, x8); const ymm s08 = subpz2(x0, x8);
            const ymm a4c = addpz2(x4, xc); const ymm s4c = subpz2(x4, xc);
            const ymm a2a = addpz2(x2, xa); const ymm s2a = subpz2(x2, xa);
            const ymm a6e = addpz2(x6, xe); const ymm s6e = subpz2(x6, xe);
            const ymm a19 = addpz2(x1, x9); const ymm s19 = subpz2(x1, x9);
            const ymm a5d = addpz2(x5, xd); const ymm s5d = subpz2(x5, xd);
            const ymm a3b = addpz2(x3, xb); const ymm s3b = subpz2(x3, xb);
            const ymm a7f = addpz2(x7, xf); const ymm s7f = subpz2(x7, xf);

            const ymm js4c = jxpz2(s4c);
            const ymm js6e = jxpz2(s6e);
            const ymm js5d = jxpz2(s5d);
            const ymm js7f = jxpz2(s7f);

            const ymm a08p1a4c = addpz2(a08, a4c); const ymm s08mjs4c = subpz2(s08, js4c);
            const ymm a08m1a4c = subpz2(a08, a4c); const ymm s08pjs4c = addpz2(s08, js4c);
            const ymm a2ap1a6e = addpz2(a2a, a6e); const ymm s2amjs6e = subpz2(s2a, js6e);
            const ymm a2am1a6e = subpz2(a2a, a6e); const ymm s2apjs6e = addpz2(s2a, js6e);
            const ymm a19p1a5d = addpz2(a19, a5d); const ymm s19mjs5d = subpz2(s19, js5d);
            const ymm a19m1a5d = subpz2(a19, a5d); const ymm s19pjs5d = addpz2(s19, js5d);
            const ymm a3bp1a7f = addpz2(a3b, a7f); const ymm s3bmjs7f = subpz2(s3b, js7f);
            const ymm a3bm1a7f = subpz2(a3b, a7f); const ymm s3bpjs7f = addpz2(s3b, js7f);

            const ymm w8_s2amjs6e = w8xpz2(s2amjs6e);
            const ymm  j_a2am1a6e =  jxpz2(a2am1a6e);
            const ymm v8_s2apjs6e = v8xpz2(s2apjs6e);

            const ymm a08p1a4c_p1_a2ap1a6e = addpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_pw_s2amjs6e = addpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_mj_a2am1a6e = subpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_mv_s2apjs6e = subpz2(s08pjs4c, v8_s2apjs6e);
            const ymm a08p1a4c_m1_a2ap1a6e = subpz2(a08p1a4c,    a2ap1a6e);
            const ymm s08mjs4c_mw_s2amjs6e = subpz2(s08mjs4c, w8_s2amjs6e);
            const ymm a08m1a4c_pj_a2am1a6e = addpz2(a08m1a4c,  j_a2am1a6e);
            const ymm s08pjs4c_pv_s2apjs6e = addpz2(s08pjs4c, v8_s2apjs6e);

            const ymm w8_s3bmjs7f = w8xpz2(s3bmjs7f);
            const ymm  j_a3bm1a7f =  jxpz2(a3bm1a7f);
            const ymm v8_s3bpjs7f = v8xpz2(s3bpjs7f);

            const ymm a19p1a5d_p1_a3bp1a7f = addpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_pw_s3bmjs7f = addpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_mj_a3bm1a7f = subpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_mv_s3bpjs7f = subpz2(s19pjs5d, v8_s3bpjs7f);
            const ymm a19p1a5d_m1_a3bp1a7f = subpz2(a19p1a5d,    a3bp1a7f);
            const ymm s19mjs5d_mw_s3bmjs7f = subpz2(s19mjs5d, w8_s3bmjs7f);
            const ymm a19m1a5d_pj_a3bm1a7f = addpz2(a19m1a5d,  j_a3bm1a7f);
            const ymm s19pjs5d_pv_s3bpjs7f = addpz2(s19pjs5d, v8_s3bpjs7f);

            const ymm h1_s19mjs5d_pw_s3bmjs7f = h1xpz2(s19mjs5d_pw_s3bmjs7f);
            const ymm w8_a19m1a5d_mj_a3bm1a7f = w8xpz2(a19m1a5d_mj_a3bm1a7f);
            const ymm h3_s19pjs5d_mv_s3bpjs7f = h3xpz2(s19pjs5d_mv_s3bpjs7f);
            const ymm  j_a19p1a5d_m1_a3bp1a7f =  jxpz2(a19p1a5d_m1_a3bp1a7f);
            const ymm hd_s19mjs5d_mw_s3bmjs7f = hdxpz2(s19mjs5d_mw_s3bmjs7f);
            const ymm v8_a19m1a5d_pj_a3bm1a7f = v8xpz2(a19m1a5d_pj_a3bm1a7f);
            const ymm hf_s19pjs5d_pv_s3bpjs7f = hfxpz2(s19pjs5d_pv_s3bpjs7f);

            setpz2(xq+s*0x0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq+s*0x1, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz2(xq+s*0x2, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq+s*0x3, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq+s*0x4, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq+s*0x5, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq+s*0x6, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq+s*0x7, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

            setpz2(xq+s*0x8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(xq+s*0x9, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz2(xq+s*0xa, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(xq+s*0xb, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(xq+s*0xc, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(xq+s*0xd, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(xq+s*0xe, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(xq+s*0xf, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        }
    }
};

template <> struct invnend<16,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const noexcept
    {
        static const xmm rN = { 1.0/16, 1.0/16 };
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
            const xmm x0 = mulpd(rN, getpz(x[0x0]));
            const xmm x1 = mulpd(rN, getpz(x[0x1]));
            const xmm x2 = mulpd(rN, getpz(x[0x2]));
            const xmm x3 = mulpd(rN, getpz(x[0x3]));
            const xmm x4 = mulpd(rN, getpz(x[0x4]));
            const xmm x5 = mulpd(rN, getpz(x[0x5]));
            const xmm x6 = mulpd(rN, getpz(x[0x6]));
            const xmm x7 = mulpd(rN, getpz(x[0x7]));
            const xmm x8 = mulpd(rN, getpz(x[0x8]));
            const xmm x9 = mulpd(rN, getpz(x[0x9]));
            const xmm xa = mulpd(rN, getpz(x[0xa]));
            const xmm xb = mulpd(rN, getpz(x[0xb]));
            const xmm xc = mulpd(rN, getpz(x[0xc]));
            const xmm xd = mulpd(rN, getpz(x[0xd]));
            const xmm xe = mulpd(rN, getpz(x[0xe]));
            const xmm xf = mulpd(rN, getpz(x[0xf]));

            const xmm a08 = addpz(x0, x8); const xmm s08 = subpz(x0, x8);
            const xmm a4c = addpz(x4, xc); const xmm s4c = subpz(x4, xc);
            const xmm a2a = addpz(x2, xa); const xmm s2a = subpz(x2, xa);
            const xmm a6e = addpz(x6, xe); const xmm s6e = subpz(x6, xe);
            const xmm a19 = addpz(x1, x9); const xmm s19 = subpz(x1, x9);
            const xmm a5d = addpz(x5, xd); const xmm s5d = subpz(x5, xd);
            const xmm a3b = addpz(x3, xb); const xmm s3b = subpz(x3, xb);
            const xmm a7f = addpz(x7, xf); const xmm s7f = subpz(x7, xf);

            const xmm js4c = jxpz(s4c);
            const xmm js6e = jxpz(s6e);
            const xmm js5d = jxpz(s5d);
            const xmm js7f = jxpz(s7f);

            const xmm a08p1a4c = addpz(a08, a4c); const xmm s08mjs4c = subpz(s08, js4c);
            const xmm a08m1a4c = subpz(a08, a4c); const xmm s08pjs4c = addpz(s08, js4c);
            const xmm a2ap1a6e = addpz(a2a, a6e); const xmm s2amjs6e = subpz(s2a, js6e);
            const xmm a2am1a6e = subpz(a2a, a6e); const xmm s2apjs6e = addpz(s2a, js6e);
            const xmm a19p1a5d = addpz(a19, a5d); const xmm s19mjs5d = subpz(s19, js5d);
            const xmm a19m1a5d = subpz(a19, a5d); const xmm s19pjs5d = addpz(s19, js5d);
            const xmm a3bp1a7f = addpz(a3b, a7f); const xmm s3bmjs7f = subpz(s3b, js7f);
            const xmm a3bm1a7f = subpz(a3b, a7f); const xmm s3bpjs7f = addpz(s3b, js7f);

            const xmm w8_s2amjs6e = w8xpz(s2amjs6e);
            const xmm  j_a2am1a6e =  jxpz(a2am1a6e);
            const xmm v8_s2apjs6e = v8xpz(s2apjs6e);

            const xmm a08p1a4c_p1_a2ap1a6e = addpz(a08p1a4c,    a2ap1a6e);
            const xmm s08mjs4c_pw_s2amjs6e = addpz(s08mjs4c, w8_s2amjs6e);
            const xmm a08m1a4c_mj_a2am1a6e = subpz(a08m1a4c,  j_a2am1a6e);
            const xmm s08pjs4c_mv_s2apjs6e = subpz(s08pjs4c, v8_s2apjs6e);
            const xmm a08p1a4c_m1_a2ap1a6e = subpz(a08p1a4c,    a2ap1a6e);
            const xmm s08mjs4c_mw_s2amjs6e = subpz(s08mjs4c, w8_s2amjs6e);
            const xmm a08m1a4c_pj_a2am1a6e = addpz(a08m1a4c,  j_a2am1a6e);
            const xmm s08pjs4c_pv_s2apjs6e = addpz(s08pjs4c, v8_s2apjs6e);

            const xmm w8_s3bmjs7f = w8xpz(s3bmjs7f);
            const xmm  j_a3bm1a7f =  jxpz(a3bm1a7f);
            const xmm v8_s3bpjs7f = v8xpz(s3bpjs7f);

            const xmm a19p1a5d_p1_a3bp1a7f = addpz(a19p1a5d,    a3bp1a7f);
            const xmm s19mjs5d_pw_s3bmjs7f = addpz(s19mjs5d, w8_s3bmjs7f);
            const xmm a19m1a5d_mj_a3bm1a7f = subpz(a19m1a5d,  j_a3bm1a7f);
            const xmm s19pjs5d_mv_s3bpjs7f = subpz(s19pjs5d, v8_s3bpjs7f);
            const xmm a19p1a5d_m1_a3bp1a7f = subpz(a19p1a5d,    a3bp1a7f);
            const xmm s19mjs5d_mw_s3bmjs7f = subpz(s19mjs5d, w8_s3bmjs7f);
            const xmm a19m1a5d_pj_a3bm1a7f = addpz(a19m1a5d,  j_a3bm1a7f);
            const xmm s19pjs5d_pv_s3bpjs7f = addpz(s19pjs5d, v8_s3bpjs7f);

            const xmm h1_s19mjs5d_pw_s3bmjs7f = h1xpz(s19mjs5d_pw_s3bmjs7f);
            const xmm w8_a19m1a5d_mj_a3bm1a7f = w8xpz(a19m1a5d_mj_a3bm1a7f);
            const xmm h3_s19pjs5d_mv_s3bpjs7f = h3xpz(s19pjs5d_mv_s3bpjs7f);
            const xmm  j_a19p1a5d_m1_a3bp1a7f =  jxpz(a19p1a5d_m1_a3bp1a7f);
            const xmm hd_s19mjs5d_mw_s3bmjs7f = hdxpz(s19mjs5d_mw_s3bmjs7f);
            const xmm v8_a19m1a5d_pj_a3bm1a7f = v8xpz(a19m1a5d_pj_a3bm1a7f);
            const xmm hf_s19pjs5d_pv_s3bpjs7f = hfxpz(s19pjs5d_pv_s3bpjs7f);

            setpz(x[0x0], addpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz(x[0x1], addpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz(x[0x2], addpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz(x[0x3], addpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz(x[0x4], addpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz(x[0x5], subpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz(x[0x6], subpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz(x[0x7], subpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

            setpz(x[0x8], subpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz(x[0x9], subpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz(x[0xa], subpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz(x[0xb], subpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz(x[0xc], subpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz(x[0xd], addpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz(x[0xe], addpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz(x[0xf], addpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
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
        inv0fft<n/16,16*s,!eo>()(y, x, W);
        invcore<n,s>()(x, y, W);
    }
};

template <int s, bool eo> struct inv0fft<16,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        inv0end<16,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct inv0fft<8,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT8omp::inv0end<8,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct inv0fft<4,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT4omp::inv0end<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct inv0fft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT4omp::inv0end<2,s,eo>()(x, y);
    }
};

//-----------------------------------------------------------------------------

template <int n, int s, bool eo> struct invnfft
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        invnfft<n/16,16*s,!eo>()(y, x, W);
        invcore<n,s>()(x, y, W);
    }
};

template <int s, bool eo> struct invnfft<16,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        invnend<16,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct invnfft<8,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT8omp::invnend<8,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct invnfft<4,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT4omp::invnend<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct invnfft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIT4omp::invnend<2,s,eo>()(x, y);
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

inline void fwdno(const int log_N,
        complex_vector x, complex_vector y, const_complex_vector W) noexcept
{
#ifdef _OPENMP
    #pragma omp parallel firstprivate(x,y,W)
#endif
    switch (log_N) {
        case  0: break;
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

inline void invno(const int log_N,
        complex_vector x, complex_vector y, const_complex_vector W) noexcept
{
#ifdef _OPENMP
    #pragma omp parallel firstprivate(x,y,W)
#endif
    switch (log_N) {
        case  0: break;
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

} /////////////////////////////////////////////////////////////////////////////

#endif // otfft_avxdit16omp_h
