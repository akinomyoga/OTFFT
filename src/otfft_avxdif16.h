/******************************************************************************
*  OTFFT AVXDIF(Radix-16) Version 6.5
*
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_avxdif16_h
#define otfft_avxdif16_h

#include "otfft/otfft_misc.h"
#include "otfft_avxdif8.h"
#include "otfft_avxdif16omp.h"

namespace OTFFT_AVXDIF16 { ////////////////////////////////////////////////////

using namespace OTFFT_MISC;

#ifdef DO_SINGLE_THREAD
static const int OMP_THRESHOLD = 1<<30;
#else
static const int OMP_THRESHOLD = 1<<12;
#endif

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
        for (int p = 0; p < n1; p++) {
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

            for (int q = 0; q < s; q += 2) {
                complex_vector xq_sp   = x + q + sp;
                complex_vector yq_s16p = y + q + s16p;

                const ymm x0 = getpz2(xq_sp+N0);
                const ymm x1 = getpz2(xq_sp+N1);
                const ymm x2 = getpz2(xq_sp+N2);
                const ymm x3 = getpz2(xq_sp+N3);
                const ymm x4 = getpz2(xq_sp+N4);
                const ymm x5 = getpz2(xq_sp+N5);
                const ymm x6 = getpz2(xq_sp+N6);
                const ymm x7 = getpz2(xq_sp+N7);
                const ymm x8 = getpz2(xq_sp+N8);
                const ymm x9 = getpz2(xq_sp+N9);
                const ymm xa = getpz2(xq_sp+Na);
                const ymm xb = getpz2(xq_sp+Nb);
                const ymm xc = getpz2(xq_sp+Nc);
                const ymm xd = getpz2(xq_sp+Nd);
                const ymm xe = getpz2(xq_sp+Ne);
                const ymm xf = getpz2(xq_sp+Nf);

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

                setpz2(yq_s16p+s*0x0,             addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz2(yq_s16p+s*0x1, mulpz2(w1p, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));
                setpz2(yq_s16p+s*0x2, mulpz2(w2p, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
                setpz2(yq_s16p+s*0x3, mulpz2(w3p, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
                setpz2(yq_s16p+s*0x4, mulpz2(w4p, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
                setpz2(yq_s16p+s*0x5, mulpz2(w5p, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
                setpz2(yq_s16p+s*0x6, mulpz2(w6p, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
                setpz2(yq_s16p+s*0x7, mulpz2(w7p, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));

                setpz2(yq_s16p+s*0x8, mulpz2(w8p, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f)));
                setpz2(yq_s16p+s*0x9, mulpz2(w9p, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));
                setpz2(yq_s16p+s*0xa, mulpz2(wap, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
                setpz2(yq_s16p+s*0xb, mulpz2(wbp, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
                setpz2(yq_s16p+s*0xc, mulpz2(wcp, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
                setpz2(yq_s16p+s*0xd, mulpz2(wdp, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
                setpz2(yq_s16p+s*0xe, mulpz2(wep, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
                setpz2(yq_s16p+s*0xf, mulpz2(wfp, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));
            }
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

            const ymm x0 = getpz2(x_p+N0);
            const ymm x1 = getpz2(x_p+N1);
            const ymm x2 = getpz2(x_p+N2);
            const ymm x3 = getpz2(x_p+N3);
            const ymm x4 = getpz2(x_p+N4);
            const ymm x5 = getpz2(x_p+N5);
            const ymm x6 = getpz2(x_p+N6);
            const ymm x7 = getpz2(x_p+N7);
            const ymm x8 = getpz2(x_p+N8);
            const ymm x9 = getpz2(x_p+N9);
            const ymm xa = getpz2(x_p+Na);
            const ymm xb = getpz2(x_p+Nb);
            const ymm xc = getpz2(x_p+Nc);
            const ymm xd = getpz2(x_p+Nd);
            const ymm xe = getpz2(x_p+Ne);
            const ymm xf = getpz2(x_p+Nf);

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

#if 0
            setpz3<16>(y_16p+0x0,             addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz3<16>(y_16p+0x1, mulpz2(w1p, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));
            setpz3<16>(y_16p+0x2, mulpz2(w2p, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
            setpz3<16>(y_16p+0x3, mulpz2(w3p, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
            setpz3<16>(y_16p+0x4, mulpz2(w4p, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
            setpz3<16>(y_16p+0x5, mulpz2(w5p, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
            setpz3<16>(y_16p+0x6, mulpz2(w6p, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
            setpz3<16>(y_16p+0x7, mulpz2(w7p, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));

            setpz3<16>(y_16p+0x8, mulpz2(w8p, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f)));
            setpz3<16>(y_16p+0x9, mulpz2(w9p, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));
            setpz3<16>(y_16p+0xa, mulpz2(wap, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
            setpz3<16>(y_16p+0xb, mulpz2(wbp, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
            setpz3<16>(y_16p+0xc, mulpz2(wcp, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
            setpz3<16>(y_16p+0xd, mulpz2(wdp, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
            setpz3<16>(y_16p+0xe, mulpz2(wep, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
            setpz3<16>(y_16p+0xf, mulpz2(wfp, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));
#else
            const ymm aA =             addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f);
            const ymm bB = mulpz2(w1p, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            const ymm cC = mulpz2(w2p, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            const ymm dD = mulpz2(w3p, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            const ymm eE = mulpz2(w4p, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            const ymm fF = mulpz2(w5p, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            const ymm gG = mulpz2(w6p, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            const ymm hH = mulpz2(w7p, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

            const ymm iI = mulpz2(w8p, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            const ymm jJ = mulpz2(w9p, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            const ymm kK = mulpz2(wap, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            const ymm lL = mulpz2(wbp, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            const ymm mM = mulpz2(wcp, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            const ymm nN = mulpz2(wdp, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            const ymm oO = mulpz2(wep, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            const ymm pP = mulpz2(wfp, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

            const ymm ab = catlo(aA, bB);
            const ymm AB = cathi(aA, bB);
            const ymm cd = catlo(cC, dD);
            const ymm CD = cathi(cC, dD);
            const ymm ef = catlo(eE, fF);
            const ymm EF = cathi(eE, fF);
            const ymm gh = catlo(gG, hH);
            const ymm GH = cathi(gG, hH);

            const ymm ij = catlo(iI, jJ);
            const ymm IJ = cathi(iI, jJ);
            const ymm kl = catlo(kK, lL);
            const ymm KL = cathi(kK, lL);
            const ymm mn = catlo(mM, nN);
            const ymm MN = cathi(mM, nN);
            const ymm op = catlo(oO, pP);
            const ymm OP = cathi(oO, pP);

            setpz2(y_16p+0x00, ab);
            setpz2(y_16p+0x02, cd);
            setpz2(y_16p+0x04, ef);
            setpz2(y_16p+0x06, gh);
            setpz2(y_16p+0x08, ij);
            setpz2(y_16p+0x0a, kl);
            setpz2(y_16p+0x0c, mn);
            setpz2(y_16p+0x0e, op);

            setpz2(y_16p+0x10, AB);
            setpz2(y_16p+0x12, CD);
            setpz2(y_16p+0x14, EF);
            setpz2(y_16p+0x16, GH);
            setpz2(y_16p+0x18, IJ);
            setpz2(y_16p+0x1a, KL);
            setpz2(y_16p+0x1c, MN);
            setpz2(y_16p+0x1e, OP);
#endif
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
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;

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

            setpz2(yq+s*0x0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(yq+s*0x1, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz2(yq+s*0x2, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(yq+s*0x3, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(yq+s*0x4, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(yq+s*0x5, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(yq+s*0x6, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(yq+s*0x7, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

            setpz2(yq+s*0x8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(yq+s*0x9, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz2(yq+s*0xa, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(yq+s*0xb, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(yq+s*0xc, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(yq+s*0xd, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(yq+s*0xe, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(yq+s*0xf, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
        }
    }
};

template <> struct fwd0end<16,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
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

        setpz(y[0x0], addpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
        setpz(y[0x1], addpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        setpz(y[0x2], addpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
        setpz(y[0x3], addpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
        setpz(y[0x4], subpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
        setpz(y[0x5], subpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
        setpz(y[0x6], subpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
        setpz(y[0x7], subpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

        setpz(y[0x8], subpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
        setpz(y[0x9], subpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        setpz(y[0xa], subpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
        setpz(y[0xb], subpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
        setpz(y[0xc], addpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
        setpz(y[0xd], addpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
        setpz(y[0xe], addpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
        setpz(y[0xf], addpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct fwd0end<16,s,0>
{
    void operator()(complex_vector x, complex_vector) const noexcept
    {
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
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;

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

            setpz2(yq+s*0x0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(yq+s*0x1, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz2(yq+s*0x2, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(yq+s*0x3, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(yq+s*0x4, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(yq+s*0x5, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(yq+s*0x6, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(yq+s*0x7, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

            setpz2(yq+s*0x8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(yq+s*0x9, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
            setpz2(yq+s*0xa, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(yq+s*0xb, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(yq+s*0xc, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(yq+s*0xd, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(yq+s*0xe, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(yq+s*0xf, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
        }
    }
};

template <> struct fwdnend<16,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        zeroupper();
        static const xmm rN = { 1.0/16, 1.0/16 };
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

        setpz(y[0x0], addpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
        setpz(y[0x1], addpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        setpz(y[0x2], addpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
        setpz(y[0x3], addpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
        setpz(y[0x4], subpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
        setpz(y[0x5], subpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
        setpz(y[0x6], subpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
        setpz(y[0x7], subpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));

        setpz(y[0x8], subpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
        setpz(y[0x9], subpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        setpz(y[0xa], subpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
        setpz(y[0xb], subpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
        setpz(y[0xc], addpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
        setpz(y[0xd], addpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
        setpz(y[0xe], addpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
        setpz(y[0xf], addpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct fwdnend<16,s,0>
{
    static const int N = 16*s;

    void operator()(complex_vector x, complex_vector) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
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
        zeroupper();
        static const xmm rN = { 1.0/16, 1.0/16 };
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
        fwd0fft<n/16,16*s,!eo>()(y, x, W);
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
        OTFFT_AVXDIF8::fwd0end<8,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwd0fft<4,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIF4::fwd0end<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwd0fft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIF4::fwd0end<2,s,eo>()(x, y);
    }
};

//-----------------------------------------------------------------------------

template <int n, int s, bool eo> struct fwdnfft
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        fwdcore<n,s>()(x, y, W);
        fwdnfft<n/16,16*s,!eo>()(y, x, W);
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
        OTFFT_AVXDIF8::fwdnend<8,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwdnfft<4,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIF4::fwdnend<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwdnfft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIF4::fwdnend<2,s,eo>()(x, y);
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
        for (int p = 0; p < n1; p++) {
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

            for (int q = 0; q < s; q += 2) {
                complex_vector xq_sp   = x + q + sp;
                complex_vector yq_s16p = y + q + s16p;

                const ymm x0 = getpz2(xq_sp+N0);
                const ymm x1 = getpz2(xq_sp+N1);
                const ymm x2 = getpz2(xq_sp+N2);
                const ymm x3 = getpz2(xq_sp+N3);
                const ymm x4 = getpz2(xq_sp+N4);
                const ymm x5 = getpz2(xq_sp+N5);
                const ymm x6 = getpz2(xq_sp+N6);
                const ymm x7 = getpz2(xq_sp+N7);
                const ymm x8 = getpz2(xq_sp+N8);
                const ymm x9 = getpz2(xq_sp+N9);
                const ymm xa = getpz2(xq_sp+Na);
                const ymm xb = getpz2(xq_sp+Nb);
                const ymm xc = getpz2(xq_sp+Nc);
                const ymm xd = getpz2(xq_sp+Nd);
                const ymm xe = getpz2(xq_sp+Ne);
                const ymm xf = getpz2(xq_sp+Nf);

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

                setpz2(yq_s16p+s*0x0,             addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
                setpz2(yq_s16p+s*0x1, mulpz2(w1p, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));
                setpz2(yq_s16p+s*0x2, mulpz2(w2p, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
                setpz2(yq_s16p+s*0x3, mulpz2(w3p, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
                setpz2(yq_s16p+s*0x4, mulpz2(w4p, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
                setpz2(yq_s16p+s*0x5, mulpz2(w5p, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
                setpz2(yq_s16p+s*0x6, mulpz2(w6p, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
                setpz2(yq_s16p+s*0x7, mulpz2(w7p, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));

                setpz2(yq_s16p+s*0x8, mulpz2(w8p, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f)));
                setpz2(yq_s16p+s*0x9, mulpz2(w9p, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));
                setpz2(yq_s16p+s*0xa, mulpz2(wap, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
                setpz2(yq_s16p+s*0xb, mulpz2(wbp, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
                setpz2(yq_s16p+s*0xc, mulpz2(wcp, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
                setpz2(yq_s16p+s*0xd, mulpz2(wdp, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
                setpz2(yq_s16p+s*0xe, mulpz2(wep, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
                setpz2(yq_s16p+s*0xf, mulpz2(wfp, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));
            }
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

            const ymm x0 = getpz2(x_p+N0);
            const ymm x1 = getpz2(x_p+N1);
            const ymm x2 = getpz2(x_p+N2);
            const ymm x3 = getpz2(x_p+N3);
            const ymm x4 = getpz2(x_p+N4);
            const ymm x5 = getpz2(x_p+N5);
            const ymm x6 = getpz2(x_p+N6);
            const ymm x7 = getpz2(x_p+N7);
            const ymm x8 = getpz2(x_p+N8);
            const ymm x9 = getpz2(x_p+N9);
            const ymm xa = getpz2(x_p+Na);
            const ymm xb = getpz2(x_p+Nb);
            const ymm xc = getpz2(x_p+Nc);
            const ymm xd = getpz2(x_p+Nd);
            const ymm xe = getpz2(x_p+Ne);
            const ymm xf = getpz2(x_p+Nf);

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

#if 0
            setpz3<16>(y_16p+0x0,             addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz3<16>(y_16p+0x1, mulpz2(w1p, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));
            setpz3<16>(y_16p+0x2, mulpz2(w2p, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
            setpz3<16>(y_16p+0x3, mulpz2(w3p, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
            setpz3<16>(y_16p+0x4, mulpz2(w4p, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
            setpz3<16>(y_16p+0x5, mulpz2(w5p, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
            setpz3<16>(y_16p+0x6, mulpz2(w6p, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
            setpz3<16>(y_16p+0x7, mulpz2(w7p, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));

            setpz3<16>(y_16p+0x8, mulpz2(w8p, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f)));
            setpz3<16>(y_16p+0x9, mulpz2(w9p, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f)));
            setpz3<16>(y_16p+0xa, mulpz2(wap, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f)));
            setpz3<16>(y_16p+0xb, mulpz2(wbp, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f)));
            setpz3<16>(y_16p+0xc, mulpz2(wcp, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f)));
            setpz3<16>(y_16p+0xd, mulpz2(wdp, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f)));
            setpz3<16>(y_16p+0xe, mulpz2(wep, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f)));
            setpz3<16>(y_16p+0xf, mulpz2(wfp, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f)));
#else
            const ymm aA =             addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f);
            const ymm bB = mulpz2(w1p, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            const ymm cC = mulpz2(w2p, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            const ymm dD = mulpz2(w3p, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            const ymm eE = mulpz2(w4p, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            const ymm fF = mulpz2(w5p, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            const ymm gG = mulpz2(w6p, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            const ymm hH = mulpz2(w7p, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

            const ymm iI = mulpz2(w8p, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            const ymm jJ = mulpz2(w9p, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            const ymm kK = mulpz2(wap, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            const ymm lL = mulpz2(wbp, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            const ymm mM = mulpz2(wcp, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            const ymm nN = mulpz2(wdp, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            const ymm oO = mulpz2(wep, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            const ymm pP = mulpz2(wfp, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

            const ymm ab = catlo(aA, bB);
            const ymm AB = cathi(aA, bB);
            const ymm cd = catlo(cC, dD);
            const ymm CD = cathi(cC, dD);
            const ymm ef = catlo(eE, fF);
            const ymm EF = cathi(eE, fF);
            const ymm gh = catlo(gG, hH);
            const ymm GH = cathi(gG, hH);

            const ymm ij = catlo(iI, jJ);
            const ymm IJ = cathi(iI, jJ);
            const ymm kl = catlo(kK, lL);
            const ymm KL = cathi(kK, lL);
            const ymm mn = catlo(mM, nN);
            const ymm MN = cathi(mM, nN);
            const ymm op = catlo(oO, pP);
            const ymm OP = cathi(oO, pP);

            setpz2(y_16p+0x00, ab);
            setpz2(y_16p+0x02, cd);
            setpz2(y_16p+0x04, ef);
            setpz2(y_16p+0x06, gh);
            setpz2(y_16p+0x08, ij);
            setpz2(y_16p+0x0a, kl);
            setpz2(y_16p+0x0c, mn);
            setpz2(y_16p+0x0e, op);

            setpz2(y_16p+0x10, AB);
            setpz2(y_16p+0x12, CD);
            setpz2(y_16p+0x14, EF);
            setpz2(y_16p+0x16, GH);
            setpz2(y_16p+0x18, IJ);
            setpz2(y_16p+0x1a, KL);
            setpz2(y_16p+0x1c, MN);
            setpz2(y_16p+0x1e, OP);
#endif
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
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;

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

            setpz2(yq+s*0x0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(yq+s*0x1, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz2(yq+s*0x2, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(yq+s*0x3, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(yq+s*0x4, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(yq+s*0x5, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(yq+s*0x6, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(yq+s*0x7, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

            setpz2(yq+s*0x8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(yq+s*0x9, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz2(yq+s*0xa, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(yq+s*0xb, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(yq+s*0xc, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(yq+s*0xd, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(yq+s*0xe, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(yq+s*0xf, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        }
    }
};

template <> struct inv0end<16,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
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

        setpz(y[0x0], addpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
        setpz(y[0x1], addpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
        setpz(y[0x2], addpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
        setpz(y[0x3], addpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
        setpz(y[0x4], addpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
        setpz(y[0x5], subpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
        setpz(y[0x6], subpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
        setpz(y[0x7], subpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

        setpz(y[0x8], subpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
        setpz(y[0x9], subpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
        setpz(y[0xa], subpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
        setpz(y[0xb], subpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
        setpz(y[0xc], subpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
        setpz(y[0xd], addpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
        setpz(y[0xe], addpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
        setpz(y[0xf], addpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct inv0end<16,s,0>
{
    void operator()(complex_vector x, complex_vector) const noexcept
    {
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
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;

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

            setpz2(yq+s*0x0, addpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(yq+s*0x1, addpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz2(yq+s*0x2, addpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(yq+s*0x3, addpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(yq+s*0x4, addpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(yq+s*0x5, subpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(yq+s*0x6, subpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(yq+s*0x7, subpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

            setpz2(yq+s*0x8, subpz2(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
            setpz2(yq+s*0x9, subpz2(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
            setpz2(yq+s*0xa, subpz2(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
            setpz2(yq+s*0xb, subpz2(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
            setpz2(yq+s*0xc, subpz2(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
            setpz2(yq+s*0xd, addpz2(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
            setpz2(yq+s*0xe, addpz2(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
            setpz2(yq+s*0xf, addpz2(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
        }
    }
};

template <> struct invnend<16,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const noexcept
    {
        zeroupper();
        static const xmm rN = { 1.0/16, 1.0/16 };
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

        setpz(y[0x0], addpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
        setpz(y[0x1], addpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
        setpz(y[0x2], addpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
        setpz(y[0x3], addpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
        setpz(y[0x4], addpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
        setpz(y[0x5], subpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
        setpz(y[0x6], subpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
        setpz(y[0x7], subpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));

        setpz(y[0x8], subpz(a08p1a4c_p1_a2ap1a6e,    a19p1a5d_p1_a3bp1a7f));
        setpz(y[0x9], subpz(s08pjs4c_pv_s2apjs6e, hf_s19pjs5d_pv_s3bpjs7f));
        setpz(y[0xa], subpz(a08m1a4c_pj_a2am1a6e, v8_a19m1a5d_pj_a3bm1a7f));
        setpz(y[0xb], subpz(s08mjs4c_mw_s2amjs6e, hd_s19mjs5d_mw_s3bmjs7f));
        setpz(y[0xc], subpz(a08p1a4c_m1_a2ap1a6e,  j_a19p1a5d_m1_a3bp1a7f));
        setpz(y[0xd], addpz(s08pjs4c_mv_s2apjs6e, h3_s19pjs5d_mv_s3bpjs7f));
        setpz(y[0xe], addpz(a08m1a4c_mj_a2am1a6e, w8_a19m1a5d_mj_a3bm1a7f));
        setpz(y[0xf], addpz(s08mjs4c_pw_s2amjs6e, h1_s19mjs5d_pw_s3bmjs7f));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct invnend<16,s,0>
{
    static const int N = 16*s;

    void operator()(complex_vector x, complex_vector) const noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
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
        zeroupper();
        static const xmm rN = { 1.0/16, 1.0/16 };
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
        inv0fft<n/16,16*s,!eo>()(y, x, W);
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
        OTFFT_AVXDIF8::inv0end<8,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct inv0fft<4,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIF4::inv0end<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct inv0fft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIF4::inv0end<2,s,eo>()(x, y);
    }
};

//-----------------------------------------------------------------------------

template <int n, int s, bool eo> struct invnfft
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector W) const noexcept
    {
        invcore<n,s>()(x, y, W);
        invnfft<n/16,16*s,!eo>()(y, x, W);
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
        OTFFT_AVXDIF8::invnend<8,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct invnfft<4,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIF4::invnend<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct invnfft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const noexcept
    {
        OTFFT_AVXDIF4::invnend<2,s,eo>()(x, y);
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
        else OTFFT_AVXDIF16omp::fwd(log_N, x, y, W);
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
        else OTFFT_AVXDIF16omp::fwd0(log_N, x, y, W);
    }

    inline void fwdn(complex_vector x, complex_vector y) const noexcept { fwd(x, y); }

    inline void fwd0o(complex_vector x, complex_vector y) const noexcept
    {
        if (N < OMP_THRESHOLD) {
            switch (log_N) {
                case  0: y[0] = x[0]; break;
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
        else OTFFT_AVXDIF16omp::fwd0o(log_N, x, y, W);
    }

    inline void fwdno(complex_vector x, complex_vector y) const noexcept
    {
        if (N < OMP_THRESHOLD) {
            switch (log_N) {
                case  0: y[0] = x[0]; break;
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
        else OTFFT_AVXDIF16omp::fwdno(log_N, x, y, W);
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
        else OTFFT_AVXDIF16omp::inv(log_N, x, y, W);
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
        else OTFFT_AVXDIF16omp::invn(log_N, x, y, W);
    }

    inline void inv0o(complex_vector x, complex_vector y) const noexcept
    {
        if (N < OMP_THRESHOLD) {
            switch (log_N) {
                case  0: y[0] = x[0]; break;
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
        else OTFFT_AVXDIF16omp::inv0o(log_N, x, y, W);
    }

    inline void invno(complex_vector x, complex_vector y) const noexcept
    {
        if (N < OMP_THRESHOLD) {
            switch (log_N) {
                case  0: y[0] = x[0]; break;
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
        else OTFFT_AVXDIF16omp::invno(log_N, x, y, W);
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

#endif // otfft_avxdif16_h
