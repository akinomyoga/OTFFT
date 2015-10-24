/******************************************************************************
*  OTFFT AVXDIT8 of OpenMP Version 5.3
******************************************************************************/

#ifndef otfft_avxdit8omp_h
#define otfft_avxdit8omp_h

//#include "otfft/otfft_misc.h"
//#include "otfft_avxdit4.h"

namespace OTFFT_AVXDIT8omp { //////////////////////////////////////////////////

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
    static const int N2 = N/4;
    static const int N3 = N1 + N2;
    static const int N4 = N/2;
    static const int N5 = N1 + N4;
    static const int N6 = N3 + N3;
    static const int N7 = N3 + N4;

    void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const
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
            const ymm a =             getpz2(yq_s8p+s*0);
            const ymm b = mulpz2(w1p, getpz2(yq_s8p+s*1));
            const ymm c = mulpz2(w2p, getpz2(yq_s8p+s*2));
            const ymm d = mulpz2(w3p, getpz2(yq_s8p+s*3));
            const ymm e = mulpz2(w4p, getpz2(yq_s8p+s*4));
            const ymm f = mulpz2(w5p, getpz2(yq_s8p+s*5));
            const ymm g = mulpz2(w6p, getpz2(yq_s8p+s*6));
            const ymm h = mulpz2(w7p, getpz2(yq_s8p+s*7));
            const ymm jc = jxpz2(c);
            const ymm jd = jxpz2(d);
            const ymm jg = jxpz2(g);
            const ymm jh = jxpz2(h);
            const ymm ap1c = addpz2(a,  c);
            const ymm amjc = subpz2(a, jc);
            const ymm am1c = subpz2(a,  c);
            const ymm apjc = addpz2(a, jc);
            const ymm bp1d = addpz2(b,  d);
            const ymm bmjd = subpz2(b, jd);
            const ymm bm1d = subpz2(b,  d);
            const ymm bpjd = addpz2(b, jd);
            const ymm ep1g = addpz2(e,  g);
            const ymm emjg = subpz2(e, jg);
            const ymm em1g = subpz2(e,  g);
            const ymm epjg = addpz2(e, jg);
            const ymm fp1h = addpz2(f,  h);
            const ymm fmjh = subpz2(f, jh);
            const ymm fm1h = subpz2(f,  h);
            const ymm fpjh = addpz2(f, jh);
            const ymm   ap1c_p_ep1g =        addpz2(ap1c, ep1g);
            const ymm   bp1d_p_fp1h =        addpz2(bp1d, fp1h);
            const ymm   amjc_m_emjg =        subpz2(amjc, emjg);
            const ymm w8bmjd_m_fmjh = w8xpz2(subpz2(bmjd, fmjh));
            const ymm   am1c_p_em1g =        addpz2(am1c, em1g);
            const ymm jxbm1d_p_fm1h =  jxpz2(addpz2(bm1d, fm1h));
            const ymm   apjc_m_epjg =        subpz2(apjc, epjg);
            const ymm v8bpjd_m_fpjh = v8xpz2(subpz2(bpjd, fpjh));
            setpz2(xq_sp+N0, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq_sp+N1, addpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(xq_sp+N2, subpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq_sp+N3, subpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(xq_sp+N4, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq_sp+N5, subpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(xq_sp+N6, addpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq_sp+N7, addpz2(apjc_m_epjg, v8bpjd_m_fpjh));
        }
    }
};

template <int N> struct fwdcore<N,1>
{
    static const int N0 = 0;
    static const int N1 = N/8;
    static const int N2 = N/4;
    static const int N3 = N1 + N2;
    static const int N4 = N/2;
    static const int N5 = N1 + N4;
    static const int N6 = N3 + N3;
    static const int N7 = N3 + N4;

    void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const
    {
#ifdef _OPENMP
        #pragma omp for schedule(static) nowait
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
#if 0
            const ymm a =             getpz3<8>(y_8p+0);
            const ymm b = mulpz2(w1p, getpz3<8>(y_8p+1));
            const ymm c = mulpz2(w2p, getpz3<8>(y_8p+2));
            const ymm d = mulpz2(w3p, getpz3<8>(y_8p+3));
            const ymm e = mulpz2(w4p, getpz3<8>(y_8p+4));
            const ymm f = mulpz2(w5p, getpz3<8>(y_8p+5));
            const ymm g = mulpz2(w6p, getpz3<8>(y_8p+6));
            const ymm h = mulpz2(w7p, getpz3<8>(y_8p+7));
#else
            const ymm ab = getpz2(y_8p+ 0);
            const ymm cd = getpz2(y_8p+ 2);
            const ymm ef = getpz2(y_8p+ 4);
            const ymm gh = getpz2(y_8p+ 6);
            const ymm ij = getpz2(y_8p+ 8);
            const ymm kl = getpz2(y_8p+10);
            const ymm mn = getpz2(y_8p+12);
            const ymm op = getpz2(y_8p+14);
            const ymm a =             catlo(ab, ij);
            const ymm b = mulpz2(w1p, cathi(ab, ij));
            const ymm c = mulpz2(w2p, catlo(cd, kl));
            const ymm d = mulpz2(w3p, cathi(cd, kl));
            const ymm e = mulpz2(w4p, catlo(ef, mn));
            const ymm f = mulpz2(w5p, cathi(ef, mn));
            const ymm g = mulpz2(w6p, catlo(gh, op));
            const ymm h = mulpz2(w7p, cathi(gh, op));
#endif
            const ymm jc = jxpz2(c);
            const ymm jd = jxpz2(d);
            const ymm jg = jxpz2(g);
            const ymm jh = jxpz2(h);
            const ymm ap1c = addpz2(a,  c);
            const ymm amjc = subpz2(a, jc);
            const ymm am1c = subpz2(a,  c);
            const ymm apjc = addpz2(a, jc);
            const ymm bp1d = addpz2(b,  d);
            const ymm bmjd = subpz2(b, jd);
            const ymm bm1d = subpz2(b,  d);
            const ymm bpjd = addpz2(b, jd);
            const ymm ep1g = addpz2(e,  g);
            const ymm emjg = subpz2(e, jg);
            const ymm em1g = subpz2(e,  g);
            const ymm epjg = addpz2(e, jg);
            const ymm fp1h = addpz2(f,  h);
            const ymm fmjh = subpz2(f, jh);
            const ymm fm1h = subpz2(f,  h);
            const ymm fpjh = addpz2(f, jh);
            const ymm   ap1c_p_ep1g =        addpz2(ap1c, ep1g);
            const ymm   bp1d_p_fp1h =        addpz2(bp1d, fp1h);
            const ymm   amjc_m_emjg =        subpz2(amjc, emjg);
            const ymm w8bmjd_m_fmjh = w8xpz2(subpz2(bmjd, fmjh));
            const ymm   am1c_p_em1g =        addpz2(am1c, em1g);
            const ymm jxbm1d_p_fm1h =  jxpz2(addpz2(bm1d, fm1h));
            const ymm   apjc_m_epjg =        subpz2(apjc, epjg);
            const ymm v8bpjd_m_fpjh = v8xpz2(subpz2(bpjd, fpjh));
            setpz2(x_p+N0, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(x_p+N1, addpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(x_p+N2, subpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(x_p+N3, subpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(x_p+N4, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(x_p+N5, subpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(x_p+N6, addpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(x_p+N7, addpz2(apjc_m_epjg, v8bpjd_m_fpjh));
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct fwd0end;

//-----------------------------------------------------------------------------

template <int s> struct fwd0end<8,s,1>
{
    void operator()(complex_vector x, complex_vector y) const
    {
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
            const ymm a = getpz2(yq+s*0);
            const ymm b = getpz2(yq+s*1);
            const ymm c = getpz2(yq+s*2);
            const ymm d = getpz2(yq+s*3);
            const ymm e = getpz2(yq+s*4);
            const ymm f = getpz2(yq+s*5);
            const ymm g = getpz2(yq+s*6);
            const ymm h = getpz2(yq+s*7);
            const ymm jc = jxpz2(c);
            const ymm jd = jxpz2(d);
            const ymm jg = jxpz2(g);
            const ymm jh = jxpz2(h);
            const ymm ap1c = addpz2(a,  c);
            const ymm amjc = subpz2(a, jc);
            const ymm am1c = subpz2(a,  c);
            const ymm apjc = addpz2(a, jc);
            const ymm bp1d = addpz2(b,  d);
            const ymm bmjd = subpz2(b, jd);
            const ymm bm1d = subpz2(b,  d);
            const ymm bpjd = addpz2(b, jd);
            const ymm ep1g = addpz2(e,  g);
            const ymm emjg = subpz2(e, jg);
            const ymm em1g = subpz2(e,  g);
            const ymm epjg = addpz2(e, jg);
            const ymm fp1h = addpz2(f,  h);
            const ymm fmjh = subpz2(f, jh);
            const ymm fm1h = subpz2(f,  h);
            const ymm fpjh = addpz2(f, jh);
            const ymm   ap1c_p_ep1g =        addpz2(ap1c, ep1g);
            const ymm   bp1d_p_fp1h =        addpz2(bp1d, fp1h);
            const ymm   amjc_m_emjg =        subpz2(amjc, emjg);
            const ymm w8bmjd_m_fmjh = w8xpz2(subpz2(bmjd, fmjh));
            const ymm   am1c_p_em1g =        addpz2(am1c, em1g);
            const ymm jxbm1d_p_fm1h =  jxpz2(addpz2(bm1d, fm1h));
            const ymm   apjc_m_epjg =        subpz2(apjc, epjg);
            const ymm v8bpjd_m_fpjh = v8xpz2(subpz2(bpjd, fpjh));
            setpz2(xq+s*0, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq+s*1, addpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(xq+s*2, subpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq+s*3, subpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(xq+s*4, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq+s*5, subpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(xq+s*6, addpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq+s*7, addpz2(apjc_m_epjg, v8bpjd_m_fpjh));
        }
    }
};

template <> struct fwd0end<8,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const
    {
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
            const xmm a = getpz(y[0]);
            const xmm b = getpz(y[1]);
            const xmm c = getpz(y[2]);
            const xmm d = getpz(y[3]);
            const xmm e = getpz(y[4]);
            const xmm f = getpz(y[5]);
            const xmm g = getpz(y[6]);
            const xmm h = getpz(y[7]);
            const xmm jc = jxpz(c);
            const xmm jd = jxpz(d);
            const xmm jg = jxpz(g);
            const xmm jh = jxpz(h);
            const xmm ap1c = addpz(a,  c);
            const xmm amjc = subpz(a, jc);
            const xmm am1c = subpz(a,  c);
            const xmm apjc = addpz(a, jc);
            const xmm bp1d = addpz(b,  d);
            const xmm bmjd = subpz(b, jd);
            const xmm bm1d = subpz(b,  d);
            const xmm bpjd = addpz(b, jd);
            const xmm ep1g = addpz(e,  g);
            const xmm emjg = subpz(e, jg);
            const xmm em1g = subpz(e,  g);
            const xmm epjg = addpz(e, jg);
            const xmm fp1h = addpz(f,  h);
            const xmm fmjh = subpz(f, jh);
            const xmm fm1h = subpz(f,  h);
            const xmm fpjh = addpz(f, jh);
            const xmm   ap1c_p_ep1g =       addpz(ap1c, ep1g);
            const xmm   bp1d_p_fp1h =       addpz(bp1d, fp1h);
            const xmm   amjc_m_emjg =       subpz(amjc, emjg);
            const xmm w8bmjd_m_fmjh = w8xpz(subpz(bmjd, fmjh));
            const xmm   am1c_p_em1g =       addpz(am1c, em1g);
            const xmm jxbm1d_p_fm1h =  jxpz(addpz(bm1d, fm1h));
            const xmm   apjc_m_epjg =       subpz(apjc, epjg);
            const xmm v8bpjd_m_fpjh = v8xpz(subpz(bpjd, fpjh));
            setpz(x[0], addpz(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz(x[1], addpz(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz(x[2], subpz(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz(x[3], subpz(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz(x[4], subpz(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz(x[5], subpz(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz(x[6], addpz(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz(x[7], addpz(apjc_m_epjg, v8bpjd_m_fpjh));
        }
    }
};

//-----------------------------------------------------------------------------

template <int s> struct fwd0end<8,s,0>
{
    void operator()(complex_vector x, complex_vector) const
    {
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            const ymm a = getpz2(xq+s*0);
            const ymm b = getpz2(xq+s*1);
            const ymm c = getpz2(xq+s*2);
            const ymm d = getpz2(xq+s*3);
            const ymm e = getpz2(xq+s*4);
            const ymm f = getpz2(xq+s*5);
            const ymm g = getpz2(xq+s*6);
            const ymm h = getpz2(xq+s*7);
            const ymm jc = jxpz2(c);
            const ymm jd = jxpz2(d);
            const ymm jg = jxpz2(g);
            const ymm jh = jxpz2(h);
            const ymm ap1c = addpz2(a,  c);
            const ymm amjc = subpz2(a, jc);
            const ymm am1c = subpz2(a,  c);
            const ymm apjc = addpz2(a, jc);
            const ymm bp1d = addpz2(b,  d);
            const ymm bmjd = subpz2(b, jd);
            const ymm bm1d = subpz2(b,  d);
            const ymm bpjd = addpz2(b, jd);
            const ymm ep1g = addpz2(e,  g);
            const ymm emjg = subpz2(e, jg);
            const ymm em1g = subpz2(e,  g);
            const ymm epjg = addpz2(e, jg);
            const ymm fp1h = addpz2(f,  h);
            const ymm fmjh = subpz2(f, jh);
            const ymm fm1h = subpz2(f,  h);
            const ymm fpjh = addpz2(f, jh);
            const ymm   ap1c_p_ep1g =        addpz2(ap1c, ep1g);
            const ymm   bp1d_p_fp1h =        addpz2(bp1d, fp1h);
            const ymm   amjc_m_emjg =        subpz2(amjc, emjg);
            const ymm w8bmjd_m_fmjh = w8xpz2(subpz2(bmjd, fmjh));
            const ymm   am1c_p_em1g =        addpz2(am1c, em1g);
            const ymm jxbm1d_p_fm1h =  jxpz2(addpz2(bm1d, fm1h));
            const ymm   apjc_m_epjg =        subpz2(apjc, epjg);
            const ymm v8bpjd_m_fpjh = v8xpz2(subpz2(bpjd, fpjh));
            setpz2(xq+s*0, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq+s*1, addpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(xq+s*2, subpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq+s*3, subpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(xq+s*4, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq+s*5, subpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(xq+s*6, addpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq+s*7, addpz2(apjc_m_epjg, v8bpjd_m_fpjh));
        }
    }
};

template <> struct fwd0end<8,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const
    {
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
            const xmm a = getpz(x[0]);
            const xmm b = getpz(x[1]);
            const xmm c = getpz(x[2]);
            const xmm d = getpz(x[3]);
            const xmm e = getpz(x[4]);
            const xmm f = getpz(x[5]);
            const xmm g = getpz(x[6]);
            const xmm h = getpz(x[7]);
            const xmm jc = jxpz(c);
            const xmm jd = jxpz(d);
            const xmm jg = jxpz(g);
            const xmm jh = jxpz(h);
            const xmm ap1c = addpz(a,  c);
            const xmm amjc = subpz(a, jc);
            const xmm am1c = subpz(a,  c);
            const xmm apjc = addpz(a, jc);
            const xmm bp1d = addpz(b,  d);
            const xmm bmjd = subpz(b, jd);
            const xmm bm1d = subpz(b,  d);
            const xmm bpjd = addpz(b, jd);
            const xmm ep1g = addpz(e,  g);
            const xmm emjg = subpz(e, jg);
            const xmm em1g = subpz(e,  g);
            const xmm epjg = addpz(e, jg);
            const xmm fp1h = addpz(f,  h);
            const xmm fmjh = subpz(f, jh);
            const xmm fm1h = subpz(f,  h);
            const xmm fpjh = addpz(f, jh);
            const xmm   ap1c_p_ep1g =       addpz(ap1c, ep1g);
            const xmm   bp1d_p_fp1h =       addpz(bp1d, fp1h);
            const xmm   amjc_m_emjg =       subpz(amjc, emjg);
            const xmm w8bmjd_m_fmjh = w8xpz(subpz(bmjd, fmjh));
            const xmm   am1c_p_em1g =       addpz(am1c, em1g);
            const xmm jxbm1d_p_fm1h =  jxpz(addpz(bm1d, fm1h));
            const xmm   apjc_m_epjg =       subpz(apjc, epjg);
            const xmm v8bpjd_m_fpjh = v8xpz(subpz(bpjd, fpjh));
            setpz(x[0], addpz(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz(x[1], addpz(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz(x[2], subpz(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz(x[3], subpz(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz(x[4], subpz(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz(x[5], subpz(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz(x[6], addpz(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz(x[7], addpz(apjc_m_epjg, v8bpjd_m_fpjh));
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct fwdnend;

//-----------------------------------------------------------------------------

template <int s> struct fwdnend<8,s,1>
{
    static const int N = 8*s;

    void operator()(complex_vector x, complex_vector y) const
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
            const ymm a = mulpd2(rN, getpz2(yq+s*0));
            const ymm b = mulpd2(rN, getpz2(yq+s*1));
            const ymm c = mulpd2(rN, getpz2(yq+s*2));
            const ymm d = mulpd2(rN, getpz2(yq+s*3));
            const ymm e = mulpd2(rN, getpz2(yq+s*4));
            const ymm f = mulpd2(rN, getpz2(yq+s*5));
            const ymm g = mulpd2(rN, getpz2(yq+s*6));
            const ymm h = mulpd2(rN, getpz2(yq+s*7));
            const ymm jc = jxpz2(c);
            const ymm jd = jxpz2(d);
            const ymm jg = jxpz2(g);
            const ymm jh = jxpz2(h);
            const ymm ap1c = addpz2(a,  c);
            const ymm amjc = subpz2(a, jc);
            const ymm am1c = subpz2(a,  c);
            const ymm apjc = addpz2(a, jc);
            const ymm bp1d = addpz2(b,  d);
            const ymm bmjd = subpz2(b, jd);
            const ymm bm1d = subpz2(b,  d);
            const ymm bpjd = addpz2(b, jd);
            const ymm ep1g = addpz2(e,  g);
            const ymm emjg = subpz2(e, jg);
            const ymm em1g = subpz2(e,  g);
            const ymm epjg = addpz2(e, jg);
            const ymm fp1h = addpz2(f,  h);
            const ymm fmjh = subpz2(f, jh);
            const ymm fm1h = subpz2(f,  h);
            const ymm fpjh = addpz2(f, jh);
            const ymm   ap1c_p_ep1g =        addpz2(ap1c, ep1g);
            const ymm   bp1d_p_fp1h =        addpz2(bp1d, fp1h);
            const ymm   amjc_m_emjg =        subpz2(amjc, emjg);
            const ymm w8bmjd_m_fmjh = w8xpz2(subpz2(bmjd, fmjh));
            const ymm   am1c_p_em1g =        addpz2(am1c, em1g);
            const ymm jxbm1d_p_fm1h =  jxpz2(addpz2(bm1d, fm1h));
            const ymm   apjc_m_epjg =        subpz2(apjc, epjg);
            const ymm v8bpjd_m_fpjh = v8xpz2(subpz2(bpjd, fpjh));
            setpz2(xq+s*0, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq+s*1, addpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(xq+s*2, subpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq+s*3, subpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(xq+s*4, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq+s*5, subpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(xq+s*6, addpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq+s*7, addpz2(apjc_m_epjg, v8bpjd_m_fpjh));
        }
    }
};

template <> struct fwdnend<8,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const
    {
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
            static const xmm rN = { 1.0/8, 1.0/8 };
            const xmm a = mulpd(rN, getpz(y[0]));
            const xmm b = mulpd(rN, getpz(y[1]));
            const xmm c = mulpd(rN, getpz(y[2]));
            const xmm d = mulpd(rN, getpz(y[3]));
            const xmm e = mulpd(rN, getpz(y[4]));
            const xmm f = mulpd(rN, getpz(y[5]));
            const xmm g = mulpd(rN, getpz(y[6]));
            const xmm h = mulpd(rN, getpz(y[7]));
            const xmm jc = jxpz(c);
            const xmm jd = jxpz(d);
            const xmm jg = jxpz(g);
            const xmm jh = jxpz(h);
            const xmm ap1c = addpz(a,  c);
            const xmm amjc = subpz(a, jc);
            const xmm am1c = subpz(a,  c);
            const xmm apjc = addpz(a, jc);
            const xmm bp1d = addpz(b,  d);
            const xmm bmjd = subpz(b, jd);
            const xmm bm1d = subpz(b,  d);
            const xmm bpjd = addpz(b, jd);
            const xmm ep1g = addpz(e,  g);
            const xmm emjg = subpz(e, jg);
            const xmm em1g = subpz(e,  g);
            const xmm epjg = addpz(e, jg);
            const xmm fp1h = addpz(f,  h);
            const xmm fmjh = subpz(f, jh);
            const xmm fm1h = subpz(f,  h);
            const xmm fpjh = addpz(f, jh);
            const xmm   ap1c_p_ep1g =       addpz(ap1c, ep1g);
            const xmm   bp1d_p_fp1h =       addpz(bp1d, fp1h);
            const xmm   amjc_m_emjg =       subpz(amjc, emjg);
            const xmm w8bmjd_m_fmjh = w8xpz(subpz(bmjd, fmjh));
            const xmm   am1c_p_em1g =       addpz(am1c, em1g);
            const xmm jxbm1d_p_fm1h =  jxpz(addpz(bm1d, fm1h));
            const xmm   apjc_m_epjg =       subpz(apjc, epjg);
            const xmm v8bpjd_m_fpjh = v8xpz(subpz(bpjd, fpjh));
            setpz(x[0], addpz(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz(x[1], addpz(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz(x[2], subpz(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz(x[3], subpz(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz(x[4], subpz(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz(x[5], subpz(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz(x[6], addpz(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz(x[7], addpz(apjc_m_epjg, v8bpjd_m_fpjh));
        }
    }
};

//-----------------------------------------------------------------------------

template <int s> struct fwdnend<8,s,0>
{
    static const int N = 8*s;

    void operator()(complex_vector x, complex_vector) const
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            const ymm a = mulpd2(rN, getpz2(xq+s*0));
            const ymm b = mulpd2(rN, getpz2(xq+s*1));
            const ymm c = mulpd2(rN, getpz2(xq+s*2));
            const ymm d = mulpd2(rN, getpz2(xq+s*3));
            const ymm e = mulpd2(rN, getpz2(xq+s*4));
            const ymm f = mulpd2(rN, getpz2(xq+s*5));
            const ymm g = mulpd2(rN, getpz2(xq+s*6));
            const ymm h = mulpd2(rN, getpz2(xq+s*7));
            const ymm jc = jxpz2(c);
            const ymm jd = jxpz2(d);
            const ymm jg = jxpz2(g);
            const ymm jh = jxpz2(h);
            const ymm ap1c = addpz2(a,  c);
            const ymm amjc = subpz2(a, jc);
            const ymm am1c = subpz2(a,  c);
            const ymm apjc = addpz2(a, jc);
            const ymm bp1d = addpz2(b,  d);
            const ymm bmjd = subpz2(b, jd);
            const ymm bm1d = subpz2(b,  d);
            const ymm bpjd = addpz2(b, jd);
            const ymm ep1g = addpz2(e,  g);
            const ymm emjg = subpz2(e, jg);
            const ymm em1g = subpz2(e,  g);
            const ymm epjg = addpz2(e, jg);
            const ymm fp1h = addpz2(f,  h);
            const ymm fmjh = subpz2(f, jh);
            const ymm fm1h = subpz2(f,  h);
            const ymm fpjh = addpz2(f, jh);
            const ymm   ap1c_p_ep1g =        addpz2(ap1c, ep1g);
            const ymm   bp1d_p_fp1h =        addpz2(bp1d, fp1h);
            const ymm   amjc_m_emjg =        subpz2(amjc, emjg);
            const ymm w8bmjd_m_fmjh = w8xpz2(subpz2(bmjd, fmjh));
            const ymm   am1c_p_em1g =        addpz2(am1c, em1g);
            const ymm jxbm1d_p_fm1h =  jxpz2(addpz2(bm1d, fm1h));
            const ymm   apjc_m_epjg =        subpz2(apjc, epjg);
            const ymm v8bpjd_m_fpjh = v8xpz2(subpz2(bpjd, fpjh));
            setpz2(xq+s*0, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq+s*1, addpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(xq+s*2, subpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq+s*3, subpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(xq+s*4, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq+s*5, subpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(xq+s*6, addpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq+s*7, addpz2(apjc_m_epjg, v8bpjd_m_fpjh));
        }
    }
};

template <> struct fwdnend<8,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const
    {
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
            static const xmm rN = { 1.0/8, 1.0/8 };
            const xmm a = mulpd(rN, getpz(x[0]));
            const xmm b = mulpd(rN, getpz(x[1]));
            const xmm c = mulpd(rN, getpz(x[2]));
            const xmm d = mulpd(rN, getpz(x[3]));
            const xmm e = mulpd(rN, getpz(x[4]));
            const xmm f = mulpd(rN, getpz(x[5]));
            const xmm g = mulpd(rN, getpz(x[6]));
            const xmm h = mulpd(rN, getpz(x[7]));
            const xmm jc = jxpz(c);
            const xmm jd = jxpz(d);
            const xmm jg = jxpz(g);
            const xmm jh = jxpz(h);
            const xmm ap1c = addpz(a,  c);
            const xmm amjc = subpz(a, jc);
            const xmm am1c = subpz(a,  c);
            const xmm apjc = addpz(a, jc);
            const xmm bp1d = addpz(b,  d);
            const xmm bmjd = subpz(b, jd);
            const xmm bm1d = subpz(b,  d);
            const xmm bpjd = addpz(b, jd);
            const xmm ep1g = addpz(e,  g);
            const xmm emjg = subpz(e, jg);
            const xmm em1g = subpz(e,  g);
            const xmm epjg = addpz(e, jg);
            const xmm fp1h = addpz(f,  h);
            const xmm fmjh = subpz(f, jh);
            const xmm fm1h = subpz(f,  h);
            const xmm fpjh = addpz(f, jh);
            const xmm   ap1c_p_ep1g =       addpz(ap1c, ep1g);
            const xmm   bp1d_p_fp1h =       addpz(bp1d, fp1h);
            const xmm   amjc_m_emjg =       subpz(amjc, emjg);
            const xmm w8bmjd_m_fmjh = w8xpz(subpz(bmjd, fmjh));
            const xmm   am1c_p_em1g =       addpz(am1c, em1g);
            const xmm jxbm1d_p_fm1h =  jxpz(addpz(bm1d, fm1h));
            const xmm   apjc_m_epjg =       subpz(apjc, epjg);
            const xmm v8bpjd_m_fpjh = v8xpz(subpz(bpjd, fpjh));
            setpz(x[0], addpz(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz(x[1], addpz(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz(x[2], subpz(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz(x[3], subpz(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz(x[4], subpz(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz(x[5], subpz(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz(x[6], addpz(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz(x[7], addpz(apjc_m_epjg, v8bpjd_m_fpjh));
        }
    }
};

///////////////////////////////////////////////////////////////////////////////
// Forward FFT
///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct fwd0fft
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector W) const
    {
        fwd0fft<n/8,8*s,!eo>()(y, x, W);
        fwdcore<n,s>()(x, y, W);
    }
};

template <int s, bool eo> struct fwd0fft<8,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const
    {
        fwd0end<8,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwd0fft<4,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const
    {
        OTFFT_AVXDIT4omp::fwd0end<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwd0fft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const
    {
        OTFFT_AVXDIT4omp::fwd0end<2,s,eo>()(x, y);
    }
};

//-----------------------------------------------------------------------------

template <int n, int s, bool eo> struct fwdnfft
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector W) const
    {
        fwdnfft<n/8,8*s,!eo>()(y, x, W);
        fwdcore<n,s>()(x, y, W);
    }
};

template <int s, bool eo> struct fwdnfft<8,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const
    {
        fwdnend<8,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwdnfft<4,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const
    {
        OTFFT_AVXDIT4omp::fwdnend<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwdnfft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const
    {
        OTFFT_AVXDIT4omp::fwdnend<2,s,eo>()(x, y);
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
    static const int N2 = N/4;
    static const int N3 = N1 + N2;
    static const int N4 = N/2;
    static const int N5 = N1 + N4;
    static const int N6 = N3 + N3;
    static const int N7 = N3 + N4;

    void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const
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
            const ymm a =             getpz2(yq_s8p+s*0);
            const ymm b = mulpz2(w1p, getpz2(yq_s8p+s*1));
            const ymm c = mulpz2(w2p, getpz2(yq_s8p+s*2));
            const ymm d = mulpz2(w3p, getpz2(yq_s8p+s*3));
            const ymm e = mulpz2(w4p, getpz2(yq_s8p+s*4));
            const ymm f = mulpz2(w5p, getpz2(yq_s8p+s*5));
            const ymm g = mulpz2(w6p, getpz2(yq_s8p+s*6));
            const ymm h = mulpz2(w7p, getpz2(yq_s8p+s*7));
            const ymm jc = jxpz2(c);
            const ymm jd = jxpz2(d);
            const ymm jg = jxpz2(g);
            const ymm jh = jxpz2(h);
            const ymm ap1c = addpz2(a,  c);
            const ymm amjc = subpz2(a, jc);
            const ymm am1c = subpz2(a,  c);
            const ymm apjc = addpz2(a, jc);
            const ymm bp1d = addpz2(b,  d);
            const ymm bmjd = subpz2(b, jd);
            const ymm bm1d = subpz2(b,  d);
            const ymm bpjd = addpz2(b, jd);
            const ymm ep1g = addpz2(e,  g);
            const ymm emjg = subpz2(e, jg);
            const ymm em1g = subpz2(e,  g);
            const ymm epjg = addpz2(e, jg);
            const ymm fp1h = addpz2(f,  h);
            const ymm fmjh = subpz2(f, jh);
            const ymm fm1h = subpz2(f,  h);
            const ymm fpjh = addpz2(f, jh);
            const ymm   ap1c_p_ep1g =        addpz2(ap1c, ep1g);
            const ymm   bp1d_p_fp1h =        addpz2(bp1d, fp1h);
            const ymm   amjc_m_emjg =        subpz2(amjc, emjg);
            const ymm w8bmjd_m_fmjh = w8xpz2(subpz2(bmjd, fmjh));
            const ymm   am1c_p_em1g =        addpz2(am1c, em1g);
            const ymm jxbm1d_p_fm1h =  jxpz2(addpz2(bm1d, fm1h));
            const ymm   apjc_m_epjg =        subpz2(apjc, epjg);
            const ymm v8bpjd_m_fpjh = v8xpz2(subpz2(bpjd, fpjh));
            setpz2(xq_sp+N0, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq_sp+N1, addpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(xq_sp+N2, addpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq_sp+N3, subpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(xq_sp+N4, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq_sp+N5, subpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(xq_sp+N6, subpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq_sp+N7, addpz2(amjc_m_emjg, w8bmjd_m_fmjh));
        }
    }
};

template <int N> struct invcore<N,1>
{
    static const int N0 = 0;
    static const int N1 = N/8;
    static const int N2 = N/4;
    static const int N3 = N1 + N2;
    static const int N4 = N/2;
    static const int N5 = N1 + N4;
    static const int N6 = N3 + N3;
    static const int N7 = N3 + N4;

    void operator()(
            complex_vector x, complex_vector y, const_complex_vector W) const
    {
        //const_complex_vector WN = W + N;
#ifdef _OPENMP
        #pragma omp for schedule(static) nowait
#endif
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
            const ymm a =             getpz3<8>(y_8p+0);
            const ymm b = mulpz2(w1p, getpz3<8>(y_8p+1));
            const ymm c = mulpz2(w2p, getpz3<8>(y_8p+2));
            const ymm d = mulpz2(w3p, getpz3<8>(y_8p+3));
            const ymm e = mulpz2(w4p, getpz3<8>(y_8p+4));
            const ymm f = mulpz2(w5p, getpz3<8>(y_8p+5));
            const ymm g = mulpz2(w6p, getpz3<8>(y_8p+6));
            const ymm h = mulpz2(w7p, getpz3<8>(y_8p+7));
#else
            const ymm ab = getpz2(y_8p+ 0);
            const ymm cd = getpz2(y_8p+ 2);
            const ymm ef = getpz2(y_8p+ 4);
            const ymm gh = getpz2(y_8p+ 6);
            const ymm ij = getpz2(y_8p+ 8);
            const ymm kl = getpz2(y_8p+10);
            const ymm mn = getpz2(y_8p+12);
            const ymm op = getpz2(y_8p+14);
            const ymm a =             catlo(ab, ij);
            const ymm b = mulpz2(w1p, cathi(ab, ij));
            const ymm c = mulpz2(w2p, catlo(cd, kl));
            const ymm d = mulpz2(w3p, cathi(cd, kl));
            const ymm e = mulpz2(w4p, catlo(ef, mn));
            const ymm f = mulpz2(w5p, cathi(ef, mn));
            const ymm g = mulpz2(w6p, catlo(gh, op));
            const ymm h = mulpz2(w7p, cathi(gh, op));
#endif
            const ymm jc = jxpz2(c);
            const ymm jd = jxpz2(d);
            const ymm jg = jxpz2(g);
            const ymm jh = jxpz2(h);
            const ymm ap1c = addpz2(a,  c);
            const ymm amjc = subpz2(a, jc);
            const ymm am1c = subpz2(a,  c);
            const ymm apjc = addpz2(a, jc);
            const ymm bp1d = addpz2(b,  d);
            const ymm bmjd = subpz2(b, jd);
            const ymm bm1d = subpz2(b,  d);
            const ymm bpjd = addpz2(b, jd);
            const ymm ep1g = addpz2(e,  g);
            const ymm emjg = subpz2(e, jg);
            const ymm em1g = subpz2(e,  g);
            const ymm epjg = addpz2(e, jg);
            const ymm fp1h = addpz2(f,  h);
            const ymm fmjh = subpz2(f, jh);
            const ymm fm1h = subpz2(f,  h);
            const ymm fpjh = addpz2(f, jh);
            const ymm   ap1c_p_ep1g =        addpz2(ap1c, ep1g);
            const ymm   bp1d_p_fp1h =        addpz2(bp1d, fp1h);
            const ymm   amjc_m_emjg =        subpz2(amjc, emjg);
            const ymm w8bmjd_m_fmjh = w8xpz2(subpz2(bmjd, fmjh));
            const ymm   am1c_p_em1g =        addpz2(am1c, em1g);
            const ymm jxbm1d_p_fm1h =  jxpz2(addpz2(bm1d, fm1h));
            const ymm   apjc_m_epjg =        subpz2(apjc, epjg);
            const ymm v8bpjd_m_fpjh = v8xpz2(subpz2(bpjd, fpjh));
            setpz2(x_p+N0, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(x_p+N1, addpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(x_p+N2, addpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(x_p+N3, subpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(x_p+N4, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(x_p+N5, subpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(x_p+N6, subpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(x_p+N7, addpz2(amjc_m_emjg, w8bmjd_m_fmjh));
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct inv0end;

//-----------------------------------------------------------------------------

template <int s> struct inv0end<8,s,1>
{
    void operator()(complex_vector x, complex_vector y) const
    {
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
            const ymm a = getpz2(yq+s*0);
            const ymm b = getpz2(yq+s*1);
            const ymm c = getpz2(yq+s*2);
            const ymm d = getpz2(yq+s*3);
            const ymm e = getpz2(yq+s*4);
            const ymm f = getpz2(yq+s*5);
            const ymm g = getpz2(yq+s*6);
            const ymm h = getpz2(yq+s*7);
            const ymm jc = jxpz2(c);
            const ymm jd = jxpz2(d);
            const ymm jg = jxpz2(g);
            const ymm jh = jxpz2(h);
            const ymm ap1c = addpz2(a,  c);
            const ymm amjc = subpz2(a, jc);
            const ymm am1c = subpz2(a,  c);
            const ymm apjc = addpz2(a, jc);
            const ymm bp1d = addpz2(b,  d);
            const ymm bmjd = subpz2(b, jd);
            const ymm bm1d = subpz2(b,  d);
            const ymm bpjd = addpz2(b, jd);
            const ymm ep1g = addpz2(e,  g);
            const ymm emjg = subpz2(e, jg);
            const ymm em1g = subpz2(e,  g);
            const ymm epjg = addpz2(e, jg);
            const ymm fp1h = addpz2(f,  h);
            const ymm fmjh = subpz2(f, jh);
            const ymm fm1h = subpz2(f,  h);
            const ymm fpjh = addpz2(f, jh);
            const ymm   ap1c_p_ep1g =        addpz2(ap1c, ep1g);
            const ymm   bp1d_p_fp1h =        addpz2(bp1d, fp1h);
            const ymm   amjc_m_emjg =        subpz2(amjc, emjg);
            const ymm w8bmjd_m_fmjh = w8xpz2(subpz2(bmjd, fmjh));
            const ymm   am1c_p_em1g =        addpz2(am1c, em1g);
            const ymm jxbm1d_p_fm1h =  jxpz2(addpz2(bm1d, fm1h));
            const ymm   apjc_m_epjg =        subpz2(apjc, epjg);
            const ymm v8bpjd_m_fpjh = v8xpz2(subpz2(bpjd, fpjh));
            setpz2(xq+s*0, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq+s*1, addpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(xq+s*2, addpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq+s*3, subpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(xq+s*4, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq+s*5, subpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(xq+s*6, subpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq+s*7, addpz2(amjc_m_emjg, w8bmjd_m_fmjh));
        }
    }
};

template <> struct inv0end<8,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const
    {
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
            const xmm a = getpz(y[0]);
            const xmm b = getpz(y[1]);
            const xmm c = getpz(y[2]);
            const xmm d = getpz(y[3]);
            const xmm e = getpz(y[4]);
            const xmm f = getpz(y[5]);
            const xmm g = getpz(y[6]);
            const xmm h = getpz(y[7]);
            const xmm jc = jxpz(c);
            const xmm jd = jxpz(d);
            const xmm jg = jxpz(g);
            const xmm jh = jxpz(h);
            const xmm ap1c = addpz(a,  c);
            const xmm amjc = subpz(a, jc);
            const xmm am1c = subpz(a,  c);
            const xmm apjc = addpz(a, jc);
            const xmm bp1d = addpz(b,  d);
            const xmm bmjd = subpz(b, jd);
            const xmm bm1d = subpz(b,  d);
            const xmm bpjd = addpz(b, jd);
            const xmm ep1g = addpz(e,  g);
            const xmm emjg = subpz(e, jg);
            const xmm em1g = subpz(e,  g);
            const xmm epjg = addpz(e, jg);
            const xmm fp1h = addpz(f,  h);
            const xmm fmjh = subpz(f, jh);
            const xmm fm1h = subpz(f,  h);
            const xmm fpjh = addpz(f, jh);
            const xmm   ap1c_p_ep1g =       addpz(ap1c, ep1g);
            const xmm   bp1d_p_fp1h =       addpz(bp1d, fp1h);
            const xmm   amjc_m_emjg =       subpz(amjc, emjg);
            const xmm w8bmjd_m_fmjh = w8xpz(subpz(bmjd, fmjh));
            const xmm   am1c_p_em1g =       addpz(am1c, em1g);
            const xmm jxbm1d_p_fm1h =  jxpz(addpz(bm1d, fm1h));
            const xmm   apjc_m_epjg =       subpz(apjc, epjg);
            const xmm v8bpjd_m_fpjh = v8xpz(subpz(bpjd, fpjh));
            setpz(x[0], addpz(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz(x[1], addpz(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz(x[2], addpz(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz(x[3], subpz(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz(x[4], subpz(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz(x[5], subpz(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz(x[6], subpz(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz(x[7], addpz(amjc_m_emjg, w8bmjd_m_fmjh));
        }
    }
};

//-----------------------------------------------------------------------------

template <int s> struct inv0end<8,s,0>
{
    void operator()(complex_vector x, complex_vector) const
    {
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            const ymm a = getpz2(xq+s*0);
            const ymm b = getpz2(xq+s*1);
            const ymm c = getpz2(xq+s*2);
            const ymm d = getpz2(xq+s*3);
            const ymm e = getpz2(xq+s*4);
            const ymm f = getpz2(xq+s*5);
            const ymm g = getpz2(xq+s*6);
            const ymm h = getpz2(xq+s*7);
            const ymm jc = jxpz2(c);
            const ymm jd = jxpz2(d);
            const ymm jg = jxpz2(g);
            const ymm jh = jxpz2(h);
            const ymm ap1c = addpz2(a,  c);
            const ymm amjc = subpz2(a, jc);
            const ymm am1c = subpz2(a,  c);
            const ymm apjc = addpz2(a, jc);
            const ymm bp1d = addpz2(b,  d);
            const ymm bmjd = subpz2(b, jd);
            const ymm bm1d = subpz2(b,  d);
            const ymm bpjd = addpz2(b, jd);
            const ymm ep1g = addpz2(e,  g);
            const ymm emjg = subpz2(e, jg);
            const ymm em1g = subpz2(e,  g);
            const ymm epjg = addpz2(e, jg);
            const ymm fp1h = addpz2(f,  h);
            const ymm fmjh = subpz2(f, jh);
            const ymm fm1h = subpz2(f,  h);
            const ymm fpjh = addpz2(f, jh);
            const ymm   ap1c_p_ep1g =        addpz2(ap1c, ep1g);
            const ymm   bp1d_p_fp1h =        addpz2(bp1d, fp1h);
            const ymm   amjc_m_emjg =        subpz2(amjc, emjg);
            const ymm w8bmjd_m_fmjh = w8xpz2(subpz2(bmjd, fmjh));
            const ymm   am1c_p_em1g =        addpz2(am1c, em1g);
            const ymm jxbm1d_p_fm1h =  jxpz2(addpz2(bm1d, fm1h));
            const ymm   apjc_m_epjg =        subpz2(apjc, epjg);
            const ymm v8bpjd_m_fpjh = v8xpz2(subpz2(bpjd, fpjh));
            setpz2(xq+s*0, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq+s*1, addpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(xq+s*2, addpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq+s*3, subpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(xq+s*4, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq+s*5, subpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(xq+s*6, subpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq+s*7, addpz2(amjc_m_emjg, w8bmjd_m_fmjh));
        }
    }
};

template <> struct inv0end<8,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const
    {
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
            const xmm a = getpz(x[0]);
            const xmm b = getpz(x[1]);
            const xmm c = getpz(x[2]);
            const xmm d = getpz(x[3]);
            const xmm e = getpz(x[4]);
            const xmm f = getpz(x[5]);
            const xmm g = getpz(x[6]);
            const xmm h = getpz(x[7]);
            const xmm jc = jxpz(c);
            const xmm jd = jxpz(d);
            const xmm jg = jxpz(g);
            const xmm jh = jxpz(h);
            const xmm ap1c = addpz(a,  c);
            const xmm amjc = subpz(a, jc);
            const xmm am1c = subpz(a,  c);
            const xmm apjc = addpz(a, jc);
            const xmm bp1d = addpz(b,  d);
            const xmm bmjd = subpz(b, jd);
            const xmm bm1d = subpz(b,  d);
            const xmm bpjd = addpz(b, jd);
            const xmm ep1g = addpz(e,  g);
            const xmm emjg = subpz(e, jg);
            const xmm em1g = subpz(e,  g);
            const xmm epjg = addpz(e, jg);
            const xmm fp1h = addpz(f,  h);
            const xmm fmjh = subpz(f, jh);
            const xmm fm1h = subpz(f,  h);
            const xmm fpjh = addpz(f, jh);
            const xmm   ap1c_p_ep1g =       addpz(ap1c, ep1g);
            const xmm   bp1d_p_fp1h =       addpz(bp1d, fp1h);
            const xmm   amjc_m_emjg =       subpz(amjc, emjg);
            const xmm w8bmjd_m_fmjh = w8xpz(subpz(bmjd, fmjh));
            const xmm   am1c_p_em1g =       addpz(am1c, em1g);
            const xmm jxbm1d_p_fm1h =  jxpz(addpz(bm1d, fm1h));
            const xmm   apjc_m_epjg =       subpz(apjc, epjg);
            const xmm v8bpjd_m_fpjh = v8xpz(subpz(bpjd, fpjh));
            setpz(x[0], addpz(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz(x[1], addpz(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz(x[2], addpz(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz(x[3], subpz(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz(x[4], subpz(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz(x[5], subpz(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz(x[6], subpz(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz(x[7], addpz(amjc_m_emjg, w8bmjd_m_fmjh));
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct invnend;

//-----------------------------------------------------------------------------

template <int s> struct invnend<8,s,1>
{
    static const int N  = 8*s;

    void operator()(complex_vector x, complex_vector y) const
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
            const ymm a = mulpd2(rN, getpz2(yq+s*0));
            const ymm b = mulpd2(rN, getpz2(yq+s*1));
            const ymm c = mulpd2(rN, getpz2(yq+s*2));
            const ymm d = mulpd2(rN, getpz2(yq+s*3));
            const ymm e = mulpd2(rN, getpz2(yq+s*4));
            const ymm f = mulpd2(rN, getpz2(yq+s*5));
            const ymm g = mulpd2(rN, getpz2(yq+s*6));
            const ymm h = mulpd2(rN, getpz2(yq+s*7));
            const ymm jc = jxpz2(c);
            const ymm jd = jxpz2(d);
            const ymm jg = jxpz2(g);
            const ymm jh = jxpz2(h);
            const ymm ap1c = addpz2(a,  c);
            const ymm amjc = subpz2(a, jc);
            const ymm am1c = subpz2(a,  c);
            const ymm apjc = addpz2(a, jc);
            const ymm bp1d = addpz2(b,  d);
            const ymm bmjd = subpz2(b, jd);
            const ymm bm1d = subpz2(b,  d);
            const ymm bpjd = addpz2(b, jd);
            const ymm ep1g = addpz2(e,  g);
            const ymm emjg = subpz2(e, jg);
            const ymm em1g = subpz2(e,  g);
            const ymm epjg = addpz2(e, jg);
            const ymm fp1h = addpz2(f,  h);
            const ymm fmjh = subpz2(f, jh);
            const ymm fm1h = subpz2(f,  h);
            const ymm fpjh = addpz2(f, jh);
            const ymm   ap1c_p_ep1g =        addpz2(ap1c, ep1g);
            const ymm   bp1d_p_fp1h =        addpz2(bp1d, fp1h);
            const ymm   amjc_m_emjg =        subpz2(amjc, emjg);
            const ymm w8bmjd_m_fmjh = w8xpz2(subpz2(bmjd, fmjh));
            const ymm   am1c_p_em1g =        addpz2(am1c, em1g);
            const ymm jxbm1d_p_fm1h =  jxpz2(addpz2(bm1d, fm1h));
            const ymm   apjc_m_epjg =        subpz2(apjc, epjg);
            const ymm v8bpjd_m_fpjh = v8xpz2(subpz2(bpjd, fpjh));
            setpz2(xq+s*0, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq+s*1, addpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(xq+s*2, addpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq+s*3, subpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(xq+s*4, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq+s*5, subpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(xq+s*6, subpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq+s*7, addpz2(amjc_m_emjg, w8bmjd_m_fmjh));
        }
    }
};

template <> struct invnend<8,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const
    {
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
            static const xmm rN = { 1.0/8, 1.0/8 };
            const xmm a = mulpd(rN, getpz(y[0]));
            const xmm b = mulpd(rN, getpz(y[1]));
            const xmm c = mulpd(rN, getpz(y[2]));
            const xmm d = mulpd(rN, getpz(y[3]));
            const xmm e = mulpd(rN, getpz(y[4]));
            const xmm f = mulpd(rN, getpz(y[5]));
            const xmm g = mulpd(rN, getpz(y[6]));
            const xmm h = mulpd(rN, getpz(y[7]));
            const xmm jc = jxpz(c);
            const xmm jd = jxpz(d);
            const xmm jg = jxpz(g);
            const xmm jh = jxpz(h);
            const xmm ap1c = addpz(a,  c);
            const xmm amjc = subpz(a, jc);
            const xmm am1c = subpz(a,  c);
            const xmm apjc = addpz(a, jc);
            const xmm bp1d = addpz(b,  d);
            const xmm bmjd = subpz(b, jd);
            const xmm bm1d = subpz(b,  d);
            const xmm bpjd = addpz(b, jd);
            const xmm ep1g = addpz(e,  g);
            const xmm emjg = subpz(e, jg);
            const xmm em1g = subpz(e,  g);
            const xmm epjg = addpz(e, jg);
            const xmm fp1h = addpz(f,  h);
            const xmm fmjh = subpz(f, jh);
            const xmm fm1h = subpz(f,  h);
            const xmm fpjh = addpz(f, jh);
            const xmm   ap1c_p_ep1g =       addpz(ap1c, ep1g);
            const xmm   bp1d_p_fp1h =       addpz(bp1d, fp1h);
            const xmm   amjc_m_emjg =       subpz(amjc, emjg);
            const xmm w8bmjd_m_fmjh = w8xpz(subpz(bmjd, fmjh));
            const xmm   am1c_p_em1g =       addpz(am1c, em1g);
            const xmm jxbm1d_p_fm1h =  jxpz(addpz(bm1d, fm1h));
            const xmm   apjc_m_epjg =       subpz(apjc, epjg);
            const xmm v8bpjd_m_fpjh = v8xpz(subpz(bpjd, fpjh));
            setpz(x[0], addpz(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz(x[1], addpz(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz(x[2], addpz(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz(x[3], subpz(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz(x[4], subpz(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz(x[5], subpz(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz(x[6], subpz(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz(x[7], addpz(amjc_m_emjg, w8bmjd_m_fmjh));
        }
    }
};

//-----------------------------------------------------------------------------

template <int s> struct invnend<8,s,0>
{
    static const int N = 8*s;

    void operator()(complex_vector x, complex_vector) const
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            const ymm a = mulpd2(rN, getpz2(xq+s*0));
            const ymm b = mulpd2(rN, getpz2(xq+s*1));
            const ymm c = mulpd2(rN, getpz2(xq+s*2));
            const ymm d = mulpd2(rN, getpz2(xq+s*3));
            const ymm e = mulpd2(rN, getpz2(xq+s*4));
            const ymm f = mulpd2(rN, getpz2(xq+s*5));
            const ymm g = mulpd2(rN, getpz2(xq+s*6));
            const ymm h = mulpd2(rN, getpz2(xq+s*7));
            const ymm jc = jxpz2(c);
            const ymm jd = jxpz2(d);
            const ymm jg = jxpz2(g);
            const ymm jh = jxpz2(h);
            const ymm ap1c = addpz2(a,  c);
            const ymm amjc = subpz2(a, jc);
            const ymm am1c = subpz2(a,  c);
            const ymm apjc = addpz2(a, jc);
            const ymm bp1d = addpz2(b,  d);
            const ymm bmjd = subpz2(b, jd);
            const ymm bm1d = subpz2(b,  d);
            const ymm bpjd = addpz2(b, jd);
            const ymm ep1g = addpz2(e,  g);
            const ymm emjg = subpz2(e, jg);
            const ymm em1g = subpz2(e,  g);
            const ymm epjg = addpz2(e, jg);
            const ymm fp1h = addpz2(f,  h);
            const ymm fmjh = subpz2(f, jh);
            const ymm fm1h = subpz2(f,  h);
            const ymm fpjh = addpz2(f, jh);
            const ymm   ap1c_p_ep1g =        addpz2(ap1c, ep1g);
            const ymm   bp1d_p_fp1h =        addpz2(bp1d, fp1h);
            const ymm   amjc_m_emjg =        subpz2(amjc, emjg);
            const ymm w8bmjd_m_fmjh = w8xpz2(subpz2(bmjd, fmjh));
            const ymm   am1c_p_em1g =        addpz2(am1c, em1g);
            const ymm jxbm1d_p_fm1h =  jxpz2(addpz2(bm1d, fm1h));
            const ymm   apjc_m_epjg =        subpz2(apjc, epjg);
            const ymm v8bpjd_m_fpjh = v8xpz2(subpz2(bpjd, fpjh));
            setpz2(xq+s*0, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq+s*1, addpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(xq+s*2, addpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq+s*3, subpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(xq+s*4, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(xq+s*5, subpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(xq+s*6, subpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(xq+s*7, addpz2(amjc_m_emjg, w8bmjd_m_fmjh));
        }
    }
};

template <> struct invnend<8,1,0>
{
    inline void operator()(complex_vector x, complex_vector) const
    {
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            zeroupper();
            static const xmm rN = { 1.0/8, 1.0/8 };
            const xmm a = mulpd(rN, getpz(x[0]));
            const xmm b = mulpd(rN, getpz(x[1]));
            const xmm c = mulpd(rN, getpz(x[2]));
            const xmm d = mulpd(rN, getpz(x[3]));
            const xmm e = mulpd(rN, getpz(x[4]));
            const xmm f = mulpd(rN, getpz(x[5]));
            const xmm g = mulpd(rN, getpz(x[6]));
            const xmm h = mulpd(rN, getpz(x[7]));
            const xmm jc = jxpz(c);
            const xmm jd = jxpz(d);
            const xmm jg = jxpz(g);
            const xmm jh = jxpz(h);
            const xmm ap1c = addpz(a,  c);
            const xmm amjc = subpz(a, jc);
            const xmm am1c = subpz(a,  c);
            const xmm apjc = addpz(a, jc);
            const xmm bp1d = addpz(b,  d);
            const xmm bmjd = subpz(b, jd);
            const xmm bm1d = subpz(b,  d);
            const xmm bpjd = addpz(b, jd);
            const xmm ep1g = addpz(e,  g);
            const xmm emjg = subpz(e, jg);
            const xmm em1g = subpz(e,  g);
            const xmm epjg = addpz(e, jg);
            const xmm fp1h = addpz(f,  h);
            const xmm fmjh = subpz(f, jh);
            const xmm fm1h = subpz(f,  h);
            const xmm fpjh = addpz(f, jh);
            const xmm   ap1c_p_ep1g =       addpz(ap1c, ep1g);
            const xmm   bp1d_p_fp1h =       addpz(bp1d, fp1h);
            const xmm   amjc_m_emjg =       subpz(amjc, emjg);
            const xmm w8bmjd_m_fmjh = w8xpz(subpz(bmjd, fmjh));
            const xmm   am1c_p_em1g =       addpz(am1c, em1g);
            const xmm jxbm1d_p_fm1h =  jxpz(addpz(bm1d, fm1h));
            const xmm   apjc_m_epjg =       subpz(apjc, epjg);
            const xmm v8bpjd_m_fpjh = v8xpz(subpz(bpjd, fpjh));
            setpz(x[0], addpz(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz(x[1], addpz(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz(x[2], addpz(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz(x[3], subpz(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz(x[4], subpz(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz(x[5], subpz(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz(x[6], subpz(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz(x[7], addpz(amjc_m_emjg, w8bmjd_m_fmjh));
        }
    }
};

///////////////////////////////////////////////////////////////////////////////
// Inverse FFT
///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct inv0fft
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector W) const
    {
        inv0fft<n/8,8*s,!eo>()(y, x, W);
        invcore<n,s>()(x, y, W);
    }
};

template <int s, bool eo> struct inv0fft<8,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const
    {
        inv0end<8,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct inv0fft<4,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const
    {
        OTFFT_AVXDIT4omp::inv0end<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct inv0fft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const
    {
        OTFFT_AVXDIT4omp::inv0end<2,s,eo>()(x, y);
    }
};

//-----------------------------------------------------------------------------

template <int n, int s, bool eo> struct invnfft
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector W) const
    {
        invnfft<n/8,8*s,!eo>()(y, x, W);
        invcore<n,s>()(x, y, W);
    }
};

template <int s, bool eo> struct invnfft<8,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const
    {
        invnend<8,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct invnfft<4,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const
    {
        OTFFT_AVXDIT4omp::invnend<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct invnfft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const
    {
        OTFFT_AVXDIT4omp::invnend<2,s,eo>()(x, y);
    }
};

///////////////////////////////////////////////////////////////////////////////
// 2 powered FFT routine
///////////////////////////////////////////////////////////////////////////////

inline void fwd(const int log_N,
        complex_vector x, complex_vector y, const_complex_vector W)
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
        complex_vector x, complex_vector y, const_complex_vector W)
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
        complex_vector x, complex_vector y, const_complex_vector W)
{
    fwd(log_N, x, y, W);
}

inline void fwd0o(const int log_N,
        complex_vector x, complex_vector y, const_complex_vector W)
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
        complex_vector x, complex_vector y, const_complex_vector W)
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
        complex_vector x, complex_vector y, const_complex_vector W)
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
        complex_vector x, complex_vector y, const_complex_vector W)
{
    inv(log_N, x, y, W);
}

inline void invn(const int log_N,
        complex_vector x, complex_vector y, const_complex_vector W)
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
        complex_vector x, complex_vector y, const_complex_vector W)
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
        complex_vector x, complex_vector y, const_complex_vector W)
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

#endif // otfft_avxdit8omp_h
