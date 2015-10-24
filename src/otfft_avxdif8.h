/******************************************************************************
*  OTFFT AVXDIF8 Version 5.3
******************************************************************************/

#ifndef otfft_avxdif8_h
#define otfft_avxdif8_h

#include "otfft/otfft_misc.h"
#include "otfft_avxdif4.h"
#include "otfft_avxdif8omp.h"

namespace OTFFT_AVXDIF8 { /////////////////////////////////////////////////////

using namespace OTFFT_MISC;

static const int OMP_THRESHOLD = 1<<11;

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
                const ymm a = getpz2(xq_sp+N0);
                const ymm b = getpz2(xq_sp+N1);
                const ymm c = getpz2(xq_sp+N2);
                const ymm d = getpz2(xq_sp+N3);
                const ymm e = getpz2(xq_sp+N4);
                const ymm f = getpz2(xq_sp+N5);
                const ymm g = getpz2(xq_sp+N6);
                const ymm h = getpz2(xq_sp+N7);
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
                setpz2(yq_s8p+s*0,             addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
                setpz2(yq_s8p+s*1, mulpz2(w1p, addpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
                setpz2(yq_s8p+s*2, mulpz2(w2p, subpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
                setpz2(yq_s8p+s*3, mulpz2(w3p, subpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
                setpz2(yq_s8p+s*4, mulpz2(w4p, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h)));
                setpz2(yq_s8p+s*5, mulpz2(w5p, subpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
                setpz2(yq_s8p+s*6, mulpz2(w6p, addpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
                setpz2(yq_s8p+s*7, mulpz2(w7p, addpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
            }
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
            const ymm a = getpz2(x_p+N0);
            const ymm b = getpz2(x_p+N1);
            const ymm c = getpz2(x_p+N2);
            const ymm d = getpz2(x_p+N3);
            const ymm e = getpz2(x_p+N4);
            const ymm f = getpz2(x_p+N5);
            const ymm g = getpz2(x_p+N6);
            const ymm h = getpz2(x_p+N7);
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
#if 0
            setpz3<8>(y_8p+0,             addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz3<8>(y_8p+1, mulpz2(w1p, addpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
            setpz3<8>(y_8p+2, mulpz2(w2p, subpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
            setpz3<8>(y_8p+3, mulpz2(w3p, subpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
            setpz3<8>(y_8p+4, mulpz2(w4p, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h)));
            setpz3<8>(y_8p+5, mulpz2(w5p, subpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
            setpz3<8>(y_8p+6, mulpz2(w6p, addpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
            setpz3<8>(y_8p+7, mulpz2(w7p, addpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
#else
            const ymm ab =             addpz2(ap1c_p_ep1g,   bp1d_p_fp1h);
            const ymm cd = mulpz2(w1p, addpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            const ymm ef = mulpz2(w2p, subpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            const ymm gh = mulpz2(w3p, subpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            const ymm ij = mulpz2(w4p, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            const ymm kl = mulpz2(w5p, subpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            const ymm mn = mulpz2(w6p, addpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            const ymm op = mulpz2(w7p, addpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            const ymm ac = catlo(ab, cd);
            const ymm bd = cathi(ab, cd);
            const ymm eg = catlo(ef, gh);
            const ymm fh = cathi(ef, gh);
            const ymm ik = catlo(ij, kl);
            const ymm jl = cathi(ij, kl);
            const ymm mo = catlo(mn, op);
            const ymm np = cathi(mn, op);
            setpz2(y_8p+ 0, ac);
            setpz2(y_8p+ 2, eg);
            setpz2(y_8p+ 4, ik);
            setpz2(y_8p+ 6, mo);
            setpz2(y_8p+ 8, bd);
            setpz2(y_8p+10, fh);
            setpz2(y_8p+12, jl);
            setpz2(y_8p+14, np);
#endif
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
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
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
            setpz2(yq+s*0, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(yq+s*1, addpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(yq+s*2, subpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(yq+s*3, subpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(yq+s*4, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(yq+s*5, subpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(yq+s*6, addpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(yq+s*7, addpz2(apjc_m_epjg, v8bpjd_m_fpjh));
        }
    }
};

template <> struct fwd0end<8,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const
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
        setpz(y[0], addpz(ap1c_p_ep1g,   bp1d_p_fp1h));
        setpz(y[1], addpz(amjc_m_emjg, w8bmjd_m_fmjh));
        setpz(y[2], subpz(am1c_p_em1g, jxbm1d_p_fm1h));
        setpz(y[3], subpz(apjc_m_epjg, v8bpjd_m_fpjh));
        setpz(y[4], subpz(ap1c_p_ep1g,   bp1d_p_fp1h));
        setpz(y[5], subpz(amjc_m_emjg, w8bmjd_m_fmjh));
        setpz(y[6], addpz(am1c_p_em1g, jxbm1d_p_fm1h));
        setpz(y[7], addpz(apjc_m_epjg, v8bpjd_m_fpjh));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct fwd0end<8,s,0>
{
    void operator()(complex_vector x, complex_vector) const
    {
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
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
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
            setpz2(yq+s*0, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(yq+s*1, addpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(yq+s*2, subpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(yq+s*3, subpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(yq+s*4, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(yq+s*5, subpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(yq+s*6, addpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(yq+s*7, addpz2(apjc_m_epjg, v8bpjd_m_fpjh));
        }
    }
};

template <> struct fwdnend<8,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const
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
        setpz(y[0], addpz(ap1c_p_ep1g,   bp1d_p_fp1h));
        setpz(y[1], addpz(amjc_m_emjg, w8bmjd_m_fmjh));
        setpz(y[2], subpz(am1c_p_em1g, jxbm1d_p_fm1h));
        setpz(y[3], subpz(apjc_m_epjg, v8bpjd_m_fpjh));
        setpz(y[4], subpz(ap1c_p_ep1g,   bp1d_p_fp1h));
        setpz(y[5], subpz(amjc_m_emjg, w8bmjd_m_fmjh));
        setpz(y[6], addpz(am1c_p_em1g, jxbm1d_p_fm1h));
        setpz(y[7], addpz(apjc_m_epjg, v8bpjd_m_fpjh));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct fwdnend<8,s,0>
{
    static const int N = 8*s;

    void operator()(complex_vector x, complex_vector) const
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
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
};

///////////////////////////////////////////////////////////////////////////////
// Forward FFT
///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct fwd0fft
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector W) const
    {
        fwdcore<n,s>()(x, y, W);
        fwd0fft<n/8,8*s,!eo>()(y, x, W);
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
        OTFFT_AVXDIF4::fwd0end<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwd0fft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const
    {
        OTFFT_AVXDIF4::fwd0end<2,s,eo>()(x, y);
    }
};

//-----------------------------------------------------------------------------

template <int n, int s, bool eo> struct fwdnfft
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector W) const
    {
        fwdcore<n,s>()(x, y, W);
        fwdnfft<n/8,8*s,!eo>()(y, x, W);
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
        OTFFT_AVXDIF4::fwdnend<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct fwdnfft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const
    {
        OTFFT_AVXDIF4::fwdnend<2,s,eo>()(x, y);
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
        for (int p = 0; p < n/8; p++) {
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
                const ymm a = getpz2(xq_sp+N0);
                const ymm b = getpz2(xq_sp+N1);
                const ymm c = getpz2(xq_sp+N2);
                const ymm d = getpz2(xq_sp+N3);
                const ymm e = getpz2(xq_sp+N4);
                const ymm f = getpz2(xq_sp+N5);
                const ymm g = getpz2(xq_sp+N6);
                const ymm h = getpz2(xq_sp+N7);
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
                setpz2(yq_s8p+s*0,             addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
                setpz2(yq_s8p+s*1, mulpz2(w1p, addpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
                setpz2(yq_s8p+s*2, mulpz2(w2p, addpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
                setpz2(yq_s8p+s*3, mulpz2(w3p, subpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
                setpz2(yq_s8p+s*4, mulpz2(w4p, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h)));
                setpz2(yq_s8p+s*5, mulpz2(w5p, subpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
                setpz2(yq_s8p+s*6, mulpz2(w6p, subpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
                setpz2(yq_s8p+s*7, mulpz2(w7p, addpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
            }
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
            const ymm a = getpz2(x_p+N0);
            const ymm b = getpz2(x_p+N1);
            const ymm c = getpz2(x_p+N2);
            const ymm d = getpz2(x_p+N3);
            const ymm e = getpz2(x_p+N4);
            const ymm f = getpz2(x_p+N5);
            const ymm g = getpz2(x_p+N6);
            const ymm h = getpz2(x_p+N7);
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
#if 0
            setpz3<8>(y_8p+0,             addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz3<8>(y_8p+1, mulpz2(w1p, addpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
            setpz3<8>(y_8p+2, mulpz2(w2p, addpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
            setpz3<8>(y_8p+3, mulpz2(w3p, subpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
            setpz3<8>(y_8p+4, mulpz2(w4p, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h)));
            setpz3<8>(y_8p+5, mulpz2(w5p, subpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
            setpz3<8>(y_8p+6, mulpz2(w6p, subpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
            setpz3<8>(y_8p+7, mulpz2(w7p, addpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
#else
            const ymm ab =             addpz2(ap1c_p_ep1g,   bp1d_p_fp1h);
            const ymm cd = mulpz2(w1p, addpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            const ymm ef = mulpz2(w2p, addpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            const ymm gh = mulpz2(w3p, subpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            const ymm ij = mulpz2(w4p, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            const ymm kl = mulpz2(w5p, subpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            const ymm mn = mulpz2(w6p, subpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            const ymm op = mulpz2(w7p, addpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            const ymm ac = catlo(ab, cd);
            const ymm bd = cathi(ab, cd);
            const ymm eg = catlo(ef, gh);
            const ymm fh = cathi(ef, gh);
            const ymm ik = catlo(ij, kl);
            const ymm jl = cathi(ij, kl);
            const ymm mo = catlo(mn, op);
            const ymm np = cathi(mn, op);
            setpz2(y_8p+ 0, ac);
            setpz2(y_8p+ 2, eg);
            setpz2(y_8p+ 4, ik);
            setpz2(y_8p+ 6, mo);
            setpz2(y_8p+ 8, bd);
            setpz2(y_8p+10, fh);
            setpz2(y_8p+12, jl);
            setpz2(y_8p+14, np);
#endif
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
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
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
            setpz2(yq+s*0, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(yq+s*1, addpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(yq+s*2, addpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(yq+s*3, subpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(yq+s*4, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(yq+s*5, subpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(yq+s*6, subpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(yq+s*7, addpz2(amjc_m_emjg, w8bmjd_m_fmjh));
        }
    }
};

template <> struct inv0end<8,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const
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
        setpz(y[0], addpz(ap1c_p_ep1g,   bp1d_p_fp1h));
        setpz(y[1], addpz(apjc_m_epjg, v8bpjd_m_fpjh));
        setpz(y[2], addpz(am1c_p_em1g, jxbm1d_p_fm1h));
        setpz(y[3], subpz(amjc_m_emjg, w8bmjd_m_fmjh));
        setpz(y[4], subpz(ap1c_p_ep1g,   bp1d_p_fp1h));
        setpz(y[5], subpz(apjc_m_epjg, v8bpjd_m_fpjh));
        setpz(y[6], subpz(am1c_p_em1g, jxbm1d_p_fm1h));
        setpz(y[7], addpz(amjc_m_emjg, w8bmjd_m_fmjh));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct inv0end<8,s,0>
{
    void operator()(complex_vector x, complex_vector) const
    {
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
        for (int q = 0; q < s; q += 2) {
            complex_vector xq = x + q;
            complex_vector yq = y + q;
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
            setpz2(yq+s*0, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(yq+s*1, addpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(yq+s*2, addpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(yq+s*3, subpz2(amjc_m_emjg, w8bmjd_m_fmjh));
            setpz2(yq+s*4, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
            setpz2(yq+s*5, subpz2(apjc_m_epjg, v8bpjd_m_fpjh));
            setpz2(yq+s*6, subpz2(am1c_p_em1g, jxbm1d_p_fm1h));
            setpz2(yq+s*7, addpz2(amjc_m_emjg, w8bmjd_m_fmjh));
        }
    }
};

template <> struct invnend<8,1,1>
{
    inline void operator()(complex_vector x, complex_vector y) const
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
        setpz(y[0], addpz(ap1c_p_ep1g,   bp1d_p_fp1h));
        setpz(y[1], addpz(apjc_m_epjg, v8bpjd_m_fpjh));
        setpz(y[2], addpz(am1c_p_em1g, jxbm1d_p_fm1h));
        setpz(y[3], subpz(amjc_m_emjg, w8bmjd_m_fmjh));
        setpz(y[4], subpz(ap1c_p_ep1g,   bp1d_p_fp1h));
        setpz(y[5], subpz(apjc_m_epjg, v8bpjd_m_fpjh));
        setpz(y[6], subpz(am1c_p_em1g, jxbm1d_p_fm1h));
        setpz(y[7], addpz(amjc_m_emjg, w8bmjd_m_fmjh));
    }
};

//-----------------------------------------------------------------------------

template <int s> struct invnend<8,s,0>
{
    static const int N = 8*s;

    void operator()(complex_vector x, complex_vector) const
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
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
};

///////////////////////////////////////////////////////////////////////////////
// Inverse FFT
///////////////////////////////////////////////////////////////////////////////

template <int n, int s, bool eo> struct inv0fft
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector W) const
    {
        invcore<n,s>()(x, y, W);
        inv0fft<n/8,8*s,!eo>()(y, x, W);
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
        OTFFT_AVXDIF4::inv0end<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct inv0fft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const
    {
        OTFFT_AVXDIF4::inv0end<2,s,eo>()(x, y);
    }
};

//-----------------------------------------------------------------------------

template <int n, int s, bool eo> struct invnfft
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector W) const
    {
        invcore<n,s>()(x, y, W);
        invnfft<n/8,8*s,!eo>()(y, x, W);
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
        OTFFT_AVXDIF4::invnend<4,s,eo>()(x, y);
    }
};

template <int s, bool eo> struct invnfft<2,s,eo>
{
    inline void operator()(
        complex_vector x, complex_vector y, const_complex_vector) const
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

    FFT0() : N(0), log_N(0), W(0) {}
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

    inline void fwd(complex_vector x, complex_vector y) const
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
        else OTFFT_AVXDIF8omp::fwd(log_N, x, y, W);
    }

    inline void fwd0(complex_vector x, complex_vector y) const
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
        else OTFFT_AVXDIF8omp::fwd0(log_N, x, y, W);
    }

    inline void fwdn(complex_vector x, complex_vector y) const { fwd(x, y); }

    inline void fwd0o(complex_vector x, complex_vector y) const
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
        else OTFFT_AVXDIF8omp::fwd0o(log_N, x, y, W);
    }

    inline void fwdno(complex_vector x, complex_vector y) const
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
        else OTFFT_AVXDIF8omp::fwdno(log_N, x, y, W);
    }

    ///////////////////////////////////////////////////////////////////////////

    inline void inv(complex_vector x, complex_vector y) const
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
        else OTFFT_AVXDIF8omp::inv(log_N, x, y, W);
    }

    inline void inv0(complex_vector x, complex_vector y) const { inv(x, y); }

    inline void invn(complex_vector x, complex_vector y) const
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
        else OTFFT_AVXDIF8omp::invn(log_N, x, y, W);
    }

    inline void inv0o(complex_vector x, complex_vector y) const
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
        else OTFFT_AVXDIF8omp::inv0o(log_N, x, y, W);
    }

    inline void invno(complex_vector x, complex_vector y) const
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
        else OTFFT_AVXDIF8omp::invno(log_N, x, y, W);
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

    inline void fwd(complex_vector x)  const { fft.fwd(x, y);  }
    inline void fwd0(complex_vector x) const { fft.fwd0(x, y); }
    inline void fwdn(complex_vector x) const { fft.fwdn(x, y); }
    inline void inv(complex_vector x)  const { fft.inv(x, y);  }
    inline void inv0(complex_vector x) const { fft.inv0(x, y); }
    inline void invn(complex_vector x) const { fft.invn(x, y); }
};
#endif

} /////////////////////////////////////////////////////////////////////////////

#endif // otfft_avxdif8_h
