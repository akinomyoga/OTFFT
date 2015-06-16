/******************************************************************************
*  OTFFT DITAVX8 Normalized Version 4.0
******************************************************************************/

#ifndef otfft_ditavx8n_h
#define otfft_ditavx8n_h

template <int n, int s, int eo> struct fwdnbut;
template <int n, int s, int eo> struct invnbut;

///////////////////////////////////////////////////////////////////////////////

template <> struct fwdnbut<2,1,0>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        OTFFT_DITAVX::fwdnbut<2,1,0>()(x, y, W);
    }
};

template <> struct fwdnbut<2,1,1>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        OTFFT_DITAVX::fwdnbut<2,1,1>()(x, y, W);
    }
};

template <> struct invnbut<2,1,0>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        OTFFT_DITAVX::invnbut<2,1,0>()(x, y, W);
    }
};

template <> struct invnbut<2,1,1>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        OTFFT_DITAVX::invnbut<2,1,1>()(x, y, W);
    }
};

///////////////////////////////////////////////////////////////////////////////

template <> struct fwdnbut<4,1,0>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        OTFFT_DITAVX::fwdnbut<4,1,0>()(x, y, W);
    }
};

template <> struct fwdnbut<4,1,1>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        OTFFT_DITAVX::fwdnbut<4,1,1>()(x, y, W);
    }
};

template <> struct invnbut<4,1,0>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        OTFFT_DITAVX::invnbut<4,1,0>()(x, y, W);
    }
};

template <> struct invnbut<4,1,1>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        OTFFT_DITAVX::invnbut<4,1,1>()(x, y, W);
    }
};

///////////////////////////////////////////////////////////////////////////////

template <> struct fwdnbut<8,1,0>
{
    void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        static const xmm r8 = { 1.0/8, 1.0/8 };
        const xmm a = mulpd(r8, getpz(x[0]));
        const xmm b = mulpd(r8, getpz(x[1]));
        const xmm c = mulpd(r8, getpz(x[2]));
        const xmm d = mulpd(r8, getpz(x[3]));
        const xmm e = mulpd(r8, getpz(x[4]));
        const xmm f = mulpd(r8, getpz(x[5]));
        const xmm g = mulpd(r8, getpz(x[6]));
        const xmm h = mulpd(r8, getpz(x[7]));
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

template <> struct fwdnbut<8,1,1>
{
    void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        static const xmm r8 = { 1.0/8, 1.0/8 };
        const xmm a = mulpd(r8, getpz(y[0]));
        const xmm b = mulpd(r8, getpz(y[1]));
        const xmm c = mulpd(r8, getpz(y[2]));
        const xmm d = mulpd(r8, getpz(y[3]));
        const xmm e = mulpd(r8, getpz(y[4]));
        const xmm f = mulpd(r8, getpz(y[5]));
        const xmm g = mulpd(r8, getpz(y[6]));
        const xmm h = mulpd(r8, getpz(y[7]));
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

template <> struct invnbut<8,1,0>
{
    void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        static const xmm r8 = { 1.0/8, 1.0/8 };
        const xmm a = mulpd(r8, getpz(x[0]));
        const xmm b = mulpd(r8, getpz(x[1]));
        const xmm c = mulpd(r8, getpz(x[2]));
        const xmm d = mulpd(r8, getpz(x[3]));
        const xmm e = mulpd(r8, getpz(x[4]));
        const xmm f = mulpd(r8, getpz(x[5]));
        const xmm g = mulpd(r8, getpz(x[6]));
        const xmm h = mulpd(r8, getpz(x[7]));
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

template <> struct invnbut<8,1,1>
{
    void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        static const xmm r8 = { 1.0/8, 1.0/8 };
        const xmm a = mulpd(r8, getpz(y[0]));
        const xmm b = mulpd(r8, getpz(y[1]));
        const xmm c = mulpd(r8, getpz(y[2]));
        const xmm d = mulpd(r8, getpz(y[3]));
        const xmm e = mulpd(r8, getpz(y[4]));
        const xmm f = mulpd(r8, getpz(y[5]));
        const xmm g = mulpd(r8, getpz(y[6]));
        const xmm h = mulpd(r8, getpz(y[7]));
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

template <int N, int eo> struct fwdnbut<N,1,eo>
{
    static const int N0 = 0;
    static const int N1 = N/8;
    static const int N2 = N/4;
    static const int N3 = N1 + N2;
    static const int N4 = N/2;
    static const int N5 = N2 + N3;
    static const int N6 = N3 + N3;
    static const int N7 = N3 + N4;

    void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        if (N < OMP_THRESHOLD) {
            fwd0but<N/8,8,!eo>()(y, x, W);
            for (int p = 0; p < N/8; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_8p = y + 8*p;
                const ymm w1p = getpz2(W+p);
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm w4p = mulpz2(w2p, w2p);
                const ymm w5p = mulpz2(w2p, w3p);
                const ymm w6p = mulpz2(w3p, w3p);
                const ymm w7p = mulpz2(w3p, w4p);
                const ymm a =             getpz3<8>(y_8p+0);
                const ymm b = mulpz2(w1p, getpz3<8>(y_8p+1));
                const ymm c = mulpz2(w2p, getpz3<8>(y_8p+2));
                const ymm d = mulpz2(w3p, getpz3<8>(y_8p+3));
                const ymm e = mulpz2(w4p, getpz3<8>(y_8p+4));
                const ymm f = mulpz2(w5p, getpz3<8>(y_8p+5));
                const ymm g = mulpz2(w6p, getpz3<8>(y_8p+6));
                const ymm h = mulpz2(w7p, getpz3<8>(y_8p+7));
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
                setpz2(x_p+N0, mulpd2(rN, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h)));
                setpz2(x_p+N1, mulpd2(rN, addpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
                setpz2(x_p+N2, mulpd2(rN, subpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
                setpz2(x_p+N3, mulpd2(rN, subpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
                setpz2(x_p+N4, mulpd2(rN, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h)));
                setpz2(x_p+N5, mulpd2(rN, subpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
                setpz2(x_p+N6, mulpd2(rN, addpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
                setpz2(x_p+N7, mulpd2(rN, addpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
            }
        }
        else
        #pragma omp parallel
        {
            fwd0but<N/8,8,!eo>()(y, x, W);
            #pragma omp for schedule(static) nowait
            for (int p = 0; p < N/8; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_8p = y + 8*p;
                const ymm w1p = getpz2(W+p);
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm w4p = mulpz2(w2p, w2p);
                const ymm w5p = mulpz2(w2p, w3p);
                const ymm w6p = mulpz2(w3p, w3p);
                const ymm w7p = mulpz2(w3p, w4p);
                const ymm a =             getpz3<8>(y_8p+0);
                const ymm b = mulpz2(w1p, getpz3<8>(y_8p+1));
                const ymm c = mulpz2(w2p, getpz3<8>(y_8p+2));
                const ymm d = mulpz2(w3p, getpz3<8>(y_8p+3));
                const ymm e = mulpz2(w4p, getpz3<8>(y_8p+4));
                const ymm f = mulpz2(w5p, getpz3<8>(y_8p+5));
                const ymm g = mulpz2(w6p, getpz3<8>(y_8p+6));
                const ymm h = mulpz2(w7p, getpz3<8>(y_8p+7));
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
                setpz2(x_p+N0, mulpd2(rN, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h)));
                setpz2(x_p+N1, mulpd2(rN, addpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
                setpz2(x_p+N2, mulpd2(rN, subpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
                setpz2(x_p+N3, mulpd2(rN, subpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
                setpz2(x_p+N4, mulpd2(rN, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h)));
                setpz2(x_p+N5, mulpd2(rN, subpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
                setpz2(x_p+N6, mulpd2(rN, addpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
                setpz2(x_p+N7, mulpd2(rN, addpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
            }
        }
    }
};

template <int N, int eo> struct invnbut<N,1,eo>
{
    static const int N0 = 0;
    static const int N1 = N/8;
    static const int N2 = N/4;
    static const int N3 = N1 + N2;
    static const int N4 = N/2;
    static const int N5 = N2 + N3;
    static const int N6 = N3 + N3;
    static const int N7 = N3 + N4;

    void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        if (N < OMP_THRESHOLD) {
            inv0but<N/8,8,!eo>()(y, x, W);
            for (int p = 0; p < N/8; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_8p = y + 8*p;
                //const ymm w1p = getwp2<-1>(W+N,p);
                const ymm w1p = cnjpz2(getpz2(W+p));
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm w4p = mulpz2(w2p, w2p);
                const ymm w5p = mulpz2(w2p, w3p);
                const ymm w6p = mulpz2(w3p, w3p);
                const ymm w7p = mulpz2(w3p, w4p);
                const ymm a =             getpz3<8>(y_8p+0);
                const ymm b = mulpz2(w1p, getpz3<8>(y_8p+1));
                const ymm c = mulpz2(w2p, getpz3<8>(y_8p+2));
                const ymm d = mulpz2(w3p, getpz3<8>(y_8p+3));
                const ymm e = mulpz2(w4p, getpz3<8>(y_8p+4));
                const ymm f = mulpz2(w5p, getpz3<8>(y_8p+5));
                const ymm g = mulpz2(w6p, getpz3<8>(y_8p+6));
                const ymm h = mulpz2(w7p, getpz3<8>(y_8p+7));
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
                setpz2(x_p+N0, mulpd2(rN, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h)));
                setpz2(x_p+N1, mulpd2(rN, addpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
                setpz2(x_p+N2, mulpd2(rN, addpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
                setpz2(x_p+N3, mulpd2(rN, subpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
                setpz2(x_p+N4, mulpd2(rN, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h)));
                setpz2(x_p+N5, mulpd2(rN, subpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
                setpz2(x_p+N6, mulpd2(rN, subpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
                setpz2(x_p+N7, mulpd2(rN, addpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
            }
        }
        else
        #pragma omp parallel
        {
            inv0but<N/8,8,!eo>()(y, x, W);
            #pragma omp for schedule(static) nowait
            for (int p = 0; p < N/8; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_8p = y + 8*p;
                //const ymm w1p = getwp2<-1>(W+N,p);
                const ymm w1p = cnjpz2(getpz2(W+p));
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm w4p = mulpz2(w2p, w2p);
                const ymm w5p = mulpz2(w2p, w3p);
                const ymm w6p = mulpz2(w3p, w3p);
                const ymm w7p = mulpz2(w3p, w4p);
                const ymm a =             getpz3<8>(y_8p+0);
                const ymm b = mulpz2(w1p, getpz3<8>(y_8p+1));
                const ymm c = mulpz2(w2p, getpz3<8>(y_8p+2));
                const ymm d = mulpz2(w3p, getpz3<8>(y_8p+3));
                const ymm e = mulpz2(w4p, getpz3<8>(y_8p+4));
                const ymm f = mulpz2(w5p, getpz3<8>(y_8p+5));
                const ymm g = mulpz2(w6p, getpz3<8>(y_8p+6));
                const ymm h = mulpz2(w7p, getpz3<8>(y_8p+7));
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
                setpz2(x_p+N0, mulpd2(rN, addpz2(ap1c_p_ep1g,   bp1d_p_fp1h)));
                setpz2(x_p+N1, mulpd2(rN, addpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
                setpz2(x_p+N2, mulpd2(rN, addpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
                setpz2(x_p+N3, mulpd2(rN, subpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
                setpz2(x_p+N4, mulpd2(rN, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h)));
                setpz2(x_p+N5, mulpd2(rN, subpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
                setpz2(x_p+N6, mulpd2(rN, subpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
                setpz2(x_p+N7, mulpd2(rN, addpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

#endif // otfft_ditavx8n_h