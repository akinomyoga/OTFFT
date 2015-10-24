/******************************************************************************
*  OTFFT Eightstep Version 5.3
******************************************************************************/

#ifndef otfft_eightstep_h
#define otfft_eightstep_h

namespace OTFFT_Eightstep { ///////////////////////////////////////////////////

using namespace OTFFT_MISC;
using OTFFT_Sixstep::weight_t;
using OTFFT_Sixstep::const_index_vector;

static const int OMP_THRESHOLD = 1<<13;

///////////////////////////////////////////////////////////////////////////////

template <int log_N> struct fwd0fftq
{
    static const int N = 1 << log_N;
    static const int N0 = 0;
    static const int N1 = N/8;
    static const int N2 = N/4;
    static const int N3 = N1 + N2;
    static const int N4 = N/2;
    static const int N5 = N2 + N3;
    static const int N6 = N3 + N3;
    static const int N7 = N3 + N4;

    static inline void transpose_kernel(
            const int p, complex_vector x, complex_vector y)
    {
        const ymm ab = getpz2(x+p+N0);
        const ymm cd = getpz2(x+p+N1);
        const ymm ef = getpz2(x+p+N2);
        const ymm gh = getpz2(x+p+N3);
        const ymm ij = getpz2(x+p+N4);
        const ymm kl = getpz2(x+p+N5);
        const ymm mn = getpz2(x+p+N6);
        const ymm op = getpz2(x+p+N7);
        const ymm ac = catlo(ab, cd);
        const ymm bd = cathi(ab, cd);
        const ymm eg = catlo(ef, gh);
        const ymm fh = cathi(ef, gh);
        const ymm ik = catlo(ij, kl);
        const ymm jl = cathi(ij, kl);
        const ymm mo = catlo(mn, op);
        const ymm np = cathi(mn, op);
        setpz2(y+8*p+ 0, ac);
        setpz2(y+8*p+ 2, eg);
        setpz2(y+8*p+ 4, ik);
        setpz2(y+8*p+ 6, mo);
        setpz2(y+8*p+ 8, bd);
        setpz2(y+8*p+10, fh);
        setpz2(y+8*p+12, jl);
        setpz2(y+8*p+14, np);
    }

    static inline void fft_and_mult_twiddle_factor_kernel(
            const int p, complex_vector x, complex_vector y, weight_t W)
    {
        const ymm w1p = getpz2(W+p);
        const ymm w2p = mulpz2(w1p, w1p);
        const ymm w3p = mulpz2(w1p, w2p);
        const ymm w4p = mulpz2(w2p, w2p);
        const ymm w5p = mulpz2(w2p, w3p);
        const ymm w6p = mulpz2(w3p, w3p);
        const ymm w7p = mulpz2(w3p, w4p);
        const ymm a = getpz2(x+p+N0);
        const ymm b = getpz2(x+p+N1);
        const ymm c = getpz2(x+p+N2);
        const ymm d = getpz2(x+p+N3);
        const ymm e = getpz2(x+p+N4);
        const ymm f = getpz2(x+p+N5);
        const ymm g = getpz2(x+p+N6);
        const ymm h = getpz2(x+p+N7);
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
        setpz2(y+p+N0,             addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
        setpz2(y+p+N1, mulpz2(w1p, addpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
        setpz2(y+p+N2, mulpz2(w2p, subpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
        setpz2(y+p+N3, mulpz2(w3p, subpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
        setpz2(y+p+N4, mulpz2(w4p, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h)));
        setpz2(y+p+N5, mulpz2(w5p, subpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
        setpz2(y+p+N6, mulpz2(w6p, addpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
        setpz2(y+p+N7, mulpz2(w7p, addpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
    }

    void operator()(const_index_vector ip,
            complex_vector x, complex_vector y, weight_t W, weight_t Ws) const
    {
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < N1; p += 2) {
                fft_and_mult_twiddle_factor_kernel(p, x, y, W);
            }
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N0, x+N0, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N1, x+N1, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N2, x+N2, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N3, x+N3, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N4, x+N4, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N5, x+N5, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N6, x+N6, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N7, x+N7, W, Ws);
            for (int p = 0; p < N1; p += 2) {
                transpose_kernel(p, y, x);
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(guided) firstprivate(x,y,W)
#endif
            for (int p = 0; p < N1; p += 2) {
                fft_and_mult_twiddle_factor_kernel(p, x, y, W);
            }
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N0, x+N0, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N1, x+N1, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N2, x+N2, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N3, x+N3, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N4, x+N4, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N5, x+N5, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N6, x+N6, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N7, x+N7, W, Ws);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) firstprivate(x,y)
#endif
            for (int p = 0; p < N1; p += 2) {
                transpose_kernel(p, y, x);
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int log_N> struct inv0fftq
{
    static const int N = 1 << log_N;
    static const int N0 = 0;
    static const int N1 = N/8;
    static const int N2 = N/4;
    static const int N3 = N1 + N2;
    static const int N4 = N/2;
    static const int N5 = N2 + N3;
    static const int N6 = N3 + N3;
    static const int N7 = N3 + N4;

    static inline void transpose_kernel(
            const int p, complex_vector x, complex_vector y)
    {
        fwd0fftq<log_N>::transpose_kernel(p, x, y);
    }

    static inline void fft_and_mult_twiddle_factor_kernel(
            const int p, complex_vector x, complex_vector y, weight_t W)
    {
        const ymm w1p = cnjpz2(getpz2(W+p));
        const ymm w2p = mulpz2(w1p, w1p);
        const ymm w3p = mulpz2(w1p, w2p);
        const ymm w4p = mulpz2(w2p, w2p);
        const ymm w5p = mulpz2(w2p, w3p);
        const ymm w6p = mulpz2(w3p, w3p);
        const ymm w7p = mulpz2(w3p, w4p);
        const ymm a = getpz2(x+p+N0);
        const ymm b = getpz2(x+p+N1);
        const ymm c = getpz2(x+p+N2);
        const ymm d = getpz2(x+p+N3);
        const ymm e = getpz2(x+p+N4);
        const ymm f = getpz2(x+p+N5);
        const ymm g = getpz2(x+p+N6);
        const ymm h = getpz2(x+p+N7);
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
        setpz2(y+p+N0,             addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
        setpz2(y+p+N1, mulpz2(w1p, addpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
        setpz2(y+p+N2, mulpz2(w2p, addpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
        setpz2(y+p+N3, mulpz2(w3p, subpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
        setpz2(y+p+N4, mulpz2(w4p, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h)));
        setpz2(y+p+N5, mulpz2(w5p, subpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
        setpz2(y+p+N6, mulpz2(w6p, subpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
        setpz2(y+p+N7, mulpz2(w7p, addpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
    }

    void operator()(const_index_vector ip,
            complex_vector x, complex_vector y, weight_t W, weight_t Ws) const
    {
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < N1; p += 2) {
                fft_and_mult_twiddle_factor_kernel(p, x, y, W);
            }
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N0, x+N0, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N1, x+N1, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N2, x+N2, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N3, x+N3, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N4, x+N4, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N5, x+N5, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N6, x+N6, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N7, x+N7, W, Ws);
            for (int p = 0; p < N1; p += 2) {
                transpose_kernel(p, y, x);
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(guided) firstprivate(x,y,W)
#endif
            for (int p = 0; p < N1; p += 2) {
                fft_and_mult_twiddle_factor_kernel(p, x, y, W);
            }
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N0, x+N0, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N1, x+N1, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N2, x+N2, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N3, x+N3, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N4, x+N4, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N5, x+N5, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N6, x+N6, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N7, x+N7, W, Ws);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) firstprivate(x,y)
#endif
            for (int p = 0; p < N1; p += 2) {
                transpose_kernel(p, y, x);
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int log_N> struct fwdnfftq
{
    static const int N = 1 << log_N;
    static const int N0 = 0;
    static const int N1 = N/8;
    static const int N2 = N/4;
    static const int N3 = N1 + N2;
    static const int N4 = N/2;
    static const int N5 = N2 + N3;
    static const int N6 = N3 + N3;
    static const int N7 = N3 + N4;

    static inline void transpose_kernel(
            const int p, complex_vector x, complex_vector y)
    {
        fwd0fftq<log_N>::transpose_kernel(p, x, y);
    }

    static inline void fft_and_mult_twiddle_factor_kernel(
            const int p, complex_vector x, complex_vector y, weight_t W)
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        const ymm w1p = getpz2(W+p);
        const ymm w2p = mulpz2(w1p, w1p);
        const ymm w3p = mulpz2(w1p, w2p);
        const ymm w4p = mulpz2(w2p, w2p);
        const ymm w5p = mulpz2(w2p, w3p);
        const ymm w6p = mulpz2(w3p, w3p);
        const ymm w7p = mulpz2(w3p, w4p);
        const ymm a = mulpd2(rN, getpz2(x+p+N0));
        const ymm b = mulpd2(rN, getpz2(x+p+N1));
        const ymm c = mulpd2(rN, getpz2(x+p+N2));
        const ymm d = mulpd2(rN, getpz2(x+p+N3));
        const ymm e = mulpd2(rN, getpz2(x+p+N4));
        const ymm f = mulpd2(rN, getpz2(x+p+N5));
        const ymm g = mulpd2(rN, getpz2(x+p+N6));
        const ymm h = mulpd2(rN, getpz2(x+p+N7));
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
        setpz2(y+p+N0,             addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
        setpz2(y+p+N1, mulpz2(w1p, addpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
        setpz2(y+p+N2, mulpz2(w2p, subpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
        setpz2(y+p+N3, mulpz2(w3p, subpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
        setpz2(y+p+N4, mulpz2(w4p, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h)));
        setpz2(y+p+N5, mulpz2(w5p, subpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
        setpz2(y+p+N6, mulpz2(w6p, addpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
        setpz2(y+p+N7, mulpz2(w7p, addpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
    }

    void operator()(const_index_vector ip,
            complex_vector x, complex_vector y, weight_t W, weight_t Ws) const
    {
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < N1; p += 2) {
                fft_and_mult_twiddle_factor_kernel(p, x, y, W);
            }
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N0, x+N0, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N1, x+N1, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N2, x+N2, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N3, x+N3, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N4, x+N4, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N5, x+N5, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N6, x+N6, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N7, x+N7, W, Ws);
            for (int p = 0; p < N1; p += 2) {
                transpose_kernel(p, y, x);
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(guided) firstprivate(x,y,W)
#endif
            for (int p = 0; p < N1; p += 2) {
                fft_and_mult_twiddle_factor_kernel(p, x, y, W);
            }
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N0, x+N0, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N1, x+N1, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N2, x+N2, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N3, x+N3, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N4, x+N4, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N5, x+N5, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N6, x+N6, W, Ws);
            OTFFT_Sixstep::fwd0ffts8<log_N-3>()(ip, y+N7, x+N7, W, Ws);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) firstprivate(x,y)
#endif
            for (int p = 0; p < N1; p += 2) {
                transpose_kernel(p, y, x);
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int log_N> struct invnfftq
{
    static const int N = 1 << log_N;
    static const int N0 = 0;
    static const int N1 = N/8;
    static const int N2 = N/4;
    static const int N3 = N1 + N2;
    static const int N4 = N/2;
    static const int N5 = N2 + N3;
    static const int N6 = N3 + N3;
    static const int N7 = N3 + N4;

    static inline void transpose_kernel(
            const int p, complex_vector x, complex_vector y)
    {
        fwd0fftq<log_N>::transpose_kernel(p, x, y);
    }

    static inline void fft_and_mult_twiddle_factor_kernel(
            const int p, complex_vector x, complex_vector y, weight_t W)
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        const ymm w1p = cnjpz2(getpz2(W+p));
        const ymm w2p = mulpz2(w1p, w1p);
        const ymm w3p = mulpz2(w1p, w2p);
        const ymm w4p = mulpz2(w2p, w2p);
        const ymm w5p = mulpz2(w2p, w3p);
        const ymm w6p = mulpz2(w3p, w3p);
        const ymm w7p = mulpz2(w3p, w4p);
        const ymm a = mulpd2(rN, getpz2(x+p+N0));
        const ymm b = mulpd2(rN, getpz2(x+p+N1));
        const ymm c = mulpd2(rN, getpz2(x+p+N2));
        const ymm d = mulpd2(rN, getpz2(x+p+N3));
        const ymm e = mulpd2(rN, getpz2(x+p+N4));
        const ymm f = mulpd2(rN, getpz2(x+p+N5));
        const ymm g = mulpd2(rN, getpz2(x+p+N6));
        const ymm h = mulpd2(rN, getpz2(x+p+N7));
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
        setpz2(y+p+N0,             addpz2(ap1c_p_ep1g,   bp1d_p_fp1h));
        setpz2(y+p+N1, mulpz2(w1p, addpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
        setpz2(y+p+N2, mulpz2(w2p, addpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
        setpz2(y+p+N3, mulpz2(w3p, subpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
        setpz2(y+p+N4, mulpz2(w4p, subpz2(ap1c_p_ep1g,   bp1d_p_fp1h)));
        setpz2(y+p+N5, mulpz2(w5p, subpz2(apjc_m_epjg, v8bpjd_m_fpjh)));
        setpz2(y+p+N6, mulpz2(w6p, subpz2(am1c_p_em1g, jxbm1d_p_fm1h)));
        setpz2(y+p+N7, mulpz2(w7p, addpz2(amjc_m_emjg, w8bmjd_m_fmjh)));
    }

    void operator()(const_index_vector ip,
            complex_vector x, complex_vector y, weight_t W, weight_t Ws) const
    {
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < N1; p += 2) {
                fft_and_mult_twiddle_factor_kernel(p, x, y, W);
            }
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N0, x+N0, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N1, x+N1, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N2, x+N2, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N3, x+N3, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N4, x+N4, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N5, x+N5, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N6, x+N6, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N7, x+N7, W, Ws);
            for (int p = 0; p < N1; p += 2) {
                transpose_kernel(p, y, x);
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(guided) firstprivate(x,y,W)
#endif
            for (int p = 0; p < N1; p += 2) {
                fft_and_mult_twiddle_factor_kernel(p, x, y, W);
            }
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N0, x+N0, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N1, x+N1, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N2, x+N2, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N3, x+N3, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N4, x+N4, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N5, x+N5, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N6, x+N6, W, Ws);
            OTFFT_Sixstep::inv0ffts8<log_N-3>()(ip, y+N7, x+N7, W, Ws);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) firstprivate(x,y)
#endif
            for (int p = 0; p < N1; p += 2) {
                transpose_kernel(p, y, x);
            }
        }
    }
};

} /////////////////////////////////////////////////////////////////////////////

#endif // otfft_eightstep_h
