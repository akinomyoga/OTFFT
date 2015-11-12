/******************************************************************************
*  OTFFT Eightstep Version 5.4
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
        const ymm aA = getpz2(x+p+N0);
        const ymm bB = getpz2(x+p+N1);
        const ymm cC = getpz2(x+p+N2);
        const ymm dD = getpz2(x+p+N3);
        const ymm eE = getpz2(x+p+N4);
        const ymm fF = getpz2(x+p+N5);
        const ymm gG = getpz2(x+p+N6);
        const ymm hH = getpz2(x+p+N7);
        const ymm ab = catlo(aA, bB);
        const ymm AB = cathi(aA, bB);
        const ymm cd = catlo(cC, dD);
        const ymm CD = cathi(cC, dD);
        const ymm ef = catlo(eE, fF);
        const ymm EF = cathi(eE, fF);
        const ymm gh = catlo(gG, hH);
        const ymm GH = cathi(gG, hH);
        setpz2(y+8*p+ 0, ab);
        setpz2(y+8*p+ 2, cd);
        setpz2(y+8*p+ 4, ef);
        setpz2(y+8*p+ 6, gh);
        setpz2(y+8*p+ 8, AB);
        setpz2(y+8*p+10, CD);
        setpz2(y+8*p+12, EF);
        setpz2(y+8*p+14, GH);
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
        const ymm x0 = getpz2(x+p+N0);
        const ymm x1 = getpz2(x+p+N1);
        const ymm x2 = getpz2(x+p+N2);
        const ymm x3 = getpz2(x+p+N3);
        const ymm x4 = getpz2(x+p+N4);
        const ymm x5 = getpz2(x+p+N5);
        const ymm x6 = getpz2(x+p+N6);
        const ymm x7 = getpz2(x+p+N7);
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
        setpz2(y+p+N0,             addpz2(a04_p1_a26,    a15_p1_a37));
        setpz2(y+p+N1, mulpz2(w1p, addpz2(s04_mj_s26, w8_s15_mj_s37)));
        setpz2(y+p+N2, mulpz2(w2p, subpz2(a04_m1_a26,  j_a15_m1_a37)));
        setpz2(y+p+N3, mulpz2(w3p, subpz2(s04_pj_s26, v8_s15_pj_s37)));
        setpz2(y+p+N4, mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37)));
        setpz2(y+p+N5, mulpz2(w5p, subpz2(s04_mj_s26, w8_s15_mj_s37)));
        setpz2(y+p+N6, mulpz2(w6p, addpz2(a04_m1_a26,  j_a15_m1_a37)));
        setpz2(y+p+N7, mulpz2(w7p, addpz2(s04_pj_s26, v8_s15_pj_s37)));
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
        const ymm x0 = getpz2(x+p+N0);
        const ymm x1 = getpz2(x+p+N1);
        const ymm x2 = getpz2(x+p+N2);
        const ymm x3 = getpz2(x+p+N3);
        const ymm x4 = getpz2(x+p+N4);
        const ymm x5 = getpz2(x+p+N5);
        const ymm x6 = getpz2(x+p+N6);
        const ymm x7 = getpz2(x+p+N7);
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
        setpz2(y+p+N0,             addpz2(a04_p1_a26,    a15_p1_a37));
        setpz2(y+p+N1, mulpz2(w1p, addpz2(s04_pj_s26, v8_s15_pj_s37)));
        setpz2(y+p+N2, mulpz2(w2p, addpz2(a04_m1_a26,  j_a15_m1_a37)));
        setpz2(y+p+N3, mulpz2(w3p, subpz2(s04_mj_s26, w8_s15_mj_s37)));
        setpz2(y+p+N4, mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37)));
        setpz2(y+p+N5, mulpz2(w5p, subpz2(s04_pj_s26, v8_s15_pj_s37)));
        setpz2(y+p+N6, mulpz2(w6p, subpz2(a04_m1_a26,  j_a15_m1_a37)));
        setpz2(y+p+N7, mulpz2(w7p, addpz2(s04_mj_s26, w8_s15_mj_s37)));
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
        const ymm x0 = mulpd2(rN, getpz2(x+p+N0));
        const ymm x1 = mulpd2(rN, getpz2(x+p+N1));
        const ymm x2 = mulpd2(rN, getpz2(x+p+N2));
        const ymm x3 = mulpd2(rN, getpz2(x+p+N3));
        const ymm x4 = mulpd2(rN, getpz2(x+p+N4));
        const ymm x5 = mulpd2(rN, getpz2(x+p+N5));
        const ymm x6 = mulpd2(rN, getpz2(x+p+N6));
        const ymm x7 = mulpd2(rN, getpz2(x+p+N7));
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
        setpz2(y+p+N0,             addpz2(a04_p1_a26,    a15_p1_a37));
        setpz2(y+p+N1, mulpz2(w1p, addpz2(s04_mj_s26, w8_s15_mj_s37)));
        setpz2(y+p+N2, mulpz2(w2p, subpz2(a04_m1_a26,  j_a15_m1_a37)));
        setpz2(y+p+N3, mulpz2(w3p, subpz2(s04_pj_s26, v8_s15_pj_s37)));
        setpz2(y+p+N4, mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37)));
        setpz2(y+p+N5, mulpz2(w5p, subpz2(s04_mj_s26, w8_s15_mj_s37)));
        setpz2(y+p+N6, mulpz2(w6p, addpz2(a04_m1_a26,  j_a15_m1_a37)));
        setpz2(y+p+N7, mulpz2(w7p, addpz2(s04_pj_s26, v8_s15_pj_s37)));
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
        const ymm x0 = mulpd2(rN, getpz2(x+p+N0));
        const ymm x1 = mulpd2(rN, getpz2(x+p+N1));
        const ymm x2 = mulpd2(rN, getpz2(x+p+N2));
        const ymm x3 = mulpd2(rN, getpz2(x+p+N3));
        const ymm x4 = mulpd2(rN, getpz2(x+p+N4));
        const ymm x5 = mulpd2(rN, getpz2(x+p+N5));
        const ymm x6 = mulpd2(rN, getpz2(x+p+N6));
        const ymm x7 = mulpd2(rN, getpz2(x+p+N7));
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
        setpz2(y+p+N0,             addpz2(a04_p1_a26,    a15_p1_a37));
        setpz2(y+p+N1, mulpz2(w1p, addpz2(s04_pj_s26, v8_s15_pj_s37)));
        setpz2(y+p+N2, mulpz2(w2p, addpz2(a04_m1_a26,  j_a15_m1_a37)));
        setpz2(y+p+N3, mulpz2(w3p, subpz2(s04_mj_s26, w8_s15_mj_s37)));
        setpz2(y+p+N4, mulpz2(w4p, subpz2(a04_p1_a26,    a15_p1_a37)));
        setpz2(y+p+N5, mulpz2(w5p, subpz2(s04_pj_s26, v8_s15_pj_s37)));
        setpz2(y+p+N6, mulpz2(w6p, subpz2(a04_m1_a26,  j_a15_m1_a37)));
        setpz2(y+p+N7, mulpz2(w7p, addpz2(s04_mj_s26, w8_s15_mj_s37)));
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
