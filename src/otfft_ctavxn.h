/******************************************************************************
*  OTFFT Cooley-Tukey AVX Normalized Version 4.0
******************************************************************************/

#ifndef otfft_ctavxn_h
#define otfft_ctavxn_h

template <int n, int s> struct fwdnbut;
template <int n, int s> struct invnbut;

///////////////////////////////////////////////////////////////////////////////

template <> struct fwdnbut<2,1>
{
    inline void operator()(complex_vector x, const_complex_vector W) const
    {
        static const xmm r2 = { 1.0/2, 1.0/2 };
        const xmm a = mulpd(r2, getpz(x[0]));
        const xmm b = mulpd(r2, getpz(x[1]));
        setpz(x[0], addpz(a, b));
        setpz(x[1], subpz(a, b));
    }
};

template <> struct invnbut<2,1>
{
    inline void operator()(complex_vector x, const_complex_vector W) const
    {
        fwdnbut<2,1>()(x, W);
    }
};

///////////////////////////////////////////////////////////////////////////////

template <> struct fwdnbut<4,1>
{
    inline void operator()(complex_vector x, const_complex_vector W) const
    {
        static const xmm r4 = { 1.0/4, 1.0/4 };
        const xmm a = mulpd(r4, getpz(x[0]));
        const xmm b = mulpd(r4, getpz(x[1]));
        const xmm c = mulpd(r4, getpz(x[2]));
        const xmm d = mulpd(r4, getpz(x[3]));
        const xmm  apc =      addpz(a, c);
        const xmm  amc =      subpz(a, c);
        const xmm  bpd =      addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(x[0], addpz(apc,  bpd));
        setpz(x[1], subpz(apc,  bpd));
        setpz(x[2], subpz(amc, jbmd));
        setpz(x[3], addpz(amc, jbmd));
    }
};

template <> struct invnbut<4,1>
{
    inline void operator()(complex_vector x, const_complex_vector W) const
    {
        static const xmm r4 = { 1.0/4, 1.0/4 };
        const xmm a = mulpd(r4, getpz(x[0]));
        const xmm b = mulpd(r4, getpz(x[1]));
        const xmm c = mulpd(r4, getpz(x[2]));
        const xmm d = mulpd(r4, getpz(x[3]));
        const xmm  apc =      addpz(a, c);
        const xmm  amc =      subpz(a, c);
        const xmm  bpd =      addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(x[0], addpz(apc,  bpd));
        setpz(x[1], subpz(apc,  bpd));
        setpz(x[2], addpz(amc, jbmd));
        setpz(x[3], subpz(amc, jbmd));
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int N> struct fwdnbut<N,1>
{
    static const int n = N;
    static const int n0 = 0;
    static const int n1 = n/4;
    static const int n2 = n/2;
    static const int n3 = n1 + n2;

    void operator()(complex_vector x, const_complex_vector W) const
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < n1; p += 2) {
                const ymm w1p = cmplx2(W[1*p], W[1*p+1]);
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm a = mulpd2(rN, getpz2(x+p+n0));
                const ymm b = mulpd2(rN, getpz2(x+p+n1));
                const ymm c = mulpd2(rN, getpz2(x+p+n2));
                const ymm d = mulpd2(rN, getpz2(x+p+n3));
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(x+p+n0,             addpz2(apc,  bpd));
                setpz2(x+p+n1, mulpz2(w2p, subpz2(apc,  bpd)));
                setpz2(x+p+n2, mulpz2(w1p, subpz2(amc, jbmd)));
                setpz2(x+p+n3, mulpz2(w3p, addpz2(amc, jbmd)));
            }
            fwd0but<n/4,4>()(x, W);
        }
        else
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int p = 0; p < n1; p += 2) {
                const ymm w1p = cmplx2(W[1*p], W[1*p+1]);
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm a = mulpd2(rN, getpz2(x+p+n0));
                const ymm b = mulpd2(rN, getpz2(x+p+n1));
                const ymm c = mulpd2(rN, getpz2(x+p+n2));
                const ymm d = mulpd2(rN, getpz2(x+p+n3));
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(x+p+n0,             addpz2(apc,  bpd));
                setpz2(x+p+n1, mulpz2(w2p, subpz2(apc,  bpd)));
                setpz2(x+p+n2, mulpz2(w1p, subpz2(amc, jbmd)));
                setpz2(x+p+n3, mulpz2(w3p, addpz2(amc, jbmd)));
            }
            fwd0but<n/4,4>()(x, W);
        }
    }
};

template <int N> struct invnbut<N,1>
{
    static const int n = N;
    static const int n0 = 0;
    static const int n1 = n/4;
    static const int n2 = n/2;
    static const int n3 = n1 + n2;

    void operator()(complex_vector x, const_complex_vector W) const
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < n1; p += 2) {
                const ymm w1p = cmplx2(W[N-1*p], W[N-1*p-1]);
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm a = mulpd2(rN, getpz2(x+p+n0));
                const ymm b = mulpd2(rN, getpz2(x+p+n1));
                const ymm c = mulpd2(rN, getpz2(x+p+n2));
                const ymm d = mulpd2(rN, getpz2(x+p+n3));
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(x+p+n0,             addpz2(apc,  bpd));
                setpz2(x+p+n1, mulpz2(w2p, subpz2(apc,  bpd)));
                setpz2(x+p+n2, mulpz2(w1p, addpz2(amc, jbmd)));
                setpz2(x+p+n3, mulpz2(w3p, subpz2(amc, jbmd)));
            }
            inv0but<n/4,4>()(x, W);
        }
        else
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int p = 0; p < n1; p += 2) {
                const ymm w1p = cmplx2(W[N-1*p], W[N-1*p-1]);
                const ymm w2p = mulpz2(w1p, w1p);
                const ymm w3p = mulpz2(w1p, w2p);
                const ymm a = mulpd2(rN, getpz2(x+p+n0));
                const ymm b = mulpd2(rN, getpz2(x+p+n1));
                const ymm c = mulpd2(rN, getpz2(x+p+n2));
                const ymm d = mulpd2(rN, getpz2(x+p+n3));
                const ymm  apc =       addpz2(a, c);
                const ymm  amc =       subpz2(a, c);
                const ymm  bpd =       addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz2(x+p+n0,             addpz2(apc,  bpd));
                setpz2(x+p+n1, mulpz2(w2p, subpz2(apc,  bpd)));
                setpz2(x+p+n2, mulpz2(w1p, addpz2(amc, jbmd)));
                setpz2(x+p+n3, mulpz2(w3p, subpz2(amc, jbmd)));
            }
            inv0but<n/4,4>()(x, W);
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

#endif // otfft_ctavxn_h
