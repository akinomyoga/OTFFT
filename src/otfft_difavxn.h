/******************************************************************************
*  OTFFT DIFAVX Normalized Version 4.0
******************************************************************************/

#ifndef otfft_difavxn_h
#define otfft_difavxn_h

template <int n, int s, int eo> struct fwdnbut;
template <int n, int s, int eo> struct invnbut;

///////////////////////////////////////////////////////////////////////////////

template <> struct fwdnbut<2,1,0>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        static const xmm r2 = { 1.0/2, 1.0/2 };
        const xmm a = mulpd(r2, getpz(x[0]));
        const xmm b = mulpd(r2, getpz(x[1]));
        setpz(x[0], addpz(a, b));
        setpz(x[1], subpz(a, b));
    }
};

template <> struct fwdnbut<2,1,1>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        static const xmm r2 = { 1.0/2, 1.0/2 };
        const xmm a = mulpd(r2, getpz(x[0]));
        const xmm b = mulpd(r2, getpz(x[1]));
        setpz(y[0], addpz(a, b));
        setpz(y[1], subpz(a, b));
    }
};

template <> struct invnbut<2,1,0>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        fwdnbut<2,1,0>()(x, y, W);
    }
};

template <> struct invnbut<2,1,1>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        fwdnbut<2,1,1>()(x, y, W);
    }
};

///////////////////////////////////////////////////////////////////////////////

template <> struct fwdnbut<4,1,0>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        static const xmm r4 = { 1.0/4, 1.0/4 };
        const xmm a = mulpd(r4, getpz(x[0]));
        const xmm b = mulpd(r4, getpz(x[1]));
        const xmm c = mulpd(r4, getpz(x[2]));
        const xmm d = mulpd(r4, getpz(x[3]));
        const xmm  apc = addpz(a, c);
        const xmm  amc = subpz(a, c);
        const xmm  bpd = addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(x[0], addpz(apc,  bpd));
        setpz(x[1], subpz(amc, jbmd));
        setpz(x[2], subpz(apc,  bpd));
        setpz(x[3], addpz(amc, jbmd));
    }
};

template <> struct fwdnbut<4,1,1>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        static const xmm r4 = { 1.0/4, 1.0/4 };
        const xmm a = mulpd(r4, getpz(x[0]));
        const xmm b = mulpd(r4, getpz(x[1]));
        const xmm c = mulpd(r4, getpz(x[2]));
        const xmm d = mulpd(r4, getpz(x[3]));
        const xmm  apc = addpz(a, c);
        const xmm  amc = subpz(a, c);
        const xmm  bpd = addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(y[0], addpz(apc,  bpd));
        setpz(y[1], subpz(amc, jbmd));
        setpz(y[2], subpz(apc,  bpd));
        setpz(y[3], addpz(amc, jbmd));
    }
};

template <> struct invnbut<4,1,0>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        static const xmm r4 = { 1.0/4, 1.0/4 };
        const xmm a = mulpd(r4, getpz(x[0]));
        const xmm b = mulpd(r4, getpz(x[1]));
        const xmm c = mulpd(r4, getpz(x[2]));
        const xmm d = mulpd(r4, getpz(x[3]));
        const xmm  apc = addpz(a, c);
        const xmm  amc = subpz(a, c);
        const xmm  bpd = addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(x[0], addpz(apc,  bpd));
        setpz(x[1], addpz(amc, jbmd));
        setpz(x[2], subpz(apc,  bpd));
        setpz(x[3], subpz(amc, jbmd));
    }
};

template <> struct invnbut<4,1,1>
{
    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        static const xmm r4 = { 1.0/4, 1.0/4 };
        const xmm a = mulpd(r4, getpz(x[0]));
        const xmm b = mulpd(r4, getpz(x[1]));
        const xmm c = mulpd(r4, getpz(x[2]));
        const xmm d = mulpd(r4, getpz(x[3]));
        const xmm  apc = addpz(a, c);
        const xmm  amc = subpz(a, c);
        const xmm  bpd = addpz(b, d);
        const xmm jbmd = jxpz(subpz(b, d));
        setpz(y[0], addpz(apc,  bpd));
        setpz(y[1], addpz(amc, jbmd));
        setpz(y[2], subpz(apc,  bpd));
        setpz(y[3], subpz(amc, jbmd));
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int N, int eo> struct fwdnbut<N,1,eo>
{
    static const int N0 = 0;
    static const int N1 = N/4;
    static const int N2 = N/2;
    static const int N3 = N1 + N2;

    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < N/4; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_4p = y + 4*p;
                //const ymm w1p = getwp2<1>(W,p);
                const ymm w1p = getpz2(W+p);
                const ymm w2p = getwp2<2>(W,p);
                const ymm w3p = getwp2<3>(W,p);
                const ymm a = mulpd2(rN, getpz2(x_p+N0));
                const ymm b = mulpd2(rN, getpz2(x_p+N1));
                const ymm c = mulpd2(rN, getpz2(x_p+N2));
                const ymm d = mulpd2(rN, getpz2(x_p+N3));
                const ymm  apc = addpz2(a, c);
                const ymm  amc = subpz2(a, c);
                const ymm  bpd = addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz3<4>(y_4p+0,             addpz2(apc,  bpd));
                setpz3<4>(y_4p+1, mulpz2(w1p, subpz2(amc, jbmd)));
                setpz3<4>(y_4p+2, mulpz2(w2p, subpz2(apc,  bpd)));
                setpz3<4>(y_4p+3, mulpz2(w3p, addpz2(amc, jbmd)));
            }
            fwd0but<N/4,4,!eo>()(y, x, W);
        }
        else
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int p = 0; p < N/4; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_4p = y + 4*p;
                //const ymm w1p = getwp2<1>(W,p);
                const ymm w1p = getpz2(W+p);
                const ymm w2p = getwp2<2>(W,p);
                const ymm w3p = getwp2<3>(W,p);
                const ymm a = mulpd2(rN, getpz2(x_p+N0));
                const ymm b = mulpd2(rN, getpz2(x_p+N1));
                const ymm c = mulpd2(rN, getpz2(x_p+N2));
                const ymm d = mulpd2(rN, getpz2(x_p+N3));
                const ymm  apc = addpz2(a, c);
                const ymm  amc = subpz2(a, c);
                const ymm  bpd = addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz3<4>(y_4p+0,             addpz2(apc,  bpd));
                setpz3<4>(y_4p+1, mulpz2(w1p, subpz2(amc, jbmd)));
                setpz3<4>(y_4p+2, mulpz2(w2p, subpz2(apc,  bpd)));
                setpz3<4>(y_4p+3, mulpz2(w3p, addpz2(amc, jbmd)));
            }
            fwd0but<N/4,4,!eo>()(y, x, W);
        }
    }
};

template <int N, int eo> struct invnbut<N,1,eo>
{
    static const int N0 = 0;
    static const int N1 = N/4;
    static const int N2 = N/2;
    static const int N3 = N1 + N2;

    inline void operator()(complex_vector x, complex_vector y, const_complex_vector W) const
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < N/4; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_4p = y + 4*p;
                //const ymm w1p = getwp2<-1>(W+N,p);
                const ymm w1p = cnjpz2(getpz2(W+p));
                const ymm w2p = getwp2<-2>(W+N,p);
                const ymm w3p = getwp2<-3>(W+N,p);
                const ymm a = mulpd2(rN, getpz2(x_p+N0));
                const ymm b = mulpd2(rN, getpz2(x_p+N1));
                const ymm c = mulpd2(rN, getpz2(x_p+N2));
                const ymm d = mulpd2(rN, getpz2(x_p+N3));
                const ymm  apc = addpz2(a, c);
                const ymm  amc = subpz2(a, c);
                const ymm  bpd = addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz3<4>(y_4p+0,             addpz2(apc,  bpd));
                setpz3<4>(y_4p+1, mulpz2(w1p, addpz2(amc, jbmd)));
                setpz3<4>(y_4p+2, mulpz2(w2p, subpz2(apc,  bpd)));
                setpz3<4>(y_4p+3, mulpz2(w3p, subpz2(amc, jbmd)));
            }
            inv0but<N/4,4,!eo>()(y, x, W);
        }
        else
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int p = 0; p < N/4; p += 2) {
                complex_vector x_p  = x + p;
                complex_vector y_4p = y + 4*p;
                //const ymm w1p = getwp2<-1>(W+N,p);
                const ymm w1p = cnjpz2(getpz2(W+p));
                const ymm w2p = getwp2<-2>(W+N,p);
                const ymm w3p = getwp2<-3>(W+N,p);
                const ymm a = mulpd2(rN, getpz2(x_p+N0));
                const ymm b = mulpd2(rN, getpz2(x_p+N1));
                const ymm c = mulpd2(rN, getpz2(x_p+N2));
                const ymm d = mulpd2(rN, getpz2(x_p+N3));
                const ymm  apc = addpz2(a, c);
                const ymm  amc = subpz2(a, c);
                const ymm  bpd = addpz2(b, d);
                const ymm jbmd = jxpz2(subpz2(b, d));
                setpz3<4>(y_4p+0,             addpz2(apc,  bpd));
                setpz3<4>(y_4p+1, mulpz2(w1p, addpz2(amc, jbmd)));
                setpz3<4>(y_4p+2, mulpz2(w2p, subpz2(apc,  bpd)));
                setpz3<4>(y_4p+3, mulpz2(w3p, subpz2(amc, jbmd)));
            }
            inv0but<N/4,4,!eo>()(y, x, W);
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

#endif // otfft_difavxn_h
