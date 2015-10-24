/******************************************************************************
*  OTFFT Sixstep of Square Version 5.3
******************************************************************************/

#ifndef otfft_sixstep0s_h
#define otfft_sixstep0s_h

namespace OTFFT_Sixstep { /////////////////////////////////////////////////////

template <int log_N, int s> struct fwd0ffts_body
{
    static const int log_n = log_N/2;
    static const int N = 1 << log_N;
    static const int n = 1 << log_n;
    static const int m = n/2*(n/2+1)/2;

    static void transpose_kernel(const int k, const int p, complex_vector x)
    {
        if (k == p) {
            const int k_kn = k + k*n;
            const ymm ab = getpz2(x+k_kn+0);
            const ymm cd = getpz2(x+k_kn+n);
            const ymm ac = catlo(ab, cd);
            const ymm bd = cathi(ab, cd);
            setpz2(x+k_kn+0, ac);
            setpz2(x+k_kn+n, bd);
        }
        else {
            const int p_kn = p + k*n;
            const int k_pn = k + p*n;
            const ymm ab = getpz2(x+p_kn+0);
            const ymm cd = getpz2(x+p_kn+n);
            const ymm ef = getpz2(x+k_pn+0);
            const ymm gh = getpz2(x+k_pn+n);
            const ymm ac = catlo(ab, cd);
            const ymm bd = cathi(ab, cd);
            const ymm eg = catlo(ef, gh);
            const ymm fh = cathi(ef, gh);
            setpz2(x+k_pn+0, ac);
            setpz2(x+k_pn+n, bd);
            setpz2(x+p_kn+0, eg);
            setpz2(x+p_kn+n, fh);
        }
    }

    static void mult_twiddle_factor_kernel(
            const int p, const int k, complex_vector x, weight_t W)
    {
        if (p == k) {
            const int pp = p*p;
            complex_vector x_p_pn = x + p + p*n;
            const complex_t& w = W[s*(pp+p)];
            const ymm w1 = cmplx2(W[s*(pp)], w);
            const ymm w2 = cmplx2(w, W[s*(pp+2*p+1)]);
            const ymm ab = getpz2(x_p_pn+0);
            const ymm cd = getpz2(x_p_pn+n);
            const ymm ac = mulpz2(w1, catlo(ab, cd));
            const ymm bd = mulpz2(w2, cathi(ab, cd));
            setpz2(x_p_pn+0, ac);
            setpz2(x_p_pn+n, bd);
        }
        else {
            const int kp = k*p;
            complex_vector x_k_pn = x + k + p*n;
            complex_vector x_p_kn = x + p + k*n;
            const ymm w1 = cmplx2(W[s*(kp)],   W[s*(kp+k)]);
            const ymm w2 = cmplx2(W[s*(kp+p)], W[s*(kp+k+p+1)]);
            const ymm ab = getpz2(x_k_pn+0);
            const ymm cd = getpz2(x_k_pn+n);
            const ymm ef = mulpz2(w1, getpz2(x_p_kn+0));
            const ymm gh = mulpz2(w2, getpz2(x_p_kn+n));
            const ymm ac = mulpz2(w1, catlo(ab, cd));
            const ymm bd = mulpz2(w2, cathi(ab, cd));
            const ymm eg = catlo(ef, gh);
            const ymm fh = cathi(ef, gh);
            setpz2(x_p_kn+0, ac);
            setpz2(x_p_kn+n, bd);
            setpz2(x_k_pn+0, eg);
            setpz2(x_k_pn+n, fh);
        }
    }

    void operator()(const_index_vector ip,
            complex_vector x, complex_vector y, weight_t W, weight_t Ws) const
    {
        if (N < OMP_THRESHOLD1) {
            for (int i = 0; i < m; i++) {
                const int k = ip[i].row;
                const int p = ip[i].col;
                transpose_kernel(k, p, x);
            }
            for (int p = 0; p < n; p++) {
                const int pn = p*n;
                OTFFT_AVXDIF8::fwd0fft<n,1,0>()(x + pn, y + pn, Ws);
            }
            for (int i = 0; i < m; i++) {
                const int p = ip[i].row;
                const int k = ip[i].col;
                mult_twiddle_factor_kernel(p, k, x, W);
            }
            for (int k = 0; k < n; k++) {
                const int kn = k*n;
                OTFFT_AVXDIF8::fwd0fft<n,1,0>()(x + kn, y + kn, Ws);
            }
            for (int i = 0; i < m; i++) {
                const int k = ip[i].row;
                const int p = ip[i].col;
                transpose_kernel(k, p, x);
            }
        }
        else if (N < OMP_THRESHOLD2) //////////////////////////////////////////
#ifdef _OPENMP
        #pragma omp parallel firstprivate(ip,x,y,W,Ws)
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int i = 0; i < m; i++) {
                const int k = ip[i].row;
                const int p = ip[i].col;
                transpose_kernel(k, p, x);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 0; p < n; p++) {
                const int pn = p*n;
                OTFFT_AVXDIF8::fwd0fft<n,1,0>()(x + pn, y + pn, Ws);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int i = 0; i < m; i++) {
                const int p = ip[i].row;
                const int k = ip[i].col;
                mult_twiddle_factor_kernel(p, k, x, W);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int k = 0; k < n; k++) {
                const int kn = k*n;
                OTFFT_AVXDIF8::fwd0fft<n,1,0>()(x + kn, y + kn, Ws);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static) nowait
#endif
            for (int i = 0; i < m; i++) {
                const int k = ip[i].row;
                const int p = ip[i].col;
                transpose_kernel(k, p, x);
            }
        }
        else //////////////////////////////////////////////////////////////////
#ifdef _OPENMP
        #pragma omp parallel firstprivate(ip,x,y,W,Ws)
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int i = 0; i < m; i++) {
                const int k = ip[i].row;
                const int p = ip[i].col;
                transpose_kernel(k, p, x);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int p = 0; p < n; p++) {
                const int pn = p*n;
                OTFFT_AVXDIF8::fwd0fft<n,1,0>()(x + pn, y + pn, Ws);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int i = 0; i < m; i++) {
                const int p = ip[i].row;
                const int k = ip[i].col;
                mult_twiddle_factor_kernel(p, k, x, W);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int k = 0; k < n; k++) {
                const int kn = k*n;
                OTFFT_AVXDIF8::fwd0fft<n,1,0>()(x + kn, y + kn, Ws);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided) nowait
#endif
            for (int i = 0; i < m; i++) {
                const int k = ip[i].row;
                const int p = ip[i].col;
                transpose_kernel(k, p, x);
            }
        }
    }
};

template <int log_N> struct fwd0ffts
{
    inline void operator()(const_index_vector ip,
            complex_vector x, complex_vector y, weight_t W, weight_t Ws) const
    {
        fwd0ffts_body<log_N,1>()(ip, x, y, W, Ws);
    }
};

template <int log_N> struct fwd0ffts2
{
    inline void operator()(const_index_vector ip,
            complex_vector x, complex_vector y, weight_t W, weight_t Ws) const
    {
        fwd0ffts_body<log_N,2>()(ip, x, y, W, Ws);
    }
};

template <int log_N> struct fwd0ffts8
{
    inline void operator()(const_index_vector ip,
            complex_vector x, complex_vector y, weight_t W, weight_t Ws) const
    {
        fwd0ffts_body<log_N,8>()(ip, x, y, W, Ws);
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int log_N, int s> struct inv0ffts_body
{
    static const int log_n = log_N/2;
    static const int N = 1 << log_N;
    static const int n = 1 << log_n;
    static const int m = n/2*(n/2+1)/2;
    
    static inline void transpose_kernel(
            const int k, const int p, complex_vector x)
    {
        fwd0ffts_body<log_N,s>::transpose_kernel(k, p, x);
    }

    static void mult_twiddle_factor_kernel(
            const int p, const int k, complex_vector x, weight_t W)
    {
        if (p == k) {
            const int pp = p*p;
            complex_vector x_p_pn = x + p + p*n;
            const complex_t& w = W[s*(N-pp-p)];
            const ymm w1 = cmplx2(W[s*(N-pp)], w);
            const ymm w2 = cmplx2(w, W[s*(N-pp-2*p-1)]);
            const ymm ab = getpz2(x_p_pn+0);
            const ymm cd = getpz2(x_p_pn+n);
            const ymm ac = mulpz2(w1, catlo(ab, cd));
            const ymm bd = mulpz2(w2, cathi(ab, cd));
            setpz2(x_p_pn+0, ac);
            setpz2(x_p_pn+n, bd);
        }
        else {
            const int kp = k*p;
            complex_vector x_k_pn = x + k + p*n;
            complex_vector x_p_kn = x + p + k*n;
            const ymm w1 = cmplx2(W[s*(N-kp)],   W[s*(N-kp-k)]);
            const ymm w2 = cmplx2(W[s*(N-kp-p)], W[s*(N-kp-k-p-1)]);
            const ymm ab = getpz2(x_k_pn+0);
            const ymm cd = getpz2(x_k_pn+n);
            const ymm ef = mulpz2(w1, getpz2(x_p_kn+0));
            const ymm gh = mulpz2(w2, getpz2(x_p_kn+n));
            const ymm ac = mulpz2(w1, catlo(ab, cd));
            const ymm bd = mulpz2(w2, cathi(ab, cd));
            const ymm eg = catlo(ef, gh);
            const ymm fh = cathi(ef, gh);
            setpz2(x_p_kn+0, ac);
            setpz2(x_p_kn+n, bd);
            setpz2(x_k_pn+0, eg);
            setpz2(x_k_pn+n, fh);
        }
    }

    void operator()(const_index_vector ip,
            complex_vector x, complex_vector y, weight_t W, weight_t Ws) const
    {
        if (N < OMP_THRESHOLD1) {
            for (int i = 0; i < m; i++) {
                const int k = ip[i].row;
                const int p = ip[i].col;
                transpose_kernel(k, p, x);
            }
            for (int p = 0; p < n; p++) {
                const int pn = p*n;
                OTFFT_AVXDIF8::inv0fft<n,1,0>()(x + pn, y + pn, Ws);
            }
            for (int i = 0; i < m; i++) {
                const int p = ip[i].row;
                const int k = ip[i].col;
                mult_twiddle_factor_kernel(p, k, x, W);
            }
            for (int k = 0; k < n; k++) {
                const int kn = k*n;
                OTFFT_AVXDIF8::inv0fft<n,1,0>()(x + kn, y + kn, Ws);
            }
            for (int i = 0; i < m; i++) {
                const int k = ip[i].row;
                const int p = ip[i].col;
                transpose_kernel(k, p, x);
            }
        }
        else if (N < OMP_THRESHOLD2) //////////////////////////////////////////
#ifdef _OPENMP
        #pragma omp parallel firstprivate(ip,x,y,W,Ws)
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int i = 0; i < m; i++) {
                const int k = ip[i].row;
                const int p = ip[i].col;
                transpose_kernel(k, p, x);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 0; p < n; p++) {
                const int pn = p*n;
                OTFFT_AVXDIF8::inv0fft<n,1,0>()(x + pn, y + pn, Ws);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int i = 0; i < m; i++) {
                const int p = ip[i].row;
                const int k = ip[i].col;
                mult_twiddle_factor_kernel(p, k, x, W);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int k = 0; k < n; k++) {
                const int kn = k*n;
                OTFFT_AVXDIF8::inv0fft<n,1,0>()(x + kn, y + kn, Ws);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static) nowait
#endif
            for (int i = 0; i < m; i++) {
                const int k = ip[i].row;
                const int p = ip[i].col;
                transpose_kernel(k, p, x);
            }
        }
        else //////////////////////////////////////////////////////////////////
#ifdef _OPENMP
        #pragma omp parallel firstprivate(ip,x,y,W,Ws)
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int i = 0; i < m; i++) {
                const int k = ip[i].row;
                const int p = ip[i].col;
                transpose_kernel(k, p, x);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int p = 0; p < n; p++) {
                const int pn = p*n;
                OTFFT_AVXDIF8::inv0fft<n,1,0>()(x + pn, y + pn, Ws);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int i = 0; i < m; i++) {
                const int p = ip[i].row;
                const int k = ip[i].col;
                mult_twiddle_factor_kernel(p, k, x, W);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int k = 0; k < n; k++) {
                const int kn = k*n;
                OTFFT_AVXDIF8::inv0fft<n,1,0>()(x + kn, y + kn, Ws);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided) nowait
#endif
            for (int i = 0; i < m; i++) {
                const int k = ip[i].row;
                const int p = ip[i].col;
                transpose_kernel(k, p, x);
            }
        }
    }
};

template <int log_N> struct inv0ffts
{
    inline void operator()(const_index_vector ip,
            complex_vector x, complex_vector y, weight_t W, weight_t Ws) const
    {
        inv0ffts_body<log_N,1>()(ip, x, y, W, Ws);
    }
};

template <int log_N> struct inv0ffts2
{
    inline void operator()(const_index_vector ip,
            complex_vector x, complex_vector y, weight_t W, weight_t Ws) const
    {
        inv0ffts_body<log_N,2>()(ip, x, y, W, Ws);
    }
};

template <int log_N> struct inv0ffts8
{
    inline void operator()(const_index_vector ip,
            complex_vector x, complex_vector y, weight_t W, weight_t Ws) const
    {
        inv0ffts_body<log_N,8>()(ip, x, y, W, Ws);
    }
};

} /////////////////////////////////////////////////////////////////////////////

#endif // otfft_sixstep0s_h
