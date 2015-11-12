/******************************************************************************
*  OTFFT Sixstep of Rectangle Version 5.4
******************************************************************************/

#ifndef otfft_sixstep0r_h
#define otfft_sixstep0r_h

namespace OTFFT_Sixstep { /////////////////////////////////////////////////////

template <int log_N> struct fwd0fftr
{
    static const int log_n = log_N/2;
    static const int log_m = log_N - log_n;
    static const int N = 1 << log_N;
    static const int n = 1 << log_n;
    static const int m = 1 << log_m;

    static void transpose_kernel(
            const int k, complex_vector x, complex_vector y)
    {
        for (int p = 0; p < m; p += 2) {
            complex_vector x_p_km = x + p + k*m;
            complex_vector y_k_pn = y + k + p*n;
            const ymm ab = getpz2(x_p_km+0);
            const ymm cd = getpz2(x_p_km+m);
            const ymm ac = catlo(ab, cd);
            const ymm bd = cathi(ab, cd);
            setpz2(y_k_pn+0, ac);
            setpz2(y_k_pn+n, bd);
        }
    }

    static void mult_twiddle_factor_kernel(const int p,
            complex_vector x, complex_vector y, const_complex_vector W)
    {
        for (int k = 0; k < n; k += 2) {
            const int kp = k*p;
            complex_vector x_k_pn = x + k + p*n;
            complex_vector y_p_km = y + p + k*m;
            const ymm w1 = cmplx2(W[kp],   W[kp+k]);
            const ymm w2 = cmplx2(W[kp+p], W[kp+p+k+1]);
            const ymm ab = getpz2(x_k_pn+0);
            const ymm cd = getpz2(x_k_pn+n);
            const ymm ac = catlo(ab, cd);
            const ymm bd = cathi(ab, cd);
            setpz2(y_p_km+0, mulpz2(w1, ac));
            setpz2(y_p_km+m, mulpz2(w2, bd));
        }
    }

    template <typename fft1_t, typename fft2_t>
    void operator()(const fft1_t& fft1, const fft2_t& fft2,
            complex_vector x, complex_vector y, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD1) {
            for (int k = 0; k < n; k += 2) {
                transpose_kernel(k, x, y);
            }
            for (int p = 0; p < m; p++) {
                const int pn = p*n;
                fft1.fwd0(y + pn, x + pn);
            }
            for (int p = 0; p < m; p += 2) {
                mult_twiddle_factor_kernel(p, y, x, W);
            }
            for (int k = 0; k < n; k++) {
                const int km = k*m;
                fft2.fwd0o(x + km, y + km);
            }
            for (int k = 0; k < n; k += 2) {
                transpose_kernel(k, y, x);
            }
        }
        else if (N < OMP_THRESHOLD2) //////////////////////////////////////////
#ifdef _OPENMP
        #pragma omp parallel firstprivate(x,y,W)
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int k = 0; k < n; k += 2) {
                transpose_kernel(k, x, y);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 0; p < m; p++) {
                const int pn = p*n;
                fft1.fwd0(y + pn, x + pn);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 0; p < m; p += 2) {
                mult_twiddle_factor_kernel(p, y, x, W);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int k = 0; k < n; k++) {
                const int km = k*m;
                fft2.fwd0o(x + km, y + km);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static) nowait
#endif
            for (int k = 0; k < n; k += 2) {
                transpose_kernel(k, y, x);
            }
        }
        else //////////////////////////////////////////////////////////////////
#ifdef _OPENMP
        #pragma omp parallel firstprivate(x,y,W)
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int k = 0; k < n; k += 2) {
                transpose_kernel(k, x, y);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int p = 0; p < m; p++) {
                const int pn = p*n;
                fft1.fwd0(y + pn, x + pn);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int p = 0; p < m; p += 2) {
                mult_twiddle_factor_kernel(p, y, x, W);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int k = 0; k < n; k++) {
                const int km = k*m;
                fft2.fwd0o(x + km, y + km);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided) nowait
#endif
            for (int k = 0; k < n; k += 2) {
                transpose_kernel(k, y, x);
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int log_N> struct inv0fftr
{
    static const int log_n = log_N/2;
    static const int log_m = log_N - log_n;
    static const int N = 1 << log_N;
    static const int n = 1 << log_n;
    static const int m = 1 << log_m;

    static inline void transpose_kernel(
            const int k, complex_vector x, complex_vector y)
    {
        fwd0fftr<log_N>::transpose_kernel(k, x, y);
    }

    static void mult_twiddle_factor_kernel(const int p,
            complex_vector x, complex_vector y, const_complex_vector W)
    {
        for (int k = 0; k < n; k += 2) {
            const int kp = k*p;
            complex_vector x_k_pn = x + k + p*n;
            complex_vector y_p_km = y + p + k*m;
            const ymm w1 = cmplx2(W[N-kp],   W[N-kp-k]);
            const ymm w2 = cmplx2(W[N-kp-p], W[N-kp-p-k-1]);
            const ymm ab = getpz2(x_k_pn+0);
            const ymm cd = getpz2(x_k_pn+n);
            const ymm ac = catlo(ab, cd);
            const ymm bd = cathi(ab, cd);
            setpz2(y_p_km+0, mulpz2(w1, ac));
            setpz2(y_p_km+m, mulpz2(w2, bd));
        }
    }

    template <typename fft1_t, typename fft2_t>
    void operator()(const fft1_t& fft1, const fft2_t& fft2,
            complex_vector x, complex_vector y, const_complex_vector W) const
    {
        if (N < OMP_THRESHOLD1) {
            for (int k = 0; k < n; k += 2) {
                transpose_kernel(k, x, y);
            }
            for (int p = 0; p < m; p++) {
                const int pn = p*n;
                fft1.inv0(y + pn, x + pn);
            }
            for (int p = 0; p < m; p += 2) {
                mult_twiddle_factor_kernel(p, y, x, W);
            }
            for (int k = 0; k < n; k++) {
                const int km = k*m;
                fft2.inv0o(x + km, y + km);
            }
            for (int k = 0; k < n; k += 2) {
                transpose_kernel(k, y, x);
            }
        }
        else if (N < OMP_THRESHOLD2) //////////////////////////////////////////
#ifdef _OPENMP
        #pragma omp parallel firstprivate(x,y,W)
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int k = 0; k < n; k += 2) {
                transpose_kernel(k, x, y);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 0; p < m; p++) {
                const int pn = p*n;
                fft1.inv0(y + pn, x + pn);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 0; p < m; p += 2) {
                mult_twiddle_factor_kernel(p, y, x, W);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int k = 0; k < n; k++) {
                const int km = k*m;
                fft2.inv0o(x + km, y + km);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static) nowait
#endif
            for (int k = 0; k < n; k += 2) {
                transpose_kernel(k, y, x);
            }
        }
        else //////////////////////////////////////////////////////////////////
#ifdef _OPENMP
        #pragma omp parallel firstprivate(x,y,W)
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int k = 0; k < n; k += 2) {
                transpose_kernel(k, x, y);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int p = 0; p < m; p++) {
                const int pn = p*n;
                fft1.inv0(y + pn, x + pn);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int p = 0; p < m; p += 2) {
                mult_twiddle_factor_kernel(p, y, x, W);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int k = 0; k < n; k++) {
                const int km = k*m;
                fft2.inv0o(x + km, y + km);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided) nowait
#endif
            for (int k = 0; k < n; k += 2) {
                transpose_kernel(k, y, x);
            }
        }
    }
};

} /////////////////////////////////////////////////////////////////////////////

#endif // otfft_sixstep0r_h
