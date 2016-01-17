/******************************************************************************
*  OTFFT Sixstep of Normalized Square Version 6.4
*
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_sixstepns_h
#define otfft_sixstepns_h

namespace OTFFT_Sixstep { /////////////////////////////////////////////////////

template <int log_N> struct fwdnffts
{
    static const int log_n = log_N/2;
    static const int N = 1 << log_N;
    static const int n = 1 << log_n;
    static const int m = n/2*(n/2+1)/2;

    static inline void transpose_kernel(
            const int k, const int p, complex_vector x) noexcept
    {
        fwd0ffts_body<log_N,1>::transpose_kernel(k, p, x);
    }

    static void mult_twiddle_factor_kernel(
            const int p, const int k, complex_vector x, weight_t W) noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        if (p == k) {
            const int pp = p*p;
            complex_vector x_p_pn = x + p + p*n;
            const complex_t& w = W[pp+p];
            const ymm w1 = mulpd2(rN, cmplx2(W[pp], w));
            const ymm w2 = mulpd2(rN, cmplx2(w, W[pp+2*p+1]));
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
            const ymm w1 = mulpd2(rN, cmplx2(W[kp],   W[kp+k]));
            const ymm w2 = mulpd2(rN, cmplx2(W[kp+p], W[kp+k+p+1]));
            const ymm ab = getpz2(x_k_pn+0);
            const ymm cd = getpz2(x_k_pn+n);
            const ymm ac = mulpz2(w1, catlo(ab, cd));
            const ymm bd = mulpz2(w2, cathi(ab, cd));
            const ymm ef = mulpz2(w1, getpz2(x_p_kn+0));
            const ymm gh = mulpz2(w2, getpz2(x_p_kn+n));
            const ymm eg = catlo(ef, gh);
            const ymm fh = cathi(ef, gh);
            setpz2(x_p_kn+0, ac);
            setpz2(x_p_kn+n, bd);
            setpz2(x_k_pn+0, eg);
            setpz2(x_k_pn+n, fh);
        }
    }

    void operator()(const_index_vector ip,
            complex_vector x, complex_vector y, weight_t W, weight_t Ws) const noexcept
    {
        if (N < OMP_THRESHOLD1) {
            for (int i = 0; i < m; i++) {
                const int k = ip[i].row;
                const int p = ip[i].col;
                transpose_kernel(k, p, x);
            }
            for (int p = 0; p < n; p++) {
                const int pn = p*n;
                OTFFT_AVXDIF16::fwd0fft<n,1,0>()(x + pn, y + pn, Ws);
            }
            for (int i = 0; i < m; i++) {
                const int p = ip[i].row;
                const int k = ip[i].col;
                mult_twiddle_factor_kernel(p, k, x, W);
            }
            for (int k = 0; k < n; k++) {
                const int kn = k*n;
                OTFFT_AVXDIF16::fwd0fft<n,1,0>()(x + kn, y + kn, Ws);
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
                OTFFT_AVXDIF16::fwd0fft<n,1,0>()(x + pn, y + pn, Ws);
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
                OTFFT_AVXDIF16::fwd0fft<n,1,0>()(x + kn, y + kn, Ws);
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
                OTFFT_AVXDIF16::fwd0fft<n,1,0>()(x + pn, y + pn, Ws);
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
                OTFFT_AVXDIF16::fwd0fft<n,1,0>()(x + kn, y + kn, Ws);
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

///////////////////////////////////////////////////////////////////////////////

template <int log_N> struct invnffts
{
    static const int log_n = log_N/2;
    static const int N = 1 << log_N;
    static const int n = 1 << log_n;
    static const int m = n/2*(n/2+1)/2;

    static inline void transpose_kernel(
            const int k, const int p, complex_vector x) noexcept
    {
        fwdnffts<log_N>::transpose_kernel(k, p, x);
    }

    static void mult_twiddle_factor_kernel(
            const int p, const int k, complex_vector x, weight_t W) noexcept
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        if (p == k) {
            const int pp = p*p;
            complex_vector x_p_pn = x + p + p*n;
            const complex_t& w = W[N-pp-p];
            const ymm w1 = mulpd2(rN, cmplx2(W[N-pp], w));
            const ymm w2 = mulpd2(rN, cmplx2(w, W[N-pp-2*p-1]));
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
            const ymm w1 = mulpd2(rN, cmplx2(W[N-kp],   W[N-kp-k]));
            const ymm w2 = mulpd2(rN, cmplx2(W[N-kp-p], W[N-kp-k-p-1]));
            const ymm ab = getpz2(x_k_pn+0);
            const ymm cd = getpz2(x_k_pn+n);
            const ymm ac = mulpz2(w1, catlo(ab, cd));
            const ymm bd = mulpz2(w2, cathi(ab, cd));
            const ymm ef = mulpz2(w1, getpz2(x_p_kn+0));
            const ymm gh = mulpz2(w2, getpz2(x_p_kn+n));
            const ymm eg = catlo(ef, gh);
            const ymm fh = cathi(ef, gh);
            setpz2(x_p_kn+0, ac);
            setpz2(x_p_kn+n, bd);
            setpz2(x_k_pn+0, eg);
            setpz2(x_k_pn+n, fh);
        }
    }

    void operator()(const_index_vector ip,
            complex_vector x, complex_vector y, weight_t W, weight_t Ws) const noexcept
    {
        if (N < OMP_THRESHOLD1) {
            for (int i = 0; i < m; i++) {
                const int k = ip[i].row;
                const int p = ip[i].col;
                transpose_kernel(k, p, x);
            }
            for (int p = 0; p < n; p++) {
                const int pn = p*n;
                OTFFT_AVXDIF16::inv0fft<n,1,0>()(x + pn, y + pn, Ws);
            }
            for (int i = 0; i < m; i++) {
                const int p = ip[i].row;
                const int k = ip[i].col;
                mult_twiddle_factor_kernel(p, k, x, W);
            }
            for (int k = 0; k < n; k++) {
                const int kn = k*n;
                OTFFT_AVXDIF16::inv0fft<n,1,0>()(x + kn, y + kn, Ws);
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
                OTFFT_AVXDIF16::inv0fft<n,1,0>()(x + pn, y + pn, Ws);
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
                OTFFT_AVXDIF16::inv0fft<n,1,0>()(x + kn, y + kn, Ws);
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
                OTFFT_AVXDIF16::inv0fft<n,1,0>()(x + pn, y + pn, Ws);
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
                OTFFT_AVXDIF16::inv0fft<n,1,0>()(x + kn, y + kn, Ws);
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

} /////////////////////////////////////////////////////////////////////////////

#endif // otfft_sixstepns_h
