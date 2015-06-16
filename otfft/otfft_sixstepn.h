/******************************************************************************
*  OTFFT Sixstep Normalized Version 4.0
******************************************************************************/

#ifndef otfft_sixstepn_h
#define otfft_sixstepn_h

///////////////////////////////////////////////////////////////////////////////

template <int log_N> struct fwdnffte
{
    static const int log_n = log_N / 2;
    static const int N = 1 << log_N;
    static const int n = 1 << log_n;
    
    template <class fft_t1, class fft_t2>
    void operator()(complex_vector x, complex_vector y, const_complex_vector W, const fft_t1& fft1, const fft_t2& fft2)
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        if (N < OMP_THRESHOLD1) {
            for (int k = 0; k < n; k += 2) {
                const int k_kn = k + k*n;
                std::swap(x[k_kn+1], x[k_kn+n]);
                for (int p = k+2; p < n; p += 2) {
                    const int p_kn = p + k*n;
                    const int k_pn = k + p*n;
                    std::swap(x[p_kn],     x[k_pn]);
                    std::swap(x[p_kn+1],   x[k_pn+n]);
                    std::swap(x[p_kn+n],   x[k_pn+1]);
                    std::swap(x[p_kn+1+n], x[k_pn+1+n]);
                }
            }
            for (int p = 0; p < n; p++) {
                const int pn = p*n;
                fft1.fwd0(x + pn, y + pn);
            }
            for (int p = 0; p < n; p += 2) {
                const int pp = p*p;
                const complex_t w = W[pp+p];
                complex_vector x_p_pn = x + p + p*n;
                const ymm w1 = mulpd2(rN, cmplx2(W[pp], w));
                const ymm w2 = mulpd2(rN, cmplx2(w, W[pp+2*p+1]));
                const ymm a = mulpz2(w1, getpz3<n>(x_p_pn));
                const ymm b = mulpz2(w2, getpz3<n>(x_p_pn+1));
                setpz2(x_p_pn,   a);
                setpz2(x_p_pn+n, b);
                for (int k = p+2; k < n; k += 2) {
                    const int kp = k*p;
                    complex_vector x_k_pn = x + k + p*n;
                    complex_vector x_p_kn = x + p + k*n;
                    const ymm w1 = mulpd2(rN, cmplx2(W[kp], W[kp+p]));
                    const ymm w2 = mulpd2(rN, cmplx2(W[kp+k], W[kp+k+p+1]));
                    const ymm a = mulpz2(w1, getpz3<n>(x_p_kn));
                    const ymm b = mulpz2(w2, getpz3<n>(x_p_kn+1));
                    const ymm c = mulpz2(w1, getpz2(x_k_pn));
                    const ymm d = mulpz2(w2, getpz2(x_k_pn+n));
                    setpz2(x_k_pn,   a);
                    setpz2(x_k_pn+n, b);
                    setpz3<n>(x_p_kn,   c);
                    setpz3<n>(x_p_kn+1, d);
                }
            }
            for (int k = 0; k < n; k++) {
                const int kn = k*n;
                fft1.fwd0(x + kn, y + kn);
            }
            for (int k = 0; k < n; k += 2) {
                const int k_kn = k + k*n;
                std::swap(x[k_kn+1], x[k_kn+n]);
                for (int p = k+2; p < n; p += 2) {
                    const int p_kn = p + k*n;
                    const int k_pn = k + p*n;
                    std::swap(x[p_kn],     x[k_pn]);
                    std::swap(x[p_kn+1],   x[k_pn+n]);
                    std::swap(x[p_kn+n],   x[k_pn+1]);
                    std::swap(x[p_kn+1+n], x[k_pn+1+n]);
                }
            }
        }
        else if (N < OMP_THRESHOLD2) //////////////////////////////////////////
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int k = 0; k < n; k += 2) {
                const int k_kn = k + k*n;
                std::swap(x[k_kn+1], x[k_kn+n]);
                for (int p = k+2; p < n; p += 2) {
                    const int p_kn = p + k*n;
                    const int k_pn = k + p*n;
                    std::swap(x[p_kn],     x[k_pn]);
                    std::swap(x[p_kn+1],   x[k_pn+n]);
                    std::swap(x[p_kn+n],   x[k_pn+1]);
                    std::swap(x[p_kn+1+n], x[k_pn+1+n]);
                }
            }
            #pragma omp for schedule(static)
            for (int p = 0; p < n; p++) {
                const int pn = p*n;
                fft1.fwd0(x + pn, y + pn);
            }
            #pragma omp for schedule(static)
            for (int p = 0; p < n; p += 2) {
                const int pp = p*p;
                const complex_t w = W[pp+p];
                complex_vector x_p_pn = x + p + p*n;
                const ymm w1 = mulpd2(rN, cmplx2(W[pp], w));
                const ymm w2 = mulpd2(rN, cmplx2(w, W[pp+2*p+1]));
                const ymm a = mulpz2(w1, getpz3<n>(x_p_pn));
                const ymm b = mulpz2(w2, getpz3<n>(x_p_pn+1));
                setpz2(x_p_pn,   a);
                setpz2(x_p_pn+n, b);
                for (int k = p+2; k < n; k += 2) {
                    const int kp = k*p;
                    complex_vector x_k_pn = x + k + p*n;
                    complex_vector x_p_kn = x + p + k*n;
                    const ymm w1 = mulpd2(rN, cmplx2(W[kp], W[kp+p]));
                    const ymm w2 = mulpd2(rN, cmplx2(W[kp+k], W[kp+k+p+1]));
                    const ymm a = mulpz2(w1, getpz3<n>(x_p_kn));
                    const ymm b = mulpz2(w2, getpz3<n>(x_p_kn+1));
                    const ymm c = mulpz2(w1, getpz2(x_k_pn));
                    const ymm d = mulpz2(w2, getpz2(x_k_pn+n));
                    setpz2(x_k_pn,   a);
                    setpz2(x_k_pn+n, b);
                    setpz3<n>(x_p_kn,   c);
                    setpz3<n>(x_p_kn+1, d);
                }
            }
            #pragma omp for schedule(static)
            for (int k = 0; k < n; k++) {
                const int kn = k*n;
                fft1.fwd0(x + kn, y + kn);
            }
            #pragma omp for schedule(static) nowait
            for (int k = 0; k < n; k += 2) {
                const int k_kn = k + k*n;
                std::swap(x[k_kn+1], x[k_kn+n]);
                for (int p = k+2; p < n; p += 2) {
                    const int p_kn = p + k*n;
                    const int k_pn = k + p*n;
                    std::swap(x[p_kn],     x[k_pn]);
                    std::swap(x[p_kn+1],   x[k_pn+n]);
                    std::swap(x[p_kn+n],   x[k_pn+1]);
                    std::swap(x[p_kn+1+n], x[k_pn+1+n]);
                }
            }
        }
        else //////////////////////////////////////////////////////////////////
        #pragma omp parallel
        {
            #pragma omp for schedule(guided)
            for (int k = 0; k < n; k += 2) {
                const int k_kn = k + k*n;
                std::swap(x[k_kn+1], x[k_kn+n]);
                for (int p = k+2; p < n; p += 2) {
                    const int p_kn = p + k*n;
                    const int k_pn = k + p*n;
                    std::swap(x[p_kn],     x[k_pn]);
                    std::swap(x[p_kn+1],   x[k_pn+n]);
                    std::swap(x[p_kn+n],   x[k_pn+1]);
                    std::swap(x[p_kn+1+n], x[k_pn+1+n]);
                }
            }
            #pragma omp for schedule(guided)
            for (int p = 0; p < n; p++) {
                const int pn = p*n;
                fft1.fwd0(x + pn, y + pn);
            }
            #pragma omp for schedule(guided)
            for (int p = 0; p < n; p += 2) {
                const int pp = p*p;
                const complex_t w = W[pp+p];
                complex_vector x_p_pn = x + p + p*n;
                const ymm w1 = mulpd2(rN, cmplx2(W[pp], w));
                const ymm w2 = mulpd2(rN, cmplx2(w, W[pp+2*p+1]));
                const ymm a = mulpz2(w1, getpz3<n>(x_p_pn));
                const ymm b = mulpz2(w2, getpz3<n>(x_p_pn+1));
                setpz2(x_p_pn,   a);
                setpz2(x_p_pn+n, b);
                for (int k = p+2; k < n; k += 2) {
                    const int kp = k*p;
                    complex_vector x_k_pn = x + k + p*n;
                    complex_vector x_p_kn = x + p + k*n;
                    const ymm w1 = mulpd2(rN, cmplx2(W[kp], W[kp+p]));
                    const ymm w2 = mulpd2(rN, cmplx2(W[kp+k], W[kp+k+p+1]));
                    const ymm a = mulpz2(w1, getpz3<n>(x_p_kn));
                    const ymm b = mulpz2(w2, getpz3<n>(x_p_kn+1));
                    const ymm c = mulpz2(w1, getpz2(x_k_pn));
                    const ymm d = mulpz2(w2, getpz2(x_k_pn+n));
                    setpz2(x_k_pn,   a);
                    setpz2(x_k_pn+n, b);
                    setpz3<n>(x_p_kn,   c);
                    setpz3<n>(x_p_kn+1, d);
                }
            }
            #pragma omp for schedule(guided)
            for (int k = 0; k < n; k++) {
                const int kn = k*n;
                fft1.fwd0(x + kn, y + kn);
            }
            #pragma omp for schedule(guided) nowait
            for (int k = 0; k < n; k += 2) {
                const int k_kn = k + k*n;
                std::swap(x[k_kn+1], x[k_kn+n]);
                for (int p = k+2; p < n; p += 2) {
                    const int p_kn = p + k*n;
                    const int k_pn = k + p*n;
                    std::swap(x[p_kn],     x[k_pn]);
                    std::swap(x[p_kn+1],   x[k_pn+n]);
                    std::swap(x[p_kn+n],   x[k_pn+1]);
                    std::swap(x[p_kn+1+n], x[k_pn+1+n]);
                }
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int log_N> struct invnffte
{
    static const int log_n = log_N / 2;
    static const int N = 1 << log_N;
    static const int n = 1 << log_n;
    
    template <class fft_t1, class fft_t2>
    void operator()(complex_vector x, complex_vector y, const_complex_vector W, const fft_t1& fft1, const fft_t2& fft2)
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        if (N < OMP_THRESHOLD1) {
            for (int k = 0; k < n; k += 2) {
                const int k_kn = k + k*n;
                std::swap(x[k_kn+1], x[k_kn+n]);
                for (int p = k+2; p < n; p += 2) {
                    const int p_kn = p + k*n;
                    const int k_pn = k + p*n;
                    std::swap(x[p_kn],     x[k_pn]);
                    std::swap(x[p_kn+1],   x[k_pn+n]);
                    std::swap(x[p_kn+n],   x[k_pn+1]);
                    std::swap(x[p_kn+1+n], x[k_pn+1+n]);
                }
            }
            for (int p = 0; p < n; p++) {
                const int pn = p*n;
                fft1.inv0(x + pn, y + pn);
            }
            for (int p = 0; p < n; p += 2) {
                const int N_pp = N-p*p;
                const complex_t w = W[N_pp-p];
                complex_vector x_p_pn = x + p + p*n;
                const ymm w1 = mulpd2(rN, cmplx2(W[N_pp], w));
                const ymm w2 = mulpd2(rN, cmplx2(w, W[N_pp-2*p-1]));
                const ymm a = mulpz2(w1, getpz3<n>(x_p_pn));
                const ymm b = mulpz2(w2, getpz3<n>(x_p_pn+1));
                setpz2(x_p_pn,   a);
                setpz2(x_p_pn+n, b);
                for (int k = p+2; k < n; k += 2) {
                    const int N_kp = N-k*p;
                    complex_vector x_k_pn = x + k + p*n;
                    complex_vector x_p_kn = x + p + k*n;
                    const ymm w1 = mulpd2(rN, cmplx2(W[N_kp], W[N_kp-p]));
                    const ymm w2 = mulpd2(rN, cmplx2(W[N_kp-k], W[N_kp-k-p-1]));
                    const ymm a = mulpz2(w1, getpz3<n>(x_p_kn));
                    const ymm b = mulpz2(w2, getpz3<n>(x_p_kn+1));
                    const ymm c = mulpz2(w1, getpz2(x_k_pn));
                    const ymm d = mulpz2(w2, getpz2(x_k_pn+n));
                    setpz2(x_k_pn,   a);
                    setpz2(x_k_pn+n, b);
                    setpz3<n>(x_p_kn,   c);
                    setpz3<n>(x_p_kn+1, d);
                }
            }
            for (int k = 0; k < n; k++) {
                const int kn = k*n;
                fft1.inv0(x + kn, y + kn);
            }
            for (int k = 0; k < n; k += 2) {
                const int k_kn = k + k*n;
                std::swap(x[k_kn+1], x[k_kn+n]);
                for (int p = k+2; p < n; p += 2) {
                    const int p_kn = p + k*n;
                    const int k_pn = k + p*n;
                    std::swap(x[p_kn],     x[k_pn]);
                    std::swap(x[p_kn+1],   x[k_pn+n]);
                    std::swap(x[p_kn+n],   x[k_pn+1]);
                    std::swap(x[p_kn+1+n], x[k_pn+1+n]);
                }
            }
        }
        else if (N < OMP_THRESHOLD2) //////////////////////////////////////////
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int k = 0; k < n; k += 2) {
                const int k_kn = k + k*n;
                std::swap(x[k_kn+1], x[k_kn+n]);
                for (int p = k+2; p < n; p += 2) {
                    const int p_kn = p + k*n;
                    const int k_pn = k + p*n;
                    std::swap(x[p_kn],     x[k_pn]);
                    std::swap(x[p_kn+1],   x[k_pn+n]);
                    std::swap(x[p_kn+n],   x[k_pn+1]);
                    std::swap(x[p_kn+1+n], x[k_pn+1+n]);
                }
            }
            #pragma omp for schedule(static)
            for (int p = 0; p < n; p++) {
                const int pn = p*n;
                fft1.inv0(x + pn, y + pn);
            }
            #pragma omp for schedule(static)
            for (int p = 0; p < n; p += 2) {
                const int N_pp = N-p*p;
                const complex_t w = W[N_pp-p];
                complex_vector x_p_pn = x + p + p*n;
                const ymm w1 = mulpd2(rN, cmplx2(W[N_pp], w));
                const ymm w2 = mulpd2(rN, cmplx2(w, W[N_pp-2*p-1]));
                const ymm a = mulpz2(w1, getpz3<n>(x_p_pn));
                const ymm b = mulpz2(w2, getpz3<n>(x_p_pn+1));
                setpz2(x_p_pn,   a);
                setpz2(x_p_pn+n, b);
                for (int k = p+2; k < n; k += 2) {
                    const int N_kp = N-k*p;
                    complex_vector x_k_pn = x + k + p*n;
                    complex_vector x_p_kn = x + p + k*n;
                    const ymm w1 = mulpd2(rN, cmplx2(W[N_kp], W[N_kp-p]));
                    const ymm w2 = mulpd2(rN, cmplx2(W[N_kp-k], W[N_kp-k-p-1]));
                    const ymm a = mulpz2(w1, getpz3<n>(x_p_kn));
                    const ymm b = mulpz2(w2, getpz3<n>(x_p_kn+1));
                    const ymm c = mulpz2(w1, getpz2(x_k_pn));
                    const ymm d = mulpz2(w2, getpz2(x_k_pn+n));
                    setpz2(x_k_pn,   a);
                    setpz2(x_k_pn+n, b);
                    setpz3<n>(x_p_kn,   c);
                    setpz3<n>(x_p_kn+1, d);
                }
            }
            #pragma omp for schedule(static)
            for (int k = 0; k < n; k++) {
                const int kn = k*n;
                fft1.inv0(x + kn, y + kn);
            }
            #pragma omp for schedule(static) nowait
            for (int k = 0; k < n; k += 2) {
                const int k_kn = k + k*n;
                std::swap(x[k_kn+1], x[k_kn+n]);
                for (int p = k+2; p < n; p += 2) {
                    const int p_kn = p + k*n;
                    const int k_pn = k + p*n;
                    std::swap(x[p_kn],     x[k_pn]);
                    std::swap(x[p_kn+1],   x[k_pn+n]);
                    std::swap(x[p_kn+n],   x[k_pn+1]);
                    std::swap(x[p_kn+1+n], x[k_pn+1+n]);
                }
            }
        }
        else //////////////////////////////////////////////////////////////////
        #pragma omp parallel
        {
            #pragma omp for schedule(guided)
            for (int k = 0; k < n; k += 2) {
                const int k_kn = k + k*n;
                std::swap(x[k_kn+1], x[k_kn+n]);
                for (int p = k+2; p < n; p += 2) {
                    const int p_kn = p + k*n;
                    const int k_pn = k + p*n;
                    std::swap(x[p_kn],     x[k_pn]);
                    std::swap(x[p_kn+1],   x[k_pn+n]);
                    std::swap(x[p_kn+n],   x[k_pn+1]);
                    std::swap(x[p_kn+1+n], x[k_pn+1+n]);
                }
            }
            #pragma omp for schedule(guided)
            for (int p = 0; p < n; p++) {
                const int pn = p*n;
                fft1.inv0(x + pn, y + pn);
            }
            #pragma omp for schedule(guided)
            for (int p = 0; p < n; p += 2) {
                const int N_pp = N-p*p;
                const complex_t w = W[N_pp-p];
                complex_vector x_p_pn = x + p + p*n;
                const ymm w1 = mulpd2(rN, cmplx2(W[N_pp], w));
                const ymm w2 = mulpd2(rN, cmplx2(w, W[N_pp-2*p-1]));
                const ymm a = mulpz2(w1, getpz3<n>(x_p_pn));
                const ymm b = mulpz2(w2, getpz3<n>(x_p_pn+1));
                setpz2(x_p_pn,   a);
                setpz2(x_p_pn+n, b);
                for (int k = p+2; k < n; k += 2) {
                    const int N_kp = N-k*p;
                    complex_vector x_k_pn = x + k + p*n;
                    complex_vector x_p_kn = x + p + k*n;
                    const ymm w1 = mulpd2(rN, cmplx2(W[N_kp], W[N_kp-p]));
                    const ymm w2 = mulpd2(rN, cmplx2(W[N_kp-k], W[N_kp-k-p-1]));
                    const ymm a = mulpz2(w1, getpz3<n>(x_p_kn));
                    const ymm b = mulpz2(w2, getpz3<n>(x_p_kn+1));
                    const ymm c = mulpz2(w1, getpz2(x_k_pn));
                    const ymm d = mulpz2(w2, getpz2(x_k_pn+n));
                    setpz2(x_k_pn,   a);
                    setpz2(x_k_pn+n, b);
                    setpz3<n>(x_p_kn,   c);
                    setpz3<n>(x_p_kn+1, d);
                }
            }
            #pragma omp for schedule(guided)
            for (int k = 0; k < n; k++) {
                const int kn = k*n;
                fft1.inv0(x + kn, y + kn);
            }
            #pragma omp for schedule(guided) nowait
            for (int k = 0; k < n; k += 2) {
                const int k_kn = k + k*n;
                std::swap(x[k_kn+1], x[k_kn+n]);
                for (int p = k+2; p < n; p += 2) {
                    const int p_kn = p + k*n;
                    const int k_pn = k + p*n;
                    std::swap(x[p_kn],     x[k_pn]);
                    std::swap(x[p_kn+1],   x[k_pn+n]);
                    std::swap(x[p_kn+n],   x[k_pn+1]);
                    std::swap(x[p_kn+1+n], x[k_pn+1+n]);
                }
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int log_N> struct fwdnffto
{
    static const int log_n = log_N / 2;
    static const int log_m = log_N - log_n;
    static const int N = 1 << log_N;
    static const int n = 1 << log_n;
    static const int m = 1 << log_m;
    
    template <class fft_t1, class fft_t2>
    void operator()(complex_vector x, complex_vector y, const_complex_vector W, const fft_t1& fft1, const fft_t2& fft2)
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        if (N < OMP_THRESHOLD1) {
            for (int p = 0; p < m; p += 2) {
                for (int k = 0; k < n; k += 2) {
                    const int k_pn = k + p*n;
                    const int p_km = p + k*m;
                    y[k_pn]     = x[p_km];
                    y[k_pn+1]   = x[p_km+m];
                    y[k_pn+n]   = x[p_km+1];
                    y[k_pn+1+n] = x[p_km+1+m];
                }
            }
            for (int p = 0; p < m; p++) {
                const int pn = p*n;
                fft1.fwd0(y + pn, x + pn);
            }
            for (int k = 0; k < n; k += 2) {
                for (int p = 0; p < m; p += 2) {
                    const int kp = k*p;
                    complex_vector x_p_km = x + p + k*m;
                    complex_vector y_k_pn = y + k + p*n;
                    const ymm w1 = mulpd2(rN, cmplx2(W[kp], W[kp+k]));
                    const ymm w2 = mulpd2(rN, cmplx2(W[kp+p], W[kp+p+k+1]));
                    setpz2(x_p_km,   mulpz2(w1, getpz3<n>(y_k_pn)));
                    setpz2(x_p_km+m, mulpz2(w2, getpz3<n>(y_k_pn+1)));
                }
            }
            for (int k = 0; k < n; k++) {
                const int km = k*m;
                fft2.fwd0o(x + km, y + km);
            }
            for (int p = 0; p < m; p += 2) {
                for (int k = 0; k < n; k += 2) {
                    const int k_pn = k + p*n;
                    const int p_km = p + k*m;
                    x[k_pn]     = y[p_km];
                    x[k_pn+1]   = y[p_km+m];
                    x[k_pn+n]   = y[p_km+1];
                    x[k_pn+1+n] = y[p_km+1+m];
                }
            }
        }
        else if (N < OMP_THRESHOLD2) //////////////////////////////////////////
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int p = 0; p < m; p += 2) {
                for (int k = 0; k < n; k += 2) {
                    const int k_pn = k + p*n;
                    const int p_km = p + k*m;
                    y[k_pn]     = x[p_km];
                    y[k_pn+1]   = x[p_km+m];
                    y[k_pn+n]   = x[p_km+1];
                    y[k_pn+1+n] = x[p_km+1+m];
                }
            }
            #pragma omp for schedule(static)
            for (int p = 0; p < m; p++) {
                const int pn = p*n;
                fft1.fwd0(y + pn, x + pn);
            }
            #pragma omp for schedule(static)
            for (int k = 0; k < n; k += 2) {
                for (int p = 0; p < m; p += 2) {
                    const int kp = k*p;
                    complex_vector x_p_km = x + p + k*m;
                    complex_vector y_k_pn = y + k + p*n;
                    const ymm w1 = mulpd2(rN, cmplx2(W[kp], W[kp+k]));
                    const ymm w2 = mulpd2(rN, cmplx2(W[kp+p], W[kp+p+k+1]));
                    setpz2(x_p_km,   mulpz2(w1, getpz3<n>(y_k_pn)));
                    setpz2(x_p_km+m, mulpz2(w2, getpz3<n>(y_k_pn+1)));
                }
            }
            #pragma omp for schedule(static)
            for (int k = 0; k < n; k++) {
                const int km = k*m;
                fft2.fwd0o(x + km, y + km);
            }
            #pragma omp for schedule(static) nowait
            for (int p = 0; p < m; p += 2) {
                for (int k = 0; k < n; k += 2) {
                    const int k_pn = k + p*n;
                    const int p_km = p + k*m;
                    x[k_pn]     = y[p_km];
                    x[k_pn+1]   = y[p_km+m];
                    x[k_pn+n]   = y[p_km+1];
                    x[k_pn+1+n] = y[p_km+1+m];
                }
            }
        }
        else //////////////////////////////////////////////////////////////////
        #pragma omp parallel
        {
            #pragma omp for schedule(guided)
            for (int p = 0; p < m; p += 2) {
                for (int k = 0; k < n; k += 2) {
                    const int k_pn = k + p*n;
                    const int p_km = p + k*m;
                    y[k_pn]     = x[p_km];
                    y[k_pn+1]   = x[p_km+m];
                    y[k_pn+n]   = x[p_km+1];
                    y[k_pn+1+n] = x[p_km+1+m];
                }
            }
            #pragma omp for schedule(guided)
            for (int p = 0; p < m; p++) {
                const int pn = p*n;
                fft1.fwd0(y + pn, x + pn);
            }
            #pragma omp for schedule(guided)
            for (int k = 0; k < n; k += 2) {
                for (int p = 0; p < m; p += 2) {
                    const int kp = k*p;
                    complex_vector x_p_km = x + p + k*m;
                    complex_vector y_k_pn = y + k + p*n;
                    const ymm w1 = mulpd2(rN, cmplx2(W[kp], W[kp+k]));
                    const ymm w2 = mulpd2(rN, cmplx2(W[kp+p], W[kp+p+k+1]));
                    setpz2(x_p_km,   mulpz2(w1, getpz3<n>(y_k_pn)));
                    setpz2(x_p_km+m, mulpz2(w2, getpz3<n>(y_k_pn+1)));
                }
            }
            #pragma omp for schedule(guided)
            for (int k = 0; k < n; k++) {
                const int km = k*m;
                fft2.fwd0o(x + km, y + km);
            }
            #pragma omp for schedule(guided) nowait
            for (int p = 0; p < m; p += 2) {
                for (int k = 0; k < n; k += 2) {
                    const int k_pn = k + p*n;
                    const int p_km = p + k*m;
                    x[k_pn]     = y[p_km];
                    x[k_pn+1]   = y[p_km+m];
                    x[k_pn+n]   = y[p_km+1];
                    x[k_pn+1+n] = y[p_km+1+m];
                }
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

template <int log_N> struct invnffto
{
    static const int log_n = log_N / 2;
    static const int log_m = log_N - log_n;
    static const int N = 1 << log_N;
    static const int n = 1 << log_n;
    static const int m = 1 << log_m;
    
    template <class fft_t1, class fft_t2>
    void operator()(complex_vector x, complex_vector y, const_complex_vector W, const fft_t1& fft1, const fft_t2& fft2)
    {
        static const ymm rN = { 1.0/N, 1.0/N, 1.0/N, 1.0/N };
        if (N < OMP_THRESHOLD1) {
            for (int p = 0; p < m; p += 2) {
                for (int k = 0; k < n; k += 2) {
                    const int k_pn = k + p*n;
                    const int p_km = p + k*m;
                    y[k_pn]     = x[p_km];
                    y[k_pn+1]   = x[p_km+m];
                    y[k_pn+n]   = x[p_km+1];
                    y[k_pn+1+n] = x[p_km+1+m];
                }
            }
            for (int p = 0; p < m; p++) {
                const int pn = p*n;
                fft1.inv0(y + pn, x + pn);
            }
            for (int k = 0; k < n; k += 2) {
                for (int p = 0; p < m; p += 2) {
                    const int N_kp = N-k*p;
                    complex_vector x_p_km = x + p + k*m;
                    complex_vector y_k_pn = y + k + p*n;
                    const ymm w1 = mulpd2(rN, cmplx2(W[N_kp], W[N_kp-k]));
                    const ymm w2 = mulpd2(rN, cmplx2(W[N_kp-p], W[N_kp-p-k-1]));
                    setpz2(x_p_km,   mulpz2(w1, getpz3<n>(y_k_pn)));
                    setpz2(x_p_km+m, mulpz2(w2, getpz3<n>(y_k_pn+1)));
                }
            }
            for (int k = 0; k < n; k++) {
                const int km = k*m;
                fft2.inv0o(x + km, y + km);
            }
            for (int p = 0; p < m; p += 2) {
                for (int k = 0; k < n; k += 2) {
                    const int k_pn = k + p*n;
                    const int p_km = p + k*m;
                    x[k_pn]     = y[p_km];
                    x[k_pn+1]   = y[p_km+m];
                    x[k_pn+n]   = y[p_km+1];
                    x[k_pn+1+n] = y[p_km+1+m];
                }
            }
        }
        else if (N < OMP_THRESHOLD2) //////////////////////////////////////////
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int p = 0; p < m; p += 2) {
                for (int k = 0; k < n; k += 2) {
                    const int k_pn = k + p*n;
                    const int p_km = p + k*m;
                    y[k_pn]     = x[p_km];
                    y[k_pn+1]   = x[p_km+m];
                    y[k_pn+n]   = x[p_km+1];
                    y[k_pn+1+n] = x[p_km+1+m];
                }
            }
            #pragma omp for schedule(static)
            for (int p = 0; p < m; p++) {
                const int pn = p*n;
                fft1.inv0(y + pn, x + pn);
            }
            #pragma omp for schedule(static)
            for (int k = 0; k < n; k += 2) {
                for (int p = 0; p < m; p += 2) {
                    const int N_kp = N-k*p;
                    complex_vector x_p_km = x + p + k*m;
                    complex_vector y_k_pn = y + k + p*n;
                    const ymm w1 = mulpd2(rN, cmplx2(W[N_kp], W[N_kp-k]));
                    const ymm w2 = mulpd2(rN, cmplx2(W[N_kp-p], W[N_kp-p-k-1]));
                    setpz2(x_p_km,   mulpz2(w1, getpz3<n>(y_k_pn)));
                    setpz2(x_p_km+m, mulpz2(w2, getpz3<n>(y_k_pn+1)));
                }
            }
            #pragma omp for schedule(static)
            for (int k = 0; k < n; k++) {
                const int km = k*m;
                fft2.inv0o(x + km, y + km);
            }
            #pragma omp for schedule(static) nowait
            for (int p = 0; p < m; p += 2) {
                for (int k = 0; k < n; k += 2) {
                    const int k_pn = k + p*n;
                    const int p_km = p + k*m;
                    x[k_pn]     = y[p_km];
                    x[k_pn+1]   = y[p_km+m];
                    x[k_pn+n]   = y[p_km+1];
                    x[k_pn+1+n] = y[p_km+1+m];
                }
            }
        }
        else //////////////////////////////////////////////////////////////////
        #pragma omp parallel
        {
            #pragma omp for schedule(guided)
            for (int p = 0; p < m; p += 2) {
                for (int k = 0; k < n; k += 2) {
                    const int k_pn = k + p*n;
                    const int p_km = p + k*m;
                    y[k_pn]     = x[p_km];
                    y[k_pn+1]   = x[p_km+m];
                    y[k_pn+n]   = x[p_km+1];
                    y[k_pn+1+n] = x[p_km+1+m];
                }
            }
            #pragma omp for schedule(guided)
            for (int p = 0; p < m; p++) {
                const int pn = p*n;
                fft1.inv0(y + pn, x + pn);
            }
            #pragma omp for schedule(guided)
            for (int k = 0; k < n; k += 2) {
                for (int p = 0; p < m; p += 2) {
                    const int N_kp = N-k*p;
                    complex_vector x_p_km = x + p + k*m;
                    complex_vector y_k_pn = y + k + p*n;
                    const ymm w1 = mulpd2(rN, cmplx2(W[N_kp], W[N_kp-k]));
                    const ymm w2 = mulpd2(rN, cmplx2(W[N_kp-p], W[N_kp-p-k-1]));
                    setpz2(x_p_km,   mulpz2(w1, getpz3<n>(y_k_pn)));
                    setpz2(x_p_km+m, mulpz2(w2, getpz3<n>(y_k_pn+1)));
                }
            }
            #pragma omp for schedule(guided)
            for (int k = 0; k < n; k++) {
                const int km = k*m;
                fft2.inv0o(x + km, y + km);
            }
            #pragma omp for schedule(guided) nowait
            for (int p = 0; p < m; p += 2) {
                for (int k = 0; k < n; k += 2) {
                    const int k_pn = k + p*n;
                    const int p_km = p + k*m;
                    x[k_pn]     = y[p_km];
                    x[k_pn+1]   = y[p_km+m];
                    x[k_pn+n]   = y[p_km+1];
                    x[k_pn+1+n] = y[p_km+1+m];
                }
            }
        }
    }
};

///////////////////////////////////////////////////////////////////////////////

#endif // otfft_sixstepn_h
