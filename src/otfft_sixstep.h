/******************************************************************************
*  OTFFT Sixstep Version 4.0
******************************************************************************/

#ifndef otfft_sixstep_h
#define otfft_sixstep_h

#include <cmath>
#include <algorithm>

#include "otfft/otfft_misc.h"
//#include "otfft_difavx.h"
//#include "otfft_ditavx.h"
#include "otfft_difavx8.h"
//#include "otfft_ditavx8.h"

namespace OTFFT_Sixstep { /////////////////////////////////////////////////////

using namespace OTFFT_MISC;

const int OMP_THRESHOLD1  = 1<<13;
const int OMP_THRESHOLD2  = 1<<18;

///////////////////////////////////////////////////////////////////////////////

template <int log_N> struct fwd0ffte
{
    static const int log_n = log_N / 2;
    static const int N = 1 << log_N;
    static const int n = 1 << log_n;
    
    template <class fft_t1, class fft_t2>
    void operator()(complex_vector x, complex_vector y, const_complex_vector W, const fft_t1& fft1, const fft_t2& fft2)
    {
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
                const ymm w1 = cmplx2(W[pp], w);
                const ymm w2 = cmplx2(w, W[pp+2*p+1]);
                const ymm a = mulpz2(w1, getpz3<n>(x_p_pn));
                const ymm b = mulpz2(w2, getpz3<n>(x_p_pn+1));
                setpz2(x_p_pn,   a);
                setpz2(x_p_pn+n, b);
                for (int k = p+2; k < n; k += 2) {
                    const int kp = k*p;
                    complex_vector x_k_pn = x + k + p*n;
                    complex_vector x_p_kn = x + p + k*n;
                    const ymm w1 = cmplx2(W[kp], W[kp+p]);
                    const ymm w2 = cmplx2(W[kp+k], W[kp+k+p+1]);
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
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
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
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 0; p < n; p++) {
                const int pn = p*n;
                fft1.fwd0(x + pn, y + pn);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 0; p < n; p += 2) {
                const int pp = p*p;
                const complex_t w = W[pp+p];
                complex_vector x_p_pn = x + p + p*n;
                const ymm w1 = cmplx2(W[pp], w);
                const ymm w2 = cmplx2(w, W[pp+2*p+1]);
                const ymm a = mulpz2(w1, getpz3<n>(x_p_pn));
                const ymm b = mulpz2(w2, getpz3<n>(x_p_pn+1));
                setpz2(x_p_pn,   a);
                setpz2(x_p_pn+n, b);
                for (int k = p+2; k < n; k += 2) {
                    const int kp = k*p;
                    complex_vector x_k_pn = x + k + p*n;
                    complex_vector x_p_kn = x + p + k*n;
                    const ymm w1 = cmplx2(W[kp], W[kp+p]);
                    const ymm w2 = cmplx2(W[kp+k], W[kp+k+p+1]);
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
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int k = 0; k < n; k++) {
                const int kn = k*n;
                fft1.fwd0(x + kn, y + kn);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static) nowait
#endif
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
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
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
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int p = 0; p < n; p++) {
                const int pn = p*n;
                fft1.fwd0(x + pn, y + pn);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int p = 0; p < n; p += 2) {
                const int pp = p*p;
                const complex_t w = W[pp+p];
                complex_vector x_p_pn = x + p + p*n;
                const ymm w1 = cmplx2(W[pp], w);
                const ymm w2 = cmplx2(w, W[pp+2*p+1]);
                const ymm a = mulpz2(w1, getpz3<n>(x_p_pn));
                const ymm b = mulpz2(w2, getpz3<n>(x_p_pn+1));
                setpz2(x_p_pn,   a);
                setpz2(x_p_pn+n, b);
                for (int k = p+2; k < n; k += 2) {
                    const int kp = k*p;
                    complex_vector x_k_pn = x + k + p*n;
                    complex_vector x_p_kn = x + p + k*n;
                    const ymm w1 = cmplx2(W[kp], W[kp+p]);
                    const ymm w2 = cmplx2(W[kp+k], W[kp+k+p+1]);
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
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int k = 0; k < n; k++) {
                const int kn = k*n;
                fft1.fwd0(x + kn, y + kn);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided) nowait
#endif
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

template <int log_N> struct inv0ffte
{
    static const int log_n = log_N / 2;
    static const int N = 1 << log_N;
    static const int n = 1 << log_n;
    
    template <class fft_t1, class fft_t2>
    void operator()(complex_vector x, complex_vector y, const_complex_vector W, const fft_t1& fft1, const fft_t2& fft2)
    {
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
                const ymm w1 = cmplx2(W[N_pp], w);
                const ymm w2 = cmplx2(w, W[N_pp-2*p-1]);
                const ymm a = mulpz2(w1, getpz3<n>(x_p_pn));
                const ymm b = mulpz2(w2, getpz3<n>(x_p_pn+1));
                setpz2(x_p_pn,   a);
                setpz2(x_p_pn+n, b);
                for (int k = p+2; k < n; k += 2) {
                    const int N_kp = N-k*p;
                    complex_vector x_k_pn = x + k + p*n;
                    complex_vector x_p_kn = x + p + k*n;
                    const ymm w1 = cmplx2(W[N_kp], W[N_kp-p]);
                    const ymm w2 = cmplx2(W[N_kp-k], W[N_kp-k-p-1]);
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
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
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
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 0; p < n; p++) {
                const int pn = p*n;
                fft1.inv0(x + pn, y + pn);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 0; p < n; p += 2) {
                const int N_pp = N-p*p;
                const complex_t w = W[N_pp-p];
                complex_vector x_p_pn = x + p + p*n;
                const ymm w1 = cmplx2(W[N_pp], w);
                const ymm w2 = cmplx2(w, W[N_pp-2*p-1]);
                const ymm a = mulpz2(w1, getpz3<n>(x_p_pn));
                const ymm b = mulpz2(w2, getpz3<n>(x_p_pn+1));
                setpz2(x_p_pn,   a);
                setpz2(x_p_pn+n, b);
                for (int k = p+2; k < n; k += 2) {
                    const int N_kp = N-k*p;
                    complex_vector x_k_pn = x + k + p*n;
                    complex_vector x_p_kn = x + p + k*n;
                    const ymm w1 = cmplx2(W[N_kp], W[N_kp-p]);
                    const ymm w2 = cmplx2(W[N_kp-k], W[N_kp-k-p-1]);
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
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int k = 0; k < n; k++) {
                const int kn = k*n;
                fft1.inv0(x + kn, y + kn);
            }
#ifdef _OPENMP
            #pragma omp for schedule(static) nowait
#endif
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
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
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
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int p = 0; p < n; p++) {
                const int pn = p*n;
                fft1.inv0(x + pn, y + pn);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int p = 0; p < n; p += 2) {
                const int N_pp = N-p*p;
                const complex_t w = W[N_pp-p];
                complex_vector x_p_pn = x + p + p*n;
                const ymm w1 = cmplx2(W[N_pp], w);
                const ymm w2 = cmplx2(w, W[N_pp-2*p-1]);
                const ymm a = mulpz2(w1, getpz3<n>(x_p_pn));
                const ymm b = mulpz2(w2, getpz3<n>(x_p_pn+1));
                setpz2(x_p_pn,   a);
                setpz2(x_p_pn+n, b);
                for (int k = p+2; k < n; k += 2) {
                    const int N_kp = N-k*p;
                    complex_vector x_k_pn = x + k + p*n;
                    complex_vector x_p_kn = x + p + k*n;
                    const ymm w1 = cmplx2(W[N_kp], W[N_kp-p]);
                    const ymm w2 = cmplx2(W[N_kp-k], W[N_kp-k-p-1]);
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
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
            for (int k = 0; k < n; k++) {
                const int kn = k*n;
                fft1.inv0(x + kn, y + kn);
            }
#ifdef _OPENMP
            #pragma omp for schedule(guided) nowait
#endif
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

template <int log_N> struct fwd0ffto
{
    static const int log_n = log_N / 2;
    static const int log_m = log_N - log_n;
    static const int N = 1 << log_N;
    static const int n = 1 << log_n;
    static const int m = 1 << log_m;
    
    template <class fft_t1, class fft_t2>
    void operator()(complex_vector x, complex_vector y, const_complex_vector W, const fft_t1& fft1, const fft_t2& fft2)
    {
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
                    const ymm w1 = cmplx2(W[kp], W[kp+k]);
                    const ymm w2 = cmplx2(W[kp+p], W[kp+p+k+1]);
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
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
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
            for (int k = 0; k < n; k += 2) {
                for (int p = 0; p < m; p += 2) {
                    const int kp = k*p;
                    complex_vector x_p_km = x + p + k*m;
                    complex_vector y_k_pn = y + k + p*n;
                    const ymm w1 = cmplx2(W[kp], W[kp+k]);
                    const ymm w2 = cmplx2(W[kp+p], W[kp+p+k+1]);
                    setpz2(x_p_km,   mulpz2(w1, getpz3<n>(y_k_pn)));
                    setpz2(x_p_km+m, mulpz2(w2, getpz3<n>(y_k_pn+1)));
                }
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
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
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
            for (int k = 0; k < n; k += 2) {
                for (int p = 0; p < m; p += 2) {
                    const int kp = k*p;
                    complex_vector x_p_km = x + p + k*m;
                    complex_vector y_k_pn = y + k + p*n;
                    const ymm w1 = cmplx2(W[kp], W[kp+k]);
                    const ymm w2 = cmplx2(W[kp+p], W[kp+p+k+1]);
                    setpz2(x_p_km,   mulpz2(w1, getpz3<n>(y_k_pn)));
                    setpz2(x_p_km+m, mulpz2(w2, getpz3<n>(y_k_pn+1)));
                }
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

template <int log_N> struct inv0ffto
{
    static const int log_n = log_N / 2;
    static const int log_m = log_N - log_n;
    static const int N = 1 << log_N;
    static const int n = 1 << log_n;
    static const int m = 1 << log_m;
    
    template <class fft_t1, class fft_t2>
    void operator()(complex_vector x, complex_vector y, const_complex_vector W, const fft_t1& fft1, const fft_t2& fft2)
    {
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
                    const ymm w1 = cmplx2(W[N_kp], W[N_kp-k]);
                    const ymm w2 = cmplx2(W[N_kp-p], W[N_kp-p-k-1]);
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
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
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
            for (int k = 0; k < n; k += 2) {
                for (int p = 0; p < m; p += 2) {
                    const int N_kp = N-k*p;
                    complex_vector x_p_km = x + p + k*m;
                    complex_vector y_k_pn = y + k + p*n;
                    const ymm w1 = cmplx2(W[N_kp], W[N_kp-k]);
                    const ymm w2 = cmplx2(W[N_kp-p], W[N_kp-p-k-1]);
                    setpz2(x_p_km,   mulpz2(w1, getpz3<n>(y_k_pn)));
                    setpz2(x_p_km+m, mulpz2(w2, getpz3<n>(y_k_pn+1)));
                }
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
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(guided)
#endif
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
            for (int k = 0; k < n; k += 2) {
                for (int p = 0; p < m; p += 2) {
                    const int N_kp = N-k*p;
                    complex_vector x_p_km = x + p + k*m;
                    complex_vector y_k_pn = y + k + p*n;
                    const ymm w1 = cmplx2(W[N_kp], W[N_kp-k]);
                    const ymm w2 = cmplx2(W[N_kp-p], W[N_kp-p-k-1]);
                    setpz2(x_p_km,   mulpz2(w1, getpz3<n>(y_k_pn)));
                    setpz2(x_p_km+m, mulpz2(w2, getpz3<n>(y_k_pn+1)));
                }
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

#include "otfft_sixstepn.h"

///////////////////////////////////////////////////////////////////////////////

struct FFT0
{
    int N, log_N;
    simd_array<complex_t> weight;
    complex_t* W;
    //OTFFT_DIFAVX::FFT0 fft1, fft2;
    //OTFFT_DITAVX::FFT0 fft1, fft2;
    OTFFT_DIFAVX8::FFT0 fft1, fft2;
    //OTFFT_DITAVX8::FFT0 fft1, fft2;

    FFT0() : N(0), log_N(0), W(0) {}
    FFT0(int n)
    {
        for (log_N = 0; n > 1; n >>= 1) log_N++;
        setup2(log_N);
    }

    void setup(int n)
    {
        for (log_N = 0; n > 1; n >>= 1) log_N++;
        setup2(log_N);
    }

    void setup2(int n)
    {
        log_N = n; N = 1 << n;
        weight.setup(N+1); W = &weight;
        if (N < 4) fft1.setup2(n);
        else { fft1.setup2(n/2); fft2.setup2(n - n/2); }
        init_W(N, W);
    }

    inline void fwd0(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: fft1.fwd0(x, y); break;
        case  2: fwd0ffte< 2>()(x, y, W, fft1, fft2); break;
        case  3: fwd0ffto< 3>()(x, y, W, fft1, fft2); break;
        case  4: fwd0ffte< 4>()(x, y, W, fft1, fft2); break;
        case  5: fwd0ffto< 5>()(x, y, W, fft1, fft2); break;
        case  6: fwd0ffte< 6>()(x, y, W, fft1, fft2); break;
        case  7: fwd0ffto< 7>()(x, y, W, fft1, fft2); break;
        case  8: fwd0ffte< 8>()(x, y, W, fft1, fft2); break;
        case  9: fwd0ffto< 9>()(x, y, W, fft1, fft2); break;
        case 10: fwd0ffte<10>()(x, y, W, fft1, fft2); break;
        case 11: fwd0ffto<11>()(x, y, W, fft1, fft2); break;
        case 12: fwd0ffte<12>()(x, y, W, fft1, fft2); break;
        case 13: fwd0ffto<13>()(x, y, W, fft1, fft2); break;
        case 14: fwd0ffte<14>()(x, y, W, fft1, fft2); break;
        case 15: fwd0ffto<15>()(x, y, W, fft1, fft2); break;
        case 16: fwd0ffte<16>()(x, y, W, fft1, fft2); break;
        case 17: fwd0ffto<17>()(x, y, W, fft1, fft2); break;
        case 18: fwd0ffte<18>()(x, y, W, fft1, fft2); break;
        case 19: fwd0ffto<19>()(x, y, W, fft1, fft2); break;
        case 20: fwd0ffte<20>()(x, y, W, fft1, fft2); break;
        case 21: fwd0ffto<21>()(x, y, W, fft1, fft2); break;
        case 22: fwd0ffte<22>()(x, y, W, fft1, fft2); break;
        case 23: fwd0ffto<23>()(x, y, W, fft1, fft2); break;
        case 24: fwd0ffte<24>()(x, y, W, fft1, fft2); break;
        }
    }

    inline void fwd(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: fft1.fwd(x, y); break;
        case  2: fwdnffte< 2>()(x, y, W, fft1, fft2); break;
        case  3: fwdnffto< 3>()(x, y, W, fft1, fft2); break;
        case  4: fwdnffte< 4>()(x, y, W, fft1, fft2); break;
        case  5: fwdnffto< 5>()(x, y, W, fft1, fft2); break;
        case  6: fwdnffte< 6>()(x, y, W, fft1, fft2); break;
        case  7: fwdnffto< 7>()(x, y, W, fft1, fft2); break;
        case  8: fwdnffte< 8>()(x, y, W, fft1, fft2); break;
        case  9: fwdnffto< 9>()(x, y, W, fft1, fft2); break;
        case 10: fwdnffte<10>()(x, y, W, fft1, fft2); break;
        case 11: fwdnffto<11>()(x, y, W, fft1, fft2); break;
        case 12: fwdnffte<12>()(x, y, W, fft1, fft2); break;
        case 13: fwdnffto<13>()(x, y, W, fft1, fft2); break;
        case 14: fwdnffte<14>()(x, y, W, fft1, fft2); break;
        case 15: fwdnffto<15>()(x, y, W, fft1, fft2); break;
        case 16: fwdnffte<16>()(x, y, W, fft1, fft2); break;
        case 17: fwdnffto<17>()(x, y, W, fft1, fft2); break;
        case 18: fwdnffte<18>()(x, y, W, fft1, fft2); break;
        case 19: fwdnffto<19>()(x, y, W, fft1, fft2); break;
        case 20: fwdnffte<20>()(x, y, W, fft1, fft2); break;
        case 21: fwdnffto<21>()(x, y, W, fft1, fft2); break;
        case 22: fwdnffte<22>()(x, y, W, fft1, fft2); break;
        case 23: fwdnffto<23>()(x, y, W, fft1, fft2); break;
        case 24: fwdnffte<24>()(x, y, W, fft1, fft2); break;
        }
    }

    inline void fwdn(complex_vector x, complex_vector y) const { fwd(x, y); }

    inline void inv0(complex_vector x, complex_vector y) const { inv(x, y); }

    inline void inv(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: fft1.inv0(x, y); break;
        case  2: inv0ffte< 2>()(x, y, W, fft1, fft2); break;
        case  3: inv0ffto< 3>()(x, y, W, fft1, fft2); break;
        case  4: inv0ffte< 4>()(x, y, W, fft1, fft2); break;
        case  5: inv0ffto< 5>()(x, y, W, fft1, fft2); break;
        case  6: inv0ffte< 6>()(x, y, W, fft1, fft2); break;
        case  7: inv0ffto< 7>()(x, y, W, fft1, fft2); break;
        case  8: inv0ffte< 8>()(x, y, W, fft1, fft2); break;
        case  9: inv0ffto< 9>()(x, y, W, fft1, fft2); break;
        case 10: inv0ffte<10>()(x, y, W, fft1, fft2); break;
        case 11: inv0ffto<11>()(x, y, W, fft1, fft2); break;
        case 12: inv0ffte<12>()(x, y, W, fft1, fft2); break;
        case 13: inv0ffto<13>()(x, y, W, fft1, fft2); break;
        case 14: inv0ffte<14>()(x, y, W, fft1, fft2); break;
        case 15: inv0ffto<15>()(x, y, W, fft1, fft2); break;
        case 16: inv0ffte<16>()(x, y, W, fft1, fft2); break;
        case 17: inv0ffto<17>()(x, y, W, fft1, fft2); break;
        case 18: inv0ffte<18>()(x, y, W, fft1, fft2); break;
        case 19: inv0ffto<19>()(x, y, W, fft1, fft2); break;
        case 20: inv0ffte<20>()(x, y, W, fft1, fft2); break;
        case 21: inv0ffto<21>()(x, y, W, fft1, fft2); break;
        case 22: inv0ffte<22>()(x, y, W, fft1, fft2); break;
        case 23: inv0ffto<23>()(x, y, W, fft1, fft2); break;
        case 24: inv0ffte<24>()(x, y, W, fft1, fft2); break;
        }
    }

    inline void invn(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: fft1.invn(x, y); break;
        case  2: invnffte< 2>()(x, y, W, fft1, fft2); break;
        case  3: invnffto< 3>()(x, y, W, fft1, fft2); break;
        case  4: invnffte< 4>()(x, y, W, fft1, fft2); break;
        case  5: invnffto< 5>()(x, y, W, fft1, fft2); break;
        case  6: invnffte< 6>()(x, y, W, fft1, fft2); break;
        case  7: invnffto< 7>()(x, y, W, fft1, fft2); break;
        case  8: invnffte< 8>()(x, y, W, fft1, fft2); break;
        case  9: invnffto< 9>()(x, y, W, fft1, fft2); break;
        case 10: invnffte<10>()(x, y, W, fft1, fft2); break;
        case 11: invnffto<11>()(x, y, W, fft1, fft2); break;
        case 12: invnffte<12>()(x, y, W, fft1, fft2); break;
        case 13: invnffto<13>()(x, y, W, fft1, fft2); break;
        case 14: invnffto<14>()(x, y, W, fft1, fft2); break;
        case 15: invnffto<15>()(x, y, W, fft1, fft2); break;
        case 16: invnffte<16>()(x, y, W, fft1, fft2); break;
        case 17: invnffto<17>()(x, y, W, fft1, fft2); break;
        case 18: invnffte<18>()(x, y, W, fft1, fft2); break;
        case 19: invnffto<19>()(x, y, W, fft1, fft2); break;
        case 20: invnffte<20>()(x, y, W, fft1, fft2); break;
        case 21: invnffto<21>()(x, y, W, fft1, fft2); break;
        case 22: invnffte<22>()(x, y, W, fft1, fft2); break;
        case 23: invnffto<23>()(x, y, W, fft1, fft2); break;
        case 24: invnffte<24>()(x, y, W, fft1, fft2); break;
        }
    }
};

struct FFT1
{
    int N, log_N;
    simd_array<complex_t> weight;
    complex_t* W;
    //OTFFT_DIFAVX::FFT0 fft1, fft2;
    //OTFFT_DITAVX::FFT0 fft1, fft2;
    OTFFT_DIFAVX8::FFT0 fft1, fft2;
    //OTFFT_DITAVX8::FFT0 fft1, fft2;

    FFT1() : N(0), log_N(0), W(0) {}
    FFT1(int n)
    {
        for (log_N = 0; n > 1; n >>= 1) log_N++;
        setup2(log_N);
    }

    void setup(int n)
    {
        for (log_N = 0; n > 1; n >>= 1) log_N++;
        setup2(log_N);
    }

    void setup2(int n)
    {
        log_N = n; N = 1 << n;
        weight.setup(N+1); W = &weight;
        if (N < 4) fft1.setup2(n);
        else { fft1.setup2(n/2); fft2.setup2(n - n/2); }
        init_W(N, W);
    }

    inline void fwd0(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: fft1.fwd0(x, y); break;
        case  2: fwd0ffto< 2>()(x, y, W, fft1, fft2); break;
        case  3: fwd0ffto< 3>()(x, y, W, fft1, fft2); break;
        case  4: fwd0ffto< 4>()(x, y, W, fft1, fft2); break;
        case  5: fwd0ffto< 5>()(x, y, W, fft1, fft2); break;
        case  6: fwd0ffto< 6>()(x, y, W, fft1, fft2); break;
        case  7: fwd0ffto< 7>()(x, y, W, fft1, fft2); break;
        case  8: fwd0ffto< 8>()(x, y, W, fft1, fft2); break;
        case  9: fwd0ffto< 9>()(x, y, W, fft1, fft2); break;
        case 10: fwd0ffto<10>()(x, y, W, fft1, fft2); break;
        case 11: fwd0ffto<11>()(x, y, W, fft1, fft2); break;
        case 12: fwd0ffto<12>()(x, y, W, fft1, fft2); break;
        case 13: fwd0ffto<13>()(x, y, W, fft1, fft2); break;
        case 14: fwd0ffto<14>()(x, y, W, fft1, fft2); break;
        case 15: fwd0ffto<15>()(x, y, W, fft1, fft2); break;
        case 16: fwd0ffto<16>()(x, y, W, fft1, fft2); break;
        case 17: fwd0ffto<17>()(x, y, W, fft1, fft2); break;
        case 18: fwd0ffto<18>()(x, y, W, fft1, fft2); break;
        case 19: fwd0ffto<19>()(x, y, W, fft1, fft2); break;
        case 20: fwd0ffto<20>()(x, y, W, fft1, fft2); break;
        case 21: fwd0ffto<21>()(x, y, W, fft1, fft2); break;
        case 22: fwd0ffto<22>()(x, y, W, fft1, fft2); break;
        case 23: fwd0ffto<23>()(x, y, W, fft1, fft2); break;
        case 24: fwd0ffto<24>()(x, y, W, fft1, fft2); break;
        }
    }

    inline void fwd(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: fft1.fwd(x, y); break;
        case  2: fwdnffto< 2>()(x, y, W, fft1, fft2); break;
        case  3: fwdnffto< 3>()(x, y, W, fft1, fft2); break;
        case  4: fwdnffto< 4>()(x, y, W, fft1, fft2); break;
        case  5: fwdnffto< 5>()(x, y, W, fft1, fft2); break;
        case  6: fwdnffto< 6>()(x, y, W, fft1, fft2); break;
        case  7: fwdnffto< 7>()(x, y, W, fft1, fft2); break;
        case  8: fwdnffto< 8>()(x, y, W, fft1, fft2); break;
        case  9: fwdnffto< 9>()(x, y, W, fft1, fft2); break;
        case 10: fwdnffto<10>()(x, y, W, fft1, fft2); break;
        case 11: fwdnffto<11>()(x, y, W, fft1, fft2); break;
        case 12: fwdnffto<12>()(x, y, W, fft1, fft2); break;
        case 13: fwdnffto<13>()(x, y, W, fft1, fft2); break;
        case 14: fwdnffto<14>()(x, y, W, fft1, fft2); break;
        case 15: fwdnffto<15>()(x, y, W, fft1, fft2); break;
        case 16: fwdnffto<16>()(x, y, W, fft1, fft2); break;
        case 17: fwdnffto<17>()(x, y, W, fft1, fft2); break;
        case 18: fwdnffto<18>()(x, y, W, fft1, fft2); break;
        case 19: fwdnffto<19>()(x, y, W, fft1, fft2); break;
        case 20: fwdnffto<20>()(x, y, W, fft1, fft2); break;
        case 21: fwdnffto<21>()(x, y, W, fft1, fft2); break;
        case 22: fwdnffto<22>()(x, y, W, fft1, fft2); break;
        case 23: fwdnffto<23>()(x, y, W, fft1, fft2); break;
        case 24: fwdnffto<24>()(x, y, W, fft1, fft2); break;
        }
    }

    inline void fwdn(complex_vector x, complex_vector y) const { fwd(x, y); }

    inline void inv0(complex_vector x, complex_vector y) const { inv(x, y); }

    inline void inv(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: fft1.inv0(x, y); break;
        case  2: inv0ffto< 2>()(x, y, W, fft1, fft2); break;
        case  3: inv0ffto< 3>()(x, y, W, fft1, fft2); break;
        case  4: inv0ffto< 4>()(x, y, W, fft1, fft2); break;
        case  5: inv0ffto< 5>()(x, y, W, fft1, fft2); break;
        case  6: inv0ffto< 6>()(x, y, W, fft1, fft2); break;
        case  7: inv0ffto< 7>()(x, y, W, fft1, fft2); break;
        case  8: inv0ffto< 8>()(x, y, W, fft1, fft2); break;
        case  9: inv0ffto< 9>()(x, y, W, fft1, fft2); break;
        case 10: inv0ffto<10>()(x, y, W, fft1, fft2); break;
        case 11: inv0ffto<11>()(x, y, W, fft1, fft2); break;
        case 12: inv0ffto<12>()(x, y, W, fft1, fft2); break;
        case 13: inv0ffto<13>()(x, y, W, fft1, fft2); break;
        case 14: inv0ffto<14>()(x, y, W, fft1, fft2); break;
        case 15: inv0ffto<15>()(x, y, W, fft1, fft2); break;
        case 16: inv0ffto<16>()(x, y, W, fft1, fft2); break;
        case 17: inv0ffto<17>()(x, y, W, fft1, fft2); break;
        case 18: inv0ffto<18>()(x, y, W, fft1, fft2); break;
        case 19: inv0ffto<19>()(x, y, W, fft1, fft2); break;
        case 20: inv0ffto<20>()(x, y, W, fft1, fft2); break;
        case 21: inv0ffto<21>()(x, y, W, fft1, fft2); break;
        case 22: inv0ffto<22>()(x, y, W, fft1, fft2); break;
        case 23: inv0ffto<23>()(x, y, W, fft1, fft2); break;
        case 24: inv0ffto<24>()(x, y, W, fft1, fft2); break;
        }
    }

    inline void invn(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
        switch (log_N) {
        case  1: fft1.invn(x, y); break;
        case  2: invnffto< 2>()(x, y, W, fft1, fft2); break;
        case  3: invnffto< 3>()(x, y, W, fft1, fft2); break;
        case  4: invnffto< 4>()(x, y, W, fft1, fft2); break;
        case  5: invnffto< 5>()(x, y, W, fft1, fft2); break;
        case  6: invnffto< 6>()(x, y, W, fft1, fft2); break;
        case  7: invnffto< 7>()(x, y, W, fft1, fft2); break;
        case  8: invnffto< 8>()(x, y, W, fft1, fft2); break;
        case  9: invnffto< 9>()(x, y, W, fft1, fft2); break;
        case 10: invnffto<10>()(x, y, W, fft1, fft2); break;
        case 11: invnffto<11>()(x, y, W, fft1, fft2); break;
        case 12: invnffto<12>()(x, y, W, fft1, fft2); break;
        case 13: invnffto<13>()(x, y, W, fft1, fft2); break;
        case 14: invnffto<14>()(x, y, W, fft1, fft2); break;
        case 15: invnffto<15>()(x, y, W, fft1, fft2); break;
        case 16: invnffto<16>()(x, y, W, fft1, fft2); break;
        case 17: invnffto<17>()(x, y, W, fft1, fft2); break;
        case 18: invnffto<18>()(x, y, W, fft1, fft2); break;
        case 19: invnffto<19>()(x, y, W, fft1, fft2); break;
        case 20: invnffto<20>()(x, y, W, fft1, fft2); break;
        case 21: invnffto<21>()(x, y, W, fft1, fft2); break;
        case 22: invnffto<22>()(x, y, W, fft1, fft2); break;
        case 23: invnffto<23>()(x, y, W, fft1, fft2); break;
        case 24: invnffto<24>()(x, y, W, fft1, fft2); break;
        }
    }
};

#if 0
struct FFT
{
    FFT0 fft;
    simd_array<complex_t> work;
    complex_t* y;

    FFT() : fft(), work(), y(0) {}
    FFT(int n) : fft(n), work(n), y(&work) {}

    inline void setup(const int n) { fft.setup(n); work.setup(n); y = &work; }

    inline void fwd0(complex_vector x) const { fft.fwd0(x, y); }
    inline void fwd(complex_vector x)  const { fft.fwd(x, y);  }
    inline void fwdn(complex_vector x) const { fft.fwd(x, y);  }
    inline void inv0(complex_vector x) const { fft.inv0(x, y); }
    inline void inv(complex_vector x)  const { fft.inv(x, y);  }
    inline void invn(complex_vector x) const { fft.invn(x, y);  }
};
#endif

} /////////////////////////////////////////////////////////////////////////////

#endif // otfft_sixstep_h
