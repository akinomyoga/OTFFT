/******************************************************************************
*  OTFFT Implementation Version 4.0
******************************************************************************/

#include <stdint.h>
#include "otfft/otfft.h"
#include "otfft_difavx.h"
#include "otfft_ditavx.h"
#include "otfft_difavx8.h"
#include "otfft_ditavx8.h"
#include "otfft_sixstep.h"

namespace OTFFT { /////////////////////////////////////////////////////////////

/******************************************************************************
*  Complex FFT
******************************************************************************/

    FFT0::FFT0() try : N(0), log_N(0),
        fft1(0), fft2(0), fft3(0), fft4(0), fft5(0), fft6(0)
    {
        fft1 = new OTFFT_DIFAVX::FFT0();
        fft2 = new OTFFT_DITAVX::FFT0();
        fft3 = new OTFFT_DIFAVX8::FFT0();
        fft4 = new OTFFT_DITAVX8::FFT0();
        fft5 = new OTFFT_Sixstep::FFT0();
        fft6 = new OTFFT_Sixstep::FFT1();
    }
    catch (...) { this->~FFT0(); throw; }

    FFT0::FFT0(int n) try :
        fft1(0), fft2(0), fft3(0), fft4(0), fft5(0), fft6(0)
    {
        fft1 = new OTFFT_DIFAVX::FFT0();
        fft2 = new OTFFT_DITAVX::FFT0();
        fft3 = new OTFFT_DIFAVX8::FFT0();
        fft4 = new OTFFT_DITAVX8::FFT0();
        fft5 = new OTFFT_Sixstep::FFT0();
        fft6 = new OTFFT_Sixstep::FFT1();
        setup(n);
    }
    catch (...) { this->~FFT0(); throw; }

    FFT0::~FFT0()
    {
        delete fft1;
        delete fft2;
        delete fft3;
        delete fft4;
        delete fft5;
        delete fft6;
    }

    void FFT0::setup(int n)
    {
        for (log_N = 0; n > 1; n >>= 1) log_N++;
        setup2(log_N);
    }

    void FFT0::setup2(int n)
    {
        log_N = n; N = 1 << n;
#include "otfft_setup.h"
    }

    void FFT0::fwd0(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
#include "otfft_fwd0.h"
    }

    void FFT0::fwd(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
#include "otfft_fwd.h"
    }

    void FFT0::fwdn(complex_vector x, complex_vector y) const { fwd(x, y); }

    void FFT0::inv0(complex_vector x, complex_vector y) const { inv(x, y); }

    void FFT0::inv(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
#include "otfft_inv.h"
    }

    void FFT0::invn(complex_vector x, complex_vector y) const
    {
        if (N < 2) return;
#include "otfft_invn.h"
    }

/******************************************************************************
*  Real FFT
******************************************************************************/

    RFFT::RFFT() : N(0), U(0) {}
    RFFT::RFFT(int n) { setup(n); }

    void RFFT::setup(int n)
    {
        N = n;
        fft.setup(N/2);
        weight.setup(N+1); U = &weight;
        const double theta0 = 2*M_PI/N;
#if 0
        for (int p = 0; p <= N/2; p++) {
            const double theta = p * theta0;
            const double c =  cos(theta);
            const double s = -sin(theta);
            U[p]   = complex_t(1 - s, c)/2;
            U[N-p] = complex_t(1 + s, c)/2;
        }
#else
        const int Nh = N/2;
        const int Nq = N/4;
        const int Ne = N/8;
        const int Nd = N - Nq;
        if (N < 1) {}
        else if (N < 2) { U[0] = U[1] = complex_t(1, 1)/2; }
        else if (N < 4) {
            U[0] = U[2] = complex_t(1,  1)/2;
            U[1]        = complex_t(1, -1)/2;
        }
        else if (N < 8) {
            U[0] = complex_t(1 + 0,  1)/2;
            U[1] = complex_t(1 + 1,  0)/2;
            U[2] = complex_t(1 + 0, -1)/2;
            U[3] = complex_t(1 - 1,  0)/2;
            U[4] = complex_t(1 + 0,  1)/2;
        }
        else if (N < OMP_THRESHOLD_W) for (int p = 0; p <= Ne; p++) {
            const double theta = p * theta0;
            const double c =  cos(theta);
            const double s = -sin(theta);
            U[p]    = complex_t(1 - s,  c)/2;
            U[Nq-p] = complex_t(1 + c, -s)/2;
            U[Nq+p] = complex_t(1 + c,  s)/2;
            U[Nh-p] = complex_t(1 - s, -c)/2;
            U[Nh+p] = complex_t(1 + s, -c)/2;
            U[Nd-p] = complex_t(1 - c,  s)/2;
            U[Nd+p] = complex_t(1 - c, -s)/2;
            U[N-p]  = complex_t(1 + s,  c)/2;
        }
        else
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int p = 0; p <= Ne; p++) {
            const double theta = p * theta0;
            const double c =  cos(theta);
            const double s = -sin(theta);
            U[p]    = complex_t(1 - s,  c)/2;
            U[Nq-p] = complex_t(1 + c, -s)/2;
            U[Nq+p] = complex_t(1 + c,  s)/2;
            U[Nh-p] = complex_t(1 - s, -c)/2;
            U[Nh+p] = complex_t(1 + s, -c)/2;
            U[Nd-p] = complex_t(1 - c,  s)/2;
            U[Nd+p] = complex_t(1 - c, -s)/2;
            U[N-p]  = complex_t(1 + s,  c)/2;
        }
#endif
    }

    void RFFT::fwd0(const_double_vector x, complex_vector y) const
    {
        if (N < 1) return;
        else if (N < 2) { y[0] = x[0]; return; }
        const int Nh = N/2;
        const int Nq = N/4;
        complex_vector z = y + Nh;
        for (int p = 0; p < Nh; p++) setpz(z[p], getpz(x + 2*p));
        fft.fwd0(z, y);
        y[0] = z[0].Re + z[0].Im;
        z[0] = z[0].Re - z[0].Im;
        if (N < OMP_THRESHOLD) {
            for (int k = 1; k <= Nq; k++) {
                const xmm a = getpz(z[k]);
                const xmm b = cnjpz(getpz(z[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), subpz(a, b));
                setpz(y[k],          subpz(a, c));
                setpz(y[Nh-k], cnjpz(addpz(b, c)));
            }
            for (int k = 1; k < Nh; k++) setpz(y[N-k], cnjpz(getpz(y[k])));
        }
        else
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int k = 1; k <= Nq; k++) {
                const xmm a = getpz(z[k]);
                const xmm b = cnjpz(getpz(z[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), subpz(a, b));
                setpz(y[k],          subpz(a, c));
                setpz(y[Nh-k], cnjpz(addpz(b, c)));
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int k = 1; k < Nh; k++) setpz(y[N-k], cnjpz(getpz(y[k])));
        }
    }

    void RFFT::fwd(const_double_vector x, complex_vector y) const
    {
        if (N < 1) return;
        else if (N < 2) { y[0] = x[0]; return; }
        const xmm rN = cmplx(1.0/N, 1.0/N);
        const int Nh = N/2;
        const int Nq = N/4;
        complex_vector z = y + Nh;
        for (int p = 0; p < Nh; p++) setpz(z[p], getpz(x + 2*p));
        fft.fwd0(z, y);
        y[0] = (z[0].Re + z[0].Im) / N;
        z[0] = (z[0].Re - z[0].Im) / N;
        if (N < OMP_THRESHOLD) {
            for (int k = 1; k <= Nq; k++) {
                const xmm a = getpz(z[k]);
                const xmm b = cnjpz(getpz(z[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), subpz(a, b));
                setpz(y[k],    mulpd(rN,       subpz(a, c)));
                setpz(y[Nh-k], mulpd(rN, cnjpz(addpz(b, c))));
            }
            for (int k = 1; k < Nh; k++) setpz(y[N-k], cnjpz(getpz(y[k])));
        }
        else
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int k = 1; k <= Nq; k++) {
                const xmm a = getpz(z[k]);
                const xmm b = cnjpz(getpz(z[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), subpz(a, b));
                setpz(y[k],    mulpd(rN,       subpz(a, c)));
                setpz(y[Nh-k], mulpd(rN, cnjpz(addpz(b, c))));
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int k = 1; k < Nh; k++) setpz(y[N-k], cnjpz(getpz(y[k])));
        }
    }

    void RFFT::fwdn(const_double_vector x, complex_vector y) const { fwd(x, y); }

    void RFFT::inv0(complex_vector x, double_vector y) const { inv(x, y); }

    void RFFT::inv(complex_vector x, double_vector y) const
    {
        if (N < 1) return;
        else if (N < 2) { y[0] = x[0].Re; return; }
        static const xmm x2 = cmplx(2, 2);
        const int Nh = N/2;
        complex_vector z = x + Nh;
        x[Nh] = x[0].Re + x[Nh].Re + jx(x[0].Re - x[Nh].Re);
        if (N < OMP_THRESHOLD) {
            for (int k = 1; k < Nh; k++) {
                const xmm a = cnjpz(getpz(x[k]));
                const xmm b = subpz(a, getpz(x[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), b);
                setpz(z[k], mulpd(x2, cnjpz(subpz(a, c))));
            }
            fft.inv0(z, x);
            for (int p = 0; p < Nh; p++) setpz(y+2*p, getpz(z[p]));
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int k = 1; k < Nh; k++) {
                const xmm a = cnjpz(getpz(x[k]));
                const xmm b = subpz(a, getpz(x[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), b);
                setpz(z[k], mulpd(x2, cnjpz(subpz(a, c))));
            }
            fft.inv0(z, x);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int p = 0; p < Nh; p++) setpz(y+2*p, getpz(z[p]));
        }
    }

    void RFFT::invn(complex_vector x, double_vector y) const
    {
        if (N < 1) return;
        else if (N < 2) { y[0] = x[0].Re; return; }
        const int Nh = N/2;
        complex_vector z = x + Nh;
        x[Nh] = (x[0].Re + x[Nh].Re + jx(x[0].Re - x[Nh].Re))/2;
        if (N < OMP_THRESHOLD) {
            for (int k = 1; k < Nh; k++) {
                const xmm a = cnjpz(getpz(x[k]));
                const xmm b = subpz(a, getpz(x[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), b);
                setpz(z[k], cnjpz(subpz(a, c)));
            }
            fft.invn(z, x);
            for (int p = 0; p < Nh; p++) setpz(y+2*p, getpz(z[p]));
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int k = 1; k < Nh; k++) {
                const xmm a = cnjpz(getpz(x[k]));
                const xmm b = subpz(a, getpz(x[Nh-k]));
                const xmm c = mulpz(getpz(U[k]), b);
                setpz(z[k], cnjpz(subpz(a, c)));
            }
            fft.invn(z, x);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int p = 0; p < Nh; p++) setpz(y+2*p, getpz(z[p]));
        }
    }

/******************************************************************************
*  DCT
******************************************************************************/

    DCT0::DCT0() : N(0), V(0) {}
    DCT0::DCT0(int n) { setup(n); }

    void DCT0::setup(int n)
    {
        N = n;
        rfft.setup(N);
        weight.setup(N+1); V = &weight;
        const double theta0 = M_PI/2/N;
        const int Nh = N/2;
        if (N < 1) {}
        if (N < 2) { V[0] = 1; V[1] = complex_t(0, 1); }
        if (N < 4) {
            V[0] = complex_t(        1,         0);
            V[1] = complex_t(M_SQRT1_2, M_SQRT1_2);
            V[2] = complex_t(        0,         1);
        }
        else if (N < OMP_THRESHOLD_W) for (int p = 0; p <= Nh; p++) {
            const double theta = p * theta0;
            const double c = cos(theta);
            const double s = sin(theta);
            V[p]    = complex_t(c, s);
            V[N-p]  = complex_t(s, c);
        }
        else
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int p = 0; p <= Nh; p++) {
            const double theta = p * theta0;
            const double c = cos(theta);
            const double s = sin(theta);
            V[p]    = complex_t(c, s);
            V[N-p]  = complex_t(s, c);
        }
    }

    void DCT0::fwd0(double_vector x, double_vector y, complex_vector z) const
    {
        if (N < 2) return;
        const int Nh = N/2;
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < Nh; p++) {
                y[p]     = x[2*p+0];
                y[N-p-1] = x[2*p+1];
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int p = 0; p < Nh; p++) {
                y[p]     = x[2*p+0];
                y[N-p-1] = x[2*p+1];
            }
        }
        rfft.fwd0(y, z);
        //for (int k = 0; k < N; k++)
        //    x[k] = V[k].Re*z[k].Re + V[k].Im*z[k].Im;
        if (N < OMP_THRESHOLD) {
            for (int k = 0; k < N; k += 2) {
                const xmm a = mulpd(getpz(V[k+0]), getpz(z[k+0]));
                const xmm b = mulpd(getpz(V[k+1]), getpz(z[k+1]));
                setpz(x+k, haddpz(a, b));
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int k = 0; k < N; k += 2) {
                const xmm a = mulpd(getpz(V[k+0]), getpz(z[k+0]));
                const xmm b = mulpd(getpz(V[k+1]), getpz(z[k+1]));
                setpz(x+k, haddpz(a, b));
            }
        }
    }

    void DCT0::fwd(double_vector x, double_vector y, complex_vector z) const
    {
        if (N < 2) return;
        const int Nh = N/2;
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < Nh; p++) {
                y[p]     = x[2*p+0];
                y[N-p-1] = x[2*p+1];
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int p = 0; p < Nh; p++) {
                y[p]     = x[2*p+0];
                y[N-p-1] = x[2*p+1];
            }
        }
        rfft.fwd(y, z);
        if (N < OMP_THRESHOLD) {
            for (int k = 0; k < N; k += 2) {
                const xmm a = mulpd(getpz(V[k+0]), getpz(z[k+0]));
                const xmm b = mulpd(getpz(V[k+1]), getpz(z[k+1]));
                setpz(x+k, haddpz(a, b));
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int k = 0; k < N; k += 2) {
                const xmm a = mulpd(getpz(V[k+0]), getpz(z[k+0]));
                const xmm b = mulpd(getpz(V[k+1]), getpz(z[k+1]));
                setpz(x+k, haddpz(a, b));
            }
        }
    }

    void DCT0::fwdn(double_vector x, double_vector y, complex_vector z) const { fwd(x, y, z); }

    void DCT0::inv0(double_vector x, double_vector y, complex_vector z) const { inv(x, y, z); }

    void DCT0::inv(double_vector x, double_vector y, complex_vector z) const
    {
        if (N < 2) return;
        const int Nh = N/2;
        z[0] = x[0];
        if (N < OMP_THRESHOLD) {
            for (int k = 1; k < N; k++) z[k] = V[k]*complex_t(x[k], -x[N-k]);
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int k = 1; k < N; k++) z[k] = V[k]*complex_t(x[k], -x[N-k]);
        }
        rfft.inv(z, y);
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < Nh; p++) {
                x[2*p+0] = y[p];
                x[2*p+1] = y[N-p-1];
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int p = 0; p < Nh; p++) {
                x[2*p+0] = y[p];
                x[2*p+1] = y[N-p-1];
            }
        }
    }

    void DCT0::invn(double_vector x, double_vector y, complex_vector z) const
    {
        if (N < 2) return;
        const int Nh = N/2;
        z[0] = x[0];
        if (N < OMP_THRESHOLD) {
            for (int k = 1; k < N; k++) z[k] = V[k]*complex_t(x[k], -x[N-k]);
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int k = 1; k < N; k++) z[k] = V[k]*complex_t(x[k], -x[N-k]);
        }
        rfft.invn(z, y);
        if (N < OMP_THRESHOLD) {
            for (int p = 0; p < Nh; p++) {
                x[2*p+0] = y[p];
                x[2*p+1] = y[N-p-1];
            }
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int p = 0; p < Nh; p++) {
                x[2*p+0] = y[p];
                x[2*p+1] = y[N-p-1];
            }
        }
    }

/******************************************************************************
*  Bluestein's FFT
******************************************************************************/

    Bluestein::Bluestein() : N(0), L(0), a(0), b(0), W(0) {}
    Bluestein::Bluestein(int n) { setup(n); }

    void Bluestein::setup(int n)
    {
        N = n;
        const int N2 = 2*N;
        for (L = 1; L < N2 - 1; L *= 2);
        fft.setup(L);
        work1.setup(L); a = &work1;
        work2.setup(L); b = &work2;
        weight.setup(N2+1); W = &weight;
        const double theta0 = M_PI/N;
        W[0] = W[N2] = 1; W[N] = -1;
        if (N < OMP_THRESHOLD_W) for (int p = 1; p < N; p++) {
            const double theta = p * theta0;
            const double c =  cos(theta);
            const double s = -sin(theta);
            W[p]    = complex_t(c,  s);
            W[N2-p] = complex_t(c, -s);
        }
        else
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int p = 1; p < N; p++) {
            const double theta = p * theta0;
            const double c =  cos(theta);
            const double s = -sin(theta);
            W[p]    = complex_t(c,  s);
            W[N2-p] = complex_t(c, -s);
        }
    }

    void Bluestein::fwd0(complex_vector x) const
    {
        if (N < 2) return;
        const int N2 = 2*N;
        a[0] = x[0]; b[0] = x[0] = 1;
        if (N < OMP_THRESHOLD) {
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[pp];
                b[p] = x[p] = W[N2-pp];
                b[L-p] = b[p];
            }
        }
        else
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[pp];
                b[p] = x[p] = W[N2-pp];
                b[L-p] = b[p];
            }
        }
        fft.fwd0(a); fft.fwd0(b);
        if (N < OMP_THRESHOLD) {
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpz(getpz(a[p]), cnjpz(getpz(x[p]))));
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpz(getpz(a[p]), cnjpz(getpz(x[p]))));
        }
    }

    void Bluestein::fwd(complex_vector x) const
    {
        if (N < 2) return;
        const xmm rN = cmplx(1.0/N, 1.0/N);
        const int N2 = 2*N;
        a[0] = x[0]; b[0] = x[0] = 1;
        if (N < OMP_THRESHOLD) {
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[pp];
                b[p] = x[p] = W[N2-pp];
                b[L-p] = b[p];
            }
        }
        else
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[pp];
                b[p] = x[p] = W[N2-pp];
                b[L-p] = b[p];
            }
        }
        fft.fwd0(a); fft.fwd0(b);
        if (N < OMP_THRESHOLD) {
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpd(rN, mulpz(getpz(a[p]), cnjpz(getpz(x[p])))));
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpd(rN, mulpz(getpz(a[p]), cnjpz(getpz(x[p])))));
        }
    }

    void Bluestein::fwdn(complex_vector x) const { fwd(x); }

    void Bluestein::inv0(complex_vector x) const { inv(x); }

    void Bluestein::inv(complex_vector x) const
    {
        if (N < 2) return;
        const int N2 = 2*N;
        a[0] = x[0]; b[0] = x[0] = 1;
        if (N < OMP_THRESHOLD) {
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[N2-pp];
                b[p] = x[p] = W[pp];
                b[L-p] = b[p];
            }
        }
        else
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[N2-pp];
                b[p] = x[p] = W[pp];
                b[L-p] = b[p];
            }
        }
        fft.fwd0(a); fft.fwd0(b);
        if (N < OMP_THRESHOLD) {
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpz(getpz(a[p]), cnjpz(getpz(x[p]))));
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpz(getpz(a[p]), cnjpz(getpz(x[p]))));
        }
    }

    void Bluestein::invn(complex_vector x) const
    {
        if (N < 2) return;
        const xmm rN = cmplx(1.0/N, 1.0/N);
        const int N2 = 2*N;
        a[0] = x[0]; b[0] = x[0] = 1;
        if (N < OMP_THRESHOLD) {
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[N2-pp];
                b[p] = x[p] = W[pp];
                b[L-p] = b[p];
            }
        }
        else
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 1; p < L; p++) a[p] = b[p] = 0;
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int p = 1; p < N; p++) {
                const int64_t q = p;
                const int pp = static_cast<int>(q*q % N2);
                a[p] = x[p]*W[N2-pp];
                b[p] = x[p] = W[pp];
                b[L-p] = b[p];
            }
        }
        fft.fwd0(a); fft.fwd0(b);
        if (N < OMP_THRESHOLD) {
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpd(rN, mulpz(getpz(a[p]), cnjpz(getpz(x[p])))));
        }
        else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int k = 0; k < L; k++)
                setpz(a[k], mulpz(getpz(a[k]), getpz(b[k])));
            fft.invn(a);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int p = 0; p < N; p++)
                setpz(x[p], mulpd(rN, mulpz(getpz(a[p]), cnjpz(getpz(x[p])))));
        }
    }

} /////////////////////////////////////////////////////////////////////////////
