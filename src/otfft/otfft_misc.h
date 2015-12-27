/******************************************************************************
*  FFT Miscellaneous Routines Version 6.0
*
*  Copyright (c) 2015 OK Ojisan(Takuya OKAHISA)
*  Released under the MIT license
*  http://opensource.org/licenses/mit-license.php
******************************************************************************/

#ifndef otfft_misc_h
#define otfft_misc_h

#include <complex>
#include <cmath>
#include <new>

#define USE_INTRINSIC 1
//#define __OPTIMIZE__ 1

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524400844362104849039
#endif

#define RSQRT2PSQRT2 0.541196100146196984405268931572763336
#define H1X ( 0.923879532511286762010323247995557949)
#define H1Y (-0.382683432365089757574419179753100195)

#if (__GNUC__ >= 3)
//#define force_inline
//#define force_inline2
//#define force_inline3
//#define force_inline  __attribute__((const))
//#define force_inline2 __attribute__((pure))
//#define force_inline3
#define force_inline  __attribute__((const,always_inline))
#define force_inline2 __attribute__((pure,always_inline))
#define force_inline3 __attribute__((always_inline))
#else
#define force_inline
#define force_inline2
#define force_inline3
#endif

#ifdef _MSC_VER

#ifdef USE_AVX2
#ifndef __AVX2__
#define __AVX2__
#define __FMA__
#endif
#ifndef __AVX__
#define __AVX__
#endif
#define __SSE3__
#define __SSE2__
#endif

#ifdef USE_AVX
#ifndef __AVX__
#define __AVX__
#endif
#define __SSE3__
#define __SSE2__
#endif

#ifdef USE_SSE3
#define __SSE3__
#define __SSE2__
#endif

#if defined(USE_SSE2) || _M_IX86_FP >= 2
#define __SSE2__
#endif

#if _MSC_VER < 1800
static inline double rint(const double& x) { return floor(x + 0.5); }
#endif

#endif // _MSC_VER

#ifdef __MINGW64__
//#undef _MM_MALLOC_H_INCLUDED
//#undef _mm_malloc
//#undef _mm_free
//#include <mm_malloc.h>
#elif defined(__MINGW32__)
#include <malloc.h>
#endif

/*****************************************************************************/

namespace OTFFT_MISC {

struct complex_t
{
    double Re, Im;

    complex_t() : Re(0), Im(0) {}
    complex_t(const double& x) : Re(x), Im(0) {}
    complex_t(const double& x, const double& y) : Re(x), Im(y) {}
    complex_t(const std::complex<double>& z) : Re(z.real()), Im(z.imag()) {}
    operator std::complex<double>() const { return std::complex<double>(Re, Im); }

    complex_t& operator+=(const complex_t& z) {
        Re += z.Re;
        Im += z.Im;
        return *this;
    }

    complex_t& operator-=(const complex_t& z) {
        Re -= z.Re;
        Im -= z.Im;
        return *this;
    }

    complex_t& operator*=(const double& x)
    {
        Re *= x;
        Im *= x;
        return *this;
    }

    complex_t& operator*=(const complex_t& z)
    {
        const double tmp = Re*z.Re - Im*z.Im;
        Im = Re*z.Im + Im*z.Re;
        Re = tmp;
        return *this;
    }
};

typedef double* __restrict const double_vector;
typedef const double* __restrict const const_double_vector;
typedef complex_t* __restrict const complex_vector;
typedef const complex_t* __restrict const const_complex_vector;

static inline double Re(const complex_t& z) force_inline;
static inline double Re(const complex_t& z) { return z.Re; }
static inline double Im(const complex_t& z) force_inline;
static inline double Im(const complex_t& z) { return z.Im; }
static inline double norm(const complex_t& z) force_inline;
static inline double norm(const complex_t& z) { return z.Re*z.Re + z.Im*z.Im; }
static inline complex_t conj(const complex_t& z) force_inline;
static inline complex_t conj(const complex_t& z) { return complex_t(z.Re, -z.Im); }
static inline complex_t jx(const complex_t& z) force_inline;
static inline complex_t jx(const complex_t& z) { return complex_t(-z.Im, z.Re); }
static inline complex_t neg(const complex_t& z) force_inline;
static inline complex_t neg(const complex_t& z) { return complex_t(-z.Im, -z.Re); }
static inline complex_t v8x(const complex_t& z) force_inline;
static inline complex_t v8x(const complex_t& z) { return complex_t(M_SQRT1_2*(z.Re-z.Im), M_SQRT1_2*(z.Re+z.Im)); }
static inline complex_t w8x(const complex_t& z) force_inline;
static inline complex_t w8x(const complex_t& z) { return complex_t(M_SQRT1_2*(z.Re+z.Im), M_SQRT1_2*(z.Im-z.Re)); }

static inline complex_t operator+(const complex_t& a, const complex_t& b) force_inline;
static inline complex_t operator+(const complex_t& a, const complex_t& b)
{
    return complex_t(a.Re + b.Re, a.Im + b.Im);
}

static inline complex_t operator-(const complex_t& a, const complex_t& b) force_inline;
static inline complex_t operator-(const complex_t& a, const complex_t& b)
{
    return complex_t(a.Re - b.Re, a.Im - b.Im);
}

static inline complex_t operator*(const double& a, const complex_t& b) force_inline;
static inline complex_t operator*(const double& a, const complex_t& b)
{
    return complex_t(a*b.Re, a*b.Im);
}

static inline complex_t operator*(const complex_t& a, const complex_t& b) force_inline;
static inline complex_t operator*(const complex_t& a, const complex_t& b)
{
    return complex_t(a.Re*b.Re - a.Im*b.Im, a.Re*b.Im + a.Im*b.Re);
}

static inline complex_t operator/(const complex_t& a, const double& b) force_inline;
static inline complex_t operator/(const complex_t& a, const double& b)
{
    return complex_t(a.Re / b, a.Im / b);
}

static inline complex_t operator/(const complex_t& a, const complex_t& b) force_inline;
static inline complex_t operator/(const complex_t& a, const complex_t& b)
{
    const double b2 = b.Re*b.Re + b.Im*b.Im;
    return (a * conj(b)) / b2;
}

static inline complex_t expj(const double& theta) force_inline;
static inline complex_t expj(const double& theta)
{
    return complex_t(cos(theta), sin(theta));
}

static void init_W(int N, complex_vector W)
{
    static const int OMP_THRESHOLD_W = 1<<16;
    const double theta0 = 2*M_PI/N;
#if 0
    for (int p = 0; p <= N/2; p++) {
        const double theta = p * theta0;
        const double c =  cos(theta);
        const double s = -sin(theta);
        W[p]   = complex_t(c,  s);
        W[N-p] = complex_t(c, -s);
    }
#else
    const int Nh = N/2;
    const int Nq = N/4;
    const int Ne = N/8;
    const int Nd = N - Nq;
    if (N < 1) {}
    else if (N < 2) { W[0] = W[1] = 1; }
    else if (N < 4) { W[0] = W[2] = 1; W[1] = -1; }
    else if (N < 8) {
        W[0] = complex_t( 1,  0);
        W[1] = complex_t( 0, -1);
        W[2] = complex_t(-1,  0);
        W[3] = complex_t( 0,  1);
        W[4] = complex_t( 1,  0);
    }
    else if (N < OMP_THRESHOLD_W) for (int p = 0; p <= Ne; p++) {
        const double theta = p * theta0;
        const double c =  cos(theta);
        const double s = -sin(theta);
        W[p]    = complex_t( c,  s);
        W[Nq-p] = complex_t(-s, -c);
        W[Nq+p] = complex_t( s, -c);
        W[Nh-p] = complex_t(-c,  s);
        W[Nh+p] = complex_t(-c, -s);
        W[Nd-p] = complex_t( s,  c);
        W[Nd+p] = complex_t(-s,  c);
        W[N-p]  = complex_t( c, -s);
    }
    else
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int p = 0; p <= Ne; p++) {
        const double theta = p * theta0;
        const double c =  cos(theta);
        const double s = -sin(theta);
        W[p]    = complex_t( c,  s);
        W[Nq-p] = complex_t(-s, -c);
        W[Nq+p] = complex_t( s, -c);
        W[Nh-p] = complex_t(-c,  s);
        W[Nh+p] = complex_t(-c, -s);
        W[Nd-p] = complex_t( s,  c);
        W[Nd+p] = complex_t(-s,  c);
        W[N-p]  = complex_t( c, -s);
    }
#endif
}

static void speedup_magic(const int N = 1 << 18)
{
    const double theta0 = 2*M_PI/N;
    volatile double sum = 0;
    for (int p = 0; p < N; p++) {
        sum += cos(p * theta0);
    }
}

} // namespace OTFFT_MISC

/*****************************************************************************/

#if defined(__SSE2__) && defined(USE_INTRINSIC)

extern "C" {
#include <emmintrin.h>
}

namespace OTFFT_MISC {

typedef __m128d xmm;

static inline xmm cmplx(const double& x, const double& y) force_inline;
static inline xmm cmplx(const double& x, const double& y) { return _mm_setr_pd(x, y); }
static inline xmm getpz(const complex_t& z) force_inline;
static inline xmm getpz(const complex_t& z) { return _mm_load_pd(&z.Re); }
static inline xmm getpz(const_double_vector x) force_inline;
static inline xmm getpz(const_double_vector x) { return _mm_load_pd(x); }
static inline void setpz(complex_t& z, const xmm x) force_inline3;
static inline void setpz(complex_t& z, const xmm x) { _mm_store_pd(&z.Re, x); }
static inline void setpz(double_vector x, const xmm z) force_inline3;
static inline void setpz(double_vector x, const xmm z) { _mm_store_pd(x, z); }

static inline void swappz(complex_t& x, complex_t& y)
{
    const xmm z = getpz(x); setpz(x, getpz(y)); setpz(y, z);
}

static inline xmm cnjpz(const xmm xy) force_inline;
static inline xmm cnjpz(const xmm xy)
{
    static const xmm zm = { 0.0, -0.0 };
    return _mm_xor_pd(zm, xy);
}

static inline xmm jxpz(const xmm xy) force_inline;
static inline xmm jxpz(const xmm xy)
{
    const xmm xmy = cnjpz(xy);
    return _mm_shuffle_pd(xmy, xmy, 1);
}

static inline xmm negpz(const xmm xy) force_inline;
static inline xmm negpz(const xmm xy)
{
    static const xmm mm = { -0.0, -0.0 };
    return _mm_xor_pd(mm, xy);
}

static inline xmm addpz(const xmm a, const xmm b) force_inline;
static inline xmm addpz(const xmm a, const xmm b) { return _mm_add_pd(a, b); }
static inline xmm subpz(const xmm a, const xmm b) force_inline;
static inline xmm subpz(const xmm a, const xmm b) { return _mm_sub_pd(a, b); }
static inline xmm mulpd(const xmm a, const xmm b) force_inline;
static inline xmm mulpd(const xmm a, const xmm b) { return _mm_mul_pd(a, b); }
static inline xmm divpd(const xmm a, const xmm b) force_inline;
static inline xmm divpd(const xmm a, const xmm b) { return _mm_div_pd(a, b); }
static inline xmm xorpd(const xmm a, const xmm b) force_inline;
static inline xmm xorpd(const xmm a, const xmm b) { return _mm_xor_pd(a, b); }

#if defined(__SSE3__) && defined(USE_INTRINSIC)
} // namespace OTFFT_MISC

extern "C" {
#include <pmmintrin.h>
}

namespace OTFFT_MISC {

static inline xmm haddpz(const xmm ab, const xmm xy) force_inline;
static inline xmm haddpz(const xmm ab, const xmm xy)
{
    return _mm_hadd_pd(ab, xy); // (a + b, x + y)
}

static inline xmm mulpz(const xmm ab, const xmm xy) force_inline;
static inline xmm mulpz(const xmm ab, const xmm xy)
{
    const xmm aa = _mm_unpacklo_pd(ab, ab);
    const xmm bb = _mm_unpackhi_pd(ab, ab);
    const xmm yx = _mm_shuffle_pd(xy, xy, 1);
    return _mm_addsub_pd(_mm_mul_pd(aa, xy), _mm_mul_pd(bb, yx));
}

static inline xmm divpz(const xmm ab, const xmm xy) force_inline;
static inline xmm divpz(const xmm ab, const xmm xy)
{
    const xmm x2y2 = _mm_mul_pd(xy, xy);
    const xmm r2r2 = _mm_hadd_pd(x2y2, x2y2);
    return _mm_div_pd(mulpz(ab, cnjpz(xy)), r2r2);
}

static inline xmm v8xpz(const xmm xy) force_inline;
static inline xmm v8xpz(const xmm xy)
{
    static const xmm rr = { M_SQRT1_2, M_SQRT1_2 };
    const xmm yx = _mm_shuffle_pd(xy, xy, 1);
    return _mm_mul_pd(rr, _mm_addsub_pd(xy, yx));
}

static inline xmm w8xpz(const xmm xy) force_inline;
static inline xmm w8xpz(const xmm xy)
{
    static const xmm rr = { M_SQRT1_2, M_SQRT1_2 };
    const xmm ymx = cnjpz(_mm_shuffle_pd(xy, xy, 1));
    return _mm_mul_pd(rr, _mm_add_pd(xy, ymx));
}

static inline xmm h1xpz(const xmm xy) force_inline;
static inline xmm h1xpz(const xmm xy)
{
    static const xmm rr = { RSQRT2PSQRT2, RSQRT2PSQRT2 };
    const xmm w8xy = w8xpz(xy);
    return _mm_mul_pd(rr, _mm_add_pd(xy, w8xy));
}

static inline xmm h3xpz(const xmm xy) force_inline;
static inline xmm h3xpz(const xmm xy)
{
    static const xmm r1 = { M_SQRT1_2, M_SQRT1_2 };
    static const xmm r2 = { RSQRT2PSQRT2, RSQRT2PSQRT2 };
    const xmm ymx = cnjpz(_mm_shuffle_pd(xy, xy, 5));
    const xmm w8xy = _mm_mul_pd(r1, _mm_add_pd(xy, ymx));
    return _mm_mul_pd(r2, _mm_add_pd(ymx, w8xy));
}

static inline xmm hfxpz(const xmm xy) force_inline;
static inline xmm hfxpz(const xmm xy)
{
    static const xmm rr = { RSQRT2PSQRT2, RSQRT2PSQRT2 };
    const xmm v8xy = v8xpz(xy);
    return _mm_mul_pd(rr, _mm_add_pd(xy, v8xy));
}

static inline xmm hdxpz(const xmm xy) force_inline;
static inline xmm hdxpz(const xmm xy)
{
    static const xmm r1 = { M_SQRT1_2, M_SQRT1_2 };
    static const xmm r2 = { RSQRT2PSQRT2, RSQRT2PSQRT2 };
    const xmm myx = jxpz(xy);
    const xmm v8xy = _mm_mul_pd(r1, _mm_add_pd(xy, myx));
    return _mm_mul_pd(r2, _mm_add_pd(myx, v8xy));
}

#else // __SSE2__

static inline xmm haddpz(const xmm ab, const xmm xy) force_inline;
static inline xmm haddpz(const xmm ab, const xmm xy)
{
    const xmm ba = _mm_shuffle_pd(ab, ab, 1);
    const xmm yx = _mm_shuffle_pd(xy, xy, 1);
    const xmm apb = _mm_add_sd(ab, ba);
    const xmm xpy = _mm_add_sd(xy, yx);
    return _mm_shuffle_pd(apb, xpy, 0); // (a + b, x + y)
}

static inline xmm mulpz(const xmm ab, const xmm xy) force_inline;
static inline xmm mulpz(const xmm ab, const xmm xy)
{
    const xmm aa = _mm_unpacklo_pd(ab, ab);
    const xmm bb = _mm_unpackhi_pd(ab, ab);
    return _mm_add_pd(_mm_mul_pd(aa, xy), _mm_mul_pd(bb, jxpz(xy)));
}

static inline xmm divpz(const xmm ab, const xmm xy) force_inline;
static inline xmm divpz(const xmm ab, const xmm xy)
{
    const xmm x2y2 = _mm_mul_pd(xy, xy);
    const xmm y2x2 = _mm_shuffle_pd(x2y2, x2y2, 1);
    const xmm r2r2 = _mm_add_pd(x2y2, y2x2);
    return _mm_div_pd(mulpz(ab, cnjpz(xy)), r2r2);
}

static inline xmm v8xpz(const xmm xy) force_inline;
static inline xmm v8xpz(const xmm xy)
{
    static const xmm rr = { M_SQRT1_2, M_SQRT1_2 };
    return _mm_mul_pd(rr, _mm_add_pd(xy, jxpz(xy)));
}

static inline xmm w8xpz(const xmm xy) force_inline;
static inline xmm w8xpz(const xmm xy)
{
    static const xmm rr = { M_SQRT1_2, M_SQRT1_2 };
    const xmm ymx = cnjpz(_mm_shuffle_pd(xy, xy, 1));
    return _mm_mul_pd(rr, _mm_add_pd(xy, ymx));
}

static inline xmm h1xpz(const xmm xy) force_inline;
static inline xmm h1xpz(const xmm xy)
{
    static const xmm rr = { RSQRT2PSQRT2, RSQRT2PSQRT2 };
    const xmm w8xy = w8xpz(xy);
    return _mm_mul_pd(rr, _mm_add_pd(xy, w8xy));
}

static inline xmm h3xpz(const xmm xy) force_inline;
static inline xmm h3xpz(const xmm xy)
{
    static const xmm r1 = { M_SQRT1_2, M_SQRT1_2 };
    static const xmm r2 = { RSQRT2PSQRT2, RSQRT2PSQRT2 };
    const xmm ymx = cnjpz(_mm_shuffle_pd(xy, xy, 1));
    const xmm w8xy = _mm_mul_pd(r1, _mm_add_pd(xy, ymx));
    return _mm_mul_pd(r2, _mm_add_pd(ymx, w8xy));
}

static inline xmm hfxpz(const xmm xy) force_inline;
static inline xmm hfxpz(const xmm xy)
{
    static const xmm rr = { RSQRT2PSQRT2, RSQRT2PSQRT2 };
    const xmm v8xy = v8xpz(xy);
    return _mm_mul_pd(rr, _mm_add_pd(xy, v8xy));
}

static inline xmm hdxpz(const xmm xy) force_inline;
static inline xmm hdxpz(const xmm xy)
{
    static const xmm r1 = { M_SQRT1_2, M_SQRT1_2 };
    static const xmm r2 = { RSQRT2PSQRT2, RSQRT2PSQRT2 };
    const xmm myx = jxpz(xy);
    const xmm v8xy = _mm_mul_pd(r1, _mm_add_pd(xy, myx));
    return _mm_mul_pd(r2, _mm_add_pd(myx, v8xy));
}

#endif // __SSE3__

#if !defined(__AVX__) && defined(__SSE2__) && defined(USE_INTRINSIC)
static inline void* simd_malloc(int ns) { return _mm_malloc(ns, 16); }
static inline void simd_free(void* p) { _mm_free(p); }
#endif // !__AVX__ && __SSE2__

} // namespace OTFFT_MISC

#else // !__SSE2__ && !__SSE3__ && !__AVX__

#include <cstdlib>

namespace OTFFT_MISC {

struct xmm { double Re, Im; };

static inline xmm cmplx(const double& x, const double& y) force_inline;
static inline xmm cmplx(const double& x, const double& y)
{
    const xmm z = { x, y };
    return z;
}

static inline xmm getpz(const complex_t& z) force_inline;
static inline xmm getpz(const complex_t& z)
{
    const xmm x = { z.Re, z.Im };
    return x;
}

static inline xmm getpz(const_double_vector x) force_inline;
static inline xmm getpz(const_double_vector x)
{
    const xmm z = { x[0], x[1] };
    return z;
}

static inline void setpz(complex_t& z, const xmm& x) force_inline3;
static inline void setpz(complex_t& z, const xmm& x) { z.Re = x.Re; z.Im = x.Im; }
static inline void setpz(double_vector x, const xmm z) force_inline3;
static inline void setpz(double_vector x, const xmm z) { x[0] = z.Re; x[1] = z.Im; }

static inline void swappz(complex_t& x, complex_t& y)
{
    const xmm z = getpz(x); setpz(x, getpz(y)); setpz(y, z);
}

static inline xmm cnjpz(const xmm& z) force_inline;
static inline xmm cnjpz(const xmm& z) { const xmm x = { z.Re, -z.Im }; return x; }
static inline xmm jxpz(const xmm& z) force_inline;
static inline xmm jxpz(const xmm& z) { const xmm x = { -z.Im, z.Re }; return x; }
static inline xmm negpz(const xmm& z) force_inline;
static inline xmm negpz(const xmm& z) { const xmm x = { -z.Re, -z.Im }; return x; }

static inline xmm v8xpz(const xmm& z) force_inline;
static inline xmm v8xpz(const xmm& z)
{
    const xmm x = { M_SQRT1_2*(z.Re - z.Im), M_SQRT1_2*(z.Re + z.Im) };
    return x;
}

static inline xmm w8xpz(const xmm& z) force_inline;
static inline xmm w8xpz(const xmm& z)
{
    const xmm x = { M_SQRT1_2*(z.Re + z.Im), M_SQRT1_2*(z.Im - z.Re) };
    return x;
}

static inline xmm addpz(const xmm& a, const xmm& b) force_inline;
static inline xmm addpz(const xmm& a, const xmm& b)
{
    const xmm x = { a.Re + b.Re, a.Im + b.Im };
    return x;
}

static inline xmm subpz(const xmm& a, const xmm& b) force_inline;
static inline xmm subpz(const xmm& a, const xmm& b)
{
    const xmm x = { a.Re - b.Re, a.Im - b.Im };
    return x;
}

static inline xmm mulpz(const xmm& a, const xmm& b) force_inline;
static inline xmm mulpz(const xmm& a, const xmm& b)
{
    const xmm x = { a.Re*b.Re - a.Im*b.Im, a.Re*b.Im + a.Im*b.Re };
    return x;
}

static inline xmm divpz(const xmm& a, const xmm& b) force_inline;
static inline xmm divpz(const xmm& a, const xmm& b)
{
    const double b2 = b.Re*b.Re + b.Im*b.Im;
    const xmm acb = mulpz(a, cnjpz(b));
    const xmm x = { acb.Re/b2, acb.Im/b2 };
    return x;
}

static inline xmm mulpd(const xmm& a, const xmm& b) force_inline;
static inline xmm mulpd(const xmm& a, const xmm& b)
{
    const xmm x = { a.Re*b.Re, a.Im*b.Im };
    return x;
}

static inline xmm divpd(const xmm& a, const xmm& b) force_inline;
static inline xmm divpd(const xmm& a, const xmm& b)
{
    const xmm x = { a.Re/b.Re, a.Im/b.Im };
    return x;
}

static inline xmm haddpz(const xmm& ab, const xmm& xy) force_inline;
static inline xmm haddpz(const xmm& ab, const xmm& xy)
{
    const xmm x = { ab.Re + ab.Im, xy.Re + xy.Im };
    return x;
}

static inline xmm h1xpz(const xmm& z) force_inline;
static inline xmm h1xpz(const xmm& z)
{
    static const xmm r = { RSQRT2PSQRT2, RSQRT2PSQRT2 };
    const xmm w8z = w8xpz(z);
    return mulpd(r, addpz(z, w8z));
}

static inline xmm h3xpz(const xmm& z) force_inline;
static inline xmm h3xpz(const xmm& z)
{
    static const xmm r1 = { M_SQRT1_2, M_SQRT1_2 };
    static const xmm r2 = { RSQRT2PSQRT2, RSQRT2PSQRT2 };
    const xmm mjz = { z.Im, -z.Re };
    const xmm w8z = mulpd(r1, addpz(z, mjz));
    return mulpd(r2, addpz(mjz, w8z));
}

static inline xmm hfxpz(const xmm& z) force_inline;
static inline xmm hfxpz(const xmm& z)
{
    static const xmm r = { RSQRT2PSQRT2, RSQRT2PSQRT2 };
    const xmm v8z = v8xpz(z);
    return mulpd(r, addpz(z, v8z));
}

static inline xmm hdxpz(const xmm& z) force_inline;
static inline xmm hdxpz(const xmm& z)
{
    static const xmm r1 = { M_SQRT1_2, M_SQRT1_2 };
    static const xmm r2 = { RSQRT2PSQRT2, RSQRT2PSQRT2 };
    const xmm jz = jxpz(z);
    const xmm v8z = mulpd(r1, addpz(z, jz));
    return mulpd(r2, addpz(jz, v8z));
}

static inline void* simd_malloc(int ns) { return malloc(ns); }
static inline void simd_free(void* p) { free(p); }

} // namespace OTFFT_MISC

#endif // __SSE2__

/*****************************************************************************/

#if defined(__AVX__) && defined(USE_INTRINSIC)

extern "C" {
#include <immintrin.h>
}

namespace OTFFT_MISC {

typedef __m256d ymm;

static inline void zeroupper() force_inline;
static inline void zeroupper() { _mm256_zeroupper(); }

static inline ymm cmplx2(const double& a, const double& b, const double& c, const double& d) force_inline;
static inline ymm cmplx2(const double& a, const double& b, const double& c, const double& d)
{
    return _mm256_setr_pd(a, b, c, d);
}

static inline ymm cmplx2(const complex_t& x, const complex_t& y) force_inline;
static inline ymm cmplx2(const complex_t& x, const complex_t& y)
{
    const xmm a = getpz(x);
    const xmm b = getpz(y);
    const ymm ax = _mm256_castpd128_pd256(a);
    const ymm bx = _mm256_castpd128_pd256(b);
    return _mm256_permute2f128_pd(ax, bx, 0x20);
//    return _mm256_insertf128_pd(_mm256_castpd128_pd256(a), b, 1);
}

static inline ymm cmplx3(const complex_t& x, const complex_t& y) force_inline;
static inline ymm cmplx3(const complex_t& x, const complex_t& y)
{
#if 1
    const ymm ax = _mm256_load_pd(&x.Re);
    const ymm bx = _mm256_load_pd(&y.Re);
    return _mm256_permute2f128_pd(ax, bx, 0x20);
#else
    const ymm ax = _mm256_load_pd(&x.Re);
    const xmm b  = getpz(y);
    return _mm256_insertf128_pd(ax, b, 1);
#endif
}

static inline ymm getpz2(const_complex_vector z) force_inline2;
static inline ymm getpz2(const_complex_vector z)
{
    return _mm256_load_pd(&z->Re);
}

static inline void setpz2(complex_vector z, const ymm x) force_inline3;
static inline void setpz2(complex_vector z, const ymm x)
{
    _mm256_store_pd(&z->Re, x);
}

static inline ymm cnjpz2(const ymm xy) force_inline;
static inline ymm cnjpz2(const ymm xy)
{
    static const ymm zm = { 0.0, -0.0, 0.0, -0.0 };
    return _mm256_xor_pd(zm, xy);
}

static inline ymm jxpz2(const ymm xy) force_inline;
static inline ymm jxpz2(const ymm xy)
{
    const ymm xmy = cnjpz2(xy);
    return _mm256_shuffle_pd(xmy, xmy, 5);
}

static inline ymm negpz2(const ymm xy) force_inline;
static inline ymm negpz2(const ymm xy)
{
    static const ymm mm = { -0.0, -0.0, -0.0, -0.0 };
    return _mm256_xor_pd(mm, xy);
}

static inline ymm v8xpz2(const ymm xy) force_inline;
static inline ymm v8xpz2(const ymm xy)
{
    static const ymm rr = { M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, M_SQRT1_2 };
    const ymm yx = _mm256_shuffle_pd(xy, xy, 5);
    return _mm256_mul_pd(rr, _mm256_addsub_pd(xy, yx));
}

static inline ymm w8xpz2(const ymm xy) force_inline;
static inline ymm w8xpz2(const ymm xy)
{
    static const ymm rr = { M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, M_SQRT1_2 };
    const ymm ymx = cnjpz2(_mm256_shuffle_pd(xy, xy, 5));
    return _mm256_mul_pd(rr, _mm256_add_pd(xy, ymx));
}

static inline ymm addpz2(const ymm a, const ymm b) force_inline;
static inline ymm addpz2(const ymm a, const ymm b) { return _mm256_add_pd(a, b); }
static inline ymm subpz2(const ymm a, const ymm b) force_inline;
static inline ymm subpz2(const ymm a, const ymm b) { return _mm256_sub_pd(a, b); }
static inline ymm mulpd2(const ymm a, const ymm b) force_inline;
static inline ymm mulpd2(const ymm a, const ymm b) { return _mm256_mul_pd(a, b); }
static inline ymm divpd2(const ymm a, const ymm b) force_inline;
static inline ymm divpd2(const ymm a, const ymm b) { return _mm256_div_pd(a, b); }

static inline ymm mulpz2(const ymm ab, const ymm xy) force_inline;
static inline ymm mulpz2(const ymm ab, const ymm xy)
{
    const ymm aa = _mm256_unpacklo_pd(ab, ab);
    const ymm bb = _mm256_unpackhi_pd(ab, ab);
    const ymm yx = _mm256_shuffle_pd(xy, xy, 5);
#ifdef __FMA__
    return _mm256_fmaddsub_pd(aa, xy, _mm256_mul_pd(bb, yx));
#else
    return _mm256_addsub_pd(_mm256_mul_pd(aa, xy), _mm256_mul_pd(bb, yx));
#endif // __FMA__
}

static inline ymm divpz2(const ymm ab, const ymm xy) force_inline;
static inline ymm divpz2(const ymm ab, const ymm xy)
{
    const ymm x2y2 = _mm256_mul_pd(xy, xy);
    const ymm r2r2 = _mm256_hadd_pd(x2y2, x2y2);
    return _mm256_div_pd(mulpz2(ab, cnjpz2(xy)), r2r2);
}

static inline ymm h1xpz2(const ymm xy) force_inline;
static inline ymm h1xpz2(const ymm xy)
{
#if 1
    static const ymm rr = { RSQRT2PSQRT2, RSQRT2PSQRT2, RSQRT2PSQRT2, RSQRT2PSQRT2 };
    const ymm w8xy = w8xpz2(xy);
    return _mm256_mul_pd(rr, _mm256_add_pd(xy, w8xy));
#else
    static const ymm h1 = { H1X, H1Y, H1X, H1Y };
    return mulpz2(h1, xy);
#endif
}

static inline ymm h3xpz2(const ymm xy) force_inline;
static inline ymm h3xpz2(const ymm xy)
{
#if 1
    static const ymm r1 = { M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, M_SQRT1_2 };
    static const ymm r2 = { RSQRT2PSQRT2, RSQRT2PSQRT2, RSQRT2PSQRT2, RSQRT2PSQRT2 };
    const ymm ymx = cnjpz2(_mm256_shuffle_pd(xy, xy, 5));
    const ymm w8xy = _mm256_mul_pd(r1, _mm256_add_pd(xy, ymx));
    return _mm256_mul_pd(r2, _mm256_add_pd(ymx, w8xy));
#else
    static const ymm h3 = { -H1Y, -H1X, -H1Y, -H1X };
    return mulpz2(h3, xy);
#endif
}

static inline ymm hfxpz2(const ymm xy) force_inline;
static inline ymm hfxpz2(const ymm xy)
{
#if 1
    static const ymm rr = { RSQRT2PSQRT2, RSQRT2PSQRT2, RSQRT2PSQRT2, RSQRT2PSQRT2 };
    const ymm v8xy = v8xpz2(xy);
    return _mm256_mul_pd(rr, _mm256_add_pd(xy, v8xy));
#else
    static const ymm hf = { H1X, -H1Y, H1X, -H1Y };
    return mulpz2(hf, xy);
#endif
}

static inline ymm hdxpz2(const ymm xy) force_inline;
static inline ymm hdxpz2(const ymm xy)
{
#if 1
    static const ymm r1 = { M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, M_SQRT1_2 };
    static const ymm r2 = { RSQRT2PSQRT2, RSQRT2PSQRT2, RSQRT2PSQRT2, RSQRT2PSQRT2 };
    const ymm myx = jxpz2(xy);
    const ymm v8xy = _mm256_mul_pd(r1, _mm256_add_pd(xy, myx));
    return _mm256_mul_pd(r2, _mm256_add_pd(myx, v8xy));
#else
    static const ymm hd = { -H1Y, H1X, -H1Y, H1X };
    return mulpz2(hd, xy);
#endif
}

static inline ymm duppz2(const xmm x) force_inline;
static inline ymm duppz2(const xmm x) { return _mm256_broadcast_pd(&x); }

static inline ymm duppz3(const complex_t& z) force_inline;
static inline ymm duppz3(const complex_t& z)
{
#if 1
    const ymm x = getpz2(&z);
    return _mm256_permute2f128_pd(x, x, 0);
#else
    const xmm x = getpz(z);
    return _mm256_broadcast_pd(&x);
#endif
}

static inline ymm cat(const xmm a, const xmm b) force_inline;
static inline ymm cat(const xmm a, const xmm b)
{
    const ymm ax = _mm256_castpd128_pd256(a);
    const ymm bx = _mm256_castpd128_pd256(b);
    return _mm256_permute2f128_pd(ax, bx, 0x20);
    //return _mm256_insertf128_pd(_mm256_castpd128_pd256(a), b, 1);
}

static inline ymm catlo(const ymm ax, const ymm by) force_inline;
static inline ymm catlo(const ymm ax, const ymm by)
{
    return _mm256_permute2f128_pd(ax, by, 0x20); // == ab
}

static inline ymm cathi(const ymm ax, const ymm by) force_inline;
static inline ymm cathi(const ymm ax, const ymm by)
{
    return _mm256_permute2f128_pd(ax, by, 0x31); // == xy
}

template <int s> static inline ymm getwp2(const_complex_vector W, const int p) force_inline2;
template <int s> static inline ymm getwp2(const_complex_vector W, const int p)
{
#if 1
    const int sp0 = s * p;
    const int sp1 = s * (p + 1);
    //return cmplx2(W[sp0], W[sp1]);
    return cmplx3(W[sp0], W[sp1]);
#else
    const int sp0 = 2*s * p;
    const int sp1 = 2*s * (p + 1);
    const_double_vector r = &W[0].Re;
    return _mm256_i64gather_pd(r, _mm256_set_epi64x(sp1+1,sp1,sp0+1,sp0), 8);
#endif
}

template <int s> static inline ymm cnj_getwp2(const_complex_vector W, const int p) force_inline2;
template <int s> static inline ymm cnj_getwp2(const_complex_vector W, const int p)
{
    const int sp0 = s * p;
    const int sp1 = s * (p + 1);
    return cnjpz2(cmplx3(W[sp0], W[sp1]));
    //return cnjpz2(cat(getpz(W[sp0]), getpz(W[sp1])));
    //return cat(cnjpz(getpz(W[sp0])), cnjpz(getpz(W[sp1])));
}

static inline ymm swaplohi(const ymm a_b) force_inline;
static inline ymm swaplohi(const ymm a_b)
{
    return _mm256_permute2f128_pd(a_b, a_b, 1);
}

static inline xmm getlo(const ymm a_b) force_inline;
static inline xmm getlo(const ymm a_b) { return _mm256_castpd256_pd128(a_b); }
static inline xmm gethi(const ymm a_b) force_inline;
static inline xmm gethi(const ymm a_b) { return _mm256_extractf128_pd(a_b, 1); }

static inline ymm cnjlohi(const ymm a_b) force_inline;
static inline ymm cnjlohi(const ymm a_b)
{
    static const ymm zzmm = { 0.0, 0.0, -0.0, -0.0 };
    return _mm256_xor_pd(zzmm, a_b);
}

template <int s> static inline ymm getpz3(const_complex_vector z) force_inline2;
template <int s> static inline ymm getpz3(const_complex_vector z)
{
#if 1
    return cmplx2(z[0], z[s]);
#else
    static const __m256i idx = _mm256_set_epi64x(2*s+1,2*s,1,0);
    return _mm256_i64gather_pd(&z->Re, idx, 8);
#endif
}

template <int s> static inline void setpz3(complex_vector z, const ymm x) force_inline3;
template <int s> static inline void setpz3(complex_vector z, const ymm x)
{
    const xmm b = gethi(x);
    const xmm a = getlo(x);
    setpz(z[0], a);
    setpz(z[s], b);
    //setpz(z[0], getlo(x));
    //setpz(z[s], gethi(x));
}

static inline void* simd_malloc(int ns) { return _mm_malloc(ns, 32); }
static inline void simd_free(void* p) { _mm_free(p); }

} // namespace OTFFT_MISC

#else // !__AVX__

namespace OTFFT_MISC {

struct ymm { xmm lo, hi; };

static inline void zeroupper() force_inline;
static inline void zeroupper() {}

static inline ymm cmplx2(const double& a, const double& b, const double& c, const double &d) force_inline;
static inline ymm cmplx2(const double& a, const double& b, const double& c, const double &d)
{
    const ymm y = { cmplx(a, b), cmplx(c, d) };
    return y;
}

static inline ymm cmplx2(const complex_t& a, const complex_t& b) force_inline;
static inline ymm cmplx2(const complex_t& a, const complex_t& b)
{
    const ymm y = { getpz(a), getpz(b) };
    return y;
}

static inline ymm getpz2(const_complex_vector z) force_inline2;
static inline ymm getpz2(const_complex_vector z)
{
    const ymm y = { getpz(z[0]), getpz(z[1]) };
    return y;
}

static inline void setpz2(complex_vector z, const ymm& y) force_inline3;
static inline void setpz2(complex_vector z, const ymm& y)
{
    setpz(z[0], y.lo);
    setpz(z[1], y.hi);
}

static inline ymm cnjpz2(const ymm& xy) force_inline;
static inline ymm cnjpz2(const ymm& xy)
{
    const ymm y = { cnjpz(xy.lo), cnjpz(xy.hi) };
    return y;
}

static inline ymm jxpz2(const ymm& xy) force_inline;
static inline ymm jxpz2(const ymm& xy)
{
    const ymm y = { jxpz(xy.lo), jxpz(xy.hi) };
    return y;
}

static inline ymm v8xpz2(const ymm& xy) force_inline;
static inline ymm v8xpz2(const ymm& xy)
{
    const ymm y = { v8xpz(xy.lo), v8xpz(xy.hi) };
    return y;
}

static inline ymm w8xpz2(const ymm& xy) force_inline;
static inline ymm w8xpz2(const ymm& xy)
{
    const ymm y = { w8xpz(xy.lo), w8xpz(xy.hi) };
    return y;
}

static inline ymm h1xpz2(const ymm& xy) force_inline;
static inline ymm h1xpz2(const ymm& xy)
{
    const ymm y = { h1xpz(xy.lo), h1xpz(xy.hi) };
    return y;
}

static inline ymm h3xpz2(const ymm& xy) force_inline;
static inline ymm h3xpz2(const ymm& xy)
{
    const ymm y = { h3xpz(xy.lo), h3xpz(xy.hi) };
    return y;
}

static inline ymm hfxpz2(const ymm& xy) force_inline;
static inline ymm hfxpz2(const ymm& xy)
{
    const ymm y = { hfxpz(xy.lo), hfxpz(xy.hi) };
    return y;
}

static inline ymm hdxpz2(const ymm& xy) force_inline;
static inline ymm hdxpz2(const ymm& xy)
{
    const ymm y = { hdxpz(xy.lo), hdxpz(xy.hi) };
    return y;
}

static inline ymm addpz2(const ymm& a, const ymm& b) force_inline;
static inline ymm addpz2(const ymm& a, const ymm& b)
{
    const ymm y = { addpz(a.lo, b.lo), addpz(a.hi, b.hi) };
    return y;
}

static inline ymm subpz2(const ymm& a, const ymm& b) force_inline;
static inline ymm subpz2(const ymm& a, const ymm& b)
{
    const ymm y = { subpz(a.lo, b.lo), subpz(a.hi, b.hi) };
    return y;
}

static inline ymm mulpz2(const ymm& a, const ymm& b) force_inline;
static inline ymm mulpz2(const ymm& a, const ymm& b)
{
    const ymm y = { mulpz(a.lo, b.lo), mulpz(a.hi, b.hi) };
    return y;
}

static inline ymm divpz2(const ymm& a, const ymm& b) force_inline;
static inline ymm divpz2(const ymm& a, const ymm& b)
{
    const ymm y = { divpz(a.lo, b.lo), divpz(a.hi, b.hi) };
    return y;
}

static inline ymm mulpd2(const ymm& a, const ymm& b) force_inline;
static inline ymm mulpd2(const ymm& a, const ymm& b)
{
    const ymm y = { mulpd(a.lo, b.lo), mulpd(a.hi, b.hi) };
    return y;
}

static inline ymm divpd2(const ymm& a, const ymm& b) force_inline;
static inline ymm divpd2(const ymm& a, const ymm& b)
{
    const ymm y = { divpd(a.lo, b.lo), divpd(a.hi, b.hi) };
    return y;
}

static inline ymm duppz2(const xmm& x) force_inline;
static inline ymm duppz2(const xmm& x) { const ymm y = { x, x }; return y; }

static inline ymm duppz3(const complex_t& z) force_inline;
static inline ymm duppz3(const complex_t& z) {
    const xmm x = getpz(z);
    const ymm y = { x, x };
    return y;
}

static inline ymm cat(const xmm& a, const xmm& b) force_inline;
static inline ymm cat(const xmm& a, const xmm& b)
{
    const ymm y = { a, b };
    return y;
}

static inline ymm catlo(const ymm& ax, const ymm& by) force_inline;
static inline ymm catlo(const ymm& ax, const ymm& by)
{
    const ymm ab = { ax.lo, by.lo };
    return ab;
}

static inline ymm cathi(const ymm ax, const ymm by) force_inline;
static inline ymm cathi(const ymm ax, const ymm by)
{
    const ymm xy = { ax.hi, by.hi };
    return xy;
}

template <int s> static inline ymm getwp2(const_complex_vector W, const int p) force_inline2;
template <int s> static inline ymm getwp2(const_complex_vector W, const int p)
{
    const int sp0 = s * p;
    const int sp1 = s * (p + 1);
    return cmplx2(W[sp0], W[sp1]);
}

template <int s> static inline ymm cnj_getwp2(const_complex_vector W, const int p) force_inline2;
template <int s> static inline ymm cnj_getwp2(const_complex_vector W, const int p)
{
#if 1
    const int sp0 = s * p;
    const int sp1 = s * (p + 1);
    return cnjpz2(cat(getpz(W[sp0]), getpz(W[sp1])));
    //return cat(cnjpz(getpz(W[sp0])), cnjpz(getpz(W[sp1])));
#else
    const int sp0 = s * p;
    const int sp1 = s * (p + 1);
    const double& ax = W[sp0].Re;
    const double& ay = W[sp0].Im;
    const double& bx = W[sp1].Re;
    const double& by = W[sp1].Im;
    return cmplx2(ax, -ay, bx, -by);
#endif
}

static inline ymm swaplohi(const ymm& a_b) force_inline;
static inline ymm swaplohi(const ymm& a_b)
{
    const ymm b_a = { a_b.hi, a_b.lo };
    return b_a;
}

static inline xmm getlo(const ymm& a_b) force_inline;
static inline xmm getlo(const ymm& a_b) { return a_b.lo; }
static inline xmm gethi(const ymm& a_b) force_inline;
static inline xmm gethi(const ymm& a_b) { return a_b.hi; }

static inline ymm cnjlohi(const ymm& a_b) force_inline;
static inline ymm cnjlohi(const ymm& a_b)
{
    const ymm a_mb = { a_b.lo, negpz(a_b.hi) };
    return a_mb;
}

template <int s> static inline ymm getpz3(const_complex_vector z) force_inline2;
template <int s> static inline ymm getpz3(const_complex_vector z)
{
    return cmplx2(z[0], z[s]);
}

template <int s> static inline void setpz3(complex_vector z, const ymm& y) force_inline3;
template <int s> static inline void setpz3(complex_vector z, const ymm& y)
{
    setpz(z[0], getlo(y));
    setpz(z[s], gethi(y));
}

} // namespace OTFFT_MISC

#endif // __AVX__

/*****************************************************************************/

namespace OTFFT_MISC {

template <class T> struct simd_array
{
    T* p;

    simd_array() : p(0) {}
    simd_array(int n) : p((T*) simd_malloc(n*sizeof(T)))
    {
        if (p == 0) throw std::bad_alloc();
    }

    ~simd_array() { if (p) simd_free(p); }

    void setup(int n)
    {
        if (p) simd_free(p);
        p = (T*) simd_malloc(n*sizeof(T));
        if (p == 0) throw std::bad_alloc();
    }

    void destroy() { if (p) simd_free(p); p = 0; }

    T& operator[](int i) { return p[i]; }
    const T& operator[](int i) const { return p[i]; }
    T* operator&() const { return p; }
};

} // namespace OTFFT_MISC

namespace fftmisc = OTFFT_MISC;

/*****************************************************************************/

#endif // otfft_misc_h
