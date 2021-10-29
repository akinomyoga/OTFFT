#include "util.h"
#include <cassert>
#include <cmath>

namespace otfft {

  inline void otfft_real::initialize_psi() {
    int const Nh = this->size / 2;
    int const Nq = (Nh + 1) / 2;
    this->psi_fwd.setup(Nq);
    this->psi_inv.setup(Nq);

    double const theta0 = 2.0 * M_PI / this->size;
    for(int p = 0; p < Nq; p++) {
      double const theta = p * theta0;
      double const hfcos = 0.5 * std::cos(theta);
      double const hfsin = 0.5 * std::sin(theta);
      this->psi_fwd[p] = complex_t(0.5 + hfsin, hfcos);
      this->psi_inv[p] = complex_t(0.5 - hfsin, hfcos);
    }
  }

  void otfft_real::resize(std::size_t n) {
    assert(n % 2 == 0);
    this->size = n;
    this->instance.setup((int) n / 2);
    this->initialize_psi();
  }

  inline void otfft_real::c2r_encode_half(complex_vector half, const_complex_vector src, const_complex_vector psi, dft_flags type) const {
    int const Nh = this->size / 2;
    int const Nq = (Nh + 1) / 2;

    // src[0], src[n/2]
    {
      // because: psi[0] = (1+i)/2, src[0] and src[Nh]: real.
      double const& realPartAtMiddle = (type & _rfft_complex_mask) == rfft_compact_complex ? src[0].Im : src[Nh].Re;
      half[0].Re = src[0].Re + realPartAtMiddle;
      half[0].Im = src[0].Re - realPartAtMiddle;
    }

    // normalization
    double factor = 2.0;
    if ((type & _normalization_mask) != normalization_none) {
      double const scale = (type & _normalization_mask) == normalization_sqrt ? std::sqrt(1.0 / this->size) : 1.0 / this->size;
      half[0].Re *= scale;
      half[0].Im *= scale;
      factor     *= scale;
    }
#ifdef otfft_misc_h
    using namespace OTFFT_MISC;
    xmm const xfactor = cmplx(factor, factor);
#endif

    // src[1] ... src[n/2-1] (except for src[n/4])
    for (int p = 1; p < Nq; p++) {
#ifdef otfft_misc_h
      xmm const src1  = getpz(src[p     ]);
      xmm const src2  = getpz(src[Nh - p]);
      xmm const alpha = mulpz(getpz(psi[p]), subpz(cnjpz(src1), src2));
      setpz(half[p     ], mulpd(xfactor, subpz(src1, cnjpz(alpha))));
      setpz(half[Nh - p], mulpd(xfactor, addpz(src2,       alpha )));
#else
      complex_t const src1  = src[p     ];
      complex_t const src2  = src[Nh - p];
      complex_t const alpha = psi[p] * (conj(src1) - src2);
      half[p     ] = factor * (src1 - conj(alpha));
      half[Nh - p] = factor * (src2 +      alpha );
#endif
    }

    // src[n/4]
    if (Nh % 2 == 0) {
      half[Nq].Re = src[Nq].Re;
      if (psi == &psi_fwd)
        half[Nq].Im = -src[Nq].Im;
      else
        half[Nq].Im =  src[Nq].Im;
      half[Nq].Re *= factor;
      half[Nq].Im *= factor;
    }
  }

  inline void otfft_real::r2c_decode_half(const_complex_vector half, complex_vector dst, const_complex_vector psi, dft_flags type) const {
    int const Nh = this->size / 2;
    int const Nq = (Nh + 1) / 2;

    // normalization
    if ((type & _normalization_mask) == normalization_none) {

      if ((type & _rfft_complex_mask) == rfft_compact_complex) {
        // rfft_compact_complex の場合は、
        // dst[0].Re に係数 0 の実部を格納し、
        // dst[0].Im に係数 Nh の実部を格納する。
        dst[0].Re = half[0].Re + half[0].Im;
        dst[0].Im = half[0].Re - half[0].Im;
      } else {
        dst[0 ] = half[0].Re + half[0].Im;
        dst[Nh] = half[0].Re - half[0].Im;
      }

      for (int p = 1; p < Nq; p++) {
#ifdef otfft_misc_h
        using namespace OTFFT_MISC;
        xmm const half1 = getpz(half[p     ]);
        xmm const half2 = getpz(half[Nh - p]);
        xmm const alpha = mulpz(getpz(psi[p]), subpz(cnjpz(half2), half1));
        setpz(dst[p     ], addpz(half1,       alpha ));
        setpz(dst[Nh - p], subpz(half2, cnjpz(alpha)));
#else
        complex_t const half1 = half[p     ];
        complex_t const half2 = half[Nh - p];
        complex_t const alpha = psi[p] * (conj(half2) - half1);
        dst[p     ] = half1 +      alpha ;
        dst[Nh - p] = half2 - conj(alpha);
#endif
      }

      if (Nh % 2 == 0) {
        dst[Nq].Re = half[Nq].Re;
        if (psi == &psi_fwd)
          dst[Nq].Im = -half[Nq].Im;
        else
          dst[Nq].Im =  half[Nq].Im;
      }
    } else {
      double const scale = 1.0 / ((type & _normalization_mask) == normalization_sqrt ? std::sqrt(this->size) : this->size);
#ifdef otfft_misc_h
      using namespace OTFFT_MISC;
      xmm const xscale = cmplx(scale, scale);
#endif

      if ((type & _rfft_complex_mask) == rfft_compact_complex) {
        // rfft_compact_complex の場合は、
        // dst[0].Re に係数 0 の実部を格納し、
        // dst[0].Im に係数 Nh の実部を格納する。
        dst[0].Re = scale * (half[0].Re + half[0].Im);
        dst[0].Im = scale * (half[0].Re - half[0].Im);
      } else {
        dst[0 ]   = scale * (half[0].Re + half[0].Im);
        dst[Nh]   = scale * (half[0].Re - half[0].Im);
      }

      for (int p = 1; p < Nq; p++) {
#ifdef otfft_misc_h
        using namespace OTFFT_MISC;
        xmm const half1 = getpz(half[p     ]);
        xmm const half2 = getpz(half[Nh - p]);
        xmm const alpha = mulpz(getpz(psi[p]), subpz(cnjpz(half2), half1));
        setpz(dst[p     ], mulpd(xscale, addpz(half1,       alpha )));
        setpz(dst[Nh - p], mulpd(xscale, subpz(half2, cnjpz(alpha))));
#else
        complex_t const half1 = half[p     ];
        complex_t const half2 = half[Nh - p];
        complex_t const alpha = psi[p] * (conj(half2) - half1);
        dst[p     ] = scale * (half1 +      alpha );
        dst[Nh - p] = scale * (half2 - conj(alpha));
#endif
      }

      if (Nh % 2 == 0) {
        dst[Nq].Re = half[Nq].Re;
        if (psi == &psi_fwd)
          dst[Nq].Im = -half[Nq].Im;
        else
          dst[Nq].Im =  half[Nq].Im;

        dst[Nq].Re *= scale;
        dst[Nq].Im *= scale;
      }
    }
  }

  void otfft_real::r2c_fwd(double const* src, std::complex<double>* _dst, otfft_buffer& work, dft_flags type) {
    complex_vector const dst = reinterpret_cast<complex_vector>(_dst);
    work.ensure<double>(this->size);
    std::copy(src, src + this->size, work.get<double>());

    complex_vector const half = work.get<complex_t>();
    instance.fwd0(half, dst);
    r2c_decode_half(half, dst, &psi_fwd, type);

    if ((type & _rfft_complex_mask) == rfft_full_complex)
      this->r2c_extend_complex(dst);
  }

  void otfft_real::r2c_inv(double const* src, std::complex<double>* _dst, otfft_buffer& work, dft_flags type) {
    complex_vector const dst = reinterpret_cast<complex_vector>(_dst);
    work.ensure<double>(this->size);
    std::copy(src, src + this->size, work.get<double>());

    complex_vector const half = work.get<complex_t>();
    instance.inv0(half, dst);
    r2c_decode_half(half, dst, &psi_inv, type);

    if ((type & _rfft_complex_mask) == rfft_full_complex)
      this->r2c_extend_complex(dst);
  }

  void otfft_real::c2r_fwd(std::complex<double> const* src, double* dst, otfft_buffer& work, dft_flags type) const {
    complex_vector const half = reinterpret_cast<complex_vector>(dst);
    this->c2r_encode_half(half, reinterpret_cast<const_complex_vector>(src), &psi_inv, type);

    int const Nh = this->size / 2;
    work.ensure<complex_t>(Nh);
    instance.fwd0(half, work.get<complex_t>());
  }

  void otfft_real::c2r_inv(std::complex<double> const* src, double* dst, otfft_buffer& work, dft_flags type) const {
    complex_vector const half = reinterpret_cast<complex_vector>(dst);
    this->c2r_encode_half(half, reinterpret_cast<const_complex_vector>(src), &psi_fwd, type);

    int const Nh = this->size / 2;
    work.ensure<complex_t>(Nh);
    instance.inv0(half, work.get<complex_t>());
  }
}
