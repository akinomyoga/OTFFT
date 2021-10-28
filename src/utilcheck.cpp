#include <cstdlib>
#include <cstdio>
#include "util.h"

bool test_otfft_real() {
  static const int size = 64;

  // reference data
  otfft::simd_array<double               > rdata(size);
  otfft::simd_array<std::complex<double> > cdata(size);
  for (int i = 0; i < size; i++) {
    double const value = (std::rand() + 0.5) / (RAND_MAX + 1.0);
    rdata[i] = value;
    cdata[i] = value;
  }
  OTFFT::FFT fft(size);
  fft.fwd0(reinterpret_cast<otfft::complex_vector>(&cdata));

  otfft::otfft_real rfft(size);
  otfft::otfft_buffer work;

  double err2 = 0.0;

  otfft::simd_array<std::complex<double> > r2cdata(size / 2 + 1);
  rfft.r2c_fwd(&rdata, &r2cdata, work, otfft::rfft_half_complex);
  otfft::simd_array<double> c2rdata(size);
  rfft.c2r_inv(&r2cdata, &c2rdata, work, otfft::rfft_half_complex);

  for (int i = 0; i < size; i++) {
    std::complex<double> const c   = cdata[i];
    std::complex<double> const r2c = i < size / 2 + 1 ? r2cdata[i] : conj(r2cdata[size - i]);
    double const erc = norm(c - r2c);
    double const c2r = c2rdata[i] / size;
    double const err = (rdata[i] - c2r) * (rdata[i] - c2r);
    err2 += erc + err;
  }

  if (std::sqrt(err2) > 1e-10) {
    std::fprintf(stderr, "== large error in otfft_real ==\n");

    for (int i = 0; i < size; i++) {
      std::complex<double> const c   = cdata[i];
      std::complex<double> const r2c = i < size / 2 + 1 ? r2cdata[i] : conj(r2cdata[size - i]);
      double const erc = norm(c - r2c);

      double const c2r = c2rdata[i] / size;
      double const err = (rdata[i] - c2r) * (rdata[i] - c2r);

      err2 += erc + err;
      std::fprintf(
        stderr, "--r2c-> %12g %+12g i (ERR %12g) --c2r-> %12g (ERR %12g)\n",
        real(r2c), imag(r2c), std::sqrt(erc),
        c2r, std::sqrt(err)
      );
    }

    std::fprintf(stderr, "ERR_TOT = %g\n", std::sqrt(err2));
    std::fflush(stderr);
    return false;
  }

  return true;
}

int main() {
  if (!test_otfft_real())
    return 1;

  std::printf("done\n");
  return 0;
}
