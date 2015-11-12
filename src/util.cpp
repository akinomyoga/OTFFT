#include "util.h"

namespace otfft{

  inline void otfft_real::c2r_encode_half(complex_vector half,const_complex_vector src,const_complex_vector psi) const{
    int const Nh=this->size/2;
    int const Nq=this->size/4;

    {
      // because: psi[0] = (1+i)/2, src[0] and src[Nh]: real.
      half[0].Re = src[0].Re + src[Nh].Re;
      half[0].Im = src[0].Re - src[Nh].Re;
    }
#ifdef otfft_misc_h
    using namespace OTFFT_MISC;
    xmm const factor = cmplx(2.0, 2.0);
#endif
    for(int p=1;p<Nq;p++){
#ifdef otfft_misc_h
      xmm const src1  = getpz(src[p   ]);
      xmm const src2  = getpz(src[Nh-p]);
      xmm const alpha = mulpz(getpz(psi[p]), subpz(cnjpz(src1), src2));
      setpz(half[p   ], mulpd(factor, subpz(src1, cnjpz(alpha))));
      setpz(half[Nh-p], mulpd(factor, addpz(src2,       alpha )));
#else
      complex_t const src1  = src[p   ];
      complex_t const src2  = src[Nh-p];
      complex_t const alpha = psi[p] * (conj(src1) - src2);
      half[p   ] = 2.0*(src1 - conj(alpha));
      half[Nh-p] = 2.0*(src2 +      alpha );
#endif
    }
    if(Nh%2==0){
      half[Nq].Re = 2.0*src[Nq].Re;
      if(psi==&psi_fwd)
        half[Nq].Im = -2.0*src[Nq].Im;
      else
        half[Nq].Im =  2.0*src[Nq].Im;
    }
  }

  inline void otfft_real::r2c_decode_half(const_complex_vector half,complex_vector dst,const_complex_vector psi) const{
    int const Nh=this->size/2;
    int const Nq=this->size/4;

    {
      dst[0 ] = half[0].Re + half[0].Im;
      dst[Nh] = half[0].Re - half[0].Im;
    }
    for(int p=1;p<Nq;p++){
#ifdef otfft_misc_h
      using namespace OTFFT_MISC;
      xmm const half1 = getpz(half[p   ]);
      xmm const half2 = getpz(half[Nh-p]);
      xmm const alpha = mulpz(getpz(psi[p]), subpz(cnjpz(half2), half1));
      setpz(dst[p   ], addpz(half1,       alpha ));
      setpz(dst[Nh-p], subpz(half2, cnjpz(alpha)));
#else
      complex_t const half1 = half[p   ];
      complex_t const half2 = half[Nh-p];
      complex_t const alpha = psi[p] * (conj(half2) - half1);
      dst[p   ] = half1 +      alpha ;
      dst[Nh-p] = half2 - conj(alpha);
#endif
    }
    if(Nh%2==0){
      dst[Nq].Re = half[Nq].Re;
      if(psi==&psi_fwd)
        dst[Nq].Im = -half[Nq].Im;
      else
        dst[Nq].Im =  half[Nq].Im;
    }
  }

  void otfft_real::r2c_fwd(double const* src,std::complex<double>* _dst,otfft_buffer& work,bool isFullComplex){
    complex_vector const dst=reinterpret_cast<complex_vector>(_dst);
    work.ensure<double>(this->size);
    std::copy(src,src+this->size,work.get<double>());

    complex_vector const half=work.get<complex_t>();
    instance.fwd0(half,dst);
    r2c_decode_half(half,dst,&psi_fwd);

    if(isFullComplex)
      this->r2c_extend_complex(dst);
  }

  void otfft_real::r2c_inv(double const* src,std::complex<double>* _dst,otfft_buffer& work,bool isFullComplex){
    complex_vector const dst=reinterpret_cast<complex_vector>(_dst);
    work.ensure<double>(this->size);
    std::copy(src,src+this->size,work.get<double>());

    complex_vector const half=work.get<complex_t>();
    instance.inv0(half,dst);
    r2c_decode_half(half,dst,&psi_inv);

    if(isFullComplex)
      this->r2c_extend_complex(dst);
  }

  void otfft_real::c2r_fwd(std::complex<double> const* src,double* dst,otfft_buffer& work) const{
    complex_vector const half=reinterpret_cast<complex_vector>(dst);
    this->c2r_encode_half(half,reinterpret_cast<const_complex_vector>(src),&psi_inv);

    int const Nh=this->size/2;
    work.ensure<complex_t>(Nh);
    instance.fwd0(half,work.get<complex_t>());
  }

  void otfft_real::c2r_inv(std::complex<double> const* src,double* dst,otfft_buffer& work) const{
    complex_vector const half=reinterpret_cast<complex_vector>(dst);
    this->c2r_encode_half(half,reinterpret_cast<const_complex_vector>(src),&psi_fwd);

    int const Nh=this->size/2;
    work.ensure<complex_t>(Nh);
    instance.inv0(half,work.get<complex_t>());
  }
}
