// -*- mode:c++ -*-
#pragma once
#ifndef otfft_util_h
#define otfft_util_h
#include <otfft/otfft_misc.h>
#include <otfft/otfft.h>
#include <complex>
#include <cassert>

#define OTFFT_USE_SIMD_ARRAY

#ifndef OTFFT_USE_SIMD_ARRAY
# include <memory>
#endif

namespace otfft{
  using OTFFT_MISC::complex_t;
  using OTFFT_MISC::complex_vector;
  using OTFFT_MISC::const_complex_vector;
  using OTFFT_MISC::simd_array;


  class otfft_buffer{
#ifdef OTFFT_USE_SIMD_ARRAY
    simd_array<char> data;
#else
    struct simd_deleter{
      void operator()(void* ptr) const{
        OTFFT_MISC::simd_free(ptr);
      }
    };
    std::unique_ptr<void,simd_deleter> data;
#endif
    std::size_t m_size;

  public:
    otfft_buffer():m_size(0){}
    void* get() const{
#ifdef OTFFT_USE_SIMD_ARRAY
      return &this->data;
#else
      return this->data.get();
#endif
    }
    std::size_t size() const{return this->m_size;}

    void ensure(std::size_t size){
      if(this->m_size<size)
        this->resize(size);
    }
    void resize(std::size_t size){
#ifdef OTFFT_USE_SIMD_ARRAY
      this->data.setup(size);
#else
      this->data.reset(OTFFT_MISC::simd_malloc(size));
#endif
      this->m_size=size;
    }

    template<typename T>
    T* get() const{
      return reinterpret_cast<T*>(get());
    }
    template<typename T>
    void ensure(std::size_t size){
      this->ensure(sizeof(T)*size);
    }
    template<typename T>
    void resize(std::size_t size){
      this->resize(sizeof(T)*size);
    }
  };

  // 現在の実装における仮定
  //   requirement: n は 2 の累乗である。
  //   requirement: n は 4 以上である。
  //   complex<double> と complex_t は binary compatible である。
  class otfft_real{
    std::size_t size;
    OTFFT::FFT0 instance;
    simd_array<complex_t> psi_fwd;
    simd_array<complex_t> psi_inv;

  public:
    otfft_real():size(0){}
    otfft_real(std::size_t n){
      this->resize(n);
    }

    void resize(std::size_t n){
      assert(n%2==0);
      this->size=n;
      this->instance.setup((int)n/2);
      this->initialize_psi();
    }

  private:
    void initialize_psi(){
      int const Nq=this->size/4;
      this->psi_fwd.setup(Nq);
      this->psi_inv.setup(Nq);

      double const theta0=2.0*M_PI/this->size;
      for(int p=0;p<Nq;p++){
        double const theta=p*theta0;
        double const hfcos=0.5*std::cos(theta);
        double const hfsin=0.5*std::sin(theta);
        this->psi_fwd[p]=complex_t(0.5+hfsin,hfcos);
        this->psi_inv[p]=complex_t(0.5-hfsin,hfcos);
      }
    }

    void c2r_encode_half(complex_vector half,const_complex_vector src,const_complex_vector psi) const;
    void r2c_decode_half(const_complex_vector half,complex_vector dst,const_complex_vector psi) const;

    void r2c_extend_complex(complex_vector dst){
      int const Nh=this->size/2;
      for(int p=Nh+1;p<size;p++)
        dst[p]=conj(dst[size-p]);
    }

  public:
    /**@fn r2c_fwd
     *   exp(-i*k/N) による実数から複素数への変換、規格化なし。
     * @param[in]  src  フーリエ変換前の実数値を指定します。
     * @param[out] dst  フーリエ変換後の複素係数を格納します。
     *   isFullComplex == false の時は dst[0],...,dst[int(n/2)] の int(n/2) + 1 の要素に値が格納されます。
     *   isFullComplex == true の時は dst[0],...,dst[n-1] の n の要素に値が格納されます。
     *   実数の Fourier 変換なので dst[n-k] = conj(dst[k]) for k = 1, ..., n/2-1 です。
     * @param[out] work 作業用バッファを指定します。
     * @param[in]  isFullComplex フーリエ変換後の複素係数の後半も計算するかどうかを指定します。
     */
    void r2c_fwd(double const* src,std::complex<double>* _dst,otfft_buffer& work,bool isFullComplex=false);

    /**@fn r2c_inv
     *   exp(+i*k/N) による実数から複素数への変換、規格化なし。
     * @param[in]  src  フーリエ変換前の実数値を指定します。
     * @param[out] dst  フーリエ変換後の複素係数を格納します。
     *   isFullComplex == false の時は dst[0],...,dst[int(n/2)] の int(n/2) + 1 の要素に値が格納されます。
     *   isFullComplex == true の時は dst[0],...,dst[n-1] の n の要素に値が格納されます。
     *   実数の Fourier 変換なので dst[n-k] = conj(dst[k]) for k = 1, ..., n/2-1 です。
     * @param[out] work 作業用バッファを指定します。
     * @param[in]  isFullComplex フーリエ変換後の複素係数の後半も計算するかどうかを指定します。
     */
    void r2c_inv(double const* src,std::complex<double>* _dst,otfft_buffer& work,bool isFullComplex=false);

    /**@fn c2r_fwd
     *   \f$\exp(-2\pi i jk/N)\f$ による複素数から実数への変換、規格化なし。
     * @param[in] src 複素数配列を指定します。
     *   src[0],...,src[int(n/2)] の int(n/2)+1 の要素を計算に使用します。
     *   src[int(n/2)+1],...,src[n-1] の後半の要素は計算に使用しません。
     *   また本来 0 でなければならない src[0], src[int(n/2)] の虚部も無視します。
     */
    void c2r_fwd(std::complex<double> const* src,double* dst,otfft_buffer& work) const;

    /**@fn c2r_fwd
     *   \f$\exp(+2\pi i jk/N)\f$ による複素数から実数への変換、規格化なし。
     * @param[in] src 複素数配列を指定します。
     *   src[0],...,src[int(n/2)] の int(n/2)+1 の要素を計算に使用します。
     *   src[int(n/2)+1],...,src[n-1] の後半の要素は計算に使用しません。
     *   また本来 0 でなければならない src[0], src[int(n/2)] の虚部も無視します。
     */
    void c2r_inv(std::complex<double> const* src,double* dst,otfft_buffer& work) const;

  };

}
#endif
