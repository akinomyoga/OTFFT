// -*- mode:c++ -*-
#pragma once
#ifndef otfft_util_h
#define otfft_util_h
#include <cstdlib>
#include <complex>
#include <otfft/otfft_misc.h>
#include <otfft/otfft.h>

namespace otfft{
  using OTFFT_MISC::complex_t;
  using OTFFT_MISC::complex_vector;
  using OTFFT_MISC::const_complex_vector;
  using OTFFT_MISC::simd_array;

  /**@class otfft_buffer
   * 適切に align された作業用メモリスペースを提供します。
   */
  class otfft_buffer{
    simd_array<char> data;
    std::size_t m_size;

  public:
    /**@fn otfft_buffer::otfft_buffer();
     * 既定のコンストラクタです。
     * otfft_buffer::resize(std::size_t) で領域を確保してから使用する必要があります。
     */
    otfft_buffer():m_size(0){}

    /**@fn void* otfft_buffer::get() const;
     * 確保した領域へのポインタを取得します。
     */
    void* get() const{
      return &this->data;
    }
    /**@fn std::size_t otfft_buffer::size() const;
     * 確保した領域の大きさをバイト単位で取得します。
     * @return 確保済の領域のサイズを返します。
     */
    std::size_t size() const{return this->m_size;}
    /**@fn void* otfft_buffer::ensure(std::size_t size);
     * 確保する領域の最小サイズを指定します。
     * 確保済の領域が指定サイズより小さい場合には領域を確保し直します。
     * データのコピーは実行されません。
     * 既に指定したサイズ以上の領域が確保されている場合は何もしません。
     * @param[in] size 最小確保サイズをバイト単位で指定します。
     */
    void ensure(std::size_t size){
      if(this->m_size<size)
        this->resize(size);
    }
    /**@fn void* otfft_buffer::ensure(std::size_t size);
     * 領域を指定サイズで確保し直します。データのコピーは実行されません。
     * @param[in] size 新しく確保する領域のサイズをバイト単位で指定します。
     */
    void resize(std::size_t size){
      this->data.setup(size);
      this->m_size=size;
    }

    /**@fn void* otfft_buffer::get<T>() const;
     * 確保した領域へのポインタを、指定した型のポインタとして取得します。
     * @tparam T 確保した領域に配置する要素の型を指定します。
     */
    template<typename T>
    T* get() const{
      return reinterpret_cast<T*>(get());
    }
    /**@fn std::size_t otfft_buffer::size<T>() const;
     * 確保した領域の大きさを要素の個数単位で取得します。
     * @tparam T 確保した領域に配置する要素の型を指定します。
     * @return 確保済の領域のサイズを要素の個数単位で返します。
     */
    template<typename T>
    std::size_t size() const{
      return this->m_size/sizeof(T);
    }
    /**@fn void* otfft_buffer::ensure<T>(std::size_t size);
     * 確保する領域の最小サイズを指定します。
     * 確保済の領域が指定サイズより小さい場合には領域を確保し直します。
     * データのコピーは実行されません。
     * 既に指定したサイズ以上の領域が確保されている場合は何もしません。
     * @tparam T 確保した領域に配置する要素の型を指定します。
     * @param[in] size 最小確保サイズを要素の個数単位で指定します。
     */
    template<typename T>
    void ensure(std::size_t size){
      this->ensure(sizeof(T)*size);
    }
    /**@fn void* otfft_buffer::ensure<T>(std::size_t size);
     * 領域を指定サイズで確保し直します。データのコピーは実行されません。
     * @tparam T 確保した領域に配置する要素の型を指定します。
     * @param[in] size 新しく確保する領域のサイズを要素の個数単位で指定します。
     */
    template<typename T>
    void resize(std::size_t size){
      this->resize(sizeof(T)*size);
    }
  };

  /**@class otfft_real
   * 実離散フーリエ変換を行う為のクラスです。
   */
  // 現在の実装における仮定
  //   requirement: n は 2 の累乗である。
  //   complex<double> と complex_t は binary compatible である。
  class otfft_real{
    std::size_t size;
    OTFFT::FFT0 instance;
    simd_array<complex_t> psi_fwd;
    simd_array<complex_t> psi_inv;

  public:
    /**@fn otfft_real::otfft_real();
     *   既定コンストラクタ。後で otfft_real::resize(n) を用いて
     *   離散フーリエ変換の大きさを指定してから使用する必要があります。
     */
    otfft_real():size(0){}
    /**@fn otfft_real::otfft_real(std::size_t n);
     *   離散フーリエ変換の大きさを指定して初期化を行います。
     * @param[in] n 変換の大きさを指定します。
     */
    otfft_real(std::size_t n){
      this->resize(n);
    }

    /**@fn void otfft_real::resize(std::size_t n);
     *   離散フーリエ変換のサイズを変更します。
     * @param[in] n 変更後のサイズを指定します。
     */
    void resize(std::size_t n);

  private:
    void initialize_psi();

    void c2r_encode_half(complex_vector half,const_complex_vector src,const_complex_vector psi) const;
    void r2c_decode_half(const_complex_vector half,complex_vector dst,const_complex_vector psi) const;

    void r2c_extend_complex(complex_vector dst){
      int const Nh=this->size/2;
      for(std::size_t p=Nh+1;p<size;p++)
        dst[p]=conj(dst[size-p]);
    }

  public:
    /**@fn r2c_fwd
     *   exp(-i*k/N) による実数から複素数への変換、規格化なし。
     * @param[in]  src  フーリエ変換前の実数値を指定します。
     * @param[out] dst  フーリエ変換後の複素係数の格納先を指定します。
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
     * @param[out] dst  フーリエ変換後の複素係数の格納先を指定します。
     *   isFullComplex == false の時は dst[0],...,dst[int(n/2)] の int(n/2) + 1 の要素に値が格納されます。
     *   isFullComplex == true の時は dst[0],...,dst[n-1] の n の要素に値が格納されます。
     *   実数の Fourier 変換なので dst[n-k] = conj(dst[k]) for k = 1, ..., n/2-1 です。
     * @param[out] work 作業用バッファを指定します。
     * @param[in]  isFullComplex フーリエ変換後の複素係数の後半も計算するかどうかを指定します。
     */
    void r2c_inv(double const* src,std::complex<double>* _dst,otfft_buffer& work,bool isFullComplex=false);

    /**@fn c2r_fwd
     *   \f$\exp(-2\pi i jk/N)\f$ による複素数から実数への変換、規格化なし。
     * @param[in] src 変換元の複素数配列を指定します。
     *   src[0],...,src[int(n/2)] の int(n/2)+1 の要素を計算に使用します。
     *   src[int(n/2)+1],...,src[n-1] の後半の要素は計算に使用しません。
     *   また本来 0 でなければならない src[0], src[int(n/2)] の虚部も無視します。
     * @param[out] dst 変換結果を格納する実数配列を指定します。
     * @param[out] work 作業用バッファを指定します。
     */
    void c2r_fwd(std::complex<double> const* src,double* dst,otfft_buffer& work) const;

    /**@fn c2r_fwd
     *   \f$\exp(+2\pi i jk/N)\f$ による複素数から実数への変換、規格化なし。
     * @param[in] src 変換元の複素数配列を指定します。
     *   src[0],...,src[int(n/2)] の int(n/2)+1 の要素を計算に使用します。
     *   src[int(n/2)+1],...,src[n-1] の後半の要素は計算に使用しません。
     *   また本来 0 でなければならない src[0], src[int(n/2)] の虚部も無視します。
     * @param[out] dst 変換結果を格納する実数配列を指定します。
     * @param[out] work 作業用バッファを指定します。
     */
    void c2r_inv(std::complex<double> const* src,double* dst,otfft_buffer& work) const;

  };

}
#endif
