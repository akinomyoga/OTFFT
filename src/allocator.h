#ifndef OTFFT_ALLOCATOR_H
#define OTFFT_ALLOCATOR_H
#include <cstddef>
#include <memory>
#include <otfft/otfft_misc.h>

#define OTFFT_ALLOCATOR_H__BROKEN_SIMD_MALLOC

namespace otfft{

  template<typename T>
  class simd_allocator:public std::allocator<T>{
    typedef std::allocator<T> base;
    typedef typename base::pointer pointer;
    typedef typename base::const_pointer const_pointer;
    typedef typename base::size_type size_type;
  public:
    simd_allocator(){}
    simd_allocator(const simd_allocator&){}
    template<typename U>
    simd_allocator(const simd_allocator<U>&){}

    static const int alignment = 32; // should be larger than sizeof(std::ptrdiff_t)
    pointer allocate(size_type n, const_pointer hint = 0){
#ifdef OTFFT_ALLOCATOR_H__BROKEN_SIMD_MALLOC
      int mask=alignment-1;
      std::ptrdiff_t const mem=(std::ptrdiff_t)OTFFT::simd_malloc(n*sizeof(T)+alignment+sizeof(std::ptrdiff_t));
      std::ptrdiff_t const ptr=(mem+mask+sizeof(std::ptrdiff_t))&~(std::ptrdiff_t)mask;
      reinterpret_cast<std::ptrdiff_t*>(ptr)[-1]=mem;
      return reinterpret_cast<pointer>(ptr);
#else
      return reinterpret_cast<pointer>(OTFFT::simd_malloc(n*sizeof(T))); // simd_malloc が壊れている
#endif
    }

    void deallocate(pointer ptr, size_type n) {
#ifdef OTFFT_ALLOCATOR_H__BROKEN_SIMD_MALLOC
      std::ptrdiff_t const mem=reinterpret_cast<std::ptrdiff_t*>(ptr)[-1];
      OTFFT::simd_free(reinterpret_cast<void*>(mem));
#else
      OTFFT::simd_free(ptr);
#endif
    }

    template<typename U>
    struct rebind{typedef simd_allocator<U> other;};
  };

}
#endif
