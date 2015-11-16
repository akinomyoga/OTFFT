# -*- mode:makefile-gmake -*-

# configurations

#USE_OPENMP := yes
USE_OPENMP := no

CXXFLAGS:=

#CXX = icpc
#CXX = g++-mp-5# for Mac with MacPorts and Xcode
#CXX = g++-5
#CXX = clang++-mp-3.7                                                                                                                                                                                           
#CXX = x86_64-w64-mingw32-g++

ifeq ($(HOSTNAME),bessel)
  CXX := /home/murase/opt/gcc/4.9.2/bin/g++-4.9
else
  ifeq ($(CXX),icpc)
    CXX = icpc
    CXXFLAGS = -O3 -march=native -xHost
  else
    #CXXFLAGS += -Ofast
    CXXFLAGS += -O3
  endif
endif

#CXXFLAGS += -Wa,-q# for Mac (required: cd /opt/local/bin; ln -s /usr/bin/clang)
#CXXFLAGS += -std=c++11
CXXFLAGS += -msse2 -mfpmath=sse#+387
CXXFLAGS += -march=native -mtune=native# -mno-fma
#CXXFLAGS += -march=core-avx-i -mtune=core-avx-i
#CXXFLAGS += -Xclang -fopenmp
#CXXFLAGS += -Xclang -fopenmp=libiomp5
CXXFLAGS += -ffast-math
#CXXFLAGS += -funroll-loops
CXXFLAGS += -funroll-all-loops
#CXXFLAGS += -fpeel-loops
#CXXFLAGS += -floop-flatten
#CXXFLAGS += -floop-optimize
#CXXFLAGS += -funswitch-loops
#CXXFLAGS += -fomit-frame-pointer
#CXXFLAGS += -fforce-addr
#CXXFLAGS += -ftree-vectorize
