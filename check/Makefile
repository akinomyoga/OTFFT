# -*- makefile-gmake -*-
# otfft/check/Makefile

#CXX = g++-mp-4.9# for Mac with macports and Xcode
#CXX = x86_64-w64-mingw32-g++
#CXXFLAGS = -Ofast
CXXFLAGS = -O3
#CXXFLAGS += -Wa,-q# for Mac (required: cd /opt/local/bin; ln -s /usr/bin/clang)
#CXXFLAGS += -std=c++11 -fpermissive
CXXFLAGS += -msse2 -mfpmath=sse#+387
CXXFLAGS += -march=native -mtune=native# -mno-fma
#CXXFLAGS += -march=core-avx-i -mtune=core-avx-i
CXXFLAGS += -fopenmp
CXXFLAGS += -ffast-math
CXXFLAGS += -funroll-all-loops
CXXFLAGS += -fpeel-loops
#CXXFLAGS += -fomit-frame-pointer
#CXXFLAGS += -floop-flatten
#CXXFLAGS += -funswitch-loops
#CXXFLAGS += -fforce-addr
#CXXFLAGS += -ftree-vectorize
#CXXFLAGS += -floop-optimize
#CXXFLAGS += -floop-nest-optimize
#CXXFLAGS += -floop-block
#CXXFLAGS += -floop-interchange
#CXXFLAGS += -floop-parallelize-all
#CXXFLAGS += -funsafe-loop-optimizations
#CXXFLAGS += -fargument-noalias
#CXXFLAGS += -fargument-noalias-global

ifeq ($(CXX),icpc)
CXX = icpc
CXXFLAGS = -openmp -O3 -march=native -xHost
endif

CXXFLAGS += -I ../src

FFTW_THREADS := yes
ifeq ($(FFTW_THREADS),yes)
  LDLIBS := -lfftw3_threads
  CXXFLAGS += -D USE_FFTW_THREADS
endif

bench_LDFLAGS := -L $(HOME)/local/lib

#bench_CXXFLAGS = -I/alt/include
#LDLIBS += -L/alt/lib -lfftw3_threads -lfftw3 -lpthread
#LDLIBS += -L/alt/lib -lfftw3_omp -lfftw3 -lpthread
#bench_CXXFLAGS = -I/alt-mingw/include
#LDLIBS += -L/alt-mingw/lib -lfftw3_threads -lfftw3 -lpthread
#LDLIBS += -lfftw3_threads -lfftw3 -lpthread
LDLIBS += -lfftw3 -lpthread


HEADERS1 = cpp_fftw3.h ooura1.h simple_fft.h
HEADERS2 = cpp_fftw3.h ooura2.h simple_fft.h


#LDFLAGS:=otfft/otfft.o
#LDFLAGS:=-L out -lotfft -Wl,-R,'$ORIGIN/out'
LDFLAGS:=-L ../out -lotfft -Wl,-R,$(PWD)/../out

all:
.PHONY: clean  benchmark check

clean:
	-rm -f fftcheck rfftcheck dctcheck bstcheck fftbench1 fftbench2 *.exe

check: fftcheck rfftcheck dctcheck bstcheck
fftcheck.exe: fftcheck.cpp simple_fft.h
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@
rfftcheck.exe: rfftcheck.cpp simple_fft.h
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@
dctcheck.exe: dctcheck.cpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@
bstcheck.exe: bstcheck.cpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@
%check: %check.exe
	./$< && touch $@

benchmark: fftbench1 fftbench2

fft.tgz:
	wget 'http://www.kurims.kyoto-u.ac.jp/~ooura/fft.tgz'
fftsg.c: fft.tgz
	tar xOf fft.tgz fft/$@ > $@
fftbench1.exe: fftbench1.cpp fftsg.c ../src/otfft/stopwatch.h $(HEADERS1)
	$(CXX) $(CXXFLAGS) $(bench_CXXFLAGS) -I ext -I . $< $(LDFLAGS) $(bench_LDFLAGS) $(LDLIBS) -o $@
fftbench2.exe: fftbench2.cpp fftsg.c ../src/otfft/stopwatch.h $(HEADERS2)
	$(CXX) $(CXXFLAGS) $(bench_CXXFLAGS) -I ext -I . $< $(LDFLAGS) $(bench_LDFLAGS) $(LDLIBS) -o $@
fftbench%: fftbench%.exe
	./$< | tee $@.txt && touch $@
