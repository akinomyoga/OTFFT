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

#CXXFLAGS2 = $(CXXFLAGS) -I/alt/include
#LDLIBS = -L/alt/lib -lfftw3_threads -lfftw3 -lpthread
#LDLIBS = -L/alt/lib -lfftw3_omp -lfftw3 -lpthread
#CXXFLAGS2 = $(CXXFLAGS) -I/alt-mingw/include
#LDLIBS = -L/alt-mingw/lib -lfftw3_threads -lfftw3 -lpthread
LDLIBS = -lfftw3_threads -lfftw3 -lpthread

HEADERS1 = ext/cpp_fftw3.h ext/ooura1.h simple_fft.h
HEADERS2 = ext/cpp_fftw3.h ext/ooura2.h simple_fft.h

all:
.PHONY: all clean install benchmark check

clean:
	-rm -f fftcheck rfftcheck dctcheck bstcheck fftbench1 fftbench2 *.exe

.PHONY: otfft_all otfft_install
otfft_all:
	make -C otfft all
otfft_install:
	make -C otfft install
all: otfft_all
install: otfft_install

check: fftcheck rfftcheck dctcheck bstcheck

benchmark: fftbench1 fftbench2

#LDFLAGS_libotfft:=otfft/otfft.o
#LDFLAGS_libotfft:=-L out -lotfft -Wl,-R,'$ORIGIN/out'
LDFLAGS_libotfft:=-L out -lotfft -Wl,-R,$(PWD)/out

fftcheck: fftcheck.cpp simple_fft.h | otfft_all
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS_libotfft) -o $@

rfftcheck: rfftcheck.cpp simple_fft.h | otfft_all
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS_libotfft) -o $@

dctcheck: dctcheck.cpp | otfft_all
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS_libotfft) -o $@

bstcheck: bstcheck.cpp | otfft_all
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS_libotfft) -o $@

fftbench1: fftbench1.cpp otfft/stopwatch.h $(HEADERS1) | otfft_all
	$(CXX) $(CXXFLAGS2) -I ext -I . $< $(LDFLAGS_libotfft) $(LDLIBS) -o $@

fftbench2: fftbench2.cpp otfft/stopwatch.h $(HEADERS2) | otfft_all
	$(CXX) $(CXXFLAGS2) -I ext -I . $< $(LDFLAGS_libotfft) $(LDLIBS) -o $@
