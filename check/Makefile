# -*- makefile-gmake -*-
# otfft/check/Makefile

# configurations

FFTW_THREADS := yes

#------------------------------------------------------------------------------

#CXX = g++-mp-5# for Mac with MacPorts and Xcode
#CXX = g++-5
#CXX = clang++-mp-3.7
#CXX = x86_64-w64-mingw32-g++

#CXXFLAGS = -Ofast
CXXFLAGS = -O3
#CXXFLAGS += -Wa,-q# for Mac (required: cd /opt/local/bin; ln -s /usr/bin/clang)
#CXXFLAGS += -std=c++11
CXXFLAGS += -msse2 -mfpmath=sse#+387
CXXFLAGS += -march=native -mtune=native# -mno-fma
#CXXFLAGS += -march=core-avx-i -mtune=core-avx-i
CXXFLAGS += -fopenmp
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

ifeq ($(CXX),icpc)
CXX = icpc
CXXFLAGS = -openmp -O3 -march=native -xHost
endif

CXXFLAGS += -I ../src

bench_LDFLAGS := -L $(HOME)/local/lib

#CXXFLAGS += -I/alt/include
#CXXFLAGS += -I/opt/local/include
#CXXFLAGS += -I/opt/local/include/libomp
#CXXFLAGS += -I/alt-mingw/include

# Enable library: gomp
OMPLIB = -lgomp
#OMPLIB = -L/alt/lib -liomp5
#OMPLIB = -L$(HOME)/lib -liomp5
#OMPLIB = -L/opt/local/lib/libomp -lomp

LDLIBS = $(OMPLIB)

# Enable library: fftw3_threads
ifeq ($(FFTW_THREADS),yes)
  bench_LIBS := -lfftw3_threads
  bench_CXXFLAGS += -D USE_FFTW_THREADS
endif

bench_LIBS += -lfftw3 -lpthread

#LDLIBS = -L/alt/lib -lfftw3_threads -lfftw3 -lpthread $(OMPLIB)
#LDLIBS = -L/alt/lib -lfftw3_omp -lfftw3 -lpthread $(OMPLIB)
#LDLIBS = -L/opt/local/lib -lfftw3_threads -lfftw3 -lpthread $(OMPLIB)
#LDLIBS = -L/alt-mingw/lib -lfftw3_threads -lfftw3 -lpthread -lgomp   

HEADERS1 = cpp_fftw3.h ooura1.h simple_fft.h
HEADERS2 = cpp_fftw3.h ooura2.h simple_fft.h

#LDFLAGS:=otfft/otfft.o
#LDFLAGS:=-L out -lotfft -Wl,-R,'$ORIGIN/out'
LDFLAGS:=-L ../out -lotfft -Wl,-R,$(PWD)/../out

all:
.PHONY: clean  benchmark check

clean:
	-rm -f fftcheck rfftcheck dctcheck bstcheck fftbench{1,2} *.exe *.o

#------------------------------------------------------------------------------
# check

check: fftcheck rfftcheck dctcheck bstcheck

fftcheck.o: fftcheck.cpp simple_fft.h
rfftcheck.o: rfftcheck.cpp simple_fft.h
dctcheck.o: dctcheck.cpp
bstcheck.o: bstcheck.cpp
fftcheck.o rfftcheck.o dctcheck.o bstcheck.o:
	$(CXX) -c $(CXXFLAGS) $<

fftcheck.exe: fftcheck.o
rfftcheck.exe: rfftcheck.o
dctcheck.exe: dctcheck.o
bstcheck.exe: bstcheck.o
fftcheck.exe rfftcheck.exe dctcheck.exe bstcheck.exe:
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

%check: %check.exe
	./$< && touch $@

#------------------------------------------------------------------------------
# benchmark

benchmark: fftbench1 fftbench2

fft.tgz:
	wget 'http://www.kurims.kyoto-u.ac.jp/~ooura/fft.tgz'
fftsg.c: fft.tgz
	tar xOf fft.tgz fft/$@ > $@
fftbench1.exe: fftbench1.cpp fftsg.c ../src/otfft/stopwatch.h $(HEADERS1)
	$(CXX) $(CXXFLAGS) $(bench_CXXFLAGS) -I ext -I . $< $(LDFLAGS) $(bench_LDFLAGS) $(LDLIBS) $(bench_LIBS) -o $@
fftbench2.exe: fftbench2.cpp fftsg.c ../src/otfft/stopwatch.h $(HEADERS2)
	$(CXX) $(CXXFLAGS) $(bench_CXXFLAGS) -I ext -I . $< $(LDFLAGS) $(bench_LDFLAGS) $(LDLIBS) $(bench_LIBS) -o $@
fftbench%: fftbench%.exe
	./$< | tee $@.txt && touch $@
