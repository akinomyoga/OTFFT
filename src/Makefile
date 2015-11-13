# -*- makefile-gmake -*-
# otfft/src/Makefile

# configurations

#USE_OPENMP := yes
USE_OPENMP := no

VERSION:=5.4
VERSION_MAJOR:=5
#------------------------------------------------------------------------------

#CXX = icpc
#CXX = g++-mp-5# for Mac with MacPorts and Xcode
#CXX = g++-5
#CXX = clang++-mp-3.7                                                                                                                                                                                           
#CXX = x86_64-w64-mingw32-g++

CXXFLAGS:=

ifeq ($(CXX),icpc)
  CXX = icpc
  CXXFLAGS = -O3 -march=native -xHost
else
  #CXXFLAGS += -Ofast
  CXXFLAGS += -O3
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

ffttune_LDLIBS:=

ifeq ($(USE_OPENMP),yes)
  ifeq ($(CXX),icpc)
    CXXFLAGS += -openmp
  else
    CXXFLAGS += -fopenmp
  endif
  ffttune_LDLIBS = -lgomp
  #ffttune_LDLIBS = -L/alt/lib -liomp5
  #ffttune_LDLIBS = -L$(HOME)/lib -liomp5
  #ffttune_LDLIBS = -L/opt/local/lib/libomp -lomp
endif

HEADERS1 = \
  otfft/otfft_misc.h \
  otfft_avxdif4.h \
  otfft_avxdif4omp.h \
  otfft_avxdif8.h \
  otfft_avxdif8omp.h \
  otfft_avxdifx.h \
  otfft_avxdit4.h \
  otfft_avxdit4omp.h \
  otfft_avxdit8.h \
  otfft_avxdit8omp.h \
  otfft_sixstep.h \
  otfft_sixstep0r.h \
  otfft_sixstepnr.h \
  otfft_sixstep0s.h \
  otfft_sixstepns.h \
  otfft_eightstep.h

OUTDIR:=../out
OUTINC:=$(OUTDIR)/include
OUTLIB:=$(OUTDIR)/lib
ifeq (x$(INSDIR),x)
INSDIR:=$(OUTDIR)/bin
endif

HEADERS2 = \
  $(OUTINC)/otfft_setup.h \
  $(OUTINC)/otfft_fwd.h \
  $(OUTINC)/otfft_fwd0.h \
  $(OUTINC)/otfft_inv.h \
  $(OUTINC)/otfft_invn.h

all:
.PHONY: all clean install

clean:
	-rm -rf $(OUTDIR)/*

$(OUTDIR) $(OUTINC) $(OUTLIB) $(INSDIR) $(INSDIR)/include/otfft $(INSDIR)/lib:
	mkdir -p $@

$(OUTDIR)/ffttune.o: ffttune.cpp $(HEADERS1) otfft/stopwatch.h | $(OUTDIR)
	@echo NOTE: This compilation may take several minutes.
	$(CXX) -c $(CXXFLAGS) $< -o $@
$(OUTDIR)/ffttune: $(OUTDIR)/ffttune.o
	$(CXX) $^ $(ffttune_LDLIBS) -o $@
$(OUTDIR)/ffttune.log: $(OUTDIR)/ffttune | $(OUTINC)
	cd $(OUTINC) && ../ffttune | tee ../ffttune.log
$(HEADERS2): $(OUTDIR)/ffttune.log

$(OUTDIR)/otfft.o: otfft.cpp $(HEADERS1) otfft/otfft.h $(HEADERS2)
	@echo NOTE: This compilation may take several minutes.
	$(CXX) $(CXXFLAGS)       -I $(OUTINC) -c -o $@ $<
$(OUTDIR)/libotfft.a: $(OUTDIR)/otfft.o
	ar cru $@ $<

libotfft_FullName:=libotfft.so.$(VERSION)
libotfft_MajorName:=libotfft.so.$(VERSION_MAJOR)
libotfft_shared_LDFLAGS:=-shared -Wl,-soname=$(libotfft_MajorName)

libotfft_object+=$(OUTDIR)/otfft_shared.o
$(OUTDIR)/otfft_shared.o: otfft.cpp $(HEADERS1) otfft/otfft.h $(HEADERS2)
	@echo NOTE: This compilation may take several minutes.
	$(CXX) $(CXXFLAGS) -fPIC -I $(OUTINC) -c -o $@ $<

libotfft_object+=$(OUTDIR)/util.o
-include $(OUTDIR)/util.d
$(OUTDIR)/util.o: util.cpp otfft/otfft.h
	$(CXX) $(CXXFLAGS) -fPIC -I $(OUTINC) -I . -c -o $@ -MD -MF $(OUTDIR)/util.d $<

$(OUTDIR)/$(libotfft_FullName): $(libotfft_object) | $(OUTDIR)
	$(CXX) $(libotfft_shared_LDFLAGS) -o $@ $^
$(OUTDIR)/libotfft.so $(OUTDIR)/$(libotfft_MajorName): $(OUTDIR)/$(libotfft_FullName)
	ln -sf $(libotfft_FullName) $@
all: $(OUTDIR)/$(libotfft_FullName) $(OUTDIR)/$(libotfft_MajorName) $(OUTDIR)/libotfft.so

# install headers
INSINC:=$(INSDIR)/include/otfft
install: $(INSINC)/otfft.h $(INSINC)/otfft_misc.h $(INSINC)/util.h
$(INSINC)/%.h: otfft/%.h | $(INSINC)
	cp $< $@
$(INSINC)/util.h: util.h | $(INSINC)
	cp $< $@

# install library files
install:  $(INSDIR)/lib/$(libotfft_FullName) $(INSDIR)/lib/$(libotfft_MajorName) $(INSDIR)/lib/libotfft.so
$(INSDIR)/lib/$(libotfft_FullName): $(OUTDIR)/$(libotfft_FullName) | $(INSDIR)/lib
	cp $< $@
$(INSDIR)/lib/$(libotfft_MajorName) $(INSDIR)/lib/libotfft.so: $(INSDIR)/lib/$(libotfft_FullName)
	ln -sf $(libotfft_FullName) $@

# otfft.o: otfft.cpp otfft.h $(HEADERS1) $(HEADERS2)
# $(libotfft_FullName): otfft_shared.o
# 	$(CXX) $(CXXFLAGS) -shared -Wl,-soname=$(libotfft_MajorName) -o $@ $^
# otfft_shared.o: otfft.cpp otfft.h $(HEADERS1) $(HEADERS2)
# 	$(CXX) $(CXXFLAGS) -fPIC -c -o $@ $<


# check
.PHONY: utilcheck

utilcheck: $(OUTDIR)/utilcheck.exe
	$<
$(OUTDIR)/utilcheck.exe: $(OUTDIR)/utilcheck.o | $(OUTDIR)/libotfft.so
	g++ -o $@ $^ -L $(OUTDIR) -Wl,-R,$(OUTDIR) -lotfft
-include $(OUTDIR)/utilcheck.d
$(OUTDIR)/utilcheck.o: utilcheck.cpp util.h
	g++ $(CXXFLAGS) -c -o $@ -MD -MF $(OUTDIR)/utilcheck.d -I $(OUTINC) -I . $<
