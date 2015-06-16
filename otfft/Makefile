#CXX = g++-mp-4.9# for Mac with macports and Xcode
#CXX = x86_64-w64-mingw32-g++

CXXFLAGS:=
#CXXFLAGS += -Ofast
CXXFLAGS += -O3
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

HEADERS1 = \
	otfft_misc.h \
	otfft_difavx.h \
	otfft_difavxn.h \
	otfft_difavx8.h \
	otfft_difavx8n.h \
	otfft_ditavx.h \
	otfft_ditavxn.h \
	otfft_ditavx8.h \
	otfft_ditavx8n.h \
	otfft_sixstep.h \
	otfft_sixstepn.h

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

$(OUTDIR)/ffttune: ffttune.cpp $(HEADERS1) stopwatch.h | $(OUTDIR)
	@echo NOTE: This compilation may take several minutes.
	$(CXX) $(CXXFLAGS) $< -o $@
$(OUTDIR)/ffttune.log: $(OUTDIR)/ffttune | $(OUTINC)
	cd $(OUTINC) && ../ffttune | tee ../ffttune.log
$(HEADERS2): $(OUTDIR)/ffttune.log

$(OUTDIR)/otfft.o: otfft.cpp $(HEADERS1) otfft.h $(HEADERS2)
	@echo NOTE: This compilation may take several minutes.
	$(CXX) $(CXXFLAGS)       -I $(OUTINC) -c -o $@ $<
$(OUTDIR)/libotfft.a: $(OUTDIR)/otfft.o
	ar cru $@ $<

libotfft_shared_LDFLAGS:=-shared -Wl,-soname=libotfft.so.4
$(OUTDIR)/otfft_shared.o: otfft.cpp $(HEADERS1) otfft.h $(HEADERS2)
	@echo NOTE: This compilation may take several minutes.
	$(CXX) $(CXXFLAGS) -fPIC -I $(OUTINC) -c -o $@ $<
$(OUTDIR)/libotfft.so.4.0: $(OUTDIR)/otfft_shared.o | $(OUTDIR)
	$(CXX) $(libotfft_shared_LDFLAGS) -o $@ $<
$(OUTDIR)/libotfft.so $(OUTDIR)/libotfft.so.4: $(OUTDIR)/libotfft.so.4.0
	ln -s libotfft.so.4.0 $@
all: $(OUTDIR)/libotfft.so.4.0 $(OUTDIR)/libotfft.so.4 $(OUTDIR)/libotfft.so

# install headers
install: $(INSDIR)/include/otfft/otfft.h $(INSDIR)/include/otfft/otfft_misc.h
$(INSDIR)/include/otfft/%.h: %.h | $(INSDIR)/include/otfft
	cp $< $@

# install library files
install:  $(INSDIR)/lib/libotfft.so.4.0 $(INSDIR)/lib/libotfft.so.4 $(INSDIR)/lib/libotfft.so
$(INSDIR)/lib/libotfft.so.4.0: $(OUTDIR)/libotfft.so.4.0 | $(INSDIR)/lib
	cp $< $@
$(INSDIR)/lib/libotfft.so.4 $(INSDIR)/lib/libotfft.so: $(INSDIR)/lib/libotfft.so.4.0
	ln -s libotfft.so.4.0 $@

# otfft.o: otfft.cpp otfft.h $(HEADERS1) $(HEADERS2)
# libotfft.so.4.0: otfft_shared.o
# 	$(CXX) $(CXXFLAGS) -shared -Wl,-soname=libotfft.so.4 -o $@ $^
# otfft_shared.o: otfft.cpp otfft.h $(HEADERS1) $(HEADERS2)
# 	$(CXX) $(CXXFLAGS) -fPIC -c -o $@ $<

