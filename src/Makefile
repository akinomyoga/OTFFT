# -*- makefile-gmake -*-
# otfft/src/Makefile

VERSION:=6.4
VERSION_MAJOR:=6
#------------------------------------------------------------------------------

include config.mk

ffttune_LIBS:=

ifeq ($(CXX),icpc)
  CXXFLAGS+=-std=c++11
else
  CXXFLAGS+=-std=gnu++0x
endif

ifeq ($(USE_OPENMP),yes)
  ifeq ($(CXX),icpc)
    CXXFLAGS += -openmp
  else
    CXXFLAGS += -fopenmp
  endif
  ffttune_LIBS = -lgomp
  #ffttune_LIBS = -L/alt/lib -liomp5
  #ffttune_LIBS = -L$(HOME)/lib -liomp5
  #ffttune_LIBS = -L/opt/local/lib/libomp -lomp
endif

HEADERS1 = \
  otfft/otfft_misc.h \
  otfft_avxdif4.h \
  otfft_avxdif4omp.h \
  otfft_avxdif8.h \
  otfft_avxdif8omp.h \
  otfft_avxdit4.h \
  otfft_avxdit4omp.h \
  otfft_avxdit8.h \
  otfft_avxdit8omp.h \
  otfft_avxdit16.h \
  otfft_avxdit16omp.h \
  otfft_avxdif16.h \
  otfft_avxdif16omp.h \
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
.PHONY: all clean install tune

clean:
	-rm -rf $(OUTDIR)/*

$(OUTDIR) $(OUTINC) $(OUTLIB) $(INSDIR) $(INSDIR)/include/otfft $(INSDIR)/lib:
	mkdir -p $@

$(OUTDIR)/ffttune.o: ffttune.cpp $(HEADERS1) otfft/stopwatch.h | $(OUTDIR)
	@echo NOTE: This compilation may take several minutes.
	$(CXX) -c $(CXXFLAGS) $< -o $@
$(OUTDIR)/ffttune: $(OUTDIR)/ffttune.o
	$(CXX) $^ $(LDFLAGS) $(ffttune_LIBS) -o $@
$(OUTDIR)/ffttune.log: $(OUTDIR)/ffttune | $(OUTINC)
	cd $(OUTINC) && ../ffttune | tee ../ffttune.log
tune $(HEADERS2):
	make $(OUTDIR)/ffttune.log


# static library

$(OUTDIR)/otfft.o: otfft.cpp $(HEADERS1) otfft/otfft.h $(HEADERS2)
	@echo NOTE: This compilation may take several minutes.
	$(CXX) $(CXXFLAGS) -I $(OUTINC) -c -o $@ $<
$(OUTDIR)/libotfft.a: $(OUTDIR)/otfft.o $(OUTDIR)/util.o
	ar rc $@ $<


# shared library

libotfft_FullName:=libotfft.so.$(VERSION)
libotfft_MajorName:=libotfft.so.$(VERSION_MAJOR)
libotfft_shared_LIBS:=-shared -Wl,-soname=$(libotfft_MajorName)

libotfft_object+=$(OUTDIR)/otfft_shared.o
$(OUTDIR)/otfft_shared.o: otfft.cpp $(HEADERS1) otfft/otfft.h $(HEADERS2)
	@echo NOTE: This compilation may take several minutes.
	$(CXX) $(CXXFLAGS) -fPIC -I $(OUTINC) -c -o $@ $<

libotfft_object+=$(OUTDIR)/util.o
-include $(OUTDIR)/util.d
$(OUTDIR)/util.o: util.cpp otfft/otfft.h
	$(CXX) $(CXXFLAGS) -fPIC -I $(OUTINC) -I . -c -o $@ -MD -MF $(OUTDIR)/util.d $<

$(OUTDIR)/$(libotfft_FullName): $(libotfft_object) | $(OUTDIR)
	$(CXX) $(LDFLAGS) $(libotfft_shared_LIBS) -o $@ $^
$(OUTDIR)/libotfft.so $(OUTDIR)/$(libotfft_MajorName): $(OUTDIR)/$(libotfft_FullName)
	ln -sf $(libotfft_FullName) $@
all: $(OUTDIR)/$(libotfft_FullName) $(OUTDIR)/$(libotfft_MajorName) $(OUTDIR)/libotfft.so


# install headers

INSINC:=$(INSDIR)/include/otfft
install: $(INSINC)/otfft.h $(INSINC)/otfft_misc.h
$(INSINC)/otfft%.h: otfft/otfft%.h | $(INSINC)
	cp $< $@
install: $(INSINC)/util.h
$(INSINC)/util.h: util.h | $(INSINC)
	cp $< $@
install: $(INSINC)/allocator.h
$(INSINC)/allocator.h: allocator.h | $(INSINC)
	cp $< $@


# install library files

install:  $(INSDIR)/lib/$(libotfft_FullName) $(INSDIR)/lib/$(libotfft_MajorName) $(INSDIR)/lib/libotfft.so
$(INSDIR)/lib/$(libotfft_FullName): $(OUTDIR)/$(libotfft_FullName) | $(INSDIR)/lib
	cp $< $@
$(INSDIR)/lib/$(libotfft_MajorName) $(INSDIR)/lib/libotfft.so: $(INSDIR)/lib/$(libotfft_FullName)
	ln -sf $(libotfft_FullName) $@


# check

.PHONY: utilcheck

utilcheck: $(OUTDIR)/utilcheck.exe
	$<
$(OUTDIR)/utilcheck.exe: $(OUTDIR)/utilcheck.o | $(OUTDIR)/libotfft.so
	$(CXX) -o $@ $^ -L $(OUTDIR) -Wl,-R,$(OUTDIR) -lotfft
-include $(OUTDIR)/utilcheck.d
$(OUTDIR)/utilcheck.o: utilcheck.cpp util.h
	$(CXX) $(CXXFLAGS) -c -o $@ -MD -MF $(OUTDIR)/utilcheck.d -I $(OUTINC) -I . $<
