# -*- makefile-gmake -*-
# otfft/Makefile

all:
.PHONY: all clean install benchmark check

clean:
	+make -C src clean
	+make -C check clean

all install:
	+make -C src $@

check benchmark: | all
	+make -C check $@
