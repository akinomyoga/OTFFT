#!/bin/sh -

for f in otfft_setup.h otfft_fwd.h otfft_fwd0.h otfft_inv.h otfft_invn.h
do
    mv otfft/$f $f.tmp
    sed "s/fft./fft$1/" $f.tmp > otfft/$f
    rm $f.tmp
done
