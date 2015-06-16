#!/bin/sh -

TUNEDIR=../out/include

for f in otfft_setup.h otfft_fwd.h otfft_fwd0.h otfft_inv.h otfft_invn.h
do
  sed "s/fft./fft$1/" "$TUNEDIR/$f" > "$TUNEDIR/$f.part" \
    && mv -f "$TUNEDIR/$f.part" "$TUNEDIR/$f"
done
