#!/bin/sh -

for f in otfft*.h otfft.cpp ffttune.cpp
do
    mv $f $f.tmp && \
        sed "2s/Version $1/Version $2/" $f.tmp > $f && \
        rm $f.tmp
done
