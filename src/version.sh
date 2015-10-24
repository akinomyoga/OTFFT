#!/bin/sh -

for f in otfft*.h otfft.cpp ffttune.cpp
do
    mv $f $f.tmp && \
        sed "2s/Version [1-9][0-9]*[.0-9]*[a-z]*/Version $1/" $f.tmp > $f && \
        rm $f.tmp
done
