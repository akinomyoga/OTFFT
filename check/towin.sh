#!/bin/sh -

for f in *.cpp *.h
do
    mv $f $f.bak
    nkf --windows $f.bak > $f && rm $f.bak
done
