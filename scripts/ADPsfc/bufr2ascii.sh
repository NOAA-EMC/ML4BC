#!/bin/bash
###############################################################
#Purpose: This script is for converting ADPsfc bufr file to ASCII file.
#8/23/2024, Linlin Cui (linlin.cui@noaa.gov)
###############################################################

for f in `ls bufr/gdas.adpsfc.t*`;do
    prefix=`echo $f | cut -c6-30`
    echo $prefix

    ./bufrsurface.x $f ascii/$prefix.txt bufrsurface.config
done
