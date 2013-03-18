#!/bin/sh
#
# Inspired on # http://t16web.lanl.gov/Kawano/gnuplot/intro/working-e.html
# Added for loops # http://gnuplot-tricks.blogspot.com.ar/2010/01/plot-iterations-and-pseudo-files.html
#
gnuplot << EOF
set terminal postscript eps color enhanced
set output "$1-speedup.eps"
set xlabel "N"
set ylabel "speedup"
set title "speedup vs N vs Procs -- $1"
set logscale x 2
set key left top
plot for [i=2:8] "$1" using 1:(column(2)/column(i)) with linespoints title columnhead(i) linewidth 4
EOF