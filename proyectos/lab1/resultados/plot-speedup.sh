#!/bin/sh
#
# Inspired on # http://t16web.lanl.gov/Kawano/gnuplot/intro/working-e.html
# Added for loops # http://gnuplot-tricks.blogspot.com.ar/2010/01/plot-iterations-and-pseudo-files.html
#
gnuplot << EOF
set terminal postscript eps color enhanced
set output "serial-speedup.eps"
set xlabel "N"
set ylabel "speedup"
set title "speedup(contrastada con original) vs N vs version"
set key left top
plot for [i=1:3] "$1" using 1:(column(2*i+1)/column(2*i)) with linespoints title columnhead(2*i) linewidth 4, \
     "$1" using 1:(column(5)/column(8)) with linespoints title columnhead(8) linewidth 4, \
     "$1" using 1:(column(7)/column(9)) with linespoints title columnhead(9) linewidth 4
EOF
