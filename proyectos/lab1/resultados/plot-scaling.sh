#!/bin/sh
#
# Inspired on # http://t16web.lanl.gov/Kawano/gnuplot/intro/working-e.html
# Added for loops # http://gnuplot-tricks.blogspot.com.ar/2010/01/plot-iterations-and-pseudo-files.html
#
gnuplot << EOF
set terminal postscript eps color enhanced
set output "serial-scaling.eps"
set xlabel "N"
set ylabel "scaling [ns per cell]"
set title "scaling vs N vs version"
set logscale x 2
set key left top
plot for [i=2:6] "$1" using 1:i with linespoints title columnhead(i) linewidth 4
EOF
