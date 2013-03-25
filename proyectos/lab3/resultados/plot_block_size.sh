#!/bin/sh
#
# Inspired on http://gnuplot.sourceforge.net/demo/heatmaps.html
# colormaps: http://t16web.lanl.gov/Kawano/gnuplot/plotpm3d2-e.html
#
gnuplot << EOF
set terminal postscript eps color enhanced
set output "$1.eps"
unset key
set view map
set xlabel "Block size X"
set ylabel "Block size Y"
set title "Kernel gputime vs. block size -- $1"
set xtics border in scale 0,0
set ytics border in scale 0,0
set ztics border in scale 0,0
set cblabel "gputime"
set xrange [ 0.5 : 32.5 ]
set yrange [ 0.5 : 32.5 ]
set palette rgbformulae -7, -5, -15
plot "$1" using 1:2:3 with image
EOF
