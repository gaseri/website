
set term tikz standalone color

set style line 11 lc rgb '#808080' lt 1
set border 3 back ls 11

set style line 12 lc rgb '#808080' lt 0 lw 1
set grid back ls 12
set pointsize 1.0

set xlabel "r (nm)"
set ylabel "g(r)"
unset key

set output "rdf.tex"
plot 'rdf.xvg' u 1:3 w l
