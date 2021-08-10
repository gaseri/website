
set term tikz standalone color

set style line 11 lc rgb '#808080' lt 1
set border 3 back ls 11

set style line 12 lc rgb '#808080' lt 0 lw 1
set grid back ls 12
set pointsize 1.0

unset key

set ylabel "Potential Energy (kJ/mol)"
set xlabel "Step"
set output "min-pot.tex"
plot 'min-pot.xvg' u 1:2 w l

set ylabel "Potential Energy (kJ/mol)"
set xlabel "Step"
set output "min2-pot.tex"
plot 'min2-pot.xvg' u 1:2 w l

set ylabel "Temperature (K)"
set xlabel "Time (ps)"
set output "eql-tmp.tex"
plot 'eql-tmp.xvg' u 1:2 w l

set ylabel "Pressure (bar)"
set xlabel "Time (ps)"
set output "eql2-press.tex"
plot 'eql2-press.xvg' u 1:2 w l
