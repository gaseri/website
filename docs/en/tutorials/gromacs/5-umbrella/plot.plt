
set term tikz standalone color

set style line 11 lc rgb '#808080' lt 1
set border 3 back ls 11

set style line 12 lc rgb '#808080' lt 0 lw 1
set grid back ls 12
set pointsize 1.0

set xrange[0:1.4]

unset key

set xlabel "r (nm)"
set ylabel "w(r) (kJ / mol)"
set output "profile1.tex"
plot 'profile.xvg' w l

set output "profile2.tex"
plot 'profile.xvg' u 1:($2+2*8.314e-3*298.15*log($1)+77) w l

set key top right
set output "profile3.tex"
plot 'profile.xvg' u 1:($2+2*8.314e-3*298.15*log($1)+77) w l t 'umbrella', \
     '../3_methanes_in_water/rdf.xvg' u 1:(-8.314e-3*298.15*log($2*10/9)) w l t 'direct'

unset key
set output "histo.tex"
set xlabel "x"
set ylabel "count"
plot "histo.xvg" w l, \
	"histo.xvg" u 1:3 w l, \
	"histo.xvg" u 1:4 w l, \
	"histo.xvg" u 1:5 w l, \
	"histo.xvg" u 1:6 w l, \
	"histo.xvg" u 1:7 w l, \
	"histo.xvg" u 1:8 w l, \
	"histo.xvg" u 1:9 w l, \
	"histo.xvg" u 1:10 w l, \
	"histo.xvg" u 1:11 w l, \
	"histo.xvg" u 1:12 w l, \
	"histo.xvg" u 1:13 w l, \
	"histo.xvg" u 1:14 w l, \
	"histo.xvg" u 1:15 w l, \
	"histo.xvg" u 1:16 w l, \
	"histo.xvg" u 1:17 w l, \
	"histo.xvg" u 1:18 w l, \
	"histo.xvg" u 1:19 w l, \
	"histo.xvg" u 1:20 w l, \
	"histo.xvg" u 1:21 w l, \
	"histo.xvg" u 1:22 w l, \
	"histo.xvg" u 1:23 w l, \
	"histo.xvg" u 1:24 w l, \
	"histo.xvg" u 1:25 w l, \
	"histo.xvg" u 1:26 w l, \
	"histo.xvg" u 1:27 w l, \
	"histo.xvg" u 1:28 w l
