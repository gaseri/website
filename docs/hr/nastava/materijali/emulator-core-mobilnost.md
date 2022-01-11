---
author: Vedran Miletić
---

# Mobilnost

Emulator CORE nam omogućuje da za (bežične) čvorove koristimo [skripte mobilnosti mrežnog simulatora ns-2](https://www.isi.edu/nsnam/ns/doc/node172.html), [CORE API](https://coreemu.github.io/core/python.html) (specijalno se poruke mogu slati i alatom `coresendmsg`) ili [EMANE događaje](https://coreemu.github.io/core/emane.html) ([dokumentacija](https://coreemu.github.io/core/gui.html#mobility-scripting)). Koristit ćemo ns-2-ove skripte mobilnosti.

Za postavljanje tih skripti potrebno je u izborniku koji dobijemo desnim klik na bežičnu mrežu odabrati opciju `Configure`. Gumb `ns-2 mobility script` pa se na kartici `ns-2 Mobility Script Parameters` može navesti datoteka pod `mobility script file`).

!!! danger
    Pripazimo da se skripta koju učitavamo nalazi u putanji bez razmaka i posebnih znakova te da ima ime bez razmaka i posebnih znakova. Specijalno, putanja i ime datoteke `/home/korisnik/Radna površina/kretanje čvorova.tcl` ne zadovoljavaju te uvjete.

Skripte mobilnosti pisane su u [jeziku Tcl](https://www.tcl-lang.org/) i sadrže naredbe pozicija i kretanja oblika

``` tcl
$node_(2) set X_ 200.0
$node_(2) set Y_ 100.0
$ns_ at 1.0 "$node_(2) setdest 300.0 150.0 20.0"
$ns_ at 9.0 "$node_(2) setdest 500.0 240.0 35.0"
```

U ovom slučaju čvor n2 postavljen je inicijalno na poziciju (200, 100), a sekundu kasnije pomiče se na poziciju (300, 150) brzinom 20 piksela po sekundi. Osam sekundi nakon toga, odnosu u trenutku t = 9 sekundi, čvor se pomiče na poziciju (500, 240) brzinom 35 piksela po sekundi.

Obzirom da CORE skripte ponavlja za vrijeme izvođenja emulacije, dobra je praksa postaviti da se čvorovi vrate na početnu poziciju kako kod ponavljanja kretanja ne bi "skakali" po platnu. Primjerice, ovdje ćemo postaviti da se iz posljednje pozicije čvor vrati na početnu u trenutku t = 20 sekundi na način

``` tcl
$node_(2) set X_ 200.0
$node_(2) set Y_ 100.0
$ns_ at 1.0 "$node_(2) setdest 300.0 150.0 20.0"
$ns_ at 9.0 "$node_(2) setdest 500.0 240.0 35.0"
$ns_ at 20.0 "$node_(2) setdest 200.0 100.0 30.0"
```

Analogno navedenom primjeru možemo definirati mobilnost u obliku trokuta, kvadrata, kao i proizvoljnog poligona koji aproksimira kružno kretanje.

U slučaju kada definiramo mobilnost dva ili više čvorova, njihove naredbe pozicija i kretaja se mogu dati čvor po čvor pa je skripta oblika

``` tcl
$node_(2) set X_ 200.0
$node_(2) set Y_ 100.0
$ns_ at 1.0 "$node_(2) setdest 300.0 150.0 20.0"
$ns_ at 9.0 "$node_(2) setdest 500.0 240.0 35.0"
$ns_ at 20.0 "$node_(2) setdest 200.0 100.0 30.0"

$node_(3) set X_ 550.0
$node_(3) set Y_ 575.0
$ns_ at 2.0 "$node_(3) setdest 400.0 350.0 20.0"
$ns_ at 12.0 "$node_(3) setdest 550.0 575.0 25.0"
```

ili se mogu isprepletati pa je skripta oblika

``` tcl
$node_(2) set X_ 200.0
$node_(2) set Y_ 100.0
$node_(3) set X_ 550.0
$node_(3) set Y_ 575.0
$ns_ at 1.0 "$node_(2) setdest 300.0 150.0 20.0"
$ns_ at 2.0 "$node_(3) setdest 400.0 350.0 20.0"
$ns_ at 9.0 "$node_(2) setdest 500.0 240.0 35.0"
$ns_ at 12.0 "$node_(3) setdest 550.0 575.0 25.0"
$ns_ at 20.0 "$node_(2) setdest 200.0 100.0 30.0"
```

Ukoliko nam platno nije dovoljno veliko za naše potrebe, možemo ga povećati putem opcije izbornika `Canvas/Size/scale...` ([dokumentacija](https://coreemu.github.io/core/gui.html#canvas-menu)).

!!! note
    Postoje i drugi načini skriptiranja mobilnosti koje nećemo koristiti, a više detalja o njima moguće je pronaći [u dijelu o skriptiranju mobilnosti u službenoj dokumentacijiemulatora CORE](https://coreemu.github.io/core/gui.html#mobility-scripting).
