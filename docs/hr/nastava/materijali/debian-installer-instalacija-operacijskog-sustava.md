---
author: Vedran Miletić
---

# Instalacija operacijskog sustava Debian korištenjem instalacijskog alata Debian-Installer

[Debian-Installer](https://www.debian.org/devel/debian-installer/), kao što samo ime sugerira, je instalacijski program koji [operacijski sustav Debian](https://www.debian.org/) koristi za instalaciju na računalo.

Kod instalacije Debiana prvo se pokreće boot učitavač [ISOLINUX](https://wiki.syslinux.org/wiki/index.php?title=ISOLINUX) koji nudi korisniku nekoliko mogućnosti:

1. `Install` -- pokretanje instalacije u tekstualnom sučelju,
1. `Graphical install` -- slično prvoj opciji, pokretanje instalacije u grafičkom sučelju,
1. `Advanced options` -- dodatne opcije kao što su ekspertni način rada, oporavak sustava i automatizirana instalacija,
1. `Help` -- pomoć, otvara se novi ekran s uputama,
1. `Install with speech synthesis` -- instalacija uz govor za pomoć slabovidnim i slijepim osobama.

Biramo `Graphical install` i pokreće se Linux jezgra i minimalan sustav, a u njemu već spomenuti instalacijski program Debian-Installer, koji ima sučelje pisano u [GTK-u](https://www.gtk.org/). Ukoliko to ne radi iz bilo kojeg razloga, može se koristiti i opcija `Install` koja pokreće Debian-Installer sa sučeljem pisanim u [ncurses-u](https://www.gnu.org/software/ncurses/ncurses.html).

Instalacija redom sadrži iduće dijaloške okvire:

- izbor jezika,
- izbor geografske lokacije,
- izbor rasporeda tipkovnice,
- imenovanje domaćina,
- konfiguracija mreže,
- konfiguracija vremenske zone,
- particioniranje: omogućuje se brisanje i korištenje cijelog diska ili ručno stvaranje particija (moramo biti pažljivi u slučaju da na računalu već imamo prethodno instalirane Windowse ili Linux i želimo ih zadržati),
- instalacija osnovnog sustava,
- dodavanje prvog korisnika,
- konfiguracija proxy poslužitelja za preuzimanje paketa,
- izbor i instalacija softvera: obzirom da se koristi uobičajena instalacija (a ne živa), imamo mogućnost izbora paketa po skupinama, što radimo programom `tasksel`, a akon završetka odabira pokreće se instalacija odabranih paketa,
- konfiguracija UTC-a,
- ponovno pokretanje.

Prosječna instalacija Linuxa na desktopu imat će od par stotina do dvije tisuće paketa. Radne stanice za razvoj softvera, obradu videa, 3D grafiku, digitalno izdavaštvo i slično će imati više ovisno o potrebama korisnika. S druge strane, poslužitelji uglavnom nemaju grafička sučelja pa je broj paketa koji se instaliraju manji, no to uvelike ovisi o funkcijama koje poslužitelj obavlja.

Nakon ponovnog pokretanja računala, i izbacivanja instalacijskog medija, pokrenut će se GRUB na disku računala i ponuditi opciju pokretanja Debiana, i eventualno memtest86+-a i Windowsa.

!!! note
    Vrlo popularna distribucija Linuxa [Ubuntu](https://ubuntu.com/) je derivat Debiana, ali ne koristi isti softver za instalaciju operacijskog sustava. Više o instalaciji Ubuntua možete pronaći u službenim tutorialima za [Desktop](https://ubuntu.com/tutorials/install-ubuntu-desktop) (omogućuje i [isprobavanje prije instalacije](https://ubuntu.com/tutorials/try-ubuntu-before-you-install)) i [Server](https://ubuntu.com/tutorials/install-ubuntu-server).
