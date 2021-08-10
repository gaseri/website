---
author: Vedran Miletić
---

# Rad s jezgrom operacijskog sustava

- prostor korisničkih aplikacija

    - aplikacije koje korisnici koriste: GNOME, Firefox, Thunderbird, LibreOffice, GIMP, Audacity, Inkscape, VLC, Python, GCC, `top`, `bash`, `mkfifo`, `chmod`, ...
    - bibliotečne datoteke i aplikacije koji nemaju direktnu primjenu za krajnjeg korisnika (programi za konfiguraciju sustava i slično): GNU C Library, Qt, CUPS, udev, X.Org, ALSA, `init`, `mkfs`, `fdisk`, ...

- jezgra (engl. *kernel*)

    - u našem slučaju Linux, ali može biti i FreeBSD, Solaris, Mac OS X, Windows, ...
    - upravlja procesima, memorijom, raspodjelom procesorskog vremena procesima, ulazno-izlaznim uređajima (hardverom)
    - jezgri nije zadaća da korisniku izravno pruži neku funkcionalnost, to rade aplikacije
    - jezgra barata s operacijama koje rade računanje (engl. *computation*) i ulaz-izlaz -- U/I (engl. *input/output*, *I/O*) i komunicira s uređajima koji rade jedno ili drugo
    - prostor jezgre je prostor u kojem se pokreću procesi jezgre

- `dmesg` ispisuje poruke koje jezgra javlja tijekom rada

!!! todo
    Ovdje treba opisati ukratko kako je strukturirana Linux jezgra.

## Dohvaćanje informacija o hardveru

- `dmidecode` ispisuje poruke [DMI i SMBIOS sučelja](https://en.wikipedia.org/wiki/SMBIOS), daje brojne informacije o hardveru
- naredba `lscpu` i datoteka `/proc/cpuinfo` daju informacije o procesorima
- naredba `free` i datoteka `/proc/meminfo` daju informacije o memoriji

!!! admonition "Zadatak"
    - Provjerite podržava li procesor na računalu proširenje fizičke adrese (engl. *Physical Address Extension*, PAE). Pročitajte na [Wikipedijinoj stranici o PAE](https://en.wikipedia.org/wiki/Physical_Address_Extension) čemu ta ekstenzija služi.
    - Pronađite način da alatom `free` ispišete veličinu slobodne i zauzete memorije u kilobajtima, a zatim pronađite te iste vrijednosti u `/proc/meminfo`.

- `lspci` ispisuje informacije o uređajima povezanim putem PCI, AGP ili PCI Express sučelja
- `lsusb` ispisuje informacije o uređajima povezanim putem USB sučelja
- `lspcmcia` ispisuje informacije o uređajima povezanim putem PCMCIA sučelja (postoji na starijim laptopima)
- `lshw` ispis informacija o svom hardveru u računalu (nije u standardnoj instalaciji većine distribucija)

!!! admonition "Zadatak"
    Saznajte više informacija o grafičkoj kartici (`VGA compatible controller`) na računalu:

    - `Device ID` i `Vendor ID`,
    - `Subsystem Device ID` i `Subsystem Vendor ID`,
    - upravljački program (engl. *driver*) jezgre koji koristi.

    **Uputa:** sjetite se preko kojeg sučelja se grafička kartica povezuje na matičnu ploču, a zatim pronađite u man stranici odgovarajućeg alata način da ispišete više informacija o hardveru.

## Moduli jezgre

!!! todo
    Ovdje treba objasniti način korištenja naredbi `modinfo`, `modprobe`, `insmod`, `rmmod`, `lsmod`.

## Kompajliranje jezgre

!!! todo
    Ovdje treba opisati `make menuconfig` i ostalo.
