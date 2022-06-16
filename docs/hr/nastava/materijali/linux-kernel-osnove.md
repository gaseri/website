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

## Parametri jezgre

Kao i programi u korisničkom prostoru, jezgra operacijskog sustava može primati [različite parametre jezgre](https://wiki.archlinux.org/title/Kernel_parameters) kod pokretanja. Parametre pokrenute jezgre operacijskog sustava možemo pronaći u datoteci `/proc/cmdline`:

``` shell
$ cat /proc/cmdline
BOOT_IMAGE=/boot/vmlinuz-linux root=UUID=950382f8-aeef-4f3d-baea-6030a38707f0 rw net.ifnames=0 console=tty1 console=ttyS0 rootflags=compress-force=zstd
```

Za isprobavanje parametara u fazi učenja uređivat ćemo izravno datoteku `/boot/grub/grub.cfg` i time prekršiti preporuku da se tu datoteku ne uređuje, što je u fazi učenja prihvatljivo.

Parametri:

- `1` (uočimo poruku)
- `3` (uočimo izlaz naredbe `systemctl get-default`)
- `ro` (možemo li stvoriti datoteke)
- `quiet`
- `module_blacklist=snd_hda_intel` (proučite koncept [blacklistanja modula](https://wiki.archlinux.org/title/Kernel_module#Blacklisting))
- `init=/bin/sh`
- `maxcpus`, `mem`, `video`

## Kompajliranje jezgre

!!! todo
    Ovdje treba opisati `make menuconfig` i ostalo.

## Specfičnosti jezgre Arch Linuxa

Arch Linux [službeno podržava četiri vrste jezgre Linuxa](https://wiki.archlinux.org/title/Kernel):

- stabilna, paket [linux](https://archlinux.org/packages/?name=linux)
- očvrsnuta, paket [linux-hardened](https://archlinux.org/packages/?name=linux-hardened)
- dugotrajna, paket [linux-lts](https://archlinux.org/packages/?name=linux-lts)
- Zen, paket [linux-zen](https://archlinux.org/packages/?name=linux-zen)

Zadana instalacija koristi stabilnu verziju.

### Nadogradnja verzije jezgre

Kod nadogradnje svih paketa naredbom `pacman -Syu`, moguće je da će među nadogradnjama biti i novija verzija paketa `linux`. Za razliku od svih ostalih paketa, nadogradnja jezgre zahtijeva ponovno pokretanje operacijskog sustava kako bi se pokrenula. Štoviše, sve do ponovnog pokretanja operacijskog sustava neće ispravno raditi naredbe koje koriste informacije o pokrenutoj verziji jezgre kod traženja modula jezgre (kao što je, primjerice, naredba `modprobe`).

### Promjena vrste jezgre koja se koristi

Za ilustraciju korištenja vrste jezgre koja nije zadana prijeći ćemo na dugotrajnu instalacijom paketa `linux-lts`:

``` shell
$ sudo pacman -S linux-lts
```

Nakon instalacije, ali i ponovnog pokretanja uvjerit ćemo se da se novoinstalirana varijanta jezgra ne koristi:

``` shell
$ uname -a
Linux ares.miletic.net 5.17.5-arch1-1 #1 SMP PREEMPT Wed, 27 Apr 2022 20:56:11 +0000 x86_64 GNU/Linux
```

Uvjerimo se da su slika jezgre i [početni datotečni sustav za RAM](https://wiki.archlinux.org/title/Arch_boot_process#initramfs) za pakete linux i linux-lts na mjestu:

``` shell
$ ls /boot
efi  grub  initramfs-linux-fallback.img  initramfs-linux-lts-fallback.img  initramfs-linux-lts.img  initramfs-linux.img  vmlinuz-linux  vmlinuz-linux-lts
```

## Pokretanje nove verzije jezgre korištenjem kexeca

Iako pokretanje nove verzije jezgre operacijskog sustava nominalno zahtijeva ponovno pokretanje računala, Linux podržava [sustavski poziv pokretanja jezgre](https://en.wikipedia.org/wiki/Kexec) (engl. *kernel execute*, kraće kexec) koji omogućuje pokretanje novije verzije jezgre iz trenutno pokrenute jezgre. Time se preskače inicijalizacija hardvera i pokretanje bootloadera.

Prvi korak je instalacija paketa `kexec-tools`:

``` shell
$ sudo pacman -S kexec-tools
```

Ako je instalirana nova verzija jezgre, alat `kexec` je može pripremiti za pokretanje korištenjem parametra `-l` na način:

``` shell
$ kexec -l /boot/vmlinuz-linux --initrd=/boot/initramfs-linux.img --reuse-cmdline
```

Pokretanje nove verzije jezgre uz zaustavljanje svih usluga i odmontiranje svih datotečnih sustava vrši se korištenjem systemdove naredbe `systemctl kexec`:

``` shell
$ sudo systemctl kexec
```

Više detalja moguće je pronaći na [stranici kexec na ArchWikiju](https://wiki.archlinux.org/title/Kexec).
