---
author: Vedran Miletić
---

# Instalacija i konfiguracija softvera za vježbe iz kolegija Računalne mreže

Upute u nastavku pisane su za [Garuda Linux](https://garudalinux.org/), ali su vjerojatno upotrebljive i na drugim derivatima [Arch Linuxa](https://archlinux.org/) kao što su [Manjaro](https://manjaro.org/), [EndeavourOS](https://endeavouros.com/) i [KaOS](https://kaosx.us/).

Specfično za Manjaro možete koristiti:

- za nadogradnju svih softvera `pamac upgrade -a` umjesto `garuda-update`,
- za instalaciju softvera `sudo pamac install` umjesto `sudo pacman -S`, a
- za instalaciju softvera iz izvornog koda `pamac build` umjesto `paru -S`.

Instalacija softvera na Arch Linuxu je centralizirana, slično kao što su na drugim platformama [Microsoftov Windows Store](https://www.microsoft.com/en-us/store/apps/windows), [Appleov App Store](https://www.apple.com/app-store/), [Googleov Play](https://play.google.com/store/apps), [Sonyjev PlayStation Store](https://store.playstation.com/en-hr/latest) i drugi. Trgovina aplikacija se ovdje zove repozitorij paketa i, kao i druge trgovine aplikacija, dostupan je putem interneta. Stoga je za instalaciju paketa iz repozitorija koju provodimo u nastavku nužno da ste povezani na internet. Vrijedi spomenuti da će upravitelj paketima [Pacman](https://wiki.archlinux.org/title/Pacman) uz pakete čiju instalaciju zatražite preuzeti i dodatne pakete koji su potrebni za njihov rad.

## Priprema operacijskog sustava

Svakako prije instalacije paketa u nastavku instalirajte sve dostupne nadogradnje. U terminalu upišite prvo

``` shell
$ garuda-update
(...)
```

pa, kad vas sustav to pita, unesite vašu zaporku. Ova naredba je specifična za Garuda Linux; Na ostalim derivatima Arch Linuxa možete instalaciju svih dostupnih nadogradnji izvesti naredbom

``` shell
$ sudo pacman -Syu
(...)
```

Obje će naredbe osvježiti popis dostupnih paketa, a time i njihovih nadogradnji, pa zatim instalirati dostupne nadogradnje.

U nastavku za instalaciju softvera uz Pacman koristimo i [Paru](https://github.com/Morganamilo/paru), koji omogućuje instalaciju programa koji nisu pakirani iz njhovog izvornog koda. Njegove popise dostupnih softvera osvježite naredbom

``` shell
$ paru
(...)
```

!!! note
    Prilikom instalacije softvera iz izvornog koda koju provodimo u nastavku, Paru će prikazati skriptu za izgradnju traženog softvera (i, po potrebi, skripte za izgradnju dodatnih softvera o kojima taj softver ovisi). Prilikom pregleda skripte za izgradnju softvera, alat `paru` se ponaša kao preglednik priručnika `man` (za prikaz koristi `less`) pa je izlazak iz pregleda moguć tipkom `q`.

## Generator mrežnog prometa MGEN

``` shell
$ paru -S mgen
```

Je li instalacija bila uspješna isprobat ćemo narebom:

``` shell
$ mgen
(...)
```

MGEN će vjerojatno javiti pogreške oko datoteke ili direktorija koje ne može otvoriti unutar funkcije `GPSSubscribe()` oblika `GPSSubscribe(): fopen() error: No such file or directory`. Te nam pogreške nisu važne jer se unatoč njima MGEN pokreće i uredno radi.

## Emulator računalnih mreža CORE

Instalirajmo prvo softvere koje CORE zahtijeva.

### Instalacija preduvjeta

``` shell
$ paru -S tkimg
(...)
```

### Preuzimanje i instalacija

``` shell
$ paru -S core
(...)
```

Između ponuđenih verzija `1) core` i `2) core-git` odaberite `1) core`.

### Pokretanje daemona

Sada možemo pokrenuti CORE daemon naredbom (ovu naredbu moramo ponoviti svaki put kad ponovno pokrenemo računalo):

``` shell
$ sudo systemctl start core-daemon.service
```

Uvjerimo se da se usluga `core-daemon` zaista i pokrenula:

``` shell
$ systemctl status core-daemon.service
(...)
```

### Provjera instalacije CORE-a

Je li instalacija bila uspješna provjerit ćemo naredbom:

``` shell
$ core-gui-legacy --version
(...)
```

CORE bi morao javiti da je verzija 8.2.0.

Grafičko sučelje alata CORE pokrećemo naredbom `core-gui-legacy` ili ikonom iz izbornika desktop sučelja. Zatvorite CORE kao i bilo koji drugi prozor, a zatim nastavite u terminalu s instalacijama ostalih softvera.

## OpenSSH te osnovni alati za analizu i konfiguraciju računalne mreže

``` shell
$ sudo pacman -S openssh inetutils net-tools iproute2 traceroute iw wireless_tools
(...)
```

## Alati za premošćenje i filtriranje paketa bridgectl, ebtables, iptables i nftables

``` shell
$ sudo pacman -S bridge-utils iptables-nft nftables
(...)
```

## Usmjernik quagga

``` shell
$ sudo pacman -S quagga
(...)
```

Ukoliko dođe greška `Failed to write file: '...' operation not permitted`, zanemarite je.

## Alat za dinamičku dodjela IP adresa ISC DHCP

``` shell
$ sudo pacman -S dhclient dhcp
(...)
```

## Analizator mrežnog prometa Wireshark

``` shell
$ sudo pacman -S wireshark-qt
(...)
```

Zatim dodajte svog korisnika u grupu `wireshark` koja ima pravo hvatanja paketa:

``` shell
$ sudo usermod -aG wireshark $(whoami)
(...)
```

Ako koristite ljusku `fish`, ova će naredba javiti grešku u sintaksi. Ispravan oblik naredbe za ljusku `fish` je:

``` shell
$ sudo usermod -aG wireshark (whoami)
(...)
```

(U naredbama iznad ne morate ništa mijenjati jer naredba `whoami` vraća ime vašeg korisnika.)

## Alat za crtanje grafova Gnuplot

``` shell
$ sudo pacman -S gnuplot
(...)
```

## Razvojno okruženje Visual Studio Code

``` shell
$ sudo pacman -S visual-studio-code-bin
(...)
```

Visual Studio Code uključuje podršku za [Markdown](https://code.visualstudio.com/docs/languages/markdown) i [PHP](https://code.visualstudio.com/docs/languages/php).

### Alat za statičku analizu Markdowna markdownlint

Pokrenite Visual Studio Code. U dijelu `Extensions` koji se nalazi u `Side Bar`-u ([pregled sučelja](https://code.visualstudio.com/docs/getstarted/userinterface)) instalirajte proširenje [markdownlint](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint).

### Paket proširenja za PHP

U dijelu `Extensions` također instalirajte [PHP Extension Pack](https://marketplace.visualstudio.com/items?itemName=xdebug.php-pack).

## HTTP klijent cURL i sučelje Curlie

``` shell
$ sudo pacman -S curl curlie
(...)
```

## HTTP poslužitelj za razvoj aplikacija, testiranje i edukaciju PHP built-in web server, proširenje Xdebug i JSON procesor jq

``` shell
$ sudo pacman -S php xdebug jq
(...)
```

### Provjera instalacije PHP-a

U upravitelju datoteka Dolphin stvorite direktorij (mapu) `php-prvi-projekt` i otvorite ga u Visual Studio Codeu korištenjem izbornika `File\Open Folder...` ili kombinacijom tipki ++control+k++ pa ++control+o++. Nakon otvaranja direktorija, iskoristite gumb `New Folder` dostupan u pogledu Explorer ([pregled sučelja](https://code.visualstudio.com/docs/getstarted/userinterface)) za stvaranje poddirektorija (podmape) naziva `public`.

Unutar poddirektorija `public` stvorite iskoristite gumb `New File` za stvaranje datoteke imena `index.php` sa sadržajem:

``` php
<?php

phpinfo();
```

Pokrenite [ugrađeni terminal](https://code.visualstudio.com/docs/editor/integrated-terminal) putem izbornika `Terminal\New Terminal` ili kombinacije tipki ++control+shift+grave++. U njemu pokrenite ugrađeni poslužitelj PHP-ovog interpretera na adresi lokalnog domaćina `127.0.0.1`:

``` shell
$ php -S 127.0.0.1:8080 -t public
(...)
```

Korištenjem web preglednika otvorite adresu `http://127.0.0.1:8080/` i uvjerite se da ste dobili informacije o PHP-u. Uočite da vam se u terminalu prikazuje svaki HTTP zahtjev i statusni kod odgovora na njega; ta nam je značajka vrlo korisna za učenje rada web poslužitelja.

## HTTP tester i benchmark alat Siege

``` shell
$ sudo pacman -S siege
(...)
```
