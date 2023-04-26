---
author: Vedran Miletić, Matea Turalija
---

# Instalacija i konfiguracija softvera za vježbe iz kolegija Arhitektura i organizacija računala

Upute u nastavku pisane su za korisnike Microsoftovih operacijskih sustava [Windows, verzije 10 i 11](https://www.microsoft.com/en-us/windows). Ako ste korisnik Ubuntua, možete preskočiti dio uputa koji je specifičan za operacijski sustav Windows i odmah početi s upisivanjem naredbi danih naredbi u terminalu.

## Priprema operacijskog sustava Windows

Korištenjem sustava [Windows Update](https://support.microsoft.com/en-us/windows/update-windows-3c5ae7fc-9fb6-9af1-1984-b5e0412c556a) instalirajte sve dostupne nadogradnje. Po potrebi, ponovno pokrenite računalo kad Windows to zatraži. Ovaj korak je nužan preduvjet za instalaciju preostalog softvera i s njime ste gotovi tek kada vam Windowsi prestanu nuditi ikakve nadogradnje za instalaciju.

Prema uputama [Which version of Windows operating system am I running?](https://support.microsoft.com/en-us/windows/which-version-of-windows-operating-system-am-i-running-628bec99-476a-2c13-5296-9dd081cdd808) uvjerite se da imate Windows verziju 22H2 (neovisno o tome imate li 10 ili 11).

### Instalacija softvera iz Microsoft Storea

Instalirajte iz Microsoft Storea:

- [Windows Subsystem for Linux](https://apps.microsoft.com/store/detail/windows-subsystem-for-linux/9P9TQF7MRM4R)
- [Windows Terminal](https://apps.microsoft.com/store/detail/windows-terminal/9N0DX20HK701)
- [Ubuntu](https://apps.microsoft.com/store/detail/ubuntu/9PDXGNCFSCZV)

### Uključivanje i isključivanje značajki sustava Windows

U tražilicu postavki odaberite Uključivanje i isključivanje značajki sustava Windows (engl. *Turn Windows features on and off*) te uključite sljedeće značajke sustava Windows:

- Podsustav Linux za Windows (engl. *Windows Subsystem for Linux*)
- Platforma virtualnog računala (engl. *Virtual Machine PLatform*)

Ponovno pokrenite računalo.

## Konfiguracija Ubuntua

Pokrenite Ubuntu, pričekajte instalaciju i postavite ime korisnika i lozinku.

### Instalacija nadogradnji

Pokrenite Windows Terminal pa [korištenjem dropdown izbornika u njemu otvorite Ubuntu](https://docs.microsoft.com/en-us/windows/terminal/panes) ili pokrenite Terminal na Ubuntuu. Upišite naredbu:

``` shell
$ sudo apt update
(...)
```

i vašu zaporku kad vas upita. Zatim upišite naredbu:

``` shell
$ sudo apt upgrade
(...)
```

i prihvatite nadogradnje koje vam ponudi.

### Instalacija Clanga i LLVM-a

Za instalaciju [Clanga](https://clang.llvm.org/) i [LLVM-a](https://llvm.org/) upišite naredbu:

``` shell
$ sudo apt install build-essential clang llvm
(...)
```

### Stvaranje direktorija projekta

``` shell
$ mkdir mojprojekt
```

## Visual Studio Code

Preuzmite Visual Studio Code sa [službenih stranica](https://code.visualstudio.com/). Instalirajte ga sa zadanim postavkama.

### Proširenje za VS Code: Remote - WSL

Ako ste na Windowsima, u dijelu `Extensions` koji se nalazi u `Side Bar`-u ([pregled sučelja](https://code.visualstudio.com/docs/getstarted/userinterface)) instalirajte [proširenje za udaljeni rad korištenjem WSL-a](https://code.visualstudio.com/docs/remote/wsl-tutorial).

### Otvaranje direktorija projekta

U donjem lijevom kutu kliknite na *Open a Remote Window*, a zatim se povežite na Ubuntu koji ste instalirali klikom na *Connect to WSL* pa otvorite direktorij `mojprojekt`.
