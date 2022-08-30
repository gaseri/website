---
author: Vedran Miletić
---

# Instalacija i konfiguracija softvera za vježbe iz kolegija Informatika (BioTech)

Upute u nastavku pisane su za korisnike Microsoftovih operacijskih sustava [Windows, verzije 10 i 11](https://www.microsoft.com/en-us/windows). Ako ste korisnik Ubuntua, možete preskočiti dio uputa koji je specifičan za operacijski sustav Windows i odmah početi s upisivanjem naredbi danih naredbi u terminalu.

## Priprema operacijskog sustava Windows

Korištenjem sustava [Windows Update](https://support.microsoft.com/en-us/windows/update-windows-3c5ae7fc-9fb6-9af1-1984-b5e0412c556a) instalirajte sve dostupne nadogradnje. Po potrebi, ponovno pokrenite računalo kad Windows to zatraži. Ovaj korak je nužan preduvjet za instalaciju preostalog softvera i s njime ste gotovi tek kada vam Windowsi prestanu nuditi ikakve nadogradnje za instalaciju.

## Instalacija podsustava Windows Subsystem for Linux (WSL)

Instalirajte Windows Subsystem for Linux (WSL) prema [Microsoftovim uputama](https://docs.microsoft.com/en-us/windows/wsl/install), specifično odjeljku [Install WSL command](https://docs.microsoft.com/en-us/windows/wsl/install#install-wsl-command). Nemojte zaboraviti pokrenuti PowerShell ili Windows Command Prompt kao administrator prije upisivanja navedene naredbe.

### Windows Terminal

Dok čekate da WSL odradi svoje, iz [Microsoftovog Storea](https://apps.microsoft.com/) instalirajte [Windows Terminal](https://apps.microsoft.com/store/detail/windows-terminal/9N0DX20HK701).

### Ubuntu on WSL

Nakon ponovnog pokretanja operacijskog sustava Ubuntu bi se trebao pokrenuti sam. Ako se to ne dogodi, pokrenite Ubuntu i pričekajte da se instalira; instalacija će trajati najviše nekoliko minuta. Nakon završetka instalacije postavite korisničko ime, postavite zaporku (nećete vidjeti znakove koje utipkate) i ponovite unos iste zaporke. Kad dobijete pozdravnu poruku, možete zatvoriti Ubuntu.

Ako ste nehotice zatvorili Ubuntu prije unosa svih podataka i time prekinuli instalaciju, vratite instalirani Ubuntu u početno stanje prema [Microsoftovim uputama](https://support.microsoft.com/en-us/windows/repair-apps-and-programs-in-windows-e90eefe4-d0a2-7c1b-dd59-949a9030f317) pa ponovite postupak instalacije i postavljanja korisničkog imena i lozinke.

## Priprema operacijskog sustava Ubuntu

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

## Python

Upišite redom naredbe:

``` shell
$ sudo apt install python-is-python3 python3 python3-pip python3-ipykernel python3-autopep8 pylint
(...)
```

``` shell
$ sudo apt install python3-rdkit python3-sympy python3-numpy python3-scipy python3-matplotlib python3-pandas
(...)
```

## Računalna kemija, kemoinformatika i Bioinformatika

### Avogadro za Windowse

Preuzmite Avogadro sa [službenih stranica](https://avogadro.cc/). Instalirajte ga sa zadanim postavkama. Ako se kod pokretanja požali da mu nedostaju određene datoteke `.dll`, instalirajte verziju 1.90.0 za 64-bitne Windowse umjesto 1.2.0 za 32-bitne.

### OpenBabel za Windowse

Preuzmite OpenBabel sa [službenih stranica](https://openbabel.org/). Instalirajte ga sa zadanim postavkama.

### PyMOL za Windowse

Preuzmite PyMOL sa [službenih stranica](https://pymol.org/). Instalirajte ga sa zadanim postavkama. Ako se u instalaciji požali na duljinu putanje, instalirajte ga za sve korisnike uz korištenje administratorskih ovlasti umjesto samo za sebe.

### UGENE za Windowse

Preuzmite UGENE sa [službenih stranica](https://ugene.net/). Instalirajte ga sa zadanim postavkama.

### Avogadro, OpenBabel, PyMOL i UGENE za Ubuntu

U (Windows) Terminalu upišite redom naredbe:

``` shell
$ sudo apt install avogadro openbabel openbabel-gui python3-openbabel pymol ugene
(...)
```

### FastQC, Trim Galore!, Trinity RNA-Seq, Bowtie 2, Samtools, Integrative Genomics Viewer (IGV)

Upišite redom naredbe:

``` shell
$ sudo apt install fastqc trim-galore trinityrnaseq bowtie2 igv samtools
(...)
```

### FASTX-Toolkit

Prvo instalirajte cURL koji ćemo iskoristiti za preuzimanje paketa:

``` shell
$ sudo apt install curl
(...)
```

Zatim upišite redom naredbe:

``` shell
$ curl -O http://de.archive.ubuntu.com/ubuntu/pool/universe/f/fastx-toolkit/fastx-toolkit_0.0.14-5_amd64.deb
$ sudo apt install ./fastx-toolkit_0.0.14-5_amd64.deb
(...)
```

## Visual Studio Code

Preuzmite Visual Studio Code sa [službenih stranica](https://code.visualstudio.com/). Instalirajte ga sa zadanim postavkama.

### Proširenje za VS Code: Remote - WSL

Ako ste na Windowsima, u dijelu `Extensions` koji se nalazi u `Side Bar`-u ([pregled sučelja](https://code.visualstudio.com/docs/getstarted/userinterface)) instalirajte [proširenje za udaljeni rad korištenjem WSL-a](https://code.visualstudio.com/docs/remote/wsl-tutorial) i povežite se na Ubuntu koji ste instalirali.

### Proširenje za VS Code: Python

Ako ste na Windowsima, uvjerite se da ste [povezani na WSL pogledom u donji lijevi kut](https://code.visualstudio.com/docs/remote/wsl) gdje bi trebalo pisati `WSL: Ubuntu`; ako to ne piše, prvo se povežite na Ubuntu u WSL-u prema uputama danim u prethodnom dijelu.

Kad ste se uvjerili da ste povezani na Ubuntu, instalirajte [proširenje za Python](https://code.visualstudio.com/docs/languages/python).

## Obrada dokumenata

### Marp, Pandoc i LibreOffice

U (Windows) Terminalu upišite redom naredbe:

``` shell
$ sudo apt install npm pandoc libreoffice
(...)
```

``` shell
$ npm i @marp-team/marp-core @marp-team/marp-cli
(...)
```

### Proširenje za VS Code: Marp

Instalirajte Marp za Visual Studio Code prema [službenim uputama](https://marp.app/#get-started).

### Proširenje za VS Code: Live Share Extension Pack

Instalirajte i konfigurirajte Visual Studio Live Share prema [službenim uputama](https://code.visualstudio.com/learn/collaboration/live-share).
