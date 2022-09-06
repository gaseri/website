---
author: Vedran Miletić
---

# Instalacija i konfiguracija softvera za vježbe iz kolegija Sigurnost informacijskih i komunikacijskih sustava

Upute u nastavku pisane su za [Garuda Linux](https://garudalinux.org/), ali su vjerojatno upotrebljive i na drugim derivatima [Arch Linuxa](https://archlinux.org/) kao što su [Manjaro](https://manjaro.org/), [EndeavourOS](https://endeavouros.com/) i [KaOS](https://kaosx.us/).

Specfično za Manjaro možete koristiti:

- za nadogradnju svih softvera `pamac upgrade -a` umjesto `garuda-update`, a
- za instalaciju softvera `sudo pamac install` umjesto `sudo pacman -S`.

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

## Skup alata za kriptografiju i SSL/TLS OpenSSL i Python modul pyOpenSSL

``` shell
$ sudo pacman -S openssl python-pyopenssl
(...)
```

## Skup alata za upravljanje autoritetom certifikata easy-rsa

``` shell
$ sudo pacman -S easy-rsa
(...)
```

## Python modul pyca/cryptography

``` shell
$ sudo pacman -S python-cryptography
(...)
```

## Alat za instalaciju Python paketa pip

``` shell
$ sudo pacman -S python-pip
(...)
```

## Alat za statičku analizu Python koda Pylint

``` shell
$ sudo pacman -S python-pylint
(...)
```

## IPython jezgra za Jupyter

``` shell
$ sudo pacman -S python-ipykernel
(...)
```

## HTTP klijent cURL

``` shell
$ sudo pacman -S curl
(...)
```

## Sustav za virtualizaciju na razini operacijskog sustava Docker

``` shell
$ sudo pacman -S docker docker-compose
(...)
```

Zatim dodajte svog korisnika u grupu `docker` koja ima pravo pokretanja kontejnera:

``` shell
$ sudo usermod -aG docker $(whoami)
(...)
```

Ako koristite ljusku `fish`, ova će naredba javiti grešku u sintaksi. Ispravan oblik naredbe za ljusku `fish` je:

``` shell
$ sudo usermod -aG docker (whoami)
(...)
```

### Uključivanje pokretanja daemona korištenjem aktivacije utičnice

``` shell
$ sudo systemctl enable --now docker.socket
(...)
```

### Provjera instalacije

``` shell
$ sudo docker run hello-world
(...)
```

Nakon odjave i ponovne prijave bit će moguće pokretati Docker kontejnere i kao običan korisnik, bez naredbe sudo.

## Razvojno okruženje Visual Studio Code

``` shell
$ sudo pacman -S visual-studio-code-bin
(...)
```

Visual Studio Code uključuje podršku za [Markdown](https://code.visualstudio.com/docs/languages/markdown) i osnovnu podršku za [Python](https://code.visualstudio.com/docs/languages/python).

### Alat za statičku analizu Markdowna markdownlint

Pokrenite Visual Studio Code. U dijelu `Extensions` koji se nalazi u `Side Bar`-u ([pregled sučelja](https://code.visualstudio.com/docs/getstarted/userinterface)) instalirajte proširenje [markdownlint](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint).

### Proširenje Docker

Instalirajte proširenje [Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker).

### Proširenje Python

Instalirajte proširenje [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python).

U upravitelju datoteka Dolphin stvorite direktorij (mapu) `php-prvi-projekt` i otvorite ga u Visual Studio Codeu korištenjem izbornika `File\Open Folder...` ili kombinacijom tipki ++control+k++ pa ++control+o++. Nakon otvaranja direktorija, stvorite u njemu datoteku `hello.py` sadržaja:

``` python
print("hello, world")
```

Pokretanje vršimo gumbom `Run Python File in Terminal` u gornjem desnom dijelu `Editor Groups`-a ili iz ugrađenog terminala (izbornik `Terminal` pa `New Terminal` ili kombinacija tipki ++control+shift+`++) naredbom:

``` shell
$ python hello.py
(...)
```

Dodatni terminal možemo dobiti gumbom `New Terminal` ili `Split Terminal` u gornjem desnom dijelu `Panel`-a, pri čemu je ovaj drugi način preferiran u situaciji kad se istovremeno pokreću klijentska i poslužiteljska mrežna aplikacija.
