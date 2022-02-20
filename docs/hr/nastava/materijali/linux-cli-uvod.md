---
author: Vedran Miletić, Vanja Slavuj, Sanja Pavkov
---

# Uvod u komandnolinijsko sučelje

- korisnik -- identifikator (UID, user ID), korisničko ime i zaporka

    - korisnik `root` (UID 0) je poseban, ima sve ovlasti i koristi se za administraciju sustava

- grupa -- identifikator (GID, group ID), ime i korisnici u njoj
- GUI -- Graphical user interface (demonstracija)
- CLI -- Command-line interface (demonstracija)
- Secure SHell (SSH) -- protokol koji koristimo za rad na udaljenom računalu; klijenti su:

    - [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/), koristi se na Windowsima starijim od 10

        - demonstracija: Host treba biti `example.group.miletic.net`, `Translation` treba osigurati da je `UTF-8`

    - [OpenSSH klijent](https://www.openssh.com/), dolazi s većinom operacijskih sustava sličnih Unixu te Windowsima 10 i novijima

        - naredba `ssh korisnik@domacin`

- Početak rada s komandnom linijom

    - koristimo distribucije [Manjaro](https://manjaro.org/) i [Garuda Linux](https://garudalinux.org/), varijante [Arch Linuxa](https://archlinux.org/) prilagođene za korištenje na desktopima i laptopima
    - Logiranje u sustav -> čim se završi postupak pokretanja servera, rad na terminalima je moguć; nije potrebna zaporka niti korisničko ime

- `naredba -parm arg1 arg2` -- općenita struktura: naredba, parametri, argumenti (nakon naredbe opcionalno idu prvo parametri pa argumenti)

## Naredbe `echo` i `man`

- `echo` vraća korisniku uneseni tekst
- `man ime_naredbe` daje stranicu priručnika koji opisuje način korištenja naredbe

    - `q` služi za izlaz iz `man`-a

- `echo != ECHO`, `man != Man`

    - operacijski sustavi slični Unixu osjetljivi su na velika i mala slova

!!! admonition "Zadatak"
    - Ispišite na ekranu svoje ime i prezime.
    - *Jednom naredbom* ispišite svoje ime i prezime u jednom redu, a u drugom redu ispišite grad iz kojeg dolazite.
    - Na ekran ispišite sljedeće: Došao je do "Hemingway-a", ali nije nastavio dalje.

    (**Uputa:** koristite `man` stranice kao pomoć.)

## Naredbe `cal` i `date`

- `cal` prikazuje kalendar za određenu godinu i u određenom obliku
    - ovdje ćemo isprobati korištenje argumenata i parametara naredbe

!!! admonition "Zadatak"
    - Ispišite na ekranu kalendar za tekuću godinu.
    - Ispišite na ekranu kalendar za 2004. godinu.
    - Ispišite na ekranu julijanski kalendar za 3. mjesec 2004. godine.
    - Koliko argumenata prima naredba `cal` u prethodnom zadatku? A koliko parametara?

- `date` ispisuje datum u određenom formatu

!!! admonition "Zadatak"
    Na ekranu ispišite današnji datum oblika *DanUTjednu, Mjesec Dan Godina* (npr. `Ponedjeljak, Rujan 05 2013.`).

## Naredbe `ls` i `cat`

- `ls` izlistava datoteke u direktoriju
- `ls -a` izlistava sve datoteke u direktoriju, uključujući i skrivene
- `ls -l` izlistava datoteke u tzv. dugom ispisu

    - izlistava datoteke u direktoriju zajedno sa detaljnim informacijama (npr. znakovni niz dozvole, vlasništvo, veličina, datum izmjene, ...)

- `cat` ispisuje sadržaj (tekstualne) datoteke na ekran; **primjer:**

    ``` shell
    $ cat examples.desktop
    ```

    - kad se pokrene bez argumeanata radi beskonačno dugo

- `^` je oznaka za tipku ++control++
- `^C`, odnosno ++control+c++ služi za prekid izvođenja većine naredbi ([više informacija](https://en.wikipedia.org/wiki/Control-C))
- `^D`, odnosno ++control+d++ kraj rada, izlaz iz terminala ([više informacija](https://en.wikipedia.org/wiki/Control-D))

!!! admonition "Zadatak"
    - Saznajte imena svih datoteka koje postoje u vašem direktoriju (uključujući i skrivene).
    - Ispišite sadržaj datoteke `.bash_logout` na ekran.
    - Ispišite sadržaj datoteke `.profile` na ekran, ali tako da ispišete i brojeve linija.
    - Izlistajte sadržaj direktorija `.config` prema veličini datoteka i direktorija koji su u njemu, i to tako da se veličina prikaže u KB.

    (**Uputa:** koristite `man`.)

!!! admonition "Ponovimo!"
    - Što je CLI?
    - Ispišite opći oblik naredbe u komandnoj liniji.
    - Čemu služi naredba `echo`?
    - Što je manual i kako se koristi?
    - Zbog čega kažemo da `cat` != `Cat`?
    - Prisjetite se kako koristimo parametre i argumente na primjeru naredbe `date`.
