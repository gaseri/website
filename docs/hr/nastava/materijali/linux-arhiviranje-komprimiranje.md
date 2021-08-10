---
author: Vedran Miletić, Vanja Slavuj, Sanja Pavkov
---

# Arhiviranje i komprimiranje

- stilovi/standardi parametara

    - System V (UNIX) stil -- prefiksirani crticom, mogu se grupirati (npr. `ls -a -s` je isto što i `ls -as`)
    - BSD stil -- nisu prefiksirani crticom, mogu se grupirati (kod `tar`-a se moraju)
    - GNU dugački stil -- prefiksirani dvjema crticama, ne mogu se grupirati (npr. `ls --all --size`)

## Arhiviranje

- tar (naredba `tar`) je alat za arhiviranje (!= komprimiranje)

    - ime tar je skraćeno od Tape ARchiver
    - arhive koje stvara nazivamo tarball
    - podržava sva tri stila parametara; sljedeće naredbe su ekvivalentne

        ``` shell
        $ tar -c tekst.txt # System V stil
        $ tar c tekst.txt # BSD stil
        $ tar --create tekst.txt # GNU dugački stil
        ```

    - osnovni parametri su `-Acdtrux`, u šali se ponekad čita kao "acid trucks"
    - postoje i brojne druge šale o alatu, npr. [xkcd: tar](https://xkcd.com/1168/)

- stvaranje `tar` arhive vrši se pomoću parametra `-c`:

    ``` shell
    $ tar -cvf naziv_arh.tar naziv_dir/ # c = create, v = verbose, f = file; v je opcionalan
    $ tar -cvf naziv_arh1.tar naziv_dat
    ```

- izvlačenje datoteka iz `tar` arhive radi se pomoću parametra `-x` (`untar` ne postoji):

    ``` shell
    $ tar -xvf naziv_arh.tar # x = extract
    ```

- izlistavanje datoteka u `tar` arhivi:

    ``` shell
    $ tar -tvf naziv_arh.tar
    ```

- ostale mogućnosti `tar`-a:

    ``` shell
    $ tar -rvf naziv_arh.tar adresa/nove/Dat # dodavanje jedne datoteke u tar datoteku
    $ tar -rvf naziv_arh.tar adresa/novog/Dir/ # dodavanje jednog direktorija u tar datoteku
    $ tar -xvf naziv_arh.tar naziv_datoteke # izvlačenje jedne datoteke iz tar datoteke
    $ tar -xvf naziv_arh.tar naziv_direktorija # izvlačenje jednog direktorija iz tar datoteke
    ```

!!! admonition "Zadatak"
    - U svom kućnom direktoriju stvorite datoteku `dat1.txt`, direktorij `Backup` i direktorij `Arhiviranje`, i u njemu niz od 5 tekstualnih datoteka, koje redom nazovite `tekst1.txt`, `tekst2.txt`, …, `tekst5.txt`.
    - Iz `~` u direktoriju `Backup` stvorite arhivu pod nazivom `6-datoteka.tar` koja sadrži sve gore navedene datoteke.
    - Izlistajte sadržaj te `tar` datoteke, bez izvlačenja datoteka iz `tar`-a i bez izlaska iz `~`.
    - U direktoriju `Arhiviranje` stvorite datoteku `tekst6.txt` i dodajte ju u postojeću arhivu.

## Komprimirane arhive

- `gzip`, `bzip2` i `xz` su alati za komprimiranje *jedne* datoteke (dakle, ne arhiviranje)

    - veličina komprimirane datoteke varira ovisno o vrsti sadržaja polazne datoteke
    - brzina kompresije varira ovisno o veličini datoteke za kompirimiranje, parametrima kompresije i vrsti sadržaja
    - nakon komprimiranja brišu polaznu datoteku

- `gunzip`, `bunzip2` i `unxz` su alati za dekomprimiranje

    - brzina dekomprimiranja varira ovisno o veličini komprimirane datoteke, parametrima kompresije i vrsti sadržaja
    - nakon dekomprimiranja brišu komprimiranu datoteku

- `tar` može koristiti `gzip`, `bzip2` i `xz` pa kao rezultat dobijemo komprimiranu arhivu

    - `cf` -- create file, bez kompresije
    - `czf` se koristi za gzip kompresiju, `cjf` za bzip2 kompresiju, `cJf` za xz kompresiju
    - `xf` -- extract file, automatsko prepoznavanje kompresije
    - `xzf`, `xjf` i `xJf` rade raspakiravanje datoteka s odgovarajućom kompresijom

!!! admonition "Zadatak"
    - U direktoriju `Arhiviranje` stvorite poddirektorij `Pohrana`. Koristeći `bzip2` kompresiju stvorite arhivu `arhiva.tar.bz2` koja sadrži datoteke `tekst2.txt` i `tekst4.txt`, te direktorij `Pohrana`.
    - U svom kućnom direktoriju stvorite direktorij `dir-backup`, te raspakirajte arhivu `arhiva.tar.bz2` u taj direktorij bez kopiranja.
    - Izlaz naredbe `dmesg` zapišite u datoteku `izlaz.txt`. Kopirajte ju u datoteke `izlaz1.txt` i `izlaz2.txt`. (Naredba `dmesg` ispisuje poruke koje Linux jezgra javlja kod pokretanja sustava. U većini slučajeva radi se o porukama o hardveru, memoriji i procesima.)
    - Datoteku `izlaz1.txt` sažmite `gzip`-om, a datoteku `izlaz2.txt` `bzip2`-om.
    - Saznajte kako ispisati veličine dobivenih datoteka, te ih usporedite.

!!! admonition "Dodatni zadatak"
    - Ispitajte koriste li se `xz` i `unxz` na isti način kao i `gzip`, odnosno `bzip2`.
    - Izlaz naredbe `dmesg` zapišite u datoteku `izlaz.txt`. Kopirajte ju u datoteke `izlaz1.txt` i `izlaz2.txt`.
    - Datoteku `izlaz1.txt` komprimirajte `gzip`-om, a datoteku `izlaz2.txt` `bzip2`-om.
    - Saznajte kako ćete ispisati veličine dobivenih datoteka, a zatim ih usporedite. Objasnite zašto, unatoč tome što `bzip2` komprimira više od `gzip`-a, nema razlike u veličini.
    - Sada izlaz naredbe `dmesg` dopišite u praznu datoteku `izlaz2.txt` pet puta (cilj je dobiti veću datoteku). Ponovite gore opisani postupak. Ima li razlike u veličini datoteka? Objasnite zašto.
    - Isprobajte `cat` na sve tri datoteke. Objasnite dobiveni rezultat. Mogu li vam naredbe `zcat` i `bzcat` pomoći?

- `zip` i `unzip` služe za baratanje ZIP komprimiranim arhivama

!!! todo
    Ovdje nedostaje zadatak.

!!! admonition "Ponovimo!"
    - Što je `tar`?
    - Koje stilove navođenja parametara poznajete?
    - Prisjetite se nekih ranijih naredbi koje smo obradili i provjerite koju vrstu navođenja parametara podržavaju.
    - Objasnite pojam `tarball`.
    - Koje alate za sažimanje podataka poznajete?
    - Usporedite najvažnije alate za sažimanje podataka u Unix-like OS-ima.
    - Kako se sadržaj sažima koristeći naredbu `tar`?
