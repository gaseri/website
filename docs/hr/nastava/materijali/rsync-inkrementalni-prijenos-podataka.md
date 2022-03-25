---
author: Vedran Miletić
---

# Inkrementalni prijenos podataka korištenjem rsynca

[rsync](https://rsync.samba.org/) je alat otvorenog koda za efikasan inkrementalni prijenos podataka. Vrlo je koristan kao alat za izradu sigurnosne kopije podataka. Kod korištenja za backup, vrlo važni parametar je `--archive`, odnosno `-a` koji čuva metapodatke o datotekama:

``` shell
$ rsync -a /podaci /backup-podataka
```

Više detalja moguće je pronaći na [stranici rsync na ArchWikiju](https://wiki.archlinux.org/title/Rsync#As_a_backup_utility).

!!! admonition "Zadatak"
    Preuzmite zadnju verziju jezgre Linux u obliku izvornog koda s [The Linux Kernel Archives](https://www.kernel.org/) i raspakirajte ju. Izvedite backup tog direktorija korištenjem rsync-a i izmjerite koliko mu vremena treba. Izmijenite po želji sadržaje datoteka `README` i `MAINTAINTERS` pa ponovno pokrenite rsync i provjerite koliko mu vremena treba da izvede kopiranje.

!!! admonition "Zadatak"
    Napravite systemd uslugu i pripadni mjerač vremena koji će pokretati istu naredbu svakih 6 sati. Razmislite morate li putanje do direktorija navoditi kao apsolutne ili kao relativne.
