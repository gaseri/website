---
author: Vedran Miletić
---

# Nabavka, kompajliranje i instalacija softvera otvorenog koda

[Softver otvorenog koda](https://en.wikipedia.org/wiki/Open-source_software) (engl. *open-source software*) je softver dostupan u obliku izvornog koda kod kojeg su prava na izvorni kod dana pod licencom koja dozvoljava korisnicima proučavanje i promjenu izvornog koda, te eventualno i daljnju distribuciju

[GPL](https://en.wikipedia.org/wiki/GNU_General_Public_License) i [BSD licence](https://en.wikipedia.org/wiki/BSD_licenses) su najčešće korištene; predstavljaju dva moguća odgovora na pitanje o tome što znači pojam *slobodan softver*, odnosno *otvoreni kod*. Video [BSD v. GPL, Jason Dixon, NYCBSDCon 2008](https://youtu.be/mMmbjJI5su0) daje pregled razlika.

Osim GPL-a i BSD licenci, ostoje i [brojne druge licence](https://en.wikipedia.org/wiki/Comparison_of_free_software_licenses).

## Nabavka softvera

Neki izvori slobodnog softvera su: [GNU Software](https://www.gnu.org/software/), [GitHub](https://github.com/), [Bitbucket](https://bitbucket.org/), [Gitorious](https://en.wikipedia.org/wiki/Gitorious), [SourceForge](https://sourceforge.net/) i [Google Code](https://code.google.com/archive/)

Preuzimanje softvera podrazumijeva preuzimanje arhiva, za što se mogu koristiti [Wget](https://en.wikipedia.org/wiki/Wget), [cURL](https://en.wikipedia.org/wiki/CURL) i drugi alati za dohvaćanje sadržaja sa poslužitelja na webu putem HTTP-a, HTTPS-a i FTP-a.

## Kompajliranje i instalacija softvera

- `./configure` je skripta ljuske (`#!/bin/sh`) koja provjerava postoje li potrebne bibliotečne datoteke i prilagođava postavke kompajliranja našem sustavu i našim željama
- `make` vrši kompajliranje izvornog koda softvera u izvršni kod
- `make install` vrši instalaciju

    - najčešće se instalacija u zadanim postavkama vrši u `/usr/local`, kako bi se ručno instalirani softver odvojio od onoga instaliranog pomoću upravitelja paketima koji ide u `/usr` (potrebno je imati dozvolu zapisivanja u taj direktorij -- u većini slučajeva to znači biti `root`)

!!! admonition "Zadatak"
    - Na [službenim stranicama uređivača teksta GNU nano](https://www.nano-editor.org/) preuzmite arhivu s izvornim kodom.
    - Raspakirajte arhivu koju ste preuzeli, a zatim izvršite konfiguraciju i kompajliranje.
    - Pokušajte izvršiti instalaciju, da uočite što točno ne uspijeva.
    - Pokušajte pronaći `nano` unutar direktorija gdje ste izvršili kompajliranje i pokrenite ga direktno. (**Napomena:** Takav način pokretanja kompajliranih programa neće uvijek raditi, ali za GNU nano specijalno hoće.)

- `./configure --prefix=<path>` omogućuje da se putanja `<path>` koristi kao mjesto za instalaciju umjesto predefinirane

!!! admonition "Zadatak"
    - Na [službenim stranicama VLC media playera](https://www.videolan.org/) preuzmite arhivu s izvornim kodom.
    - Raspakirajte je i pokušajte izvršiti konfiguraciju. Objasnite što skripta za konfiguraciju ne nalazi na sustavu.
    - Ono što VLC-u nedostaje moguće je instalirati u proizvoljne direktorije, ali je potrebno odgovarajućim parametrom `--with-<biblioteka>` navesti gdje se nalazi. Napravite to za prve dvije nedostajuće biblioteke.

!!! admonition "Dodatni zadatak"
    - Na [službenim stranicama uređivača teksta GNU nano](https://www.nano-editor.org/) moguće je GNU nano preuzeti i putem SVN-a.
    - Pronađite kako, a zatim izvršite kompajliranje po uputama koje su tamo navedene.
    - Pokrenite `nano` na isti način kao u prethodnom zadatku da se uvjerite da se zaista radi o različitoj verziji.
    - Istražite koju naredbu pokreće skripta ljuske `autogen.sh`. U `man` stranici te naredbe proučite što rade parametri koji su navedeni u skripti i kojem skupu alata pripada navedena naredba.
