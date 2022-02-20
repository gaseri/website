---
author: Luka Vretenar
---

# Pretraživanje, prava pristupa, rad sa tekstom

## Pretraživanje datoteka i direktorija

### Filtriranje naziva datoteka po obrascima

- Komandna linija `bash` koju koristimo omogućuje nam zadavanje naziva datoteka ili direktorija preko određenih uzoraka.
- Na taj način možemo obuhvatiti više datoteka ili direkotirja odjednom.
- Ta funkcionalnost `bash` komandne linije se naziva `glob` expansion.
- Primjer:

    - brišemo više datoteka koje završavaju na `.txt`
    - ispisujemo sadržaj direktorija filtrirajući samo datoteke koje završavaju na `.txt`

    ``` shell
    $ ls *.txt
    $ rm *.txt
    ```

- Takvi uzorci se nazivaju `globs` i sastoje se od teksta po kojem filtriramo i simbola:

    - `*` -- zamjenjuje nam više znakova u nazivu
    - `?` -- zamjenjuje nam jedan znak u nazivu

!!! admonition "Zadatak"
    - Ispišite sadržaj direktorija `/usr/bin/` filtrirajući samo datoteke čiji naziv započinje znakom `b`.

### Pretraživanje datoteka u datotečnom sustavu

- Datotečni sustav možemo pretraživati naredbom `find`.
- Naredba pretražuje počevši od zadanog direktorija, sve direktorije ispod njega u potrazi za danim nazivom datoteke.
- Mogu se koristiti uzorci za pretragu.
- U najjednostavnijem obliku, pretražujemo po nazivu datoteke `naziv` u nekom direktoriju `direktorij` (`-name` je dio naredbe):

    ``` shell
    $ find direktorij -name naziv
    ```

- Postoje različiti argumenti sa kojima možemo specificirati pretragu po veličini datoteke, vremenu izmjene itd. Za opise pojedinih argumenata konzultirati dokumentaciju naredbe.

!!! admonition "Zadatak"
    - Potražiti u direktoriju `/usr/share/` sve datoteke naziva `index.html`.

## Prava pristupa datotekama i upravljanje pravima

### Način rada prava pristupa

- Operacijski sustav Linux nam omogućuje upravljanje vlasništvom i pravima pristupa datotekama i direktorijima pojedinim korisnicima.
- Za provjeru vlasništva i prava nad pojedinom datotekom možemo iskoristiti naredbu `ls -l`.
- Primjer dobivenog izlaza:

    ```
    drwxr-xr-x 2 lukav lukav 4096 Nov  9 22:38 direktorij
    -rw-r--r-- 1 lukav lukav   74 Nov  9 22:38 primjer.txt
    ```

- Od značaja su nam vrijednosti prvog, trećeg i petog stupca:

    - `-rw-r--r--` je niz znakova koji nam definira prava pristupa
    - `lukav` je korisnik koji je vlasnik prikazane datoteke
    - `lukav` koji se pojavljuje drugi put, je naziv grupe korisnika
    - `74` je veličina datoteke u bajtovima

!!! hint
    `Linux` je višekorisnički sustav te omogućuje upravljanje vlasništvom datoteka za svakog korisnika, pored korisnika postoje i grupe korisnika koje također mogu biti vlasnici datoteka.

- Niz koji definira prava pristupa možemo dalje podijeliti na četri segmenata:

    ```
    d rwx rwx rwx
    ```

- Niz prava pristupa se sastoji od četri segmenata:

    - prvi segment je samo jedan znak i može biti `-` ili `d`, u slučaju da je `d` specificira nam da se radi o direktoriju
    - drugi segment sadrži tri znaka `rwx` ili `-` na mjestu bilo kojeg znaka, specificira nam pravo pristupa vlasnika datoteke
    - treći segment je sličan drugom, specificira pravo pristupa grupi korisnika koja posjeduje datoteku
    - četvrti segment je isto sličan drugom, specificira pravo pristupa svih ostalih korisnika na sustavu

- Znakovi `rwx` imaju slijedeće značenje:

    - `r` -- mogućnost čitanja iz datoteke ili direktorija
    - `w` -- mogućnost pisanja u datoteku ili direktorij
    - `x` -- mogućnost pokretanja datoteke kao aplikaciju ili mogućnost ulaska u direktorij

- Svaki znak može biti zamjenjen sa `-`, u tom slučaju to pravo je onemogućeno na toj datoteci.

!!! admonition "Zadatak"
    - Koja su prava pristupa definirana i za koga za datoteku `primjer.txt` prikazanu u tekstu?

### Izmjena prava pristupa -- naredba `chmod`

- Ako smo vlasnik datoteke ili direktorija, možemo izmjeniti prava pristupa koristeći naredbu `chmod`.
- Naredba `chmod` prima dva argumenta:

    - modifikato kojim zadajemo koje pravo želimo izmjeniti
    - naziv datoteke ili direktorija čije pravo pristupa izmjenjujemo

    ``` shell
    $ chmod XYZ datoteka
    ```

- Modifikator `XYZ` se sastoji od tri dijela:

    - `X` -- specifikacija segmenta koji mjenjamo, `u` za vlasnika, `g` za grupu koja posjeduje datoteku, `o` za ostale korisnike, `a` za sva prava
    - `Y` -- operaciju izmjene, `+` za dodavanje prava, `-` za uklanjanje prava
    - `Z` -- pravo pristupa koje izmjenjujemo, `r`, `w` ili `x`

    ``` shell
    $ chmod o-r primjer.txt
    ```

!!! admonition "Zadatak"
    - U komandnoj liniji izvršiti naredbu `touch prazna.txt`, koja kreira datoteku `prazna.txt`.
    - Provjeriti koja prava pristupa ima nova datoteka.
    - Dati svim ostalim korisnicima pravo čitanja iz datoteke.
    - Ukloniti pravo pisanja u datoteku grupi korisnika koja je vezana na datoteku.

## Rad sa tekstualnim datotekama

### Editor teksta `nano`

- Na svim Linux sustavima u komandnoj liniji dolazi editor teksta `nano`.
- Pokrećemo izmjenu ili izradu nove datoteke pozivom alata `nano` na slijedeći način:

    ``` shell
    $ nano datoteka.txt
    ```

- Nakon što smo završili unos ili izmjenu, datoteku možemo spremiti kombinacijom tipki ++control+x++ te pratiti upite na dnu ekrana.
- Sve naredbe editora su navedene na dnu ekrana.

### Naredbe za pregled sadržaja

- Pregled sadržaja tekstualne datoteke naredbom `less`:

    - kao argument prima naziv tekstualne datoteke
    - sa tipkom `<Space>` ili tipkama gore i dolje prolazimo kroz datoteku
    - sa tipkom `q` izlazimo iz prikaza

- Pregled prvih `n` linija u tekstualnoj datoteci sa `head`:

    - kao argument prima naziv tekstualne datoteke
    - ispisuje prvih `10` linija sa vrha tekstualne datoteke
    - kao parametar `-n` se može zadati da ispisuje `n` brojeva linija

    ``` shell
    $ head datoteka.txt
    $ head -2 datoteka.txt
    ```

- Pregled `n` linija sa kraja tekstualne datoteke sa `tail`:

    - isto kao `head` samo ispisuje linije sa kraja datoteke

- Brojanje riječi i linija u tekstualnoj datoteci sa `wc`:

    - prima naziv datoteke kao argument
    - ispisuje tri broja: broj linija, broj riječi i broj znakova u datoteci

- Sortiranje linija u datoteci naredbom `sort`:

    - prima naziv datoteke kao argument
    - ispisuje sve linije iz izvorne datoteke sortirane po abecednom redu

- Konkatenacija i ispisivanje datoteka sa `cat`:

    - prima nijedan, jedan ili više naziva datoteka kao argumente
    - ispisuje datoteke zadanim redoslijedom u komandnu liniju
    - u slučaju da se pokrene bez argumenta, samo ponavlja unešeni tekst, unos se prekida sa ++control+d++

!!! admonition "Zadatak"
    - Naredbom `less` pregledati sadržaj datoteke iz prethodnog zadatka.
    - Naredbom `tail` ispisati zadnje tri linije iz iste datoteke.
    - Ispisati sve sortirane linije iz datoteke.
    - Ispisati sadržaj cijele datoteke u terminalu.

### Pretraga sadržaja datoteka sa `grep`

- Naredba `grep` nam omogućuje pretragu sadržaja neke datoteke po zadanom uzorku.
- Prima dva argumenta:

    - niz koji pretražujemo, ograđen navodnicima `""`
    - datoteku koju pretražujemo

    ``` shell
    $ grep "izraz" datoteka.txt
    ```

- Pretraga je osjetljiva na mala i velika slova, u slučaju da želimo zanemariti veličinu slova potrebno je kao argument dodati `-i`.

!!! admonition "Zadatak"
    - U datoteci iz koju smo kreirali sa `nano` pretražiti jednu od riječi ignorirajući velika i mala slova.

### Preusmjeravanje izlaza naredbi

- Komandna linija `bash` omogućuje preusmjeravanje izlaza pojedinih naredbi u datoteke ili u druge naredbe.
- Na taj način je moguće zapisati izlaz koji dobijemo ispisan na ekranu u neku datoteku ili nizati više naredbi za redom.
- Operacije preusmjeravanja su:

    - `> datoteka`, zapisuje izlaz naredbe u datoteku
    - `>> datoteka`, doda izlaz naredbe na kraj već postojeće datoteke
    - `< datoteka`, naredbi na ulaz daj sadržaj datoteke
    - `|`, preusmjeri izlaz jedne naredbe na ulaz druge

    ``` shell
    $ sort datoteka.txt | head -2
    $ cat dat1.txt dat2.txt > out.txt
    ```

!!! admonition "Zadatak"
    - U datoteku `dupla.txt` nadodati preusmjeravanjem sadržaj konkatenacije datoteke iz `nano` zadatka same sa sobom.
    - Iskoristite naredbu `cat` za izradu datoteke, izlaz naredbe preusmjerite u datoteku `izlaz.txt`. (Unos prekidate tipkama ++control+d++ na praznoj liniji)
