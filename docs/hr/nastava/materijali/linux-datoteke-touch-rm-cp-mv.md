---
author: Vedran Miletić, Vanja Slavuj, Sanja Pavkov, Anja Vrbanjac
---

# Baratanje datotekama u datotečnom sustavu

## Stvaranje, kopiranje, brisanje i preimenovanje datoteka

- `touch` stvara praznu datoteku danog imena
- `rm` briše datoteku danog imena
- `cp` kopira datoteku danog imena u datoteku drugog danog imena; **primjer:**

    ``` shell
    $ cp datoteka1 datoteka2 # kopira datoteke (prvu u drugu)
    ```

- `mv` preimenuje (miče) datoteku danog imena; **primjer:**

    ``` shell
    $ mv datoteka1 datoteka2 # premješta datoteku1 u datoteku2 i preimenuje ih
    ```

!!! admonition "Zadatak"
    - U svom kućnom direktoriju napravite direktorij `zadatak1510` i uđite u njega.
    - U njemu napravite 3 datoteke: `vjezba1`, `vjezba2` i `vjezba3` jednom naredbom (linijom). Ispišite na ekran sadržaj datoteke `vjezba2`.
    - Izbrišite zatim datoteku `vjezba1`.
    - U istom direktoriju napravite i direktorij `dir1`.
    - Kopirajte datoteku `vjezba3` u `dir1`, jednom koristeći reltivno a drugi put apsolutno referenciranje.

- `cp -r` i `rm -r` označava rekurzivno kopiranje i brisanje, briše direktorij i sve datoteke i poddirektorije u njemu

    - kod naredbe `mv`, za razliku od `cp` i `rm`, nema potrebe za rekurzijom; radi se o preimenovanju

!!! admonition "Zadatak"
    - U Vašem kućnom direktoriju napravite predloženu strukturu direktorija:

        ```
        studentXY ------ Studij --------- Preddiplomski -------- DINP.txt
                                   |                        |
                                   |----- Pravilnik.txt     |--- Raspored.txt
        ```

    - Kopirajte datoteku `Pravilnik.txt` u direktorij `Preddiplomski`.
    - Kopirajte sav sadržaj direktorija `Preddiplomski`, u direktorij `Diplomski`, koristeći apsolutno referenciranje.
    - Izbrišite direktorij `Preddiplomski`.
    - Direktorij `Diplomski` preimenujte u `Dipl`.

!!! admonition "Dodatni zadatak"
    Provjerite čemu služi i kako se koristi program [Midnight Commander](https://en.wikipedia.org/wiki/Midnight_Commander) (naredba `mc`).

## Glob uzorci

- mnogo radimo sa nazivima datoteka pa postoji mogućnost rada sa posebnim znakovima (koji nemaju doslovno značenje) da bismo brzo i lako specificirali nazive većeg broja datoteka
- [glob uzorci](https://en.wikipedia.org/wiki/Glob_(programming)), globovi ili wildcards

    - koriste se za pretraživanje uzoraka koji odgovaraju zadanome
    - način da više datoteka koje imaju slična imena povežemo jednom naredbom
    - glob != regex, samo ima donekle sličnu sintaksu i namjenu

- `?` -- jedan znak, bilo koji
- `*` -- bilo koliko bilo kojih znakova
- `[chars]` -- jedan znak, bilo koji od navedenih u zagradama, može i raspon oblika `[A-Z]`, `[a-z]` ili `[0-9]`
- `[:klasa:]` -- zamjenjuje samo jedan, bilo koji znak koji je član navedene klase

    - najčešće korištene klase su: `[:alnum:]`, `[:alpha:]`, `[:digit:]`, `[:lower:]`, `[:upper:]`

- `\` -- tzv. *prekidni znak*

!!! admonition "Zadatak"
    - U svom kućnom direktoriju stvorite poddirektorij `Zadatci` i u njemu datoteke `zadatak`, `zadatek`, `zadatuk`, `zadatak1`, `zadatak2`, `zadatakABC`, `zadatakabc`, `zadacnica`, `zadacnicA`, `zad3` i `dat05`.
    - Jednom naredbom, koristeći se glob-om, izlistajte samo:

        - `zadatak`, `zadatek` i `zadatuk`
        - `zadatek` i `zadatuk`
        - samo datoteke koje na 8 mjestu naziva imaju veliko slovo
        - samo datoteke koje počinju slovom `z`, na 5 mjestu naziva im nije ni malo ni veliko slovo koja se po abecedi nalazi nakon slova `s`, i čiji naziv završava malim slovom
        - sve datoteke čiji naziv završava brojem manjim od 4
        - sve navedene datoteke

    - Isprobajte naredbu `ls [^ad]*` i razmislite o njezinom značenju.
    - Isprobajte naredbu `ls {ab,dat,f}??` i razmislite o njezinom značenju.
    - Isprobajte naredbu `cat *[[:upper:]1-4]`. Što ona radi?

## Pretraživanje datotečnog sustava

- `find` u specificiranim direktorijima traži datoteke ili skupine datoteka

    - sintaksa: `find <direktoriji> <uvjeti>` (direktorij koji se pretražuje *mora* biti naveden prije uvjeta)
    - direktorij može biti dan korištenjem apsolutnog ili relativnog referenciranja
    - može koristiti regularne izraze parametar `-regex`
    - pregled često korištenih parametara:

        | Parametar | Ograničenje pretraživanja |
        | --------- | ------------------------- |
        | `-user <ime korisnika>` | Samo datoteke određenog korisnika |
        | `-size <veličina>` | Samo datoteke specifične veličine |
        | `-type f` | Samo datoteke (ne direktoriji) |
        | `-type d` | Samo direktoriji |
        | `-name <ime datoteke>` | Samo datoteke određenog imena |
        | `-atime <broj dana>` | Samo datoteke čije je vrijeme posljednjeg pristupa manje od navedenog u danima |
        | `-amin <broj minuta>` | Isto kao iznad samo se navode minute umjesto dana |
        | `-ctime <broj dana>` | Samo datoteke čije je vrijeme izrade manje od navedenog u danima |
        | `-cmin <broj minuta>` | Isto kao iznad samo se navode minute umjesto dana |
        | `-mtime <broj dana>` | Samo datoteke čije je vrijeme posljednje promjene manje od navedenog u danima |
        | `-mmin <broj minuta>` | Isto kao iznad samo se navode minute umjesto dana |
        | `-newer <datoteka>` | Samo datoteke stvorene prije određene datoteke |

    - naredba je spora jer mora provjeriti svaki file koji se nalazi na zadanoj putanji
    - česte primjene:

        - brisanje nađenih datoteka naredbom find: `find ... -exec rm {} \;` ili `find ... | xargs rm`
        - pretraživanje nađenih datoteka: `find ... -type f | xargs grep <izraz>`

!!! admonition "Zadatak"
    Napišite naredbu `find` kojom u svom kućnom direktoriju tražite datoteke koje počinju sa `iz-`, čiji ste vi vlasnik i kojima je pristupano u zadnjih 30 dana.

- `locate` pretražuje bazu datoteka za datoteku koja u imenu sadrži dani niz znakova

    - sintaksa: `locate <ime datoteke>`
    - rezultat pretraživanja je puna putanja do tražene datoteke
    - pretražuje brže jer stvara bazu imena datoteka koje postoje u datotećnom sustavu te nema potrebe pretraživati svaku datoteku koja postoji na datotečnom sustavu; baza se osvježava naredbom `updatedb`, na većini modernih distribucija se pokreće automatski na dnevnoj bazi

!!! admonition "Zadatak"
    Pronađite datoteku `os-release`.

- `which` se koristi kada želimo saznati punu putanju do određene izvršne datoteke datoteke koja se nalazi u nekom od direktorija navedenom u varijabli okoline `PATH`
- `whereis` pretražuje lokaciju izvornog koda, binarnih datoteka i stranica priručnika

    - pregled često korištenih parametara:

        | Parametar | Uloga |
        | --------- | ----- |
        | `-b` | traženje binarnih datoteka |
        | `-m` | traženje stranica priručnika |
        | `-s` | traženje izvršnog koda |

!!! admonition "Ponovimo!"
    - Prisjetite se naredbi za stvaranje, brisanje i kopiranje datoteka. Koja naredba može primiti više argumenata?
    - Koja se naredba koristi za pomicanje neke datoteke u datotečnom sustavu? Koja za preimenovanje? Objasnite.
    - Što su globovi?
    - Koja je razlika između znakova `?` i `*` kod globova?
