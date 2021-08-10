---
author: Vedran Miletić, Vanja Slavuj, Sanja Pavkov
---

# Dozvole i kontrola pristupa datotekama

- dozvole su dio informacijskog čvora datoteke

    - tri klase: dozvole korisnika (engl. *user*), dozvole grupe (engl. *group*) i dozvole ostalih (engl. *others*)
    - tri dozvole (u svakoj klasi): čitanje (engl. *read*), pisanje (engl. *write*), pokretanje (engl. *execute*)
    - sve kombinacije tih čine obične načine pristupa (engl. *modes*)

- značenje dozvola za datoteke

    - **read**: čitanje sadržaja datoteke
    - **write**: promjena sadržaja datoteke
    - **execute**: pokretanje datoteke (potrebno, primjerice, kod kompajliranih C/C++ programa i Bash/Perl/Python skripti)

- značenje dozvola za direktorije

    - **read**: čitanje imena datoteka i poddirektorija u direktoriju (ali ne i sadržaja, dozvola, veličine i ostalih metapodataka)
    - **write**: promjena datoteka i poddirektorija u direktoriju: stvaranje, brisanje i preimenovanje
    - **execute**: prolaz kroz direktorij i mogućnost pristupa datotekama i poddirektorijima (ali ne i čitanje sadržaja direktorija)

- `chmod` mijenja način pristupa dane datoteke

- simbolička notacija

    - `u` -- user, `g` -- group, `o` -- others, `a` -- all
    - `+` -- dodaje, `-` -- briše, `=` -- postavlja
    - `r` -- read, `w` -- write, `x` -- execute
    - npr. `u+r`, `go-rx`, `a=r`

- oktalna notacija

    - `r` -- vrijednost 4; `w` -- vrijednost 2; `x` -- vrijednost 1; `-` -- vrijednost 0
    - tri broja -- user, group, others
    - primjerice, dozvole `644`, `755`, `600`

        ```
            6  4  4
           /   |   \
        User Group Others # User -- broj 6 označava da korisnik ima dozvole za čitanje(r) i pisanje(w)
                          # Group -- broj 4 označava da grupa ima dozvolu samo za čitanje(r)
                          # Others -- broj 4 označava da ostali imaju dozvolu samo za čitanje(r)
            7  5  5
           /   |   \
        User Group Others # User -- broj 7 označava da korisnik ima dozvole za čitanje(r), pisanje(w) i pokretanje(x)
                          # Group -- broj 5 označava da grupa ima dozvolu za čitanje(r) i pokretanje(x)
                          # Others -- broj 4 označava da ostali imaju dozvolu za čitanje(r) i pokretanje(x)
            6  0  0
           /   |   \
        User Group Others # User -- broj 6 označava da korisnik ima dozvole za čitanje(r) i pisanje(w)
                          # Group -- broj 0 označava da grupi nije dodjeljena niti jedna dozvola
                          # Others -- broj 0 označava da ostalima nije dodjeljena niti jedna dozvola
        ```

    - `644`, `755`, `600` je isto što i `0644`, `0755`, `0600`

- naredba `groups` daje popis grupa kojih je korisnik član

    - korisnik može biti član više grupa
    - da bi korisnik mogao pristupiti datoteci koja pripada grupi `grupa`, dovoljno je da bude član grupe `grupa`

!!! admonition "Zadatak"
    - Koje je značenje sljedećih dozvola kod naredbe `chmod ______ dat.txt`?

        - `go-rx`
        - `a=rw`
        - `0774`
        - `665`
        - `ug+w-rx`

    - Može li korisnik u posljednjem primjeru mijenjati sadržaj datoteke `dat.txt`? Dokažite!
    - Stvorite datoteku `dat000` i u nju upišite sadržaj po želji.

        - Postavite joj oktalnom notacijom dozvole na `r-xrwx--x`.
        - Pokušajte promijeniti njen sadržaj. Objasnite zašto to ne možete, bez obzira što ste u odgovarajućoj grupi.
        - Postavite dozvolu za sebe i ostale na čitanje i promjenu, a zatim grupi obrišite dozvolu promjene sadržaja i pokretanja datoteke.

!!! admonition "Dodatni zadatak"
    - Postoje i posebni načini pristupa (engl. *special modes*), koji se dijele u tri grupe:

        - special execute,
        - setuid i setgid,
        - sticky bit.

    - Proučite `chmod(1)` i objasnite kako rade.
    - Doznajte kako se koristi naredba `group`, te čemu služi.

## Korisnička maska

- `umask` je korisnička maska (engl. *user mask*)

    - služi za ograničavanje dozvola koje dobivaju stvorene datoteke i direktoriji
    - koristi simboličku ili oktalnu notaciju, kao i `chmod`
    - zadana vrijednost u većini distribucija je `022`

- zadane dozvole

    - za datoteke: `rw-rw-rw-`, oktalno `666`
    - za direktorije: `rwxrwxrwx`, oktalno `777`

- postupak računanja efektivnih dozvola dan je na [Wikipedijinoj stranici o umask](https://en.wikipedia.org/wiki/Umask)-u

!!! admonition "Zadatak"
    - Druga često korištena vrijednost za umask je `077`. Objasnite njeno značenje.
    - Pronađite vrijednost umaska koji ostavlja sve dozvole korisniku, briše dozvolu čitanja grupi i briše dozvolu pokretanja ostalima.

## Liste kontrole pristupa

- liste kontrole pristupa (engl. *Access Control Lists*, ACL) proširuju mogućnosti dozvola

    - skup aplikacija za manipuliranje proširenim dozvolama
    - 1998. specifikacija poznata pod imenom POSIX.1e "draft 17"

- vrste ACL-a (dvije podjele)

    - prva podjela

        - minimalne (engl. *minimal*): ekvivalentne standardnim dozvolama
        - proširene (engl. *extended*)

    - druga podjela

        - pristupne (engl. *access*)
        - zadane (engl. *default*)

- zadane ACL, ako postoje, poništavaju korisničku masku
- ACL algoritam: UID ?= owner, UID ?= named, GID ?= owner, GID ?= named, other
- `getfacl` čita ACL-e datoteke
- `setfacl` postavlja ACL-e datoteke

!!! admonition "Zadatak"
    - Korisniku `profesor` dodijelite na datoteci `dat1` mogućnost čitanja i pokretanja. Uočite kako ovo djeluje na obične dozvole grupe.
    - Grupi `disk` dodijelite na istoj datoteci mogućnost čitanja i pisanja. Uočite kako ovo djeluje na ACL masku.

!!! admonition "Zadatak"
    Dodijelite direktoriju `dir1` zadanu ACL koja omogućuje čitanje, pisanje i pokretanje za korisniku `nobody`.

    - U kakvom su odnosu zadane i pristupne ACL tog direktorija? Impliciraju li zadane ACL pristupne ACL?
    - Uočite kako ovo djeluje na datoteke i direktorije koje u njemu stvarate. Objasnite zašto datoteke dobivaju `#effective:rw-`.

!!! admonition "Dodatni zadatak"
    Zadatak iz *stvarnog* života: korištenjem ACL osmislite način da troje korisnika, `domargan`, `iivakic` i `vedranm`,dijelite jedan direktorij. Svaki od njih mora biti u mogućnosti u njemu stvarati datoteke i direktorije koje moraju kod stvaranja dobiti dozvolu čitanja i pisanja za sve troje.

!!! admonition "Ponovimo!"
    - Što je informacijski čvor?
    - Objasnite razliku između različitih timestamp-ova u informacijskom čvoru.
    - Opišite način i svrhu korištenja čvrstih i simboličkih poveznica. Prisjetite se poglavlja o relativnom i apsolutnom referenciranju.
    - Čemu služe dozvole?
    - Što čini način pristupa (engl. mode)?
    - Prisjetite se razlike između oktalnog i simboličkog zapisa kod naredbe `chmod`.
    - Detaljno objasnite pojam korisničke maske. Potkrijepite konkretnim primjerima.
    - Što je ACL i čemu služi? Opišite kroz primjer.

## Osnovni i prošireni atributi datoteka

- atributi su svojstva datoteka, slični atributima na datotečnom sustavu FAT (archive, directory, hidden, read-only, system i volume)

    - Linux ih podržava, ali ne koriste se često

- dvije vrste atributa

    - mogu se mijenjati: append only (a), compressed (c), no dump (d), extent format (e), immutable (i), data journalling (j), secure deletion (s), no tail-merging (t), undeletable (u), no atime updates (A), synchronous directory updates (D), synchronous updates (S), top of directory hierarchy (T).
    - mogu se samo čitati: huge file (h), compression error (E), indexed directory (I), compression raw access (X), compressed dirty file (Z)

- `lsattr` izlistava trenutne atribute datoteka
- `chattr` mijenja atribute datoteka

!!! todo
    Ovdje nedostaje zadatak.

- prošireni atributi (engl. *extended attributes*)

    - naredbe `getfattr` i `setfattr`
    - opisani u `attr(5)`

!!! todo
    Ovdje nedostaje objašnjenje i zadatak.
