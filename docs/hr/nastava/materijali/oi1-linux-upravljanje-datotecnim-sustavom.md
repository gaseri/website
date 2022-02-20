---
author: Luka Vretenar
---

# Osnove rada sa komandnom linijom

- Komandna linija (eng. `command line interface`, `CLI`) je način upravljanja računalom unosom tekstualnih naredbi koje se predaju operacijskom sustavu za izvršavanje.
- Većina računalnih korisnika se susrela pretežito sa grafičkim sučeljima (eng. `graphical user interface`, `GUI`), koja iako najpopularnija nisu i najrasprostranjenija.
- Upravljanje računala komandnom linijom pruža veliku fleksibilnost, posebice kada su u pitanju operacije nad tekstualnim podacima, no nedostatak je ograničenje samo na tekstualni unos.
- Operacijski sustavi GNU/Linux pružaju više grafičkih korisničkih sučelja, kao i više sučelja komandne linije, no danas najrasprostranjenije sučelje za komandnu liniju je program `bash`.

## Pristup komandnoj liniji

- Iako imamo mogućnost rada na sustavu koji je podignut isključivo u komandnu liniju, najlakši način da joj pristupimo je iz grafičkog sučelja pokretanjem emulatora komande linije (eng. `terminal emulator`).
- Emulator komandne linije možete pronaći instaliran među ostalim aplikacijama na vašem sustavu.
- U slučaju računala koje imate na vježbama, emulator komandne linije se naziva `Konsole`.

!!! admonition "Zadatak"
    - Pokrenite emulator komandne linije.

- Pri pokretanju komandne linije dočekati će vas polje za unos (eng. `prompt`):

    ```
    korisnik@racunalo:~$
    ```

- Polje za unos sadrži određene korisne informacije:

    - ime trenutnog korisnika pod kojim smo logirani na sustav
    - naziv računala na kojem smo logirani
    - trenutnu putanju na direktorij u datotečnom sustavu u kojem se nalazimo

- Iza znaka `$` možemo upisivati vlastite naredbe.
- Naredbe se izvršavaju u direktoriju u kojem se nalazimo u trenutku pokretanja te naredbe!
- Naredbe su oblika:

    ``` shell
    $ naredba -parametar argument1 argument2 ...
    ```

!!! hint
    Pri pisanju naziva naredbe, možemo na tipkovnici pritisnuti dva puta tipku `<Tab>` da dobijemo popis svih naredbi koji započinju unesenim slovima.

- Svaka naredba može imati nijedan, jedan ili više parametara kao i nijedan, jedan ili više argumenata.
- Parametre prethodi simbol `-` i oni govore kako da se naredba ponaša.
- Svi argumenti iza naziva naredbe moraju biti odvojeni praznim znakom.
- Izvršavanje unesene naredbe izvodimo pritiskom tipke `Enter`.

!!! hint
    U slučaju da želimo prekinuti izvršavanje naredbe koja se odvija dugo ili se čini da se zamrznula, možemo to učiniti kombinacijom tipki ++control+c++.

!!! hint
    Za sve naredbe koje će biti spomenute može se pristupiti dokumentaciji u komandnoj liniji naredbom `man` dajući kao argument naziv aplikacije, iz dokumentacije izlazimo tipkom `q`.

## Datotečni sustav

- Datotečni sustav možemo zamisliti kao stablo direktorija i datoteka koje se grana na svakom direktoriju.
- Linux, kao i ostali Unix sustavi koristi unificirani datotečni sustav:

    - svi uređaji za pohranu se nalaze pod istim stablom direktorija

- Vrh stabla datotečnog sustava označavamo znakom `/` i nazivamo korjenom (eng. `root`).

    ```
    /
    ├── bin
    ├── boot
    ├── dev
    ├── etc
    ├── home
    │   └── lukav
    │       ├── Desktop
    │       ├── Documents
    │       ├── Downloads
    │       ├── Music
    │       ├── Pictures
    │       └── Public
    ├── lib
    ├── lib64
    .
    .
    ```

- Put do pojedinih direktorija i datoteka u datotečnom sustavu može biti zadan:

    - relativno, od direktorija u kojem se trenutno nalazimo
    - apsolutno, cjela putanja do direktorija od korjena `/`

- U slučaju da zadajemo put relativno, od direktorija u kokjem se nalazimo, zapisujemo ga:

    ```
    put/do/direktorija/
    put/do/datoteke
    ```

- Kada imamo apsolutni put od korjena `/`:

    ```
    /apsolutni/put/do/direktorija/
    /apsolutni/put/do/datoteke
    ```

- Putanje za direktorije i za datoteke se razlikuju po tome što putanja za direktorije završava znakom `/`.
- Znak `/` se pored označavanja korjena datotečnog sustava koristi i za odvajanje direktorija u putanji.

!!! hint
    Kod pisanja putanji u komandnoj liniji, možemo dva puta pritisnuti tipku `<Tab>` na dipkovnici da dobijemo popis svih datoteka u putanji čije ime započinje unesenim znakovima.

- Pored navedenih putanji u datotečnom sustavu postoje dodatne specijalne putanje:

    - `..` -- pokazuje na direktorij iznad onog u kojem se trenutno nalazimo
    - `.` -- pokazuje na trenutni direktorij u kojem se nalazimo
    - `~` -- pokazuje na `home` direktorij, to je direktorij u kojem se nalaze svi podaci trenutnog korisnika sustava
    - `/` -- korjen datotečnog sustava, u kojem se nalaze svi ostali direktoriji i datoteke

- Moguće je iz prikazanih specijalnih putanji i navedenih oblika putanji graditi kompleksne putanje spajanjem znakom `/`.

!!! hint
    Znak `~` se naziva tilda i možete je na hrvatskoj tipkovnici dobiti kombinacijom tipki ++alt-graph+1++.

!!! hint
    Svaki korisnik na sustavu ima vlastiti home direktorij koji se nazali u `/home/korisnik` gdje je korisnik naziv korisnika. Pojedini korisnik može upravljati datotekama samo u vlastitom home direktoriju.

## Upravljanje direktorijima

### Ispis trenutne pozicije -- naredba `pwd`

!!! admonition "Zadatak"
    - Ispišite putanju direktorija u kojem se trenutno nalazite uz pomoć naredbe `pwd`.

### Ispis sadržaja direktorija -- naredba `ls`

- Naredbom `ls` možemo ispisati sadržaj direktorija u kojem se trenutno nalazimo.
- Naredbi možemo postaviti dodatne argumente koje mjenjanju njezino izvršavanje:

    - argument `-a` prikazuje i sakrivene datoteke i direktorije
    - argument `-l` prikazuje detalje za svaku datoteku i direktorij, detalji su prava pristupa, veličina i vrijeme izmjene
    - putanju do nekog direktorija, govori naredbi `ls` da nam da ispis sadržaja nekog drugog direktorija koji joj specificiramo

    ``` shell
    $ ls
    $ ls -a
    $ ls -l
    $ ls -a -l
    $ ls /apsolutna/putanja/
    $ ls relativno/
    ```

!!! admonition "Zadatak"
    - Ispišite sadržaj trenutnog direktorija u kojem se nalazite.
    - Ispišite sadržaj trenutnog direktorija uključujući i sakrivene datoteke.
    - Ispišite sadržaj direktorija koji se nalazi iznad vašeg home (`~`) direktorija.
    - Ispišite sadržaj korjena (`/`) datotečnog sustava.

### Promjena pozicije u datotečnom sustavu -- naredba `cd`

- Za promjenu trenutnog direktorija u kojem se nalazimo možemo koristiti naredbu `cd`.
- Naredba prima jedan argument koji je putanja u direktorij u koji želimo ući.

    ``` shell
    $ cd direktorij
    $ cd /putanja/
    $ cd ..
    $ cd ../..
    $ cd ~
    ```

- Argument može biti putanja zadana:

    - relativno, od trenutnog direktorija
    - apsolutno

!!! admonition "Zadatak"
    - Postavite se u korjen datotečnog sustava.
    - Postavite se u direktorij `/usr/`.
    - Postavite se u direktorij `/usr/local/` relativnom putanjom.
    - Provjerite trenutnu lokaciju sa `pwd`.

- Za promjenu direktorija možete koristiti i specijalne putanje:

    - `..` da odemo u direktorij iznad
    - `~` da odemo u naš home direktorij

!!! admonition "Zadatak"
    - Postavite se u vlasiti home direktorij.
    - Odite direktorij iznad onog u kojem se trenutno nalazite.
    - Provjerite trenutnu lokaciju sa `pwd`.

### Izrada novog direktorija -- naredba `mkdir`

- Naredba `mkdir` prima jedan argument, naziv direktorija kojeg želimo kreirati.
- Putanja do direktorija može biti zadana relativno i apsolutno.

    ``` shell
    $ mkdir novidirektorij
    $ mkdir novidirektorij/drugidirektorij
    $ mkdir /apsolutno/novidirektorij
    ```

- Ukoliko se samo zada naziv direktorija, izraditi ćemo ga gdje se trenutno nalazimo u datotečnom sustavu.

!!! admonition "Zadatak"
    - Pozicionirajte se u vaš home direktorij.
    - Izradite direktorij `oi1`.
    - U direktoriju `oi1` izradite dva direktorija: `vjezbe` i `predavanja`.
    - Ispišite sadržaj direktorija `oi1`.

### Brisanje postojećeg direktorija -- naredba `rmdir`

- Naredbom `rmdir` brišemo pojedini direktorij iz datotečnog sustava, naziv direktorija zadajemo argumentom.

    ``` shell
    $ rmdir direktorij
    $ rmdir putanja/direktorij
    ```

- Važno je da je direktorij kojeg brišemo prazan, inače ga ne možemo obrisati naredbom `rmdir`!

!!! admonition "Zadatak"
    - Pobrišite sve direktorije iz prethodnog zadatka.
    - Naredbom `ls` provjerite da su svi pobrisani.

## Upravljanje datotekama

### Brisanje datoteka -- naredba `rm`

- Datoteke možemo izbrisati naredbom `rm`, na način da naziv datoteke ili putanju do nje specificiramo kao argument naredbi.

    ``` shell
    $ rm datoteka
    $ rm /put/do/datoteka
    ```

- U većini slučajeva nepovratno briše zadanu datoteku!

!!! admonition "Zadatak"
    - Prepisati i izvršiti naredbu `touch prazna.txt` u trenutnom direktoriju.
    - Provjeriti postojanje datoteke `prazna.txt` naredbom `ls`.
    - Pobrisati datoteku.

### Kopiranje datoteka -- naredba `cp`

- Datoteke kopiramo naredbom `cp`, potrebna su dva argumenta:

    - izvor
    - destinacija

- Kao drugi argument možemo zadati i put do direktorija, u tom slučaju kopiramo datoteku pod istim imenom u taj direktorij.

    ``` shell
    $ cp izvor destinacija
    $ cp /put/do/izvor /put/do/destinacije/
    ```

!!! admonition "Zadatak"
    - Kreirati praznu datoteku naredbom `touch prazna.txt` u trenutnom direktoriju.
    - Kopirati tu datoteku u isti direktorij pod nazivom `kopija.txt`.
    - Napraviti novi direktorij `pohrana` i iskopirati `prazna.txt` u njega.

### Pomicanje datoteka, -- naredba `mv`

- Naredba `mv` nam služi za pomicanje datoteke između direktorija, i kao `cp` prima dva argumenta.

    ``` shell
    $ mv izvor destinacija
    $ mv /put/do/izvor /put/do/destinacije/
    ```

- Naredbu `mv` koristimo i za preimenovanje datoteke ako radimo pomicanje u istom direktoriju pod drugo ime.

!!! admonition "Zadatak"
    - Kreirati praznu datoteku naredbom `touch prva.txt` u trenutnom direktoriju.
    - Preimenovati ju u `druga.txt`.
    - Premjestiti `druga.txt` u direktorij `pohrana` iz prethodnog zadatka.
