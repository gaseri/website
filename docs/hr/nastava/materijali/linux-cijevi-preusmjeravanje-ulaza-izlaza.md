---
author: Vedran Miletić, Vanja Slavuj, Sanja Pavkov
---

# Cijevi, preusmjeravanje ulaza i izlaza

- cijev, znak `|` (++alt-graph+w++ na hrvatskoj tipkovnici), omogućuje da izlaz jedne naredbe postane ulaz druge naredbe

    - broj cijevi u nizu je praktično neograničen
    - jedna od osnovnih vještina rada na operacijskim sustavima slični Unixu

- primjer korištenja cijevi

    - `grep rijec datoteka.txt` možemo izvesti i na način `cat datoteka.txt | grep rijec`
    - sve naredbe koje ćemo dalje navoditi mogu se koristiti s cijevima ili bez njih

## Filteri uz cijevi

- `wc`, kratica za brojanje riječi (engl. *word count*)

    - ispisuje broj linija (`-l`), riječi (`-w`), znakova (`-m`) i bajtova (`-c`) u tekstualnoj datoteci
    - može ispisati i najdulju liniju (`-L`)

!!! admonition "Zadatak"
    - Napravite direktorij `Zadatak3` i u njemu tri tekstualne datoteke, koje nazovite redom `tekst1.txt`, `tekst2.txt` i `tekst3.txt`. U prvu unesite dane u tjednu, u drugu nekoliko imena i prezimena kolega, a treću ostavite praznom.
    - Objasnite razliku u izlazu između:

        - `wc tekst*.txt` i `cat tekst*.txt | wc`
        - `wc --m tekst1.txt` i `wc --c tekst1.txt`

    - U rekurzivnom ispisu direktorija `/usr` prebrojite koliko ukupno znakova imaju sve linije koje sadrže niz `/bin`. Koristite cijevi.
    - Iz ranije stvorene datoteke `postavke.txt` provjerite točan broj zapisa IBM-ovih kodiranja u čijem nazivu postoji troznamenkasti broj. Koristite cijevi!

- `sort` reda linije u danoj tekstulanoj datoteci po danom kriteriju

!!! admonition "Zadatak"
    - Napravite datoteku `eu.txt` u koju zapišite nazive bar 10ak država EU. Nakon toga poredajte popis:

        - u abecednom poretku, bez korištenja cijevi
        - u abecednom poretku, korištenjem cijevi
        - u poretku suprotnom abecednom, korištenjem cijevi

    - Nadopunite popis tako da ispred naziva države upišete godinu njezina pridruživanja, te:

        - poredajte nazive država koje završavaju nizom `ka` prema redoslijedu njihova pridruživanja;
        - države koje su se Uniji pridružili nakon 2000. godine poredajte prema redoslijedu pridruživanja i zapišite u datoteku `2000.txt`, te prebrojite znakove koje ste upisali u datoteku. Zašto je dobiveni broj znakova jednak 0?

- `uniq` uzastopne duplicirane linije u tekstualnoj datoteci ispisuje samo jednom
- `tee` istovremeno ispisuje na standardni izlaz i u datoteku; **primjer:**

    ``` shell
    $ ls -a > zapis.txt
    $ ls -a | tee zapis.txt | less # izlaz jedne naredbe (ls -a) se profiltrira u tee i zapisuje se u datoteku zapis.txt te se nakon toga ispisuje u lessu
    $ ls -a | tee zapis.txt | grep Faks | less
    $ ls -a | tee zapis.txt | grep Faks | less > zapis.txt
    ```

    - koristimo je za fleksibilniji rad s cijevima
    - saznajte više o naredbi [na Wikipedijinoj stranici](https://en.wikipedia.org/wiki/Tee_(command))

!!! admonition "Zadatak"
    - Napišite naredbu koja iz datoteke `studenti1.txt` izdvaja sve linije koje počinju slovom `M`, i istovremeno ih šalje programu `less` i nadopisuje u datoteku `studenti-izdvojeni.txt` (**Uputa:** iskoristite naredbu `tee` i cijevi.)
    - Iz datoteke `eu.txt` *jednom naredbom* izlistajte sve države koje su se Uniji pridružili prije `2000. godine` i upišite ih u datoteku `prije2000.txt`, sortirajte taj popis i prebrojite broj država koje ne sadrže slovo `e` i broj upišite u datoteku `broj.txt`.

## Standardni ulaz, standardni izlaz i standardni izlaz za greške

Gotovo svaka naredba koju koristimo daje nam neki rezultat njezina izvođenja, izlaz; to mogu biti podaci kojima naredba treba rezultirati i poruke o greškama. Rezultati izvođenja zapisuju se u posebnu datoteku koja se naziva standardni izlaz (engl. *standard output*). Poruke o greškama zapisuju se u posebnu datoteku koja se naziva standardni izlaz za greške (engl. *standard error*).

Mnogi programi dobivaju input od standardnog ulaza (engl. *standard input*), a u zadanim postavkama to je tipkovnica. Korištenjem preusmjeravanja ulaza možemo odrediti otkud dolazi standardni ulaz, a korištenjem preusmjeravanja izlaza možemo odrediti gdje se šalju standardni izlaz i standardni izlaz za greške.

- preusmjeravanje izlaza, znak `>` omogućuje preusmjeravanje teksta koji bi se ispisao na ekranu u tekstualnu datoteku
- `>` vs `>>` -- prvi znak briše sadržaj, a drugi znak ga nadopisuje

    ``` shell
    $ cat datoteka.txt > datoteka1.txt # rezultat naredbe cat se zapisuje u datoteku datoteka1.txt
                                       # orginalni sadržaj datoteke se zamjenjuje sa sadržajem druge datoteke
    $ cat datoteka1.txt # ispis na ekran
    $ cat file1.txt >> file2.txt # dupli sadržaj; sadržaj prve datoteke se nadodaje na sadržaj druge datoteke
    ```

- preusmjeravanja izlaza za greške, znakovi `2>` i `2>>`
- preusmjeravanje ulaza, znak `<` rjeđe se koristi, većina alata može čitati tekstualne datoteke direktno

!!! admonition "Zadatak"
    - Odredite koja je razlika između `echo "XY" > tekst.txt` i `echo "XY" >> tekst.txt`.
    - Napravite datoteku `studenti1.txt` i u nju upišite nekoliko imena i prezimena vaših kolega. Izdvojite sve studente kojima prezime ima na drugom mjestu neki samoglasnik, i rezultat preusmjerite u datoteku `studenti-izdvojeni.txt`.
    - U datoteku `studenti1.txt` dodajte `John Terry`, `Arsen Dedic` i `Mile Kekin`. Zatim uz korištenje cijevi izdvojite sve studente kojima prezime na drugom mjestu ima slovo `e`, te rezultat nadopišite u datoteku `studenti-izdvojeni.txt`.
    - Izvedite `ls` tako da kao argument proslijedite nepostojeći direktorij. Radi li preusmjeravanje izlaza? Zašto? Probajte isto s `man`-om čiji je argument nepostojeća naredba.

!!! admonition "Dodatni zadatak"
    Koja je razlika između `cat > tekst1.txt` i `cat >> tekst1.txt`

    - ukoliko datoteka ne postoji,
    - ukoliko datoteka postoji i prazna je,
    - ukoliko datoteka postoji i nije prazna?

- `xargs` koristi standardni ulaz kao kao dio naredbe koju pokreće

!!! todo
    Ovdje nedostaje zadatak.

## Uspoređivanje sadržaja datoteka

- `comm` uspoređuje dvije sortirane datoteke liniju po liniju

!!! admonition "Zadatak"
    - U datoteku `naredbe.txt` preusmjerite popis sadržaja direktorija `/bin`. U istu datoteku nadopišite i popis sadržaja direktorija `/sbin`. U `naredbe2.txt` preusmjerite sadržaje direktorija `/usr/bin` i `/usr/sbin`.
    - Usporedite ove dvije datoteke i na ekran ispišite samo linije koje su jedinstvene datoteci `naredbe.txt`.
    - Sortirajte obje datoteke i spojite ih u jednu naziva `nar.txt`, te na ekran ispišite sve linije koje se ponavljaju u toj datoteci.

- `diff` uspoređuje dvije proizvoljne datoteke liniju po liniju

    - s parametrom `-u` prikazuje razliku u unificiranom diff formatu

!!! admonition "Zadatak"
    - Napišite osnovni "Hello world" program u C++-u i spremite ga u `program1.cpp`.
    - Kopirajte datoteku u `program2.cpp`. Izmijenite datoteku `program2.cpp` tako da ispisuje "Pozdrav svijete".
    - Usporedite razliku između izlaza naredbe `diff` i `diff -u`.

!!! admonition "Ponovimo!"
    - Što su cijevi i zašto su nam korisne?
    - Koje filtere najčešće koristimo uz cijevi? Koja je njihova uloga?
    - Prisjetite se parametara koje može primiti naredba `wc`.
    - Na kojem principu radi preusmjeravanje izlaza?
    - Definirajte `stdin`, `stdout` i `stderr`.
    - Objasnite prednosti naredbe `tee`.
    - Kako usporediti sadržaj dviju datoteka?
    - Razmislite kada je najkorisnije upotrijebiti naredbu `comm`?
