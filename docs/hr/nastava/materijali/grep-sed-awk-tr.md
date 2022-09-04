---
author: Vedran Miletić, Vanja Slavuj, Sanja Pavkov
---

# Pretraživanje i obrada tekstualnih datoteka

## Izdvajanje linija iz tekstualnih datoteka

- regularni izrazi (regex ili regexp) su simbolička notacija koju koristimo za pretraživanja obrazaca unutar nekog teksta

    - koriste se u mnogim alatima i programskim jezicima za rješavanje problema povezanih sa manipulacijom teksta
    - regularni izrazi mogu varirati s obzirom na alat ili programski jezik
    - razlikujemo dvije vrste znakova kod regularnih izraza
    - regex != glob

- `grep`  služi za izdvajanje linija prema određenom uzorku iz tekstualnih datoteka, **primjer:**

    ``` shell
    $ grep "x" datoteka.txt # izdvaja linije koje sadrže slovo x u tekstualnoj datoteci datoteka.txt
    ```

    - ime dolazi od naredbe `g/re/p` (global / regular expression / print) tradicionalnog Unix editora `ed`
    - navodnici su značajni: postoji *velika* razlika između `grep rij ec datoteka.txt` i `grep "rij ec" datoteka.txt`
    - više detalja o naredbi pronađite [na Wikipedijinoj stranici](https://en.wikipedia.org/wiki/Grep) i u [službenom priručniku](https://www.gnu.org/software/grep/manual/grep.html)

- `[chars]` -- jedan znak, bilo koji od navedenih u zagradama

    - kao i glob, podržava i sintaksu oblika `[A-Z]`, `[a-z]`, `[0-9]`

- `[^chars]` -- jedan znak, bilo koji *osim* navedenih u zagradama
- `.` -- jedan znak, bilo koji
- `^` -- početak retka
- `$` -- kraj retka
- poznajemo dvije vrste regularnih izraza: basic regular expressions (BRE) i extended regular expressions (ERE)

    - razlikuju se u interpretaciji posebnih znakova
    - BRE: ^ $ . [ ] *
    - ERE: ( ) { } ? + |
    - grep je program koji koristi BRE
    - `grep -E '(xd|xy)' dat.txt`

- `*` -- nijedno, jedno ili bilo koliko ponavljanja prethodnog znaka (npr. `.*` je bilo koliko ponavljanja bilo kojeg znaka)
- `+` -- jedno ili bilo koliko ponavljanja prethodnog znaka (ali ne niti jedno!)
- `?` -- jedno ili nijedno ponavljanje prethodnog znaka
- `{n,m}` -- element koji prethodi mora se pojaviti određen broj puta

    - specijalni slučajevi `{n}`, `{n,}`, `{,m}`

- `(ab|cd)` -- predstavlja alternaciju tj. uzorak mora sadržavati ili niz `ab` ili `cd`
- *puno više o regularnim izrazima čuti ćete na kolegijima Formalni jezici i jezični procesori 1 i 2*

!!! admonition "Zadatak"
    - Stvorite datoteku `mjeseci.txt` u koju ćete, u svaki red posebno, zapisati sve mjesece u godini i sve dane u tjednu. Zatim iz nje izdvojite sljedeće linije:

        - sve koje sadrže slovo `a`;
        - sve koje sadrže slovo `O` ili `o`;
        - sve koje ne sadrže slovo `i`;
        - sve koje sadrže niz znakova an i još barem jedno slovo nakon `n`;
        - sve koje počinju slovom `S` ili `s`;
        - sve koje imaju točno 8 slova.

    - Napravite datoteku `adrese1.txt` i u nju upišite sljedeće e-mail adrese (svaka u svome retku):

        ```
        peric.hrvoje@mail.net
        nekiMail@inet.hr
        mail1@t-com.net
        Miro-gavran@gmail.com
        mail25@yahoo.com
        mail18@bnet.hr
        Vedran.miletic@inf.uniri.hr
        vperovic@mail.com
        pposcic2@net.hr
        ```

    - Izdvojite iz datoteke `adrese1.txt` adrese koje:

        - koriste bar jedno veliko slovo u korisničkom imenu;
        - završavaju na `.hr` ili `.com`;
        - čije se korisničko ime sastoji od točno 8 slova;
        - počinju sa bilo kojim slovom koje se u abecedi nalazi nakon slova `o`;
        - koriste znak `.` ili `--` u korisničkom imenu;
        - čije korisničko ime ne završava brojem;
        - sadrže slovo `v`, nakon kojeg slijedi niz `ic` (koji se može ponavljati više puta), a iza kojeg odmah slijedi znak `@`;
        - su oblika `mail(broj<5)(broj>7)`.

!!! admonition "Dodatni zadatak"
    Napravite datoteku `studenti.txt` u koju napišite popis studenata prisutnih na satu u formatu `Ime Prezime`, pri čemu je svaki student u svom redu te datoteke. Izdvojite iz nje:

    - sve studente kojima prezime počinje na `P`,
    - sve kojima prezime počinje slovom koje je dio hrvatske abecede,
    - sve kojima ime počinje hrvatskim dijakritičkim znakom,
    - sve koji nemaju hrvatski dijakritički znak ni u imenu ni u prezimenu,
    - sve kojima ime počinje na `P`, a prezime završava na `ć`,
    - sve čije prezime ima točno 6 slova, ali zadnje nije `ć`.

    **Napomene:**

    - `\(abc\)` čini da se niz znakova `abc` tretira kao cjelina, može se koristiti u kombinaciji sa `*` i `+`,
    - `\(abc\|de\)` ima značenje niz `abc` ili niz `de`,
    - `.` i `-` dobijemo kao `\\.` i `\\-` respektivno (jedan escape "pojede" ljuska, drugi se prosljeđuje grepu), a unutar zagrada `[]` escape nije potreban,
    - `\` dobijemo kao `\\\\`.

!!! admonition "Dodatni zadatak"
    Iz datoteke `mjeseci.txt` u koju ste upisali nazive mjeseca u godini izdvojite:

    - sve mjesece koji sadrže slovo `a`,
    - sve mjesece koji sadrže niz znakova `an`,
    - sve mjesece koji ne sadrže slovo `e`.

## Obrada tekstualnih datoteka

- `sed` je skraćenica od *stream editor*

    - dozvoljava rad nad nizom podataka
    - podaci sa standardnog ulaza uređuju se prema prethodno napisanim uputama, naredbama spremljenim u datoteku i prosljeđuju se na standardni izlaz
    - više detalja o naredbi pronađite [na Wikipedijinoj stranici](https://en.wikipedia.org/wiki/Sed) i u [službenom priručniku](https://www.gnu.org/software/sed/manual/sed.html) ili [tutorialu Brucea Barnetta](https://www.grymoire.com/unix/sed.html)

`sed` nudi dvije osnovne mogućnosti primjene.

- `sed` za zamjenu (`s`); **primjer:**

    ``` shell
    $ sed 's/dan/noć/' text1.txt # mijenja slijed dan za noć u datoteci text1.txt
    ```

    - koristi se zajedno sa regularnim izrazima, kao i kod naredbe `grep`, za zamjenu traženog uzorka, zadanim uzorkom
    - **sintaksa:** `sed 's/traženiUzorak/zamjenskiUzorak/' file.txt`

- `sed` za transformaciju (`y`); **primjer:**

    ``` shell
    $ sed 'y/aeiou/AEIOU/' file1.txt # zamjenjuje svako od navedenih malih slova s odgovarajućim velikim slovom
    ```

    - svaki se traženi znak zamjenjuje odgovarajućim zadanim znakom
    - **sintaksa:** `sed 'y/uzorak1/uzorak2/' file.txt`

Najčešće korišteni parametri naredbe `sed` su:

- `sed -n` čini da ne ispisuje na ekranu nikakav tekst, osim ako to nije eksplicitno navedeno zastavicom (po defaultu `sed` ispisuje sve na ekran)

    - ako se `-n` opcija ne koristi, `sed` je vrlo sličan naredbi `cat`

- `sed -e` koristi se za višestruku obrada

    - koristi se kada imamo više naredbi za `sed` prije prvog delimitatora (npr. `sed -e 's/a/A/' -e 's/e/E/'`)

!!! admonition "Zadatak"
    Napravite datoteku `dani.txt` koja će sadržavati sve dane u tjednu (jedan ispod drugoga).

    - zamijenite riječ `utorak` sa `drugi`,
    - koristeći samo jednu naredbu `sed`, zamijenite riječ `subota` sa `početak`, a `nedjelja` sa `kraj`,
    - transformirajte svaki samoglasnik u slovo koje se po abecedi nalazi iza njega

    **Napomena:** poslije zadnjeg delimitatora dodajte `g`; `g s y` ne funkcionira, funkionira samo sa `s`.

!!! admonition "Zadatak"
    Iz sustava MudRi preuzmite datoteku `receniceSED.txt` i sačuvajte ju u svom kućnom direktoriju. Zatim:

    - zamijenite svaku riječ koja počinje sa `an`, a završava na samoglasnik, sa rječju `BRAVO`. Možete li to učiniti?,
    - svaku rečenicu koja sadrži broj, zamijenite sa rječju `RECENICA`,
    - svako pojavljivanje znaka `a` ili `e` zamijenite sa `SAMae`, a svaku pojavu `o` ili `i` sa `SAMoe`,
    - izbrišite sve samoglasnike iz svih rečenica.

- `sed` zastavice se dodaju nakon posljednjeg delimitatora (/)

    - `/g` = zamjena svih pojavljivanja traženog uzorka
    - `/p` = ispisuje se izmjenjena linija (usporedite rezultat sa i bez `-n`)
    - `/5` = izmjenjuje 5. uzorak koji je pronađen
    - `/5g` = izmjenjuje 5. uzorak i svaki poslije njega
    - `/w file.txt` = određuje se odredišna datoteka u koju se upisuje rezultat naredbe `sed`

- zastavice se mogu kombinirati, no `/w` mora biti na posljednjem mjestu
- u kombinaciji s `y` ne koristimo zastavice

!!! admonition "Zadatak"
    Koristite datoteku `receniceSED.txt` za sljedeći zadatak:

    - zamijenite svaku (osim prve!) riječ koja počinje sa `an` ili `An` a završava na suglasnik, sa rječju `BRAVO`. Možete li to sada učiniti?,
    - svako pojavljivanje znaka `a` ili `e` nakon njegova (ukupno) trećeg pojavljivanja zamijenite sa `SAMae`, a svaku pojavu `o` ili `i` sa `SAMoe`,
    - preuredite prvu naredbu tako da se na ekran ne ispisuje ništa osim promijenjenih linija, te da se samo izmjenjene linije upisuju u datoteku pod nazivom `receniceSED1.txt`.

!!! admonition "Zadatak"
    Napravite datoteku `drzave.txt` u kojoj upišite Hrvatska i nazive 6 zemalja s kojima graniči (svaku u svome redu). Zatim napravite sljedeće:

    - Sve nazive država koje počinju slovom `S` promijeniti u naziv sa `s`, upisati u datoteku `maledrzave.txt`, te prikazati promjene na ekranu.
    - Promijenite nazive svih država koje završavaju na `ija` u `ZAMJENA`, upisati rezultat u datoteku `zam.txt` i onemogućiti ispis na ekranu.
    - Dodajte sljedeće države na popis: Kazahstan, Kirgistan, Uzbekistan, Afganistan, Pakistan i Turkmenistan, te zamijenite sve nazive koji završavaju na `stan` sa `nije susjed`, ali samo nakon trećeg pojavljivanja tog uzorka, te ispišite sve, ne samo izmijenjene linije.

- `sed` restrikcije: navode se *ispred* izraza koji opisuje zamjenu ili transformaciju

    - restrikcija broja retka

        - `sed '3 s/[0-9]*//'` -- brisanje prvog broja u trećem retku
        - `sed '/^g/ s/Mark/Sonya/g'` -- u svakom retku koji počinje sa slovom 'g' zamijeniti *Mark* sa *Sonya*

    - restrikcija prema rasponu retka

        - `sed '1,100 s/a/A/g'` -- zamjena 'a' sa 'A' vrši se u prvih 100 redaka (uočite da *nije* iskorišten znak ^)
        - `sed '21,$ s/a/A/g'` -- zamjena 'a' sa 'A' se vrši od 21. retka do kraja datoteke

!!! admonition "Zadatak"
    - Napišite `sed` naredbu koja će prikazati sadržaj prvih 7 redaka datoteke `drzave.txt`.
    - Napišite `sed` naredbu koja će u datoteci pretražiti pojavljivanja uzorka `an` i ispisati broj redaka pojave uzoraka.
    - Napišite `sed` naredbu koja će u retcima koji završavaju slovom `n` mijenjati sva mala slova u velika slova.
    - Napišite naredbu kojom ćete u datoteci `drzave.txt` promijeniti svako `a` u `A`, te `s` u `S`, ali samo ako riječ u retku ima 8 slova.

!!! admonition "Dodatni zadatak"
    - Stvorite datoteku `file1` proizvoljnog sadržaja.
    - Napišite `sed` naredbu koja će prikazati sadržaj prvih 7 redaka datoteke `file1`.
    - Napišite `sed` naredbu koja će u recima koji završavaju sa slovom `b` mijenjati sva mala slova u velika.
    - Napišite `sed` naredbu koja će prikazati sadržaj datoteke `file1` tako da umjesto sadržaja redaka od 3. do 5. budu prazne linije.
    - Napišite `sed` naredbu koja će prije linije ispisati i broj linije u kojoj mijenja sadržaj.

    (**Uputa:** konzultirajte `sed`-ovu `man` stranicu.)

- `d`, `p` i `q` bez `-n`

    - navedene se naredbe vrlo posebno ponašaju kada se upotrebljavaju sa i bez `-n`
    - brisanje sa `d` (briše linije od početka do kraja)

        - `sed '11,$ d' file.txt` -- gleda samo prvih deset redaka, ostale zanemaruje (slično naredbi `head`)
        - `sed '1,10 !d' file.txt` -- gleda samo prvih deset redaka, ostale zanemaruje (primjetite da ! znači negaciju)

    - ispis linija na ekranu sa `p`

        - `sed 's/dan/noć/p' text1.txt` -- ispisuje sve što je u datoteci `text1.txt`, a promjene koje je napravio (prema uvjetu) ispisuje još jednom
        - `sed '1,10 p' file1.txt` -- ispisuje prvih 10 redaka datoteke `file1.txt`

    - izlazak sa `q`

        - `sed '11 q'` -- prekida se izvođenje nakon obrade 11. retka

!!! admonition "Dodatni zadatak"
    Dopunite sljedeću tablicu sa izlazima na ekranu, s obzirom na oblik `sed` naredbe:

    | Sed parametri | Raspon | Naredb | Rezultat/ispis |
    | ------------- | ------ | ------ | -------------- |
    | `sed -n` | `1,10` | `p` | Ispisuje samo prvih deset redaka |
    | `sed -n` | `11,$` | `!p` |   |
    | `sed` | `1,10` | `!d` |   |
    | `sed` | `11,$` | `d` |   |
    | `sed -n` | `1,10` | `!p` |   |
    | `sed -n` | `11,$` | `p` |   |
    | `sed` | `1,10` | `d` |   |
    | `sed` | `11,$` | `!d` |   |
    | `sed -n` | `1,10` | `d` |   |
    | `sed -n` | `1,10` | `!d` |   |
    | `sed -n` | `11,$` | `d` |   |
    | `sed -n` | `11,$` | `!d` |   |
    | `sed` | `1,10` | `p` |   |
    | `sed` | `11,$` | `!p` |   |
    | `sed` | `1,10` | `!p` |   |
    | `sed` | `11,$` | `p` |   |

## Složenije izdvajanje i obrada teksta

- `awk` je jezik za skriptiranje koji pruža složene mogućnosti uparivanja obrazaca
- `awk` se može koristiti za zamjenu riječi u nekoj tekstualnoj datoteci sa zadanim riječima ili se mogu vršiti izračuni koristeći brojeve koji su zapisani u nekoj datoteci; mi ćemo se zadržati na funkciji ispisa

    - **sintaksa:** `awk '{print <što se ispisuje>}' datoteka.txt`
    - tekst koji želite ispisati (a nije dio datoteke) stavite u navodnike (npr. "Tekst")
    - s opcijom `print` i varijablama možemo koristiti i matematičke operatore `+`, `-`, `*` i `/`

- `awk` obrađuje dokument liniju po liniju, a svakoj je riječi u liniji pridružena varijabla: prvoj $1, drugoj $2, itd.

!!! admonition "Zadatak"
    Napravite datoteku `recenice.txt` koja ima dva retka riječi. U prvom retku neka se nalaze riječi `voli`, `mrzi` i `putovati`, a u drugom `mrzi`, `kuhati`, `peglati` i `čitati`. Koristeći naredbu `awk` napravite rečenice od zadanih riječi, tako da se u svakom retku koriste samo 1. i 3. riječ.

    **Napomena:** ispis bi trebao izgledati ovako:

    ```
    Moja prijateljica voli putovati.
    Moja prijateljica mrzi peglati.
    ```

!!! admonition "Zadatak"
    Napravite datoteku `ucenici.txt` u koju ćete zapisati ime i prezime nekoliko vaših kolega, te dodajte po dvije ocjene za svakoga (od 1 do 5). Koristeći naredbu `awk` ispišite:

    - prezime i sve ocjene za svakog učenika u datoteci, ispis neka bude oblika:

        ```
        Učenik Prezime ima ocjena1 iz prvog predmeta, te ocjena2 iz drugog predmeta.
        ```

    - ime i prvu ocjenu za svakog učenika, te zbroj ocjena, ispis neka bude oblika:

        ```
        Učenik Ime ima ocjena1 iz prvog predmeta, a zbroj ocjena mu iznosi zbrojOcjena.
        ```

    - ime, prezime, te prosječnu ocjenu za svakog studenta, ispis neka bude oblika:

        ```
        Učenik Ime Prezime ima prosječnu ocjenu prosječnaOcjena.
        ```

- `tr` mijenja sve pojave jednog izraza ili znaka u neki drugi; **primjer:**

    ``` shell
    $ cat file.txt | tr a-mA-Mn-zN-Z n-ZN-Za-mA-M # mijenja sva slova iz prvog intervala u odgovarajuća slova u drugom intervalu (šifrira tekst jednostavnom šifrom)
    ```

    - primjerice, može mijenjati sva velika slova u mala i obrnuto, slaže riječi u stupac i sl.

!!! todo
    Ovdje nedostaje zadatak.

## Ostali alati za manipulaciju prikazom datoteka

- `less` je preglednik tekstualnih datoteka, koristan za hvatanje izlaza na kraju niza cijevi

    - osobito koristan za upotrebu kod ispisa veće količine podataka (`cat` bi prekrio nekoliko ekrana)
    - naziv `less` znači suprotno od `more`, starijeg alata iste namjene, za razliku od kojeg omogućuje scrollanje teksta i "prema gore" (`more` je omogućavao samo scrollanje "prema dolje")

!!! admonition "Zadatak"
    - Otvorite pomoću programa `less` datoteku `/etc/passwd` i pronađite pojavljivanje riječi `bin` u istoj datoteci. Pronađite način da pomaknete na sljedeću pojavu traženog uzorka. (**Uputa:** pronađite u man stranici kako se radi `Search forward` i `Search backward`.)
    - Iskoristite naredbu `Examine` da bi otvorili i datoteku `/etc/group`.

- `head` ispisuje početni dio datoteke (zaglavlje), primjerice prvih 10 linija
- `tail` ispisuje završni dio datoteke (podnožje), primjerice zadnjih 25 linija

!!! admonition "Zadatak"
    - Ukucajte naredbe `locale > postavke.txt` i `locale -m >> postavke.txt` u komandnu liniju, pa zatim iz datoteke `postavke.txt`:

        - ispišite posljednjih 7 linija,
        - ispišite prvih 20 linija.

    - Uočite što se događa s izlazom naredbe `tail` ukoliko vaša datoteka ima više od jedne prazne linije na kraju.

!!! admonition "Ponovimo!"
    - Što je `grep`?
    - Zašto kažemo da globalni izrazi nisu isto što i regularni izrazi? Objasnite.
    - Prisjetite se oznaka za regularne izraze.
    - Što je BRE, a što ERE?
    - Za što se koristi naredba `sed`?
    - Što su zastavice kod naredbe `sed`? Nabrojite ih nekoliko.
    - Koja zastavica kod naredbe `sed` uvijek mora biti na posljednjem mjestu kada ih više navodimo? Zbog čega?
    - Objasnite razliku između naredbi `head` i `tail`.
    - Kako biste opisali za što se koristi `less`?
