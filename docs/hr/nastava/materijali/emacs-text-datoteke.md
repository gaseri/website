---
author: Vedran Miletić, Vanja Slavuj, Sanja Pavkov
---

# Rad s tekstualnim datotekama

## Kodiranje znakova

- [kodiranje znakova](https://en.wikipedia.org/wiki/Character_encoding) je zapis znakova nizom 0 i 1

    - npr. [ASCII](https://en.wikipedia.org/wiki/ASCII), [ISO/IEC 8-bitna kodiranja](https://en.wikipedia.org/wiki/ISO/IEC_8859), [ANSI kodne stranice](https://en.wikipedia.org/wiki/Windows_code_page#ANSI), [Unicode kodiranja](https://en.wikipedia.org/wiki/Unicode)

- Unicode je industrijski standard kodiranja koji omogućuje kodiranje većinu postojećih znakova

    - najčešće se koristi [UTF-8](https://en.wikipedia.org/wiki/UTF-8), nešto rjeđe [UTF-16](https://en.wikipedia.org/wiki/UTF-16), [UTF-32](https://en.wikipedia.org/wiki/UTF-32)

- tekstualna datoteka je datoteka koja sadrži čisti tekst kodiran nekim od kodiranja

    - iako datoteke mogu biti tekstualnog oblika bez obzira na ekstenziju, radi lakšeg snalaženja, označavat ćemo ih ekstenzijom `.txt`, primjerice `file1.txt`

- `locale` ispisuje trenutne lokalne i regionalne postavke operacijskog sustava

    - `en_US.UTF-8` (zadane), `hr_HR.UTF-8` (naše), `en_DK.UTF-8` (često korištene, donekle hakeraj; Danske regionalne i lokalne postavke slične našima)

- `locale -m` ispisuje popis podržanih kodiranja na sustavu

    - osim Unicode kodiranja, ISO-8859-2 je kodiranje koje sadrži Hrvatske znakove

!!! admonition "Dodatni zadatak"
    Naredba `iconv` se koristi se za konverziju između različitih kodiranja. Mi ćemo od sada nadalje koristiti samo UTF-8, ali se u praksi često sreću datoteke kodirane i drugim kodnim stranicama.

    Proučite `man` stranicu naredbe `iconv`, a zatim je isprobajte kod prebacivanja datoteke iz `UTF-8` kodiranja u `UTF-7`, `UTF-16`, `ISO-8859-2`.

## Uređivači teksta

- prije grafičkih sučelja, programi bili dizajnirani na način da koriste plain text
- programi za rad s tekstom u komandnoj liniji nisu [programi za obradu teksta](https://en.wikipedia.org/wiki/Word_processor) (engl. *word processors*) nego [uređivači teksta](https://en.wikipedia.org/wiki/Text_editor) (engl. *text editors* )
- [GNU Emacs](https://en.wikipedia.org/wiki/GNU_Emacs), naredba `emacs`

    - iznimno moćan uređivač teksta, velike mogućnost prilagodbe i proširenja funkcionalnosti
    - podržava bojenje sintakse, kompletiranje naredbi i automatsko uvlačenje koda za desetke različitih programskih jezika
    - podržava UTF-8 i koristi ga kao zadani ako je on zadan sustavskim lokalnim i regionalnim postavkama; [više detalja o kodiranjima u Emacsu](https://www.emacswiki.org/emacs/ChangingEncodings)
    - ne sadrži modove, za razliku od popularnog [Vim](https://en.wikipedia.org/wiki/Vim_(text_editor))-a
    - [xkcd: Real Programmers](https://xkcd.com/378/)

!!! admonition "Zadatak"
    - Proučite [vodič za Emacs od Stanfordovog Centra za računalna istraživanja u glazbi i akustici](https://ccrma.stanford.edu/guides/package/emacs/emacs.html) ili papir s Emacsovim naredbama. Isprobajte iduću funkcionalnost:

        - open, close, read, save
        - mark, copy, cut, paste
        - search forward, search backward

    - Napravite u svom kućnom direktoriju datoteku `mjeseci.txt` i u nju upišite nazive mjeseci tako da je svaki naziv u svome retku.
    - Kopirajte sav sadržaj i zalijepite ga na kraj datoteke tri puta.
    - Sačuvajte Vašu datoteku pod nazivom `mjeseci3.txt`.
    - Dodajte svoje ime i prezime na kraj datoteke i sačuvajte Vaš trenutni rad.
    - Odaberite dvije stvorene kopije sadržaja, izrežite ih, a zatim spremite u novu datoteku `mjeseci2.txt`.
    - Pozicionirajte se u neki redak datoteke `mjeseci2.txt` i otkrijte broj linije u kojoj se trenutno nalazite.
    - Pronađite u datoteci string `anj` i zamijenite ga sa rječju `ZAMJENA`.

!!! admonition "Dodatni zadatak"
    - Otvorite u Emacsu datoteku nazvanu `moj_program.cpp`. Uočite u koji način rada (mode) vas postavlja.
    - Unesite kod hello world C++ programa i spremite ga, a zatim ponovno spremite datoteku pod imenom `moj_program2.cpp`.
    - Otvorite obje datoteke istovremeno i prebacite se iz jedne u drugu.
    - Uočite da Emacs u zadanim postavkama ne vrši bojenje pripadnih zagrada `()`, odnosno `[]`, odnosno `{}`, odnosno `<>`. Pronađite u "Options" način da to uključite.

    (**Napomena:** Vremenom ćemo naučiti kako koristiti program-prevoditelj za prevođenje C++ programa na Linuxu.)

!!! admonition "Dodatni zadatak"
    - Proučite program Vi IMproved (naredba `vim`).

!!! admonition "Dodatni zadatak"
    - Pokrenite GNU nano (naredba `nano`), koji je vrlo jednostavan uređivač teksta, zamjena za nekoć popularni `pico`.
    - Isprobajte u GNU nano-u iduću funkcionalnost:

        - `^X -> Exit`,
        - `^C -> Cur Pos`,
        - `^O -> WriteOut`,
        - `^R -> Read File`,
        - `^K -> Cut Text` i `^U -> UnCut Text`,
        - `^W -> Where Is` (i varijante `^W^R`, `^W^Y`, `^W^V`, `^W^T`).

!!! admonition "Ponovimo!"
    - Što je kodiranje znakova?
    - Što je UTF-8? A UTF-32?
    - Na što sve utječu lokalne i regionalne postavke sustava?
    - Što je Emacs i čemu služi?
