---
author: Luka Vretenar
---

# Skriptiranje u Perl-u

- Interpretirani skriptni programski jezik osnovne namjene za obradu teksta.
- `perl` interpreter je besplatan i slobodno dostupan sa adrese [perl.org](https://www.perl.org/).
- `perl` interpreter dolazi preinstaliran na gotovo svim `unix` sustavima, uključujući i `linux`.
- Više informacija o `perl` jeziku i o njegovoj povjesti na [Wikipediji](https://hr.wikipedia.org/wiki/Perl).
- Danas većinom na putu u zaborav, nekada je pogonio većinu skripti za generiranje sadržaja dostupnog na internetu (`cgi scripts`).
- Na modernim sustavima u potpunosti zamjenjen dinamičkim interpretiranim jezicima kao što su `python` i `ruby`.

## Pokretanje perl aplikacija

- Skripte za izvršavanje `perl` koda pišemo u običnim tekstualnim datotekama.
- Za pisanje skripti možemo koristiti klasične tekstualne editore ili neki `IDE` (`Integrated developement environment`).
- Programske skripte moraju započeti sa posebnom linijom koja definira da želimo skriptu izvršiti kao `perl` program:

    ``` perl
    #!/usr/bin/perl
    ```

- `Perl` skriptu možemo pokrenuti na dva načina, ako pretpostavimo da se naša skripta naziva `skripta.pl`:

    - pokretanje direktnim pozivanjem `perl` interpretera:

        ``` perl
        perl skripta.pl
        ```

    - dodjeljivanjem prava za pokretanje skripti te pokretanje skripte kao zasebnu aplikaciju na `Linux`-u:

        ``` perl
        # prvo datoteci dodijelimo prava pokretanja
        chmod u+x skripta.pl

        # pokrenemo skriptu
        ./skripta.pl
        ```

!!! hint
    Znak `#` na početku linije nam ujedno označava i komentar u kodu. Linije koje započinju tim znakom će biti ignorirane od strane `perl` interpretera pri izvršavanju skripte.

!!! admonition "Zadatak"
    - Napravite novu skriptu naziva `prva.pl` i u nju zapišite:

        ``` perl
        #!/usr/bin/perl
        print("Hello world!\n");
        ```

    - Postavite skripti prava pokretanja i pokrenite ju u komandnoj liniji.

## Osnove sintakse

- Svaka linija koda u `perl` skripti, koja nije komentar ili početak logičkog grananja, mora završavati znakom `;`!

### Varijable i vrijednosti

- Varijable su simboli kojima imenujemo određene vrijednosti u memoriji računala.
- U `perl`-u sve varijable počinju znakom `$` nakon čega slijedi ime varijable.
- Pri odabiru imena pojedine varijable potrebno je paziti da se zadovolje slijedeći uvjeti:

    - naziv varijable mora započeti znakom `$`
    - nakon znaka za varijablu slijedi najmanje jedno slovo iz abecede
    - može slijediti kombinacija bilo kojeg broja slova, znamenki brojeva i znakova `_`.

- Primjer ispravnih naziva varijabli:

    ``` perl
    $x
    $var
    $my_variable
    $var2
    ```

- Primjer neispravnih naziva varijabli:

    ``` perl
    $47x
    $_var
    $variable!
    $new.var
    ```

- Nazivi varijabli su osjetljivi na velika i mala slova, `$Var` nije isto što i `$var`.
- Vrijednosti varijablama pridružujemo korištenjem znaka `=`, primjeri:

    ``` perl
    $var = 42;
    $ime = "Lucija";
    $a = $b;
    ```

- Prije korištenja određene varijable u skripti, potrebno ju je definirati tako da joj pridružimo neku početnu vrijednost.
- Nije potrebno specificirati vrste podataka u varijablama pri definiciji, `perl` interpreter sam prepoznaje vrstu podataka.
- Vrijednosti koje dodijeljujemo varijabljama u `perl` skriptama su najčešće:

    - nizovi znakova, ograđeni navodnicima -- `"Hello world!"`
    - brojčane vrijednosti -- `42`, `3.14`

### Operacije sa varijablama

- Nad vrijednostima varijabli možemo izvoditi numeričke i logičke operacije.
- Numeričke operacije su primjenjive na varijablama koje sadrže brojeve i vraćaju nam novu numeričku vrijednost:

    - `+` -- zbrajanje
    - `-` -- oduzimanje
    - `*` -- množenje
    - `/` -- dijeljenje

- Logičke operacije primjenjujemo kada želimo usporediti vrijednosti dvaju varijabli i vraćaju nam logičke izraze točno ili netočno:

    - `==` -- logička usporedba jednakosti
    - `!=` -- provjera nejednakosti
    - `<`, `>` -- manje od, veće od
    - `<=`, `>=` -- manje ili jednako od, veće ili jednako od
    - `&&`, `||` -- logičko I, logičko ILI

- Logičku vrijednost točno nam predstavlja broj `1` a netočno broj `0`.
- Ako želimo koristiti vrijednost pojedine varijable u izrazu, zapišemo njezino puno ime:

    ``` perl
    $c = $a + $b;
    $tekst = "Pozdrav $ime $prezime !";
    ```

- Na isti način možemo koristiti vrijednosti varijabli u nizovima znakova: zapisom punog imena varijable gdje želimo njezinu vrijednost.

### Čitanje i pisanje u komandnu liniju

- Čitanje linija teksta iz komandne linije u naš `perl` program se izvodi preko specijalne varijable `<STDIN>`.
- Mjesto u skripti na kojem se nalazi simbol `<STDIN>` će biti tijekom izvršavanja skripte zamijenjeno jednom pročitanom linijom sa ulaza programa.
- Ispis vrijednosti jedne ili više varijabli nazad u komandnu liniju vršimo naredbom `print()` kojoj u zagradama dajemo vrijednost za ispis.
- Ulaz i izlaz u program su nam najčešće komandna linija u kojoj smo pokrenuli program.
- Primjer čitanja jedne linije i njen ispis u komandnu liniju:

    ``` perl
    $linija = <STDIN>;
    print($linija);
    ```

- Kada čitamo neki broj ili niz znakova iz jednog reda potrebno je prvo pročitanu liniju pročistiti od znaka za kraj linije sa `chomp()` naredbom:

    ``` perl
    $broj = <STDIN>;
    chomp($broj);
    $dvostruko = $broj * 2;
    print("dvostruka vrijednost unesenog broja je ", $dvostruko);
    ```

!!! admonition "Primjer"
    - Program za pretvaranje metara u kilometre i obrnuto:

        ``` perl
        #!/usr/bin/perl
        print("Unesite vrijednost udaljenosti za pretvoriti: ");
        $udaljenost = <STDIN>;
        chomp($udaljenost);

        $metri = $udaljenost * 1000;
        $kilometri = $udaljenost / 1000;

        print("$udaljenost kilometara = $metri metara\n");
        print("$udaljenost metara = $kilometri kilometara\n");

        ```

!!! admonition "Zadatak"
    - Napiši skriptu koja učitava dva broja, zbroji ih i ispiše rezultat.

## Logička grananja

### Grananja `if`, `if-else`

- Logička grananja koristimo da bi izvršili određeni dio koda samo kada vrijedi određeni logički izraz.
- Grananja se sastoje od jednog ili više logičkih izraza i od jednog ili više blokova koda koji izvršavamo.
- Blokovi koda u grananja su grupirani između znakova `{` i `}`.
- Najjednostavnije logičko grananje nam je `if`:

    ``` perl
    if (uvjet)
    {
        naredbe;
    }
    ```

- Grananje možemo proširiti i na slučaj da se ne ispuni uvjet na `else` i na provjeru više uvjeta:

    ``` perl
    if (uvjet)
    {
        naredbe;
    }
    else
    {
        naredbe;
    }


    if (uvjet1)
    {
        naredbe;
    }
    elsif (uvjet2)
    {
        naredbe;
    }
    else
    {
        naredbe;
    }
    ```

- Provjere `elsif` možemo ponavljati više puta u istom logičkom grananju.

- Uvjet se sastoji od logičkog izraza koji može imati jednu ili dvije varijable i neki od logičkih operacija između njih.
- Najčešće provjeravamu jednakost vrijednosti dvije varijable logičkim izrazom za jednakost `==`.

!!! hint
    `==` nije isto što i znak za dodjeljivanje vrijednosti varijable `=`.

!!! admonition "Primjer"
    - Primjer provjere jednakosti unesenih brojeva.

        ``` perl
        #!/usr/bin/perl
        print("Unesite broj: ");
        $broj1 = <STDIN>;
        chomp($broj1);

        print("Unesite drugi broj: ");
        $broj2 = <STDIN>;
        chomp($broj2);

        if ($broj1 == $broj2)
        {
            print("Uneseni brojevi su jednaki.\n");
        }
        else
        {
            print("Uneseni brojevi nisu jednaki.\n");
        }
        ```

### Petlje `while`, `until`

- U slučaju da određene naredbe želimo ponavljati sve dok je određeni uvijet zadovoljen koristimo petlje `while` i `until`.
- Petlja `while` se sastoji od uvijeta i bloka naredbi koje ponavljamo. Sve dok je uvijet zadovoljen ponavlja se izvršavanje tih naredbi:

    ``` perl
    while (uvjet) {
      naredbe;
    }
    ```

- Petlja `until` se razlikuje od petlje `while` po tome što je logika uvjeta obrnuta, izvršavanje naredbi se ponavlja sve dok nije zadovoljen zadani uvjet:

    ``` perl
    until (uvjet) {
      naredbe;
    }
    ```

!!! admonition "Primjer"
    - Odbrojavanje do nule:

        ``` perl
        #!/usr/bin/perl
        $brojac = 10;
        while ($brojac >= 0)
        {
            print($brojac, "\n");
            $brojac = $brojac - 1;
        }
        ```

    - Čitanje ulaza dok nije pročitan ispravan broj:

        ``` perl
        #!/usr/bin/perl
        print("Koliko iznosi 17 plus 26? ");
        $broj = <STDIN>;
        chomp($broj);

        while ($broj != 43)
        {
            print("Krivo! Pokušaj ponovo: ");
            $broj = <STDIN>;
            chomp($broj);
        }
        print("Točno!\n");
        ```

!!! admonition "Zadatak"
    - Napiši program koji koristi `until` petlju za ispis prvih 10 brojeva silazno (10-1).

!!! admonition "Zadatak"
    - Napiši program koji koristi `while` petlju za ispis prvih 10 brojeva uzlazno (1-10).

## Regularni izrazi

- Najvažnija funkcionalnost jezika `perl`.
- Regularni izrazi se zadaju kao vrijednosti ograđene sa dvije kose crte `/` i `/`:

    ``` perl
    /[0-9]+/
    /xy+/
    /$var/
    ```

- Primjena zadanog regularnog izraza nad nekom varijablom koja sadrži niz znakova se vrši specijalnim operaterom `=~`:

    ``` perl
    $tekst =~ s/123/abc/;
    $uvjet = $ime =~ /luka/i;
    ```

- Unutar ograda regularnog izraza `/` i `/` mogu se nalaziti i nazivi varijabla čije vrijednosti želimo koristiti u regularnom izrazu.

!!! admonition "Pažnja!"
    - Operater `=~` se može promijeniti vrijednost varijable nad kojom se primjenjuje. U slučaju pretrage niza regularnim izrazom, rezultat operacije je logičko točno ili netočno, ako regularni izraz koristimo za substituciju onda se mjenja sama vrijednost varijable.

### Pretraga teksta

- Regularne izraze u `perl`-u možemo, u najjednostavnijem obliku, koristiti za pretragu određenih vrijednosti u nekom tekstu.
- U tom slučaju regularni izraz poprima oblik `/regex/mod`, gdje je `regex` naš regularni izraz a `mod` može biti prazan ili sadržavati specijalan znak `i` kojim definiramo neosjetljivost na mala i velika slova.

!!! admonition "Primjer"
    - Pronalaženje određenog niza teksta u liniji pročitanoj sa ulaza:

        ``` perl
        #!/usr/bin/perl
        $recenica = "Svaka prava ptica leti.";

        print("Sto moram traziti? ");
        $trazi = <STDIN>;
        chomp($trazi);

        if ($recenica =~ /$trazi/)
        {
            print("Našao sam $trazi u $recenica.\n");
        }
        else
        {
            print("Nisam pronašao.\n");
        }
        ```

### Substitucija teksta

- Substitucija teksta primjenom regularnog izraza nam mjenja samu vrijednost varijable nad kojom radimo substituciju.
- Substitucija se izvrši na prvom nizu kojeg pronađe regularni izraz ako ne zadamo drugačije određenim modifikatorom.
- Regularni izrazi substitucije su oblika `s/regex/sub/mod`, gdje su značenja pojedinih elemenata slijedeća:

    - znak `s` -- govori da se radi o substitucijskom regularnom izrazu, mora biti zapisan kao takav
    - `regex` -- niz specijalnih znakova koji nam predstavljaju regularni izraz
    - `sub` -- niz znakova u koji mjenjamo znakove koji podudaraju zadanom regularnom izrazu
    - `mod` -- može biti prazan ili sadržavati specijalne znakove koji mjenjaju ponašanje regularnog izraza, ti znakovi su:

        - `g` -- globalno, promjeni sve pojave zadanog regularnog izraza u nizu
        - `i` -- zanemari razliku između velikih i malih slova

- Primjer regularnog izraza koji mjenja sva pojavljivanja znakova `123` za znakove `456`:

    ``` perl
    $string = "abc123def123";
    $string =~ s/123/456/g;
    print($string);
    ```

- Primjer zamjene djela niza neovisno o razlici velikih i malih slova:

    ``` perl
    $string = "abc123def123";
    $string =~ s/Abc/def/gi;
    print($string);
    ```

!!! admonition "Zadatak"
    - Dana je vrijednost varijable `$var = "abc123abc";`
    - Koje su vrijednosti `$var` nakon slijedećih zamjena?
    - Provjeri pomoću `perl` skripte, zasebno za svaku zamjenu.

        - `$var =~ s/abc/def/;`
        - `$var =~ s/[a-z]+/X/g;`
        - `$var =~ s/B/W/i;`

### Prevođenje teksta

- Regularne izraze možemo koristiti i za specificiranje tablice prevođenja pojedinih znakova.
- Oblik regularnog izraza za prevođenje znakova je `tr/niz1/niz2/`, gdje `niz1` i `niz2` moraju imati jednak broj znakova.
- Prevođenje se izvodi na način da se svaki znak iz prvog niza prevede u znak koji je na istom mjestu ali u drugom nizu.

    ``` perl
    $string = "abcdefghicba";
    $string =~ tr/abc/123/;
    print($string);
    ```

!!! admonition "Zadatak"
    - Dana je vrijednost varijable `$var = "abc123abc";`
    - Koje su vrijednosti `$var` nakon slijedećih prijevoda?
    - Provjeri pomoću `perl` skripte, zasebno za svaki prijevod.

        - `$var =~ tr/a-z/A-Z/;`
        - `$var =~ tr/123/456/;`
        - `$var =~ tr/231/564/;`

## Dodatni zadaci

!!! admonition "Zadatak"
    - Pronađi grešku/e u slijedećem kodu:

        ``` perl
        #!/usr/bin/python
        $value = <STDIN>
        if ($value = 17)
        {
            print("Unio si broj 17.\n");
        else
        {
            print("Nisi unio broj 17.\n");
        ```

!!! admonition "Zadatak"
    - Napiši skriptu koja provjerava je li korisnik unio pravilan e-mail.
    - Pretpostaviti ćemo da je pravilan e-mail oblika `tekst.tekst@inf.uniri.hr`, gdje je `tekst` bilo koji niz znakova i znamenki.
    - Ispisati `e-mail ispravan` u slučaju da je, u suprotnom ispisati `neispravan e-mail`.
