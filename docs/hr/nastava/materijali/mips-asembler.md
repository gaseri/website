---
author: Vito Weiner, Betty Žigo, Matea Turalija
---

# MIPS asembler

[MIPS asemblerski jezik](https://en.wikipedia.org/wiki/MIPS_architecture) se jednostavno odnosi na asemblerski jezik MIPS procesora. Pojam MIPS akronim je za *Microprocessor without Interlocked Pipeline Stages*. To je arhitektura smanjenog skupa instrukcija, [RISC](https://en.wikipedia.org/wiki/Reduced_instruction_set_computer) (engl. *Reduced instruction set computer*) koju je razvila organizacija pod nazivom [MIPS Technologies](https://en.wikipedia.org/wiki/MIPS_Technologies).

Važno je poznavanje MIPS jezika jer se mnoga ugrađena sklopovlja i softveri izvode na MIPS procesoru. Poznavanje ovog jezika omogućuje bolje razumijevanje rada ovih sustava na niskoj razini. Pored toga, MIPS se često koristi u akademskom okruženju kako bi se studentima omogućilo razumijevanje koncepta niskorazinske programske izvedbe. Korištenje simulatora u tu svrhu omogućava praktičnu primjenu znanja i testiranje različitih programa napisanih u asemblerskom jeziku MIPS.

Prije početka izrade koda u MIPS jeziku, važno je nabaviti dobro integrirano razvojno okruženje koje olakšava kompajliranje i izvršavanje koda. Preporučuje se korištenje softvera [MARS](https://computerscience.missouristate.edu/mars-mips-simulator.htm) (engl. *MIPS Assembler i Runtime Simulator*). U nastavku se fokusiramo na primjenu asemblerskog jezika MIPS i upotrebu simulacijskog programa MARS za istraživanje načina pokretanja i izvođenja programa napisanih u MIPS asembleru.

## Uspostava razvojnog okruženja

[MARS](https://dpetersanderson.github.io/) simulator je softver koji simulira izvođenje programa napisanih u programskom jeziku MIPS. Omogućuje izradu i testiranje programa te sadrži program za ispravljanje pogrešaka (engl. *Debugger*) koji omogućuje korisnicima da detaljno prate izvođenje programa, otklanjaju greške i optimiziraju izvedbu. U nastavku ćemo dati upute za njegovu instalaciju i pojasniti njegovo korištenje.

### Instalacija i konfiguracija softvera MARS

Na stranici [MARS download](https://github.com/dpetersanderson/MARS) preuzmite simulator klikom na gumb `Download MARS` i pričekaje dok se program instalira.

MARS zahtijeva Java SDK instaliran na vašem računalu. Na istoj stranici kliknite na gumb `Download Java` koja će vas preusmjeriti na službenu stranicu [Oracle](https://www.oracle.com/java/technologies/downloads/), gdje ćete izabrati operacijski sustav koji koristite (Linux, macOS i Windows) i preuzeti najnovije izdanje Jave.

Nakon preuzimanja instalacijske datoteke, pronađite je u mapama preuzimanja na vašem računalu i pokrenite je. Zatim slijedite upute za instalaciju Java softvera kako biste dovršili proces instalacije.

### Rad u simulatoru MARS

Kada ste uspješno dovršili instalaciju, pokrenite simulator MARS klikom na datoteku `Mars.jar`.

Simulator MARS ima intuitivno sučelje koje se sastoji od nekoliko dijelova. Na desnoj strani se nalazi prozor s registrima, gdje su vidljive vrijednosti svih registara, uključujući registre opće i posebne namjene.

Procesor tijekom izvođenja programa uzima podatke i instrukcije iz radne memorije te obrađuje podatke pomoću instrukcija. Registri u procesoru omogućuju brz pristup podacima koji se obrađuju, a obrađeni podaci mogu se pohraniti u segment podataka u memoriju ili ostati u registrima procesora za daljnju obradu.

Kada započnete novi program (++control+n++), u središtu se nalazi prazan prozor *Edit* za uređivanje koda korisnika. U kartici *Execute*, pokraj prozora *Edit*, smještena je kontrola izvođenja programa koja je omogućena tek nakon uspješnog pokretanja programa. U njoj se nalazi podjela radne memorije na *Text Segment* i *Data Segment*. Text Segment sadrži instrukcije, adrese i kod u binarnom i asemblerskom obliku, stoga ga često nazivamo i *segment koda programa*. Data Segment je mjesto gdje se mogu vidjeti korisnički podaci, poput stogova pa ga nazivamo *segmentom podataka*.

Na dnu prozora se nalaze *MARS Messages*, gdje se prikazuju poruke i greške koje su se pojavile tijekom pokretanja programa, te *Run I/O* sučelje konzole, gdje se mogu pratiti ulazi i izlazi programa.

Uz ove dijelove, postoji i traka izbornika na vrhu prozora koja omogućava pristup raznim funkcijama i opcijama te ikonica upitnika za pomoć (engl. *Help*) koja prikazuje sve moguće instrukcije za MIPS asembler.

#### Izrada novog programa

Novi program možete izraditi odabirom *File* u traci izbornika koja će otvoriti padajući izbornik. Zatim izaberite *New File* kako biste stvorili prazni uređivački prozor u kojem možete pisati i uređivati MARS program.

Alternativno, možete stvoriti novu datoteku klikom na ikonu bijelog kvadratića (engl. *Create a new file for editing*) smještenu odmah ispod trake izbornika ili najjednostavnije, koristiti kombinaciju tipki ++control+n++.

#### Spremanje programa

Prije pokretanja programa koji ste napisali, važno ga je prvo spremiti. Da biste to učinili, prvo kliknite na izbornik *File*, a zatim odaberite opciju *Save* ili *Save as ...*. Također se možete koristiti kombinacijom tipki ++control+s++ za brzi pristup opciji spremanja.

Važno je napomenuti da je uobičajeni nastavak datoteke za asemblerski jezik `.s` ili `.asm` pa je preporučljivo koristiti jedan od tih nastavaka za spremanje vašeg programa.

#### Pokretanje programa

Kada je program spremljen, možete ga pokrenuti klikom na opciju *Run* u gornjoj traci. Otvorit će se padajući izbornik, gdje ćete odabrati *Assembly*, a zatim kliknuti na *Go* u istom izborniku.

Alternativno možete kliknuti na ikonicu dvaju ključića (engl. *Assemble the current file and clear breakpoints*) ispod trake izbornika i pričekati da se uvjerite kako vaš program nema grešaka. Potom kliknite na ikonicu desno od nje (engl. *Run the current program*) za pokretanje programa. Ukoliko je sve u redu u izlaznom prozoru na dnu ekrana će se prikazati izlaz programa.

Ukoliko želite otvoriti već postojeći asembler program spremljenog na vašem računalu, jednostavno odaberite opciju *Open* u izborniku *File* na izbornoj traci, a zatim odaberite program koji želite otvoriti.

#### Zatvaranje programa

Kada ste završili s programom i želite započeti s pisanjem novog programa, prvo trebate spremiti trenutni program. Zatim u padajućem izborniku pod *File* odaberite *Close* ili kombinacijom tipki ++control+w++ zatvorite program.

##### Segment koda programa

Segment koda programa (engl. *Text Segment*) prikazuje detalje o asemblerskim instrukcijama programa prikazanih u tablici sa stupcima *Bkpt*, *Address*, *Code*, *Basic* i *Source*.

Prvi stupac tablice se naziva *Bkpt* i koristi se za postavljanje točaka prekida (engl. *Breakpoint*) u programu. Postavljanjem kvačice na određenu instrukciju, program će se zaustaviti pri izvođenju te instrukcije, omogućavajući nam pregledavanje vrijednosti registara i memorije u tom trenutku.

U stupcu *Address* vidimo adresu u radnoj memoriji gdje je pohranjena svaka pojedina instrukcija programa. U stupcu *Source* se prikazuju dijelovi instrukcija koje su napisane od strane korisnika, dok stupac *Basic* prikazuje osnovne instrukcije asemblerskog jezika koje su sastavni dijelovi tih instrukcija. U stupcu *Code* vidimo strojni izgled instrukcija u heksadecimalnom zapisu.

##### Segment podataka

Segment podataka (engl. *Data Segment*) u programu sadrži podatke koje program koristi za obradu. U tablici su prikazane adrese na kojima su ti podaci pohranjeni te sami podaci pohranjeni u obliku bajtova. U prvom stupcu tablice nalazi se adresa na kojoj je pohranjen pojedini podatak, dok se u ostalim stupcima nalaze vrijednosti pohranjenih bajtova, pri čemu se svaki sljedeći bajt nalazi na adresi koja je za četiri veća od prethodne. Svi podaci su prikazani u heksadecimalnom zapisu radi bolje čitljivosti.

##### Praćenje podataka u registrima

Na desnoj strani ekrana vidimo tablicu s registrima i podacima koji su pohranjeni u njima nakon izvođenja programa.

U gornjem dijelu ekrana, ispod trake izbornika, nalazi se opcija za podešavanje brzine izvođenja programa u instrukcijama po sekundi. Ako postavimo sporu brzinu izvođenja programa i pokrenemo ga, možemo pratiti promjene u registrima dok se instrukcije izvršavaju jedna po jedna. Trenutna izvršena instrukcija prikazana je žutom bojom u segmentu programa, a zelenom bojom su označene promjene u registrima, što olakšava praćenje izvođenja programa.

Također, postoje ikonice za korak nazad (engl. *Undo the last step*) i korak naprijed (engl. *Run one step at a time*) kako bismo detaljnije mogli pratiti izvođenje programa i vrijednosti u registrima za svaki korak.

## Struktura MIPS programa

Struktura asemblerskog programa prikazanog ispod, općenito je podijeljena u dva glavna segmenta: data i text segment koje se deklariraju pomoću asemblerskih direktiva `.data` i `.text`.

**Asemblerske direktive** su upute koje omogućuju procesoru da pravilno obradi različite dijelove koda. One se obično navode ispred dijelova koda kojima se odnose, a počinju s točkom. Primjerice, `.data`, `.asciiz`, `.text` i `.globl` su samo neke od često korištenih asemblerskih direktiva.

``` asm
        .data        # deklaracija data segmenta (statički podaci)
labela:
        .asciiz "Hello World!"
        .text        # deklaracija text segmenta (kôd programa)
        .globl main  # početak programa, treba biti globalan
main:
                     # korisnički programski kôd
```

Na početku programa deklariramo segmente. Koristimo `.data` segment za pohranu podataka za obradu, što u ovom slučaju predstavlja statičke podatke. Podaci se inicijaliziraju na zadane vrijednosti prije izvršavanja programa i pohranjuju se u memoriju. Kada se program izvršava, podaci se mogu čitati i mijenjati prema potrebi. Primjerice, ovaj dio koda može se koristiti za ispisivanje stringa na ekran, u ovom slučaju "Hello World!".

Oznaka ili labela `labela:` označava memorijsku adresu na kojoj se nalazi podatak, u ovom primjeru u segmentu podataka ili označava memorijsku adresu na koju se odnosi određena instrukcija u programskom kodu.

Za pohranu znakovnih nizova koristimo asemblerske direktive `.ascii` ili `.asciiz`. Ovim se rezervira potreban broj bajtova za spremanje znakovnog niza te ga se postavlja na zadanu vrijednost. Za deklariranje 16-bitnih podataka koristi se direktiva `.half`. Može se navesti jedan ili više podataka koji se razdvajaju zarezima: `.half 3, 10, -25.` Za deklariranje cijelih 32-bitnih podataka koristimo direktivnu `.word`.

Deklaracija segmenata programa podrazumijeva direktivu `.text` u koju se pohranjuju instrukcije, odnosno programski kod. Nakon toga slijedi `.global main` što označava da je `main` globalna funkcija, tj. početak programa. Zatim definiramo oznaku `main:` koja predstavlja početak našeg programa i krećemo s programiranjem.

### Sistemski pozivi

Sistemski poziv asemblerska je naredba kojom se poziva operacijski sustav (OS) kako bi se izvršila operacija koja nije dostupna unutar programa. Sistemski poziv se poziva naredbom `syscall`. Kada se ova naredba pozove, program šalje zahtjev operacijskom sustavu koji zatim izvršava zahtjev i vraća rezultat programu.

Osnovni sistemski pozivi obuhvaćaju funkcionalnosti kao što su ispisivanje na ekran, učitavanje podataka s tipkovnice, alokacija memorije i slično.

U asembleru, `syscall` se sastoji od spremanja određenog broja u registar `$v0`, što predstavlja kôd specifične usluge koja se poziva. Primjerice instrukcija `li $v0, 10` sprema kôd `10` u registar `$v0`. Nakon toga poziva se instrukcija `syscall` koja prenosi kontrolu operacijskom sustavu za izlaz iz programa (engl. *Exit*).

U nastavku je prikaz usluga sistemskih poziva i odgovarajućih kodova za svaku od njih:

| Syscall usluga | Kod u `$v0` | Argumenti | Rezultat |
| ---------------| ----------- | --------- | -------- |
| print_int | `1` | `$a0` – cijeli broj | Ispisuje se sadržaj `$a0` kao cjelobrojni broj. |
| print_float | `2` | `f12` – realni broj | Ispisuje se sadržaj `$f12` kao realni broj. |
| print_double | `3` | `$f12` – realni broj | Ispisuje se sadržaj `$f12` kao realni broj. |
| print_string | `4` | `$a0` – adresa niza | Ispisuje se niz znakova, početak niza je na adresi pohranjenoj u `$a0`. |
| read_int | `5` | | Učitava se cijeli broj i pohranjuje u `$v0`. |
| read_float | `6` | | Učitava se ralni broj i pohranjuje u `$f0`. |
| read_double | `7` | | Učitava se ralni broj i pohranjuje u `$f0`. |
| read_string | `8` | `$a0` – adresa međuspremnika, `$a1` – veličina međuspremnika | Učitava se niz znakova i sprema od početne adrese (`$a0`). |
| sbrk | `9` | `$a0` – broj bajtova | Dinamički se alocira (`$a0`) bajtova. Početna adresa smješta se u `$v0`. |
| exit | `10` | | Uredan izlaz iz programa. |

### Registri

Većina naredbi u MIPS arhitekturi zahtijeva operande koji se nalaze u registrima. Također se rezultati pohranjuju u registre. Zbog toga je važno pobliže razmotriti vrste registara u ovoj arhitekturi.

Svi registri su duljine 32 bita i mogu biti opće ili posebne namjene. Imamo 32 registra opće namjene koji su označeni brojevima od $0$ do $31$ ili njihovim imenima, vidi tablicu. Registri posebne namjene su primjerice programsko brojilo (engl. *Program Counter*, skraćeno PC) koji pokazuje adresu sljedeće instrukcije koju treba izvršiti. Registri `lo` (engl. *Low*) i `hi` (engl. *High*) u MIPS procesoru su također registri posebne namjene koji se koriste za pohranu rezultata operacija množenja i dijeljenja.

| Ime | Broj | Namjena |
| --- | ---- | ------- |
| `zero` | `0` | Konstanta `0`. |
| `at` | `1` | Rezervirano za asembler. |
| `v0, v1` | `2`, `3` | Rezultati potprograma. |
| `a0`–`a3` | `4`–`7` | Argumenti funkcije. |
| `t0`–`t7` | `8`–`15` | Privremeni. Ne čuvaju se kod poziva potprograma. |
| `s0`–`s7` | `16`–`23` | Spremljeni privremeni. Čuvaju se kod poziva potptograma. |
| `t8`, `t9` | `24`, `25` | Privremeni. Ne čuvaju se. |
| `k0`, `k1` | `26`, `27` | Rezervirani za jezgru operacijskog sustava. |
| `gp` | `28` | Pokazivač na globalno područje |
| `sp` | `29` | Pokazivač stoga |
| `fp` | `30` | Pokazivač okvira, ako je potreban. |
| `ra` | `31` | Povratna adresa kod izlaza iz potprograma |

Prvi registar opće namjene je `$zero` koji u sebi sadrži konstantnu vrijednost $0$ i nemoguće ju je promijeniti. Iako se na prvi pogled čini kao suvišan, međutim vrlo često se koristi kada je potrebna konstanta 0 ili kada treba odbaciti rezultat neke operacije.

Registar `$at` se koristi kao registar privremenih rezultata. Namijenjen je za pohranu vrijednosti koje se koriste samo u tijeku izvođenja nekih instrukcija i koje se ne koriste u daljnjem izvođenju programa. To omogućuje programerima da koriste druge registre opće namjene i izbjegnu moguće probleme prilikom izvršavanja instrukcija koje koriste te registre. Jedna od najčešćih primjena registra `$at` je kod pseudo-instrukcije `li` (engl. *Load Immediate*), koja služi za učitavanje neposredne vrijednosti u registar. Ova instrukcija ne postoji u strojnom jeziku MIPS arhitekture, nego se kombinira korištenjem instrukcija `addiu` i `lui`. Budući da ove instrukcije zahtijevaju registar kao izvor, vrijednost koja se učitava mora se privremeno spremiti u registar `$at` kako bi se mogla koristiti u izvođenju instrukcije.

Registri `$v0` i `$v1` služe za pohranu povratnih vrijednosti iz funkcija dok se sljedeća četiri registra od `$a0` do `$a3` koriste za pohranu argumenata funkcije.

Registri od `$t0` do `$t9` su privremeni jer se koriste za čuvanje međurezultata programa. Nasuprot tome, registri od `$s0` do `$s7` su spremljeni privremeni koji se koriste za pohranu vrijednosti koje je potrebno sačuvati između poziva funkcija.

Registri `$k0` i `$k1` su rezervirani za jezgru operacijskog sustava. Tu su i registri pokazivača kao što je `$gp` pokazivač na globalno područje, `$sp` pokazovač stoga i `$fp` pokazivač okvira stoga. Posljednji registar je `$ra` koji se koristi za pohranu adrese koja služi za povratak iz funkcije u glavni program.

Optimizacija koda u asembleru je vrlo bitna, a korištenje registara može znatno povećati brzinu izvršavanja programa. Naime, pristupanje registrima je puno brže od pristupanja memoriji. No, budući da je broj registara ograničen, važno je pažljivo birati koji se registri koriste i za što se koriste u svakom dijelu programa kako bi se postigla maksimalna efikasnost. Stoga je poznavanje dostupnih registara i njihove namjene od ključne važnosti za uspješnu optimizaciju koda u asembleru.

### Moj prvi "Hello World" program

!!! example "Zadatak"
    Napišite program koji će ispisati Hello World! na ekran. Isprobajte program u simulatoru MARS.

??? success "Rješenje"
    ``` asm
    ## Program ispisuje pozdrav na ekran.

        .data                   # deklaracija segmenta podataka
    pozdrav:
        .asciiz "Hello World!"  # niz bajtova koji odgovara ASCII kodu teksta Hello World! sprema se na adresu s labelom pozdrav
        .text                   # deklaracija text segmenta
        .globl main
    main:
        la    $a0, pozdrav      # učita adresu labele pozdrav u registar $a0
        li    $v0, 4            # postavljanje koda (4) sustava za ispisivanje teksta
        syscall                 # poziv sustava za ispisivanje teksta
        li    $v0, 10           # postavljanje koda (10) sustava za izlazak iz programa
        syscall                 # poziv sustava za izlazak iz programa
    ```

    Program koristi `.data` i `.text` segmente, gdje se u `.data` deklarira oznaka `pozdrav:` koja sadrži string "Hello World!". U ovom djelu program sprema se niz bajtova u `pozdrav:` koji odgovaraju [ASCII](https://hr.wikipedia.org/wiki/ASCII) kodu teksta "Hello World!".

    U `.text` segmentu upisuju se instrukcije programa. Prvo se deklarira glavni program direktivom `.global main` i oznaka početka main funkcije `main:`. Za ispis stringa na ekran, prvo ga je potrebno spremiti u registar `$a0` i to činimo sljedećom instrukcijom: `la $a0, pozdrav`.

    Kako bismo ispisali string na ekran koristit ćemo sistemski poziv. Prvo spremimo kod `4` u registar `$v0` instrukcijom: `li $v0,4` i pozovemo `syscall`. Tim instrukcijama pozivamo operacijski sustav da ispiše string na ekran.

    Za završetak programa koristimo instrukciju `li $v0, 10` i `syscall` te je ovime program gotov.

## Primjeri programa u MIPS asembleru

!!! example "Zadatak"
    Napišite program koji će ispisati cjelobrojnu vrijednost 5 na ekran. Isprobajte program u simulatoru MARS i proučite sadržaj registara.

??? success "Rješenje"
    ``` asm
        .text
        .global main
    main:
        li    $a0, 5    # učitavanje cjelobrojne vrijednosti 5 u registar $a0
        li    $v0, 1    # postavljanje koda (1) sustava za ispisivanje cjelobrojne vrijednosti
        syscall
        li    $v0, 10   # exit
        syscall
    ```

    Za ispisivanje cjelobrojne vrijednosti $5$ na ekran nije nam potreban segment podataka. Stoga krećemo odmah s deklaracijom segmenta programa. Vrijednost broja $5$ spremili smo instrukcijom `li $a0, 5` u registar `$a0`. Važno je napomenuti da poziv funkcije `syscall` za ispisivanje na standardni izlaz zahtijeva da se vrijednost koju želimo ispisati nalazi u registru `$a0`, baš kao i kod ispisa "Hello World!" na ekran. Stoga, prije nego što se pozove ovaj sistemski poziv, vrijednost koju želimo ispisati treba biti pohranjena u registar `$a0` kako bi se ispis mogao izvršiti ispravno. Zatim slijede instrukcije za ispis na ekran i kraj programa.

!!! example "Zadatak"
    Napišite program koji zbraja brojeve 2 i 7 i ispisuje rezultat. Isprobajte program u simulatoru MARS i proučite sadržaj registara.

??? success "Rješenje"
    ``` asm
    # Program koji zbraja brojeve 2 i 7, sprema rezultat u registar $t0 i ispisuje rezultat.
        .text
        .globl main
    main:
        li    $a0, 2        # spremanje cjelobrojne vrijednosti 2 u registar $a0
        li    $a1, 7        # spremanje cjelobrojne vrijednosti 7 u registar $a1
        add   $t0, $a0, $a1 # zbroj dvaju brojeva i spremanje u registar $t0
        move  $a0, $t0      # kopiranje sadržaja registra $t0 u $a0, pripremanje registra $a0 za ispis zbroja
        li    $v0, 1        # ispis na ekran
        syscall
        li    $v0, 10       # exit
        syscall
    ```

    Prije same aritmetičke operacije zbrajanja prvo moramo pohraniti vrijednosti 2 i 7 u neke registre. U ovom primjeru to su registri `$a0` i `$a1`. Zatim slijedi instrukcija zbrajanja `add` koja zbroj brojeva pohranjenih u registrima `$a0` i `$a1` sprema u registar `$t0`.

    U idućem koraku ne smijemo zaboraviti pripremiti registar `a0` za ispis. Iskoristit ćemo instrukciju micanja podataka `move`, kako bi premjestili vrijednost iz registra `t0` u registar `a0`. Ovo je neizbježno učiniti jer bi se u suprotnom umjesto zbroja ispisala vrijednost $2$.

    Alternativno, mogli smo napisati i `add $a0, $a0, $a1` gdje bi se zbroj odmah pohranio u registar `$a0` i na taj način smanljili liniju koda programa. Potom slijede instrukcije za ispis na ekran i izlaz iz programa.

!!! example "Zadatak"
    Napišite program koji učitava dva cijela broja i ispisuje njihov zbroj. Isprobajte program u simulatoru MARS i proučite sadržaj registara.

??? success "Rješenje"
    ``` asm
    # Program učitava dva cijela broja i ispisuje njihov zbroj

        .text
        .globl main
    main:
        li      $v0, 5        # postavljanje koda (5) sustava za unos cjelobrojne vrijednosti prvog broja
        syscall
        move    $t0, $v0      # spremamo prvi broj u registar $t0
        li      $v0, 5        # učitavanje drugog broja
        syscall

        add     $t1, $v0, $t0 # zbroj
        move    $a0, $t1      # priprema za ispis zbroja
        li      $v0, 1        # ispis
        syscall
        li      $v0, 10       # exit
        syscall
    ```

    U ovom zadatku, program treba omogućiti korisniku unos dva cijela broja putem sučelja te ih zbrojiti i ispisati na ekran.

    Za učitavanje cijelog broja koristimo sistemski poziv s kodom $5$, a unesena vrijednost se sprema u registar `$v0`. Kako bismo zadržali ovu vrijednost za daljnje korištenje, moramo ju premjestiti u neki drugi registar, npr. `$t0`. Ako to ne bismo učinili, pri novom učitavanju broja, prethodno unesena vrijednost bi se izgubila.

    Zatim učitavamo novu vrijednost i ponavljamo postupak. Nakon što su sve vrijednosti učitane i pohranjene u registre možemo ih zbrojiti. U registar `$t1` pohranjen je zbroj vrijednosti koje su pohranjene u registrima `$v0` i `$t0`.

    Nakon zbrajanja, opet koristimo instrukciju `move` kojom ćemo zbroj pohranjen u registru `$t1` premjestiti u `$a0` i ispišemo rezultat na ekran. Program je nakon ispisa gotov, stoga ga je potrebno uredno završiti instrukcijama za izlaz iz programa.

!!! example "Zadatak"
    Napišite program koji traži od korisnika unos visine s koje tijelo pada te izračunava kvadratnu brzinu slobodnog pada tijela s te visine. Isprobajte program u simulatoru MARS i postavite visinu na 5 m. Rješenje je tada $v^2 = 100\text{ m}^2/\text{s}^2$.

??? success "Rješenje"
    ``` asm
    # Program izracunava kvadratnu brzinu slobodnog pada tijela

        .data
    visina:
        .asciiz "Upiši visinu na kojoj se tijelo nalazi u metrima: "
    odgovor:
        .asciiz "Kvadratna brzina tijela pri padu je: "
    konstanta:
        .word 10    # ubrzanje sile teže, g = 10 m/s^2

        .text
        .globl main
    main:
        la $a0, visina     # spremanje adrese teksta u registar $a0
        li $v0, 4          # ispis teksta na ekran
        syscall
        li $v0, 5          # učitavanje visine tijela
        syscall

        # Izračun kvadrata brzine tijela pri padu
        lw $t0, konstanta    # spremanje broja 10 u registar $t0
        li $t1, 2            # učitavanje broja 2 u registar $t1
        mul $t2, $v0, $t1    # t2=visina*2
        mul $t0, $t2, $t0    # t0=t2*10 (g=10 m/s^2)

        # Ispis rezultata na ekran
        la $a0, odgovor
        li $v0, 4
        syscall
        move $a0, $t0
        li $v0, 1
        syscall

        li $v0, 10            # exit
        syscall
    ```

    Slobodni pad opisuje jednoliko ubrzano kretanje tijela pod utjecajem Zemljine privlačne sile, poznate i kao sila teža. Ubrzanje tijela u slobodnom padu iznosi približno $9,81\text{ m/s}^2$, što uzrokuje da se brzina pada povećava dok tijelo prelazi sve veće puteve:

    $$
    v=\sqrt{2gh}
    $$

    Radi jednostavnosti u ovom zadatku računamo kvadratnu brzinu:

    $$v^2 = 2 \cdot g \cdot h,$$

    pri čemu $v$ označava brzinu, $h$ visinu na kojoj se tijelo nalazi, a $g$ je ubrzanje sile teže kojeg smo zaokružili na $10\text{ m/s}^2$.

    Program započinjemo deklaracijom segmenta podataka u kojeg ćemo pohraniti adrese na kojima se tekst s uputama za korisnika nalazi. Koristimo oznaku `visina:` u koju pohranjujemo adresu teksta "Upiši visinu na kojoj se tijelo nalazi u metrima: ". Zatim oznaka `odgovor:` ima adresu teksta "Kvadratna brzina tijela pri padu je: ". Na kraju je definirana oznaka `konstanta:` s adresom do pohranjene vrijednosti 10 u memoriji i predstavlja ubrzanje sile teže. Direktiva `.word` rezervira točno 4 bajta u radnoj memoriji, koliko je i potrebno za pohranu cjelobrojne vrijednosti.

    Nadalje, definiramo text segment, a ispod njega glavni program. Prvo moramo ispisati uputu korisniku koja se nalazi na adresi koju predstavlja oznaka `visina:`. Nakon ispisa uputa korisniku slijedi unos visine tijela.

    Korisnik u ovom djelu unosi visinu koja se prema formuli množi s `2`, kojeg smo prethodno spremili u registar `$t1` i konstantom koja iznosi `10`. Vrijednost 10 je pohranjena u radnoj memoriji stoga ju je potrebno pohraniti u neki rgistar, u ovom primjeru to je registar `$t0` i to činimo instrukcijom `lw` (engl. *Load Word*). Nakon toga slijede instrukcije umnoška i spremanje konačnog rezultata u registar `$t0`.

    Prije ispisa rezultata na ekran slijedi ispis teksta oznake `odgovor:`, a tek onda i sam rezultat množenja. Nakon toga uredno završavamo program naredbom za izlaz.

!!! example "Zadatak"
    Modificirajte prethodni zadatak tako da uzmete za ubrzanje sile teže konstantu $9,81\text{ m/s}^2$, umjesto $10\text {m/s}^2$. Isprobajte program u simulatoru MARS.

??? success "Rješenje"
    ``` asm
        .data
    visina:
        .asciiz "Upiši visinu na kojoj se tijelo nalazi u metrima: "
    odgovor:
        .asciiz "Kvadratna brzina tijela pri padu je: "
    konstanta:
        .float 9.81
    broj:
        .float 2

        .text
        .globl main
    main:
        # ispis uputa korisniku i unos visine
        la $a0, visina
        li $v0, 4
        syscall
        li $v0, 6
        syscall

        # učitavanje brojeva u registre
        l.s $f1, broj
        l.s $f2, konstanta

        # instrukcije množenja
        mul.s $f0, $f0, $f1    # f0 = visina*2
        mul.s $f12, $f0, $f2   # f12 = f0*9,81

        # ispis rezultata na ekran
        la $a0, odgovor
        li $v0, 4     # ispis uputa korisniku na ekran
        syscall
        li $v0, 2     # ispis rezultata
        syscall
        li $v0, 10    # exit
        syscall
    ```

    Obzirom da ćemo u programu koristiti float vrijednosti brojeva, moramo koristiti posebne float registre za njihovu pohranu. MIPS ne podržava množenje različitih tipova podataka pa ako želimo množiti konstantu tipa float sa cjelobrojnim brojem 2, moramo 2 spremiti kao float vrijednost.

    Nakon prelaska na rad s float vrijednostima, morali smo izmijeniti neke instrukcije. Na primjer, umjesto uobičajene instrukcije za učitavanje vrijednosti u registar sada koristimo instrukciju `l.s`. Također, instrukcija za množenje se sada naziva `mul.s`.

    Valja napomenuti da instrukcija za unos float vrijednosti ne sprema vrijednost u registar `$v0` već se za unos float vrijednosti koristi registar `$f0`, dok se za ispis vrijednosti onda koristi registar `$f12`. Na kraju ispisa konačnog rješenja uredno završavamo program naredbom za izlaz.

!!! example "Zadatak"
    1. Napiši program koji računa hipotezu trokuta Pitagorinim poučkom $c^2 = a^2 + b^2$. Neka je zadan pravokutan trokut s katetama $a = 5$ i $b = 2$. Program na kraju ispisuje vrijednost stranice $c$. Program provjerite simulatorom MARS.
    2. Napiši program koji će izračunati kinetičku energiju $E_k = \frac{m \cdot v^2}{2}$. Program pita korisnika da unese masu u kg i brzinu u m/s te ispisuje konačnu vrijednost kinetičke energije. Program provjerite simulatorom MARS.
    3. Napiši program koji će izračunati hidrostatski tlak u vodi $p = \rho \cdot g \cdot h$. Gustoća vode iznosi $\rho = 1000\text{ kg/m}^3$, ubrzanje sile teže uzmite da je $g = 9,81\text{ m/s}^2$, a dubinu vode $h$ unosi korisnik. Program provjerite simulatorom MARS.
    4. Napišite program koji računa brzinu vertikalnog hitca $v^2 = v_0^2 + 2 \cdot g \cdot h$. Neka je početna brzina $3\text{ m/s}$, a visinu unosi korisnik. Na kraju programa ispisuje se rezultat brzine. Program provjerite simulatorom MARS.
