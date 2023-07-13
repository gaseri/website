---
autor: Petar Car, Goran Marković, Matea Turalija
---

# MIPS: grananje i petlje

Granjanje i petlje su temeljni koncepti u programiranju koji omogućavaju programima da se izvršavaju na različite načine i prilagođavaju uvjetima izvođenja. Granjanje omogućuje programima donošenje odluka na temelju uvjeta, dok petlje omogućuju ponavljanje određenih blokova koda. U nastavku ćemo se baviti ovim konceptima i njihovom primjenom u programiranju.

## Tijek izvođenja programa

Tijek izvođenja programa određuje redoslijed izvršavanja instrukcija i možemo ga podijeliti na tri osnovna tipa:

- slijedno izvođenje instrukcija,
- uvjetno grananje (*if-else*),
- petlje (*for*, *while*, *do-while*).

*Slijedno izvođenje instrukcija* podrazumijeva da se svaka instrukcija izvršava jedna za drugom, redom kako su napisane. *Uvjetno grananje* omogućava programu da preskoči određene dijelove koda u slučaju kada je odgovarajući uvjet zadovoljen, dok *petlje* omogućavaju ponavljanje određenog bloka više puta. Ovi tipovi tijeka izvođenja programa daju programerima mogućnost prilagodbe izvršavanja programa različitim situacijama i uvjetima.

### Slijedno izvođenje instrukcija

Slijedno izvođenje instrukcija predstavlja osnovni način izvršavanja programa, gdje se naredbe izvršavaju jedna za drugom. Ovaj pristup osigurava predvidljivo i logično funkcioniranje programa. Svaka instrukcija se čita i izvršava bez preskakanja ili ponavljanja dijelova koda. Nakon što se jedna instrukcija dovrši, program nastavlja s izvršavanjem sljedeće instrukcije u memoriji.

Pogledajmo primjer programa u C++ koji zbraja dva broja. Prvo se deklarira varijabla `a`, zatim varijabla `b` i `c` u koju se sprema njihov zbroj. Na kraju program završava. U ovom primjeru naredbe se izvršavaju jedna za drugom bez preskakanja ili ponavljanja dijelova koda.

``` cpp
int main() {
    int a = 5;
    int b = 10;
    int c = a + b;
    return 0;
}
```

### Uvjetno grananje

Uvjetno grananje je koncept koji omogućava programima da donose odluke na temelju određenih uvjeta. Kroz uvjetnu izjavu, program provjerava zadovoljava li određeni kriterij, poput usporedbe vrijednosti ili provjere logičkog izraza. Na temelju rezultata provjere, program odabire odgovarajuću putanju izvođenja, omogućavajući fleksibilnost i prilagodljivost u programiranju.

Pogledajmo sljedeći primjer programa u C++ koji provjerava je li uvjet `a < b` istinit:

``` cpp
int main() {
    int a = 5;
    int b = 10;
    if (a < b) {
        cout << "a je manje od b" << endl;
    } else {
        cout << "b je manje ili jednako od a" << endl;
    }
    return 0;
}
```

U ovom primjeru koristimo *if-else* naredbu za ispisivanje različitih poruka ovisno o odnosu između dvije varijable, `a` i `b`. Ako je uvjet `a < b` ispunjen, ispisuje se prva poruka, dok se u suprotnom ispisuje druga poruka. Ovaj koncept omogućava programu da dinamički odabere između različitih mogućnosti na temelju određenih uvjeta.

### Petlje

Petlje (eng. *loop*) su ključni koncept u programiranju koji omogućava izvršavanje određenog bloka naredbi više puta. One se koriste kada želimo ponavljati određeni zadatak sve dok je određeni uvjet ispunjen.

U nastavku ćemo detaljnije objasniti različite vrste petlji, kao što su petlja *for*, *while* i *do-while*, zajedno s primjerima njihove upotrebe u programskom jeziku C++.

#### Petlja *for*

Petlja *for* omogućava ponavljanje određenog bloka naredbi kada je unaprijed poznat broj ponavljanja, u našem primjeru to je 10. Sastoji se od tri dijela: inicijalizacije `int i = 1`, uvjeta `i <= 10` i ažuriranja `i++`. Inicijalizacija se koristi za postavljanje početne vrijednosti kontrolne varijable, uvjet određuje kada će se petlja prestati izvršavati, a ažuriranje se koristi za promjenu vrijednosti kontrolne varijable nakon svake iteracije petlje.

``` cpp
int main() {
    for (int i = 1; i <= 10; i++) {
        cout << i << endl;
    }
    return 0;
}
```

Pogledajmo primjer programa koji ispisuje brojeve od 1 do 10. Inicijalizacija postavlja početnu vrijednost varijable `i` na 1, uvjet petlje provjerava je li `i` manje ili jednako od 10, dok izraz inkrementacije povećava vrijednost `i` za 1 pri svakoj iteraciji petlje.

Kada se program izvrši, petlja će ispisati brojeve od 1 do 10, svaki u zasebnom redu na konzoli. U ovom primjeru možemo primijetiti da nam je unaprijed poznat broj ponavljanja petlje (10). Međutim, postoje situacije u kojima nam broj ponavljanja nije poznat unaprijed. U tim slučajevima koristimo petlje koje se izvršavaju sve dok je određeni uvjet ispunjen. Pogledajmo sada primjere programa koji koriste takve petlje.

#### Petlja *shile*

Petlja *while* se koristi kada želimo ponavljati određeni zadatak sve dok je uvjet ispunjen. Primjerice, program traži od korisnika da unese točnu zaporku. Dakle, program neće nastaviti dalje s izvođenjem dok korisnik ne unese točnu zaporku. Uvjet se provjerava prije svake iteracije petlje i ako je uvjet istinit izvršava se blok naredbi. Ako uvjet nije zadovoljen, petlja se prekida i izvršenje programa se nastavlja izvan petlje.

``` cpp
int main() {
    int i = 1;
    while (i <= 10) {
        cout << i << endl;
        i++;
    }
    return 0;
}
```

Ovaj program će također ispisati brojeve od 1 do 10. Petlja će se izvršavati sve dok je vrijednost varijable `i` manja ili jednaka 10. Nakon svake iteracije petlje provjerava se uvjet `i <= 10`, a ako je istinit, petlja se ponovno izvršava. Kada vrijednost `i` postane veća od 10, uvjet više nije istinit te se petlja prekida. Tada se izvršenje programa nastavlja izvan tijela petlje.

#### Petlja *do-while*

Petlja *do-while* slična je petlji *while*, ali s jednom ključnom razlikom. U petlji *do-while*, uvjet se provjerava nakon izvršavanja tijela petlje, što znači da će se blok naredbi uvijek izvršiti barem jednom, čak i ako uvjet na početku nije ispunjen. Nakon izvršavanja bloka narebi provjerava se uvjet. Ako je uvjet ispunjen, petlja se nastavlja iznova, a ako uvjet nije ispunjen, petlja se prekida i izvršenje programa nastavlja izvan tijela petlje.

Ova vrsta petlje korisna je kada želimo osigurati izvršavanje određenog bloka naredbi barem jednom, bez obzira na početnu vrijednost uvjeta.

Kao i u prethodnim primjerima koje smo vidjeli pogledajmo ovaj za ispis brojeva od 1 do 10:

``` cpp
int main() {
    int i = 1;
    do {
        cout << i << endl;
        i++;
    }
    while (i <= 10);
    return 0;
}
```

Na početku varijabla `i` se inicijalizira na vrijednost 1. Nakon toga se izvršava `do-while` petlja. Unutar petlje, trenutna vrijednost varijable `i` se ispisuje na ekran. Potom se vrijednost varijable `i` povećava za 1 pomoću operatora inkrementa `++`.

Nakon izvršenja tijela petlje, provjerava se uvjet `(i <= 10)`. Ako je uvjet ispunjen, petlja se ponavlja i izvršava se ponovno tijelo petlje. Ovaj korak se ponavlja sve dok je vrijednost varijable `i` manja ili jednaka 10. Kada vrijednost varijable `i` postane veća od 10, petlja se prekida, a izvršenje programa nastavlja se izvan tijela petlje.

## MIPS: grananje

Programi se obično izvršavaju slijedno, slijedeći redoslijed instrukcija. Međutim, ponekad želimo da program donese odluke i promijeni svoje ponašanje ovisno o određenim uvjetima. U takvim situacijama koristimo grananje.

Jedan od najčešćih načina grananja je korištenje *if-else* naredbi. Unutar *if* bloka, određeni blok koda izvršava se samo ako je uvjet istinit. U slučaju da uvjet nije istinit, program preskače taj blok i prelazi na sljedeći dio koda u *else* bloku.

### Jednostavno grananje naredbom *if* u MIPS-u

Proučimo primjer jednostavnog grananja u C++ koji ispisuje apsolutnu vrijednost varijable `a`:

``` cpp
if (a < 0) {
    a = -a;
}
cout << a;
```

Kako bismo preveli uvjetno grananje u MIPS asemblerski jezik, ovom primjeru možemo pristupiti na dva načina:

- 1\. način: okrenuti uvjet naredbe *if* (`a >= 0`)
- 2\. način: korištenje više oznaka i naredbi grananja

#### Okrenuti uvjet naredbe *if* u MIPS-u

Za početak, pretpostavimo da je varijabla `a` pohranjena u registru `$a0`. U ovom slučaju, koristit ćemo instrukcije uvjetnog grananja. Uvjet *if* naredbe je `a < 0`. Kako bismo okrenuli uvjet i provjerili je li `a >= 0`, koristit ćemo instrukciju `bgez` (engl. _**b**ranch if **g**reater than or **e**qual to **z**ero_) koja uspoređuje sadržaj registra `$a0` s nulom.

Ako je sadržaj veći ili jednak nuli, program će skočiti na određenu oznaku, u ovom primjeru to je oznaka `granaj`. U suprotnom, izvršit će se sljedeća instrukcija `sub a0, $0, $a0` koja oduzima sadržaj registra `$a0` od nule i rezultat pohranjuje natrag u registar `$a0`. Time se postiže efekt apsolutne vrijednosti varijable `a`.

``` asm
    bgez  $a0, granaj   # ako je $a0 >= 0 skače na oznaku granaj
    sub   $a0, $0, $a0
granaj:
    li    $v0, 1        # ispis varijable a
    syscall
    li    $v0, 10       # izlaz
    syscall
```

Nakon izvršavanja instrukcija unutar `if` bloka ili preskakanja bloka, program nastavlja s izvršavanjem instrukcija koje se nalaze nakon `if` naredbe. U ovom slučaju to su naredbe za ispis na konzolu i izlaz iz programa.

#### Korištenje više oznaka i naredbi grananja

U ovom slučaju razmatramo uvjet `a < 0` koristeći instrukciju `bltz` (engl. _**b**ranch if **l**ess **t**han **z**ero_) koja provjerava je li vrijednost registra `$a0` manja od nule. Ako je, program će skočiti na oznaku `granaj` i pohraniti apsolutnu vrijednost varijable `a` natrag u registar `$a0`.

``` asm
    bltz   $a0, granaj   # ako je a0<0 skače na granaj
    j      nastavak      # bezuvjetno grananje; skače na nastavak
granaj:
    sub    $a0, $0, $a0
nastavak:
    li     $v0, 1        # ispis varijable a
    syscall
    li     $v0, 10       # izlaz
    syscall
```

U slučaju kada uvjet nije ispunjen, nastavlja se izvršavanje sljedećih instrukcija. Prva sljedeća instrukcija je *bezuvjetno grananje* na oznaku `nastavak` iza koje se izvršavaju instrukcije ispisa i izlaz iz programa.

Primjetimo kako smo u ovom slučaju dobili više oznaka i naredbi grananja nego u prvom slučaju.

!!! admonition "Zadatak"
    Prevedite sljedeći dio koda napisanog u C++ u asemblerski jezik arhitekture MIPS. Pretpostavite da su varijable `a0`, `v0`, `t0`, `t1` i `t2` u istoimenim registrima.

    ``` cpp
    if (a0 > 0) {
        v0 = t0;
    }
    else if (a == 0){
        v0 = t1;
    }
    else{
        v0 = t2;
    }
    ```

**Rješenje:**

Ovaj zadatak riješit ćemo na drugi način koristeći više oznaka i narebi grananja. Kako bismo provjerili uvjet `a0 > 0` koristimo instrukciju `bgtz` (engl. _**b**ranch if **g**reater **t**han **z**ero_). Ukoliko je uvjet ispunjen program skače na oznaku `blok1`. U tom bloku, vrijednost registra `$t0` se kopira u registar `$v0` pomoću instrukcije `move`. Nakon toga program skače na oznaku `dalje` i izvršavaju se naredbe izvan bloka *if* naredbe.

``` asm
    bltz   $a0, blok1
    beqz   $a0, blok2
blok_else:
    move   $v0, $t2
    b      dalje
blok1:
    move   $v0, $t0
    b      dalje
blok2:
    move   $v0, $t1
dalje:
    #...
```

Ako uvjet `a0 > 0` nije ispunjen, program provjerava sljedeći uvjet `a0 == 0`. U slučaju da je uvjet istinit, program skače na blok instrukcija označen kao `blok2`. U tom bloku, vrijednost registra `$t1` se kopira u registar `$v0`, a zatim program skače na oznaku `dalje`.

Ako nijedan od prethodnih uvjeta nije ispunjen, izvršava se blok `blok_else`. U tom bloku, vrijednost registra `$t2` se kopira u registar `$v0`. Na kraju, program skače na oznaku `dalje` pri čemu se izvršavaju naredbe koje slijede izvan bloka *if* naredbe.

!!! admonition "Zadatak"
    Sljedeći isječak koda u jeziku C++ pretvori u asembler za arhitekturu MIPS. Pretpostavi da su varijable `a0` i `v0` u istoimenim registrima.

    ``` cpp
    if (a0 > 0) {
        v0 = a0 - 3;
    }
    else {
        v0 = 0;
    }
    cout << v0;
    ```

#### Instrukcije uvjetnog grananja

| Naziv | Instrukcija | Opis |
| ----- | ----------- | ---- |
| Bezuvjetno grananje | `b lab` | Bezuvjetno grananje (skok) na oznaku `lab` |
| Granaj ako je jednako | `beq Rsrc1, Rsrc2, lab` | Skače na oznaku `lab` ako je sadržaj registra `Rsrc1 = Rsrc2` |
| Granaj ako je jednako nuli | `beqz Rsrc1, lab` | Skače na oznaku `lab` ako je sadržaj registra `Rsrc1 = 0` |
| Granaj ako je veće | `bgt Rsrc1, Rsrc2, lab` | Skače na oznaku `lab` ako je sadržaj registra `Rsrc1 > Rsrc2` |
| Granaj ako je veće od 0 | `bgtz Rsrc1, lab` | Skače na oznaku `lab` ako je sadržaj registra `Rsrc1 > 0` |
| Granaj ako je veće ili jednako | `bge Rsrc1, Rsrc2, lab` | Skače na oznaku `lab` ako je sadržaj registra `Rsrc1 >= Rsrc2` |
| Granaj ako je veće ili jednako nuli | `bgez Rsrc1, lab` | Skače na oznaku `lab` ako je sadržaj registra `Rsrc1 >= 0` |
| Granaj ako je manje | `blt Rsrc1, Rsrc2, lab` | Skače na oznaku `lab` ako je sadržaj registra `Rsrc1 < Rsrc2` |
| Granaj ako je manje od 0 | `bltz Rsrc1, lab` | Skače na oznaku `lab` ako je sadržaj registra `Rsrc1 < 0` |
| Granaj ako je manje ili jednako | `ble Rsrc1, Rsrc2, lab` | Skače na oznaku `lab` ako je sadržaj registra `Rsrc1 <= Rsrc2` |
| Granaj ako je manje ili jednako nuli | `blez Rsrc1, lab` | Skače na oznaku `lab` ako je sadržaj registra `Rsrc1 <= 0` |
| Granaj ako nije jednako | `bne Rsrc1, Rsrc2, lab` | Skače na oznaku `lab` ako je sadržaj registra `Rsrc1 != Rsrc2` |
| Granaj ako nije jednako 0 | `bnez Rsrc1, lab` | Skače na oznaku `lab` ako je sadržaj registra `Rsrc1 != 0` |

### Složeni uvjeti grananja naredbom *if* u MIPS-u

U MIPS asemblerskom jeziku, tzv. [SET instrukcije](#instrukcije-usporedbe) omogućuju izračunavanje složenih uvjeta na temelju usporedbe vrijednosti registara. SET instrukcije imaju sličan format kao i uvjetno grananje, koristeći mnemonik "s" s dodanim uvjetom.

Primjerice, instrukcija `slt $Rdest, $Rsrc1, $Rsrc2` (engl. _**s**et if **l**ess **t**hen_) uspoređuje vrijednosti registara `$Rsrc1` i `$Rsrc2` te rezultat (0 ili 1) pohranjuje u registr `$Rdest`. U ovom slučaju 1 pohranjuje u registar `$Rdest` ako je `Rsrc1 < $Rsrc2`.

!!! admonition "Zadatak"
    Sljedeći isječak koda u jeziku C++ pretvori u MIPS asemblerski jezik. Pretpostavi da je varijabla `a0` u istoimenom registru.

    ``` cpp
    if (a0 < 0 || a0 > 100) {
        a0 = 1;
    }
    else {
        a0++;
    }
    ```

**Rješenje:**

Uvjet grananja možemo unaprijed izračunati. Tada možemo koristiti naredbe uvjetnog grananja kao da je u C++ ovakav kod:

``` cpp
bool t = a0 < 0 || a0 > 100;
if (t != 0) {
    a0 = 1;
}
else {
    a0++;
}
```

Uvjet grananja pohranjujemo u varijablu `t` koja je tipa `bool`. Sada je varijabla `t` postala rezultat ispitivanja istinitosti uvjeta `a0 < 0 || a0 > 100`. Vrijednost koja će se pohraniti u varijablu `t` biti će ili `true` ili `false`, ovisno o istinitosti uvjeta.

Uvjet koji se provjerava je `t != 0`, što znači da će se blok koda *if* naredbe izvršiti ako je vrijednost varijable `t` različita od nule, odnosno ako je uvjet istinit.

U suprotnom, ako uvjet nije ispunjen (vrijednost varijable `t` je nula) izvršit će se blok koda unutar `else` dijela.

Prevođenjem ovoga primjera u asemblerski, dobiti ćemo kod programa koji će izgledati ovako:

``` asm
    sltz   $t1, $a0          # t1 = a0 < 0
    li     $t2, 100          # t2 = 100
    sgt    $t0, $a0, $t2     # t0 = a0 > 100
    or     $t0, $t0, $t1     # t0 = t0 || t1
    bnez   $t0, prvislucaj   # t0 ≠ 0
    addi   $a0, $a0, 1       # a0 = a0 + 1 (a0++)
    j      dalje
prvislucaj:
    li     $a0, 1            # a0 = 1
dalje:
    #...
```

U ovome primjeru, instrukcija `sltz` (engl. _**s**et if **l**ess **t**han **z**ero_) provjerava je li sadržaj registra `$a0` manji od nule i rezultat pohranjuje u registar `$t1`. Sljedeća instrukcija `li` postavlja vrijednost 100 u registar `$t2`.

Instrukcija `sgt` (engl. _**s**et if **g**reater **t**hen_) uspoređuje sadržaj registra `$a0` s vrijednošću u registru `$t2`. Ako je sadržaj registra `$a0` veći od vrijednosti u registru `$t2`, rezultat će biti 1, inače je 0. Rezultat usporedbe pohranjuje u registar `$t0`.

Sljedeća instrukcija `or` izvodi logičku operaciju ILI (engl. *OR*) između sadržaja registara `$t0` i `$t1`. Rezultat te operacije pohranjuje se natrag u registar `$t0`. Time se postiže da je vrijednost registra `$t0` različita od 0 samo ako je jedan od uvjeta (`$a0 < 0` ili `$a0 > 100`) ispunjen.

Instrukcija `bnez` (engl. _**b**ranch if **n**ot **e**qual to **z**ero_) provjerava sadržaj registra `$t0` ako je različit od nule. U slučaju kada je `$t0` različit od nule, skočit će na oznaku `prvislucaj` i spremiti 1 u registar `$a0`.

U drugom slučaju, kada je `$t0` jednak nuli, izvršava se instrukcija `addi` koja pridodaje 1 vrijednosti u registru `$a0`. To znači da se vrijednost u registru `$a0` povećala za 1. Nakon toga, instrukcija bezuvjetnog grananja `j` izvodi skok na oznaku `dalje` i time se izvršavanje nastavlja od prve sljedeće naredbe koja se nalazi nakon oznake.

!!! admonition "Zadatak"
    Sljedeći isječak koda u jeziku C++ pretvori u MIPS asemblerski jezik. Pretpostavi da su varijable `a0` i `a1` u istoimenim registrima.

    ``` cpp
    if (a1 > 50 || a0 > 0) {
        a0 = -a0;
    }
    else {
        a1 = 2 * a1;
    }
    ```

!!! admonition "Zadatak"
    Napiši kod u asembleru MIPS koji odgovara isječku koda u C++. Program računa sumu prvih deset prirodnih brojeva. Pretpostavi da su varijable `a` i `i` u registrima `a0` i `t0`.

    ``` cpp
    int a = 0;
    for (int i = 1; i <= 10; i++) {
        a += i;
    }
    cout << a;
    ```

#### Instrukcije usporedbe

| Instrukcija | Sintaksa | Opis |
| ----------- | -------- | ---- |
| `seq` | `seq Rdest, Rsrc1, Rsrc2` | Postavlja rezultat 1 u registar `Rdest` ako je `Rsrc1 = Rsrc2`, inače u 0 |
| `sne` | `sne Rdest, Rsrc1, Rsrc2` | Postavlja rezultat 1 u registar `Rdest` ako je `Rsrc1 != Rsrc2`, inače u 0 |
| `sgt(u)` | `sgt(u) Rdest, Rsrc1, Rsrc2` | Postavlja rezultat 1 u registar `Rdest` ako je `Rsrc1 > Rsrc2`, inače u 0 |
| `sge(u)` | `sge(u) Rdest, Rsrc1, Rsrc2` | Postavlja rezultat 1 u registar `Rdest` ako je `Rsrc1 >= Rsrc2`, inače u 0 |
| `slt(u)` | `slt(u) Rdest, Rsrc1, Rsrc2` | Postavlja rezultat 1 u registar `Rdest` ako je `Rsrc1 < Rsrc2`, inače u 0 |
| `slti(u)` | `slti(u) Rdest, Rsrc1, Imm` | Postavlja rezultat 1 u registar `Rdest` ako je `Rsrc1 < Imm`, inače u 0 |
| `sle(u)` | `sle(u) Rdest, Rsrc1, Rsrc2` | Postavlja rezultat 1 u registar `Rdest` ako je `Rsrc1 <= Rsrc2`, inače u 0 |

## MIPS: petlje

### Petlja *do-while* u MIPS-u

Najjednostavniji slučaj je petlja *do-while*, gdje uvjet petlje odgovara uvjetu grananja. U slučajevima petlji *while* i *for* koristimo sličan koncept kao i kod naredbe *if*, s okrenutim uvjetom ili više oznaka i skokova. Ako je uvjet ispunjen, tijelo petlje će se izvršiti. Za razliku od naredbe *if*, kod petlji imamo bezuvjetni skok na početak, odnosno provjeru uvjeta.

!!! admonition "Zadatak"
    Sljedeći isječak koda u jeziku C++ pretvori u MIPS asemblerski jezik. Pretpostavi da je varijabla `a0` u istoimenom registru.

    ```cpp
    int a0;
    do {
        cin >> a0;
    } while (a0 < 0)
        
    cout >> a0;
    ```

**Rješenje:**

Program traži od korisnika unos broja, sve dok je taj broj manji od 0. Započnimo postavljanjem broja 5 u registar `$v0` kako bismo odabrali odgovarajući sistemski poziv za unos broja. Nakon toga, premjestimo vrijednost iz registra `$v0`, koju je korisnik unio, u registar `$a0` kako bismo je spremili za daljnje korištenje, odnosno ispis.

``` asm
pocetak:
    li     $v0, 5        # unos broja
    syscall
    move   $a0, $v0      # a0 = v0
    bltz   $a0, pocetak  # skoči na oznaku pocetak ako je $a0 < 0
    li     $v0, 1        # ispis 
    syscall
```

Koristit ćemo instrukciju `bltz` (engl. _**b**ranch if **l**ess **t**han **z**ero_) kako bismo usporedili vrijednost u registru `$a0` s nulom, `$a0 < 0`. Ako je ta vrijednost manja od nule, skočit ćemo na oznaku `pocetak` kako bismo ponovno zatražili unos broja od korisnika.

U slučaju da uvjet nije zadovoljen, postavit ćemo broj 1 u registar `$v0` kako bismo odabrali odgovarajući sistemski poziv za ispis broja.

### Petlja *while* u MIPS-u

!!! admonition "Zadatak"
    Sljedeći isječak koda u jeziku C++ pretvori u MIPS asemblerski jezik. Pretpostavi da su varijable `a0` i `t0` u istoimenim registrima.

    ``` cpp
    int a0;
    int t0 = 0;
    cin >> a0;
    while (a0 > 2) {
        t0++;
        a0 = a0 - 2;
    }
    ```

**Rješenje:**

Ako je `a > 2` program se nastavlja od tijela petlje, inače skače na naredbu iza petlje:

``` asm
    li    $t0, 0
    li    $v0, 5
    syscall
    move  $a0, $v0
    li    $t1, 2
uvjet:
    bgt   $a0, $t1, petlja
    j     dalje
petlja:
    addi  $t0, $t0, 1
    sub   $a0, $a0, $t1
    j     uvjet
dalje:
```

U drugom slučaju koristit ćemo okrenut uvjet. Kada je `a ≤ 2` preskačemo tijelo petlje:

``` asm
    li   $t0, 0
    li   $v0, 5
    syscall
    move $a0, $v0
    li   $t1, 2
uvjet:
    ble  $a0, $t1, dalje
    addi $t0, $t0, 1
    sub  $a0, $a0, $t1
    j    uvjet
dalje:
```

### Petlja *for* u MIPS-u

!!! admonition "Zadatak"
    Sljedeći isječak koda u jeziku C++ pretvori u MIPS asemblerski jezik. Pretpostavi da su varijable `a0` i `t0` u istoimenim registrima.

``` cpp
for (int a0 = 1; a0 <= t0; a0++) {
    t0 = t0*a0;
}
```

**Rješenje:**

Petlja *for* je ekvivalentna petlji *while*, ako ju zapišemo na ovaj način:

``` cpp
int a0 = 1;
while (a0 <= t0) {
    t = t0*a0;
    a0++;
}
```

Stoga, rješenje možemo zapisati na ovaj način:

``` asm
    li    $a0, 1
uvjet:
    bgt   $a0, $t0, dalje
    mul   $t0, $t0, $a0
    addi  $a0, $a0, 1
    j     uvjet
dalje:
```

### Složena petlja *for* u MIPS-u

!!! admonition "Zadatak"
    Napiši program koji računa zbroj komponenti vektora $v$, $s = \sum_{i=0}^{n-1} v_i$.

**Rješenje:**

Pretpostavimo da je varijabla `s` u registru `$s0`, `i` u registru `$t0`, `v` u registru`v0` i varjabla `n` u registru `$t1`.

Varijabla `v` je pokazivač na polje, sadrži adresu prvog člana polja.

``` cpp
int s = 0;
for (int i = 0; i < n; i++) {
    s = s + v[i];
}
```

Prvo pretvorimo *for* petlju u *while* petlju:

```cpp
int s = 0;
int i = 0;
while (i < n) {
    s = s + v[i];
    i++;
}
```

Novost je da imamo unutar petlje operator uglatih zagrada `[ ]` za pristupanje elementima niza iz memorije na odgovarajućoj adresi. Za pristup određenom elementu niza, izračunavamo adresu elementa koristeći početnu adresu niza `v` i pomak od $\text{i} \cdot \text{sizeof(int)}$. Ovdje se $\text{sizeof(int)}$ koristi za određivanje veličine jednog elementa niza u bajtovima. Drugim riječima. ako imamo `v[3]`, to znači da želimo pristupiti četvrtom elementu niza `v`. Adresa tog elementa je $\text{v} + 3 \cdot \text{sizeof(int)}$.

Nakon dohvaćanja vrijednosti elementa `v[i]`, dodajemo je trenutnoj vrijednosti varijable `s` kako bismo sumirali sve elemente niza. To se postiže izrazom `s = s + v[i]`. Nakon svake iteracije petlje, vrijednost `i` se povećava za 1, što omogućava prolazak kroz sve elemente niza. Na kraju izvođenja petlje, varijabla `s` će sadržavati zbroj svih elemenata niza.

Sada ćemo raspisat elemente polja kako bi nam lakše bilo pretvoriti u asmembler:

```cpp
int s = 0;
int i = 0;
int *v1 = v;      // inicijalizacija pokazivača v1 na početak niza v
for (i < n) {
    s = s + *v1;  // dodavanje vrijednosti koju pokazuje v1 na varijablu s
    v1++;         // pomicanje pokazivača v1 na sljedeći element niza
    i++;
}
```

Ono sto u C++ znači uvećavanje za 1, to u asmembleru, posebno u ovom kontekstu, znači uvećavanje za veličinu tipa podatka na kojeg pointer pokazuje. Ovdje imamo pokazivač na integer, a to znači da će adresa na koju pokazuje biti povećana za veličinu tipa podataka integer, što je u ovom slučaju 4 bajta.

Prilikom uvećavanja pokazivača, asembler neće automatski voditi računa o veličini podataka, već je na programeru da osigura pravilno uvećanje adrese za ispravan pristup elementima niza.

```mips
    li   $s0, 0          # int s = 0
    li   $t0, 0          # int i = 0
    move $v1, $v0        # v1 = v
uvjet:
    bge  $t0, $t1, dalje
    lw   $t3, 0($v1)     # tmp = *v1
    add  $s0, $s0, $t3   # s = s + tmp
    addi $v1, $v1, 4     # v1++ zbraja 4
    addi $t0, $t0, 1     # i++
    j    uvjet
    dalje:
```

Program koristi petlju za prolazak kroz sve komponente vektora `v`. Na početku se registri `$s0` i `$t0` inicijaliziraju na 0, što predstavlja početne vrijednosti za varijable `s` i `i`. Također, sadržaj registra `$v0` se prenosi u registar `$v1` kako bi se sačuvala adresa vektora `v`.

Unutar petlje se koristi instrukcija `bge` koja provjerava uvjet `i >= n`. Ako je uvjet ispunjen, program skoči na oznaku `dalje` i izlazi iz petlje. U suprotnom, izvršavaju se instrukcije unutar petlje.

Unutar petlje se koristi instrukcija `lw` (engl. *load word*) za učitavanje vrijednosti koju pokazuje registar `$v1` u registar `$t3`. Ta se vrijednost zatim dodaje na trenutni zbroj u registru `$s0` pomoću `add`. Adresa vektora `v` se povećava za 4 bajta, a brojač `i` za 1 pomoću instrukcije `addi`. Na kraju se skoči na oznaku `uvjet` kako bi se ponovo provjerio uvjet i izvršile instrukcije u petlji.

Nakon završetka petlje, program nastavlja izvršavati instrukcije koje slijede nakon petlje, označene oznakom `dalje`.
