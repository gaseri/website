---
author: Dino Gržinić, Vedran Rakuljić, Matea Turalija
---

# Booleova algebra

Booleova algebra, također poznata kao matematička logika, temeljni je dio računalne znanosti i elektrotehnike. Prvi ga je predstavio engleski matematičar [George Boole](https://en.wikipedia.org/wiki/George_Boole) sredinom 19. stoljeća. Boolea su zanimali temelji matematike te je razvio simbolički sustav za predstavljanje logičkih iskaza pomoću matematičke notacije. Objavio je svoje ideje u knjizi pod nazivom [The Laws of Thought](https://en.wikipedia.org/wiki/The_Laws_of_Thought) 1854. godine.

Značaj Booleove algebre u računalstvu leži u njezinoj sposobnosti predstavljanja binarnih podataka, što je srž obrade digitalnih informacija. U digitalnoj elektronici informacije su predstavljene s dva stanja: uključeno ili isključeno, istinito ili lažno, ili 1 ili 0. Booleova algebra pruža način za manipuliranje tim stanjima i njihovim kombinacijama pomoću logičkih operatora kao što su I, ILI i NE. Drugim riječima, ovi nam operatori omogućuju izvođenje operacija nad binarnim podacima.

U računalnom sklopovlju Booleova algebra koristi se za projektiranje logičkih sklopova i izgradnju digitalnih sustava. Predstavljanjem binarnih stanja sustava s logičkim izrazima, inženjeri mogu dizajnirati sklopove koji obavljaju specifične zadatke i koji rade pouzdano. Također se koristi u računalnom programiranju i dizajnu softvera, gdje daje osnovu za dizajniranje algoritama i procesa donošenja odluka.

Ako ste se ikada zapitali kako računala rade i kako mogu izvoditi složene izračune i operacije samo s jedinicama i nulama, tada je razumijevanje Booleove algebre ključno. Pa zaronimo i istražimo ovo zanimljivo područje matematičke logike!

## Osnovni pojmovi

U ovom poglavlju upoznat ćemo se s osnovnim pojmovima Booleove algebre poput logičkih izjava i varijabli te logičkih operatora koji su nam potrebni za bolje razumijevanje matematičke logike.

### Logičke izjave i varijable

Osnovni element matemtičke logike je **izjava** koja je ili istinita ili lažna. Drugim riječima, tvrdnja kojoj se ne može jednoznačno odrediti je li istinita ili lažna, nije izjava u smislu matematičke logike. Stoga, logičke izjave moraju biti jasne i nedvosmislene kako bi bile primjerene za korištenje u Booleovoj algebri, a izjave koje su subjektivne i ne mogu se provjeriti nazivamo nelogičnim izjavama.

Primjerice, izjava: *Tara je najljepša djevojka na svijetu* očito ne može biti izjava jer njezina istinitost ovisi o promatraču. Dok je izjava: *Elektron je elementarna čestica* logička izjava jer se nože utvrditi njezina točnost.

Za označavanje izjava koristimo se simbolima koje nazivamo logičkim **varijablama**. Vrijednost varijable može poprimati samo jednu od dvije moguće vrijednosti. Ako je varijabla istina označavamo je s T (engl. *True*) ili sa $1$, a ako je varijabla neistinita označavamo je s F (engl. *False*) ili sa $0$.

Stoga, logičkoj izjavi *Elektron je elementarna čestica* možemo dodjeliti varijablu $A$ i tada kraće zapisati kao $A = 1$.

### Logički operatori

Logičke izjave možemo povezati logičkim **operatorima** te time stvoriti logičke **izraze**. Razmotrit ćemo one osnovne logičke operacije koje su bitne u računalnoj primjeni, a s obzirom na prioritet izvođenja, su:

- negacija,
- konjunkcija,
- disjunkcija.

#### Negacija, NE (engl. *NOT*)

Negacija ili komplemet je logička operacija koja djeluje na jednu izjavu, čiji je rezultat uvijek suprotna vrijednost. Označava se s $¬A$ u logici, $\overline{A}$ u matematici, a riječima NOT ili znakom $!$ u programskim jezicima.

Djelovanje logičkih operacija vrlo često opisujemo tablicom kombinacija, odnosno istinitosti. Slijedi tablica istinitosti za operator negacije:

| $A$   | $\overline{A}$ |
| :---: | :------------: |
| 0 | 1 |
| 1 | 0 |

#### Konjukcija, I (engl. *AND*)

Konjunkcija je logička operacija koja djeluje na dvije ili više varijabli (izjava), a istinita je samo ako su sve izjave istinite. Simbol konjunkcije u logici je $\wedge$, u informatici znak $⋅$, a u programiranju `AND` ili `&&`. Tablica istinitosti je sljedeća:

| $A$   | $B$   | $A \cdot B$ |
| :---: | :---: | :---------: |
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

#### Disjunkcija, ILI (engl. *OR*)

Disjunkcija je logička operacija koja djeluje na dvije ili više varijabli (izjava), a istinita je kada je bar jedna izjava istinita. U logici je njezin simbol $\vee$, u informatici znak $+$, a u programskim jezicima `OR` odnosno `||`. Prema tome, tablica istinitosti za disjunkciju je:

| $A$   | $B$   | $A + B$ |
| :---: | :---: | :-----: |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

!!! question "Pitanja za ponavljanje"
    1. Koji je osnovni element u Booleovoj algebri?
    2. Koje je temeljno svojstvo logičke izjave?
    3. Smatra li se izjavom u logičkoj algebri rečenica: *Rijeka je najljepši grad na svijetu!* Obrazloži.
    4. Opiši osnovne logičke operacije.
    5. Što se prikazuje tablicom istinitosti?

## Složeni logički izrazi

Složene logičke izraze dobit ćemo kombinacijom osnovnih logičkih operacija. Tada u takvom izrazu treba pripaziti da operacije nemaju isti prioritet izvođenja jer na njihov redoslijed može se utjecati pomoću zagrada, kao i u aritmetičkim zadacima u matematici. Stoga, ako postoje zagrade, prvo se rješava izraz unutar njih, sljedeći prioritet ima negacija, zatim konjunkcija i na kraju disjunkcija.

!!! example "Zadatak"
    Putem tablice istinitosti provjerimo za koje vrijednosti $A$, $B$ i $C$ je izraz $\overline{A + B} \cdot \overline{C}$ istinit.

??? success "Rješenje"
    | $A$   | $B$   | $C$   | ${A + B}$ | $\overline{A + B}$ | $\overline{C}$ | $\overline{A + B} \cdot \overline{C}$ |
    | :---: | :---: | :---: | :-------: | :----------------: | :------------: | :-----------------------------------: |
    | 0 | 0 | 0 | 0 | 1 | 1 | **1** |
    | 0 | 0 | 1 | 0 | 1 | 0 | **0** |
    | 0 | 1 | 0 | 1 | 0 | 1 | **0** |
    | 0 | 1 | 1 | 1 | 0 | 0 | **0** |
    | 1 | 0 | 0 | 1 | 0 | 1 | **0** |
    | 1 | 0 | 1 | 1 | 0 | 0 | **0** |
    | 1 | 1 | 0 | 1 | 0 | 1 | **0** |
    | 1 | 1 | 1 | 1 | 0 | 0 | **0** |

    Iz tablice istinitosti vidimo da je izraz $\overline{A + B} \cdot \overline{C}$ istinit samo u slučaju kada su vrijednosti varijabli $A$, $B$ i $C$ jednake $0$.

!!! example "Zadatak"
    1. Napiši tablicu istinitosti za sljedeće logičke izraze: $A \cdot B + \overline{C} \cdot D$, $\overline{A \cdot B} + C$, $(A + B) + \overline{A} \cdot C$.

## Pretvaranje tablice istinitosti u logički izraz

Korištenjem disjunktivne ili konjunktivne normalne forme, iz tablice istinitosti možemo dobiti logički izraz. Uzmimo za primjer sljedeću tablicu na koju ćemo primjeniti normalne forme:

| $A$   | $B$   | $Y$   |
| :---: | :---: | :---: |
| 0 | 0 | **0** |
| 0 | 1 | **1** |
| 1 | 0 | **0** |
| 1 | 1 | **1** |

Možemo dobiti logički izraz koristeći oba normalna oblika, ali način na koji dolazimo do izraza je drugačiji i takvi dobiveni izrazi možda neće izgledati isto.

Dobiveni logički izrazi se često mogu i pojednostaviti, ali o tome ćemo pričati u poglavljima [Aksiomi i teoremi](#aksiomi-i-teoremi-booleove-algebre) i [Minimizacija](#minimizacija).

### Disjunktivna normalna forma

Skraćeno ju nazivamo i **DNF**. Ona započinje traženjem redaka u kojima je vrijednost varijable (u ovom slučaju stupac $Y$) jednak **1**. U gore navedenoj tablici to su 2. i 4. redak te ih promatramo zasebno:

| $A$   | $B$   |
| :---: | :---: |
| 0 | 1 |
| 1 | 1 |

Kod ove forme varijable povezujemo operatorom konjunkcije ($\cdot$), ali pritom moramo paziti da negiramo vrijenosti svake varijable čija je vrijednost u tom retku 0. To činimo za svaki redak, stoga sada imamo: $\overline{A} \cdot B$ za 2. redak i $A \cdot B$ za 4. redak.

Preostaje nam te retke povezati. To činimo operacijom konjunkcije ($+$). Sada dobivamo konačan logički izraz naše tablice: $Y = \overline{A} \cdot B + A \cdot B$

### Konjunktivna normalna forma

Skraćeno ju nazivamo i **KNF**. Ona je po principu rada slična disjunktivnoj normalnoj formi samo što drukčije koristimo vrijednosti i operatore iz tablice. Sada ćemo promatrati retke čija je vrijednost varijable $Y$ jednaka **0**. U našoj tablici to su 1. i 3. redak:

| $A$   | $B$   |
| :---: | :---: |
| 1 | 0 |
| 0 | 0 |

Kod ove forme varijable povezujemo operatorom disjunkcije ($+$), ali sada moramo paziti da negiramo vrijenosti svake varijable čija je vrijednost u tom retku 1. Tako imamo: $A + B$ za 1. redak i $\overline{A} + B$ za 3. redak.

U ovom slučaju retke povezujemo operatorom konjunkcije ($\cdot$) te dobijemo konačan rezultat: $Y = (A + B) \cdot (\overline{A} + B)$

Važno je naglasiti da je u KNF izraze pojedinih redaka nužno stavljati u zagrade kako bi se prije konjunkcije odvila operacija disjunkcije.

Ukoliko nam je zadana gotova tablica istinitosti i od nas se traži da odredimo njezin logički izraz možemo koristiti bilo koju od ove dvije normalne forme. Međutim, u tablici istinitosti može se dogoditi da se broj nula i jedinica u rezultatu razlikuje. Stoga će jedna forma biti mnogo učinkovitija od druge. Recimo da imamo tablicu s 4 varijable te kao vrijednosti varijable $Y$ imamo 12 jedinica i 4 nule, očigledno je kako će nam za njezino rješavanje KNF biti povoljnija. Isto vrijedi i obrnuto.

!!! example "Zadatak"
    1. Opiši postupak disjunktivne i konjunktivne normalne forme.
    2. Tablice istinitosti iz prethodnog zadatka pretvori u složene logičke izraze koristeći obje normalne forme.

## Aksiomi i teoremi Booleove algebre

Aksiome i teoreme Booleove algebre koristimo pri pojednostavljivanju složenih logičkih izraza kako bi oni bili čitljiviji i razumljiviji, ali i u konačnici kako bi računala učinkovitije radila s pojednostavljenim izrazima.

### Aksiomi

Aksiom ili postulat je temeljna istina koja se ne može dokazati i služi kao temelj matematičke ili logičke teorije, tj. smatra se pretpostavkom na kojoj se gradi teorija. U nastavku su prikazana četiri osnovna aksioma.

#### A1. Postojanje neutralnog elementa

$$A + 0 = A$$

$$A \cdot 1 = A$$

Kako bi nam bilo lakše razumjeti ova dva aksioma možemo izraditi njihove tablice istinitosti.

| $A$   | $0$   | $A + 0$ |
| :---: | :---: | :-----: |
| 0 | 0 | 0 |
| 1 | 0 | 1 |

Budući da je $0$ konstanta, u njezinom stupcu za svaki redak pišemo vrijednost 0. Možemo primjetiti kako će vrijednost izraza $A + 0$ uvijek imati identičan iznos kao i varijabla $A$. Stoga je $0$ u disjunkiciji neutralni element.

| $A$   | $1$   | $A \cdot 0$ |
| :---: | :---: | :---------: |
| 0 | 1 | 0 |
| 1 | 1 | 1 |

U ovom slučaju također možemo uočiti kako vrijednost izraza $A \cdot 0$ uvijek odgovara vrijednosti varijable $A$. Stoga je $1$ u konjunkciji neutralni element.

#### A2. Postojanje inverza ili komplementa

$$A + \overline{A} = 1$$

$$A \cdot \overline{A} = 0$$

Kao i kod prethodnog aksioma možemo napraviti tablice istinitosti.

| $A$   | $\overline{A}$ | $A + \overline{A}$ |
| :---: | :------------: | :----------------: |
| 0 | 1 | 1 |
| 1 | 0 | 1 |

Vrijednost ovog izraza uvijek će biti $1$ jer u svakom retku postoji barem jedna vrijednost $1$ zbog koje će disjunkcija poprimati vrijendost $1$.

| $A$   | $\overline{A}$ | $A \cdot \overline{A}$ |
| :---: | :------------: | :--------------------: |
| 0 | 1 | 0 |
| 1 | 0 | 0 |

Međutim, kod konjunkcije to znači da nikad obje varijable u jednom retku neće imati vrijednost $1$ zbog čega konjunkcija uvijek poprima vrijednost $0$.

#### A3. Komutativnost

$$A + B = B + A$$

$$A \cdot B = B \cdot A$$

Kao i u matematici tako i u Booleovoj algebri uočavamo svojstvo komutativnosti koje se odnosi na operacije zbrajanja (disjunkcija) i množenja (konjunkcija). Kojim god redoslijedom pisali elemente u takvim izrazima oni će uvijek imati istu tablicu istinitosti, odnosno bit će ekvivalentni.

#### A4. Distributivnost

$$A \cdot (B + C) = A \cdot B + A \cdot C$$

$$A + (B \cdot C) = (A + B) \cdot (A + C)$$

Kod distributivnosti element van zagrade će biti uparen sa svakim elementom u zagradi koristeći operator koji se nalazi u zagradi. Zatim su ta dva para povezana operatorom koji se nalazi van zagrade.

### Teoremi

Teorem ili poučak je iskaz u kojoj se utvrđuje da matematički pojam ima i druge karakteristike osim onih danih u definiciji tog pojma i ta se tvrdnja mora dokazati.

Na sve teoreme se odnosi **princip dualnosti**. On nam govori da ako u nekom teoremu zamjenimo $+$ sa $\cdot$ (i obrnuto) te vrijednosti 0 sa 1 (i obrnuto) dobit ćemo izraz koji je također teorem. Primjenu tog principa lako uočavamo u 1. teoremu.

U ovoj ćemo dokumentaciji spomenuti 8 osnovnih teorema koji će nam biti dovoljni za pojednostavljivanje većine logičkih izraza.

| Broj teorema | Ime teorema | 1. dio | 2. dio |
| :----------: | :---------: | :----: | :----: |
| T1 | Dominacija | $A + 1 = 1$ | $A \cdot 0 = 0$ |
| T2 | Idempotencija | $A + A = A$ | $A \cdot A = A$ |
| T3 | Involucija | $\overline{\overline{A}} = A$ | |
| T4 | Asocijacija | $(A + B) + C = A + (B + C)$ | $(A \cdot B) \cdot C = A \cdot (B \cdot C)$ |
| T5 | Apsorpcija | $A + A \cdot B = A$ | $A \cdot (A + B) = A$ |
| T6 | Simplifikacija | $A \cdot B + A \cdot \overline{B} = A$ | $(A + B) \cdot (A + \overline{B}) = A$ |
| T7 | de Morganov zakon | $\overline{A + B} = \overline{A} \cdot \overline{B}$ | $\overline{A \cdot B} = \overline{A} + \overline{B}$ |
| T8 | | $A + \overline{A} \cdot B = A + B$ | $A \cdot (\overline{A} + B) = A \cdot B$ |

!!! example "Zadatak"
    Dokažimo 2. teorem idempotencije.

??? success "Rješenje"
    1. $A + A = A$
    2. $A \cdot 1 + A \cdot 1 = A$ (1. aksiom)
    3. $A \cdot (1 + 1) = A$ (4. aksiom)
    4. $A \cdot 1 = A$ (jer je $1 + 1 = 1$)
    5. $A = A$ (1. aksiom)

!!! example "Zadatak"
    Dokažimo 6. teorem simplifikacije.

??? success "Rješenje"
    1. $A \cdot B + A \cdot \overline{B} = A$
    2. $A \cdot (B + \overline{B}) = A$ (4. aksiom)
    3. $A \cdot 1 = A$ (2. aksiom)
    4. $A = A$ (1. aksiom)

!!! question "Pitanja za ponavljanje"
    1. Za što su nam potrebni aksiomi i teoremi?
    2. Opiši princip dualnosti.

!!! example "Zadatak"
    Dokaži de Morganov zakon.

## Minimizacija

Pojednostavljenje složenih izraza primjenom [Aksioma](#aksiomi) i [Teorema](#teoremi) Booleove algebre nazivamo minimizacija. Prilikom njihovog korištenja u minimizaciji nije ih potrebno dokazivati.

!!! example "Zadatak"
    Minimizirajmo logički izraz iz poglavlja [Konjunktivna normalna forma](#konjunktivna-normalna-forma).

??? success "Rješenje"
    1. $Y = (A + B) \cdot (\overline{A} + B)$
    2. $Y = (B + A) \cdot (B + \overline{A})$ (3. aksiom)
    3. $Y = B$ (6. teorem)

Ovaj zadatak se mogao riješiti i korištenjem aksioma budući da su teoremi zasnovani na njima, no korištenjem teorema u manje koraka dođemo do istog rješenja.

!!! example "Zadatak"
    Minimizirajmo logički izraz $(\overline{A} + B) + \overline{A}$.

??? success "Rješenje"
    1. $(\overline{A} + B) + \overline{A}$
    2. $(\overline{A} + \overline{A}) + B$ (4. teorem)
    3. $\overline{A} + B$ (2. teorem)

!!! question "Pitanja za ponavljanje"
    1. Što je minimizacija?

!!! example "Zadatak"
    Minimiziraj sljedeći izraz: $\overline{A + B} \cdot (\overline{B} + A)$.
