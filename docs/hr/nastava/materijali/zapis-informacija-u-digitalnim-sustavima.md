---
author: Alen Hamzić, Darian Žeko, Matea Turalija
---

# Zapis informacija u digitalnim sustavima

Računalna arhitektura bavi se dizajnom i organizacijom računalnih sustava, uključujući njihove hardverske i softverske komponente, radi postizanja specifičnih performansi i funkcionalnih ciljeva. Uključuje teme kao što su arhitektura procesora, hijerarhija memorije, ulazno-izlazni sustavi i analiza performansi. Cilj proučavanja arhitekture računala je razumjeti kako računalni sustavi rade i kako dizajnirati učinkovite i djelotvorne računalne sustave da zadovolje specifične potrebe. To uključuje razumijevanje načina na koji različite komponente računalnog sustava rade zajedno i kako se mogu optimizirati za performanse, potrošnju energije i cijenu. Osim toga, arhitektura računala pruža temelj za računalno inženjerstvo, računalne znanosti i druga srodna područja te je osnovno znanje za svakog tko je uključen u dizajn, razvoj ili analizu računalnih sustava.

Tablica 1.: Razine apstrakcije za elektronički računalni sustav.

| Razina apstrakcije | Čimbenici razine |
| ------------------ | ---------------- |
| Aplikacijski softver | Računalni programi |
| Operacijski sustav | Upravljački programi |
| Arhitektura | Instrukcije, registri|
| Mikro arhitektura| Put podataka, upravljači |
| Logika | Adrese, memorija |
| Digitalni sklopovi | logički sklopovi |
| Analogni sklopovi | Pojačivači, filteri |
| Uređaji | Tranzistori, diode |
| Fizika | Ponašanje elektrona |

Tablica 1. prikazuje razine apstrakcije za elektronički računalni sustav, zajedno s tipičnim sastavnim elementima na svakoj razini. Na najnižoj razini apstrakcije je fizika, gdje je ponašanje elektrona opisano [kvantnom mehanikom](https://en.wikipedia.org/wiki/Quantum_mechanics) i [Maxwellovim jednadžbama](https://en.wikipedia.org/wiki/Maxwell%27s_equations). Elektronički uređaji kao što su [tranzistori](https://en.wikipedia.org/wiki/Transistor) zatim se koriste za izgradnju analognih i digitalnih sklopova, koji se pak kombiniraju da bi se stvorile složenije strukture poput procesora i memorijskih sustava.

Mikroarhitektura povezuje logiku i arhitektonsku razinu apstrakcije. Arhitektonska razina apstrakcije opisuje računalo iz programerove perspektive. Na primjer, arhitektura [Intel x86](https://en.wikipedia.org/wiki/X86) koju koristi većina osobnih računala definirana je skupom instrukcija i registara koje programer smije koristiti. Mikroarhitektura uključuje kombiniranje logičkih elemenata za izvršavanje instrukcija definiranih arhitekturom.

Prelazeći u područje softvera, operacijski sustav obrađuje detalje niske razine kao što je pristup tvrdom disku ili upravljanje memorijom, dok aplikacijski softver koristi te mogućnosti za rješavanje problema za korisnika.

Ovaj kolegij općenito pokriva širok raspon tema o arhitekturi i organizaciji računala, uključujući digitalne logičke sklopove, arhitekturu procesora, memorijske sustave, ulazno/izlazne uređaje i analizu performansi računala. Po završetku kolegija studenti će moći analizirati i procijeniti performanse računala, razumjeti različite arhitekture RISC i CISC procesora, napisati jednostavne programe u asemblerskom jeziku te prilagoditi programska rješenja karakteristikama funkcionalnih komponenti računala. Cilj kolegija je omogućiti studentima duboko razumijevanje načina na koji računala rade na niskoj razini i međusobno djelovanje različitih razina apstrakcije u računalnom sustavu kako bi stvorile funkcionalno i učinkovito računalno okruženje.

U ovom odjeljku razmatramo različite vrste brojevnih sustava i načine njihovog zapisivanja u tim sustavima. Brojevni sustavi su se razvijali kroz povijest, od [egipatskog](https://en.wikipedia.org/wiki/Egyptian_numerals), [babilonskog](https://en.wikipedia.org/wiki/Babylonian_cuneiform_numerals), [rimskog](https://en.wikipedia.org/wiki/Roman_numerals) i drugih, do najraširenijeg sustava u svijetu, [arapskog](https://en.wikipedia.org/wiki/Arabic_numerals), koji se temelji na decimalnom sustavu. Međutim, kada je došlo vrijeme za predstavljanje informacija u digitalnom svijetu, postavilo se pitanje kako te informacije zapisati u tom sustavu. Bez sumnje, binarni sustav se pokazao najprikladnijim za kodiranje informacija u digitalnom sustavu jer omogućuje jednostavno i pouzdano čitanje i pisanje podataka, uz minimalnu vjerojatnost grešaka.

## Brojevni sustavi

Brojevni sustav način je zapisivanja brojeva i njihovih tumačenja. U svakodnevnom životu naviknuti smo na dekadski brojevni sustav, no u digitalnim sustavima binarni i heksadekadski sustav češto su pogodniji za korištenje. Za početak objasnimo podjelu sustava na **pozicijske** i **nepozicijske** brojevne sustave.

Nepozicijski brojevni sustavi prikazuju brojeve znamenkama kojima vrijednost ne ovisi o položaju u zapisu broja. Najjedonstavniji primjer nepozicijskog brojevnog sustava je [rimski brojevni sustav](https://en.wikipedia.org/wiki/Roman_numerals). Za primjer možemo uzeti broj $XXX$ koji se sastoji od znamenke $X$ čija je vrijednost $10$ na svakoj poziciji u zapisu broja. Tek njihovim zbrajanjem znamenaka dobivamo konačnu vrijednost broja za navedeni primjer, $X + X + X = 30$. Kako bi se smanjila kompleksnost ovoga sustava izostavljena je oznaka za nulu, međutim kod obavljanja različitih aritmetičkih operacija ovakav sustav je izrazito složen.

Pozicijski brojevni sustav način je zapisivanja brojeva gdje vrijednost svake znamenke u zapisu ovisi o svom položaju. Najčešći primjer je decimalni brojevni sustav gdje svaka znamenka može poprimiti vrijednost od $0$ do $9$, ovisno o svom položaju u zapisu broja. Primjerice, broj $1991$ se sastoji od znamenki $1$ i $9$. Vrijednost prve znamenke $1$ zapravo se odnosi na vrijednost $1000$, dok posljednja znamenka $1$ označava jedinice. Slično tome, vrijednost prve znamenke $9$ tako iznosi $900$, a druge $90$. Dakle, u pozicijskom brojevnom sustavu vrijednost svake znamenke određena je njezinim položajem u zapisu.

U nastavku slijedi pregled najčešće korištenih brojevnih sustava: binarnog, oktalnog, dekadskog i heksadekadskog sustava. Svaki od ovih sustava ima svoje karakteristike i primjene, a poznavanje njihovog rada ključno je za razumijevanje računalnih sustava i tehnologija.

!!! example "Zadatak"
    1. Što je brojevni sustav?
    2. Objasni razliku između pozicijskih i nepozicijskih brojevnih sustava. Navedi primjere.

### Dekadski sustav

Dekadski sustav brojeva najčešće je korišteni sustav u svakodnevnom životu. Sastoji se od $10$ brojeva u rasponu od $0$ do $9$, stoga mu je brojevna baza $10$. Svi brojevi u ovom sustavu se prikazuju kao kombinacija ovih znamenki, gdje svaka znamenka ima težinsku vrijednost koja se određuje njezinim položajem u zapisu broja. Primjerice, pedeset i sedam piše se kombinacijom znamenaka $5$ i $7$:

| Stotisućice | Desettisućice | Tisućice | Stotice | Desetice | Jedinice |
| ----------: | ------------: | -------: | ------: | -------: | -------: |
| $0$ | $0$ | $0$ | $0$ | $5$ | $7$ |

Dekadski brojevni sustav karakterizira činjenica da svaka slijedeća dekadska jedinica ima vrijednost deset puta veću od prethodne. Stoga se za ovaj sustav kaže da mu je osnovica broj $10$. Cjelokupna numerička vrijednost dekadskog broja dobiva se zbrajanjem pojedinačnih dekadskih jedinica koje ga čine. Na primjer, za broj $57$, njegovu vrijednost (pedeset sedam) dobivamo kao zbroj pet desetica i sedam jedinica.

Ukoliko imamo zapisan broj u nekom drugom brojevnom sustavu s bazom $B$, a želimo ga prevesti u dekadski sustav, to ćemo učiniti na sljedeći način:

$$Z_{(B)} = \sum_{i=0}^{N-1} Z_i B^i,$$

gdje je broj $Z$ od $N$ znamenaka u tom brojevnom sustavu.

Raspišemo li gornji izraz dobijemo slijedeći oblik:

$$Z_{(B)} = Z_{(N-1)} \cdot B^{N-1} + Z_{(N-2)} \cdot B^{N-2} + \ldots + Z_{(1)} \cdot B^{1} + Z_{(0)} \cdot B^{0}.$$

!!! example "Zadatak"
    Broj 1136, s težinskim vrijednostima baze 7, napiši u dekadskom obliku.

??? success "Rješenje"
    $$1136_{(7)} = 1 \cdot 7^3 + 1 \cdot 7^2 + 3 \cdot 7^1 + 6 \cdot 7^0 = 343 + 49 + 21 + 6 = 419_{(10)}.$$

    Dakle, broj $1136_{(7)}$ zapisan u brojevnom sustavu s bazom $7$ ima vrijednost $419_{(10)}$ u dekadskom brojevnom sustavu.

Ako imamo broj koji sadrži decimalne znamenke, nastavljamo s računanjem pomoću negativnih eksponenata. Svaka slijedeća decimalna znamenka ima eksponent koji je manji za $N-1$ u odnosu na prethodnu znamenku. To znači da prva znamenka iza decimalnog zareza predstavlja desetinke, druga predstavlja stotinke, treća predstavlja tisućinke itd.

!!! example "Zadatak"
    Pretvori broj $5,24_{(8)}$ u dekadski oblik.

??? success "Rješenje"
    $$5,24_{(8)} = 5 \cdot 8^{0} + 2 \cdot 8^{-1} + 4 \cdot 8^{-2} = 5,3125_{(10)}.$$

!!! example "Zadatak"
    1. Pretvori broj $10110101_{(2)}$ u dekadski oblik.
    2. Pretvori broj $40,18_{(8)}$ u dekadski oblik.

### Binarni sustav

Binarni brojevni sustav jedan je od najosnovnijih i najkorištenijih brojevnih sustava u računalnoj tehnologiji. Ovaj sustav se sastoji od samo dvije znamenke, $0$ i $1$, stoga mu je baza $2$. Kako je to brojevni sustav s najmanjom bazom, iz naziva na engleskom jeziku "**BI**nary digi**T**" nastalo je ime za najmanju količinu informacije **BIT**.

Bitovi se koriste za kodiranje i prijenos informacija u računalima pa je stoga bitna jedinica mjere za brzinu prijenosa podataka. Računala obrađuju i prikazuju podatke u obliku niza bitova, a veće količine podataka prikazuju se kao kombinacije bitova, kao što su bajt, kilobajt, megabajt i tako dalje.

Kako bismo neki broj zapisali u binarnom sustavu, koristimo postupak dijeljenja s brojem $2$. Prvo, broj dijelimo s $2$ i zapisujemo ostatak. Zatim, taj ostatak dijelimo sa $2$ i opet zapisujemo ostatak. Taj postupak ponavljamo sve dok kao rezultat ne dobijemo nulu. Na kraju, sve ostatke zapišemo obrnutim redoslijedom i dobijemo prikaz broja u binarnom brojevnom sustavu.

!!! example "Zadatak"
    Pretvori broj $52_{(10)}$ u binarni oblik.

??? success "Rješenje"
    |      | Ostatak |
    | ---: | ------: |
    | $52 : 2 = 26$ | $0$ |
    | $26 : 2 = 13$ | $0$ |
    | $13 : 2 = 6$ | $1$ |
    | $6 : 2 = 3$ | $0$ |
    | $3 : 2 = 1$ | $1$ |
    | $1 : 2 = 0$ | $1$ |

    Dakle, broj $52_{(10)}$ zapisan u dekadskom sustavu ima vrijednost $110100_{(2)}$ u binarnom brojevnom sustavu.

Kada radimo s brojevima koji imaju decimalne znamenke, postupak pretvaranja u binarni zapis je nešto drugačiji. Prvo, odvajamo cijeli i decimalni dio. Cijeli dio pretvaramo u binarni zapis kao što smo već objasnili. Za pretvaranje decimalnog dijela, množimo ga s brojem $2$. Ako je rezultat jednak ili veći od $1$, zapisujemo $1$, a ako je manji od $1$, zapisujemo $0$. Potom ponavljamo postupak s decimalnim dijelom rezultata, sve dok ne dobijemo $0$ kao decimalni dio dekadskog broja ili do željenog broja decimala binarnog broja. Važno je zapamtiti da rezultat čitamo od **gore prema dolje**, što znači da je prva zapisana znamenka u bitu najviši bit, a posljednja zapisana znamenka najniži bit.

!!! example "Zadatak"
    Pretvori broj $8,125_{(10)}$ u binarni zapis.

??? success "Rješenje"
    |      | Ostatak |
    | ---: | ------: |
    | $8 : 2 = 4$ | $0$ |
    | $4 : 2 = 2$ | $0$ |
    | $2 : 2 = 1$ | $0$ |
    | $1 : 2 = 0$ | $1$ |

    |      | Ostatak |
    | ---: | ------: |
    | $0,125 \cdot 2 = 0,25$ | $0$ |
    | $0,25 \cdot 2 = 0,5$ | $0$ |
    | $0,5 \cdot 2 = 1,0$ | $1$ |

    Dakle, broj $8,125_{(10)}$ je $1000,001_{(2)}$ u binarnom zapisu.

!!! example "Zadatak"
    Pretvori broj $6,74_{(10)}$ u binarni zapis s tri decimale.

??? success "Rješenje"
    |      | Ostatak |
    | ---: | ------: |
    | $6 : 2 = 3$ | $0$ |
    | $3 : 2 = 1$ | $1$ |
    | $1 : 2 = 0$ | $1$ |

    |      | Ostatak |
    | ---: | ------: |
    | $0,74 \cdot 2 = 1,48$ | $1$ |
    | $0,48 \cdot 2 = 0,96$ | $0$ |
    | $0,96 \cdot 2 = 1,92$ | $1$ |

    Dakle, broj $6,74_{(10)}$ je $110,101_{(2)}$ u binarnom zapisu.

!!! example "Zadatak"
    1. Pretvori broj $28_{(10)}$ u binarni zapis.
    2. Pretvori broj $42,32_{(10)}$ u binarni zapis s 5 decimalnih mjesta.

### Heksadekadski sustav

Zapisivanje većih brojeva u binarnom zapisu može biti izazovno i može dovesti do pogrešaka u zapisivanju. Kako bi olakšali zapisivanje većih brojeva u binarnom zapisu, često se koristi heksadekadski brojevni sustav s bazom $16$. Ovaj sustav koristi brojeve od $0$ do $9$, ali i slova od $A$ do $F$ za predstavljanje vrijednosti.

Heksadekadski sustav ima važnu primjenu, a to je prikaz boja na računalu. Kombinacijom vrijednosti koje se kreću od 0 do 255, heksadekadski sustav omogućuje prikaz različitih nijansi boja. Na primjer, #000000 predstavlja crnu boju, dok #008000 predstavlja svijetlo zelenu boju. Gotovo svi računalni programi koji imaju mogućnost odabira boja koriste heksadekadski zapis, uključujući i popularni program za razvoj web stranica, Visual Studio Code (CSS i sl.).

## Prikaz binarnih brojeva s predznakom

Kada je riječ o binarnom brojevnom sustavu, uobičajeno je da se predstavljanje brojeva vrši korištenjem samo $0$ i $1$. U takvom sustavu, umjesto uobičajenih znakova "+" i "-", koristi se krajnji lijevi bit kako bi se odredilo je li broj pozitivan ili negativan. Ako je taj bit 0, broj je pozitivan, a ako je bit 1, broj negativan.

Kada je riječ o negativnim brojevima, postoje tri načina njihovog prikazivanja. Prvi način je prikazivanje negativnog broja pomoću predznaka i veličine. U tom slučaju, krajnji lijevi bit predstavlja predznak broja (0 za pozitivan, 1 za negativan), a preostali bitovi predstavljaju veličinu broja. Preostala dva načina prikazivanja negativnih brojeva temelje se na korištenju komplementa broja.

!!! example "Zadatak"
    Tehnikom predznaka i veličine napišite $-17_{(10)}$ u zapisu od $8$ bita.

??? success "Rješenje"
    |  |  |
    | ---: | ---: |
    | $+17$ | $00010001$ |
    | $-17$ | $10001100$ |

### Komplement broja

Komplement broja $z$ ($z'$) definira se kao dopuna broja $z$ do baze $B^n$, odnosno do baze $B^n$ umanjene za jedinicu:

$$z + z' = B^n,$$

$$z + z' = B^n - 1,$$

gdje je $n$ broj mjesta (znamenaka) za zapis broja. U nastavku ćemo kroz primjere pokazati navedene komplemente.

#### Jedinični komplement

Drugi način prikazivanja negativnih binarnih brojeva je pomoću predznaka i tzv. jediničnog komplementa (1-komplement). Jedinični komplement dobijemo iz prethodne druge jednadžbe:

$$z' = B^n - 1 - z.$$

Umjesto korištenja komplicirane formule za izračun komplementa broja, postoji jednostavniji način zapisivanja pomoću invertiranja svake znamenke broja.

!!! example "Zadatak"
    Napišite broj $-17_{(10)}$ tehnikom predznaka i jediničnog komplementa u zapisu od $8$ bita.

??? success "Rješenje"
    |  |  |
    | ---: | ---: |
    | $+17$ | $00010001$ |
    | $-17$ | $11101110$ |

    Možemo primjetiti kako se u zapisu jediničnog komplementa invertirala svaka znamenka broja $-17_{(10)}$.

Jedinični komplement se najčešće koristi kod računalnih sustava za provjeru ispravnosti prijenosa podataka.

#### Dvojni komplement

Posljedni način prikazivanja negativnih brojeva je pomoću predznaka i dvojnog komplementa (2-komplement) kojeg dobijemo na sljedeći način:

$$z' = B^n - z.$$

U ovom slučaju, negativni broj se dobiva tako da se uzme jedinični komplement broja i zbroji s $1$ ili puno jednostavnije, počevši s desne strane prepišemo sve znamenke do prve jedinice (uključujući prvu jedinicu), a ostalne znamenke invertiramo.

!!! example "Zadatak"
    Tehnikom dvojnog komplementa prikaži broj $-12_{(10)}$ u zapisu od 8 bitova.

??? success "Rješenje"
    |  |  |  |
    | ---: | ---: | --- |
    | $+12$ | $00001100$ | Broj $+12$ u zapisu od 8 bita. |
    | $-12$ | $11110100$ | Prepišemo sve znamenke do prve jedinice (i prvu jedinicu), a ostalne znamenke invertiramo. |

Svaki od ova tri načina ima svoje prednosti i nedostatke. Prednost korištenja predznaka i veličine je da se broj može jednostavno pretvoriti u drugi brojevni sustav, dok prednost korištenja dvojnog komplementa je brzina matematičkih operacija. U konačnici, odabir načina prikazivanja negativnih brojeva ovisi o primjeni, ali i osobnim preferencijama. Svaki sustav binarnog prikaza brojeva ima svoja ograničenja prikaza brojeva u određenom rasponu, što može utjecati na odabir načina prikaza negativnih brojeva:

| Sustav | Raspon |
| :----: | -----: |
| Bez predznaka | $[0, 2^N - 1]$ |
| S predznakom | $[-2^{N-1} + 1, 2^{N-1} - 1]$ |
| Dvojni komplement | $[-2^{N-1}, 2^{N-1} - 1]$ |

Uzmimo za primjer zapis 4-bitnih brojeva u svakom sustavu:

| Broj  | $-8$ | $-7$ | $-6$ | $-5$ | $-4$ | $-3$ | $-2$ | $-1$ | $0$  | $1$  | $2$  | $3$  | $4$  | $5$  | $6$  | $7$  | $8$  | $9$  | $10$ | $11$ | $12$ | $13$ | $14$ | $15$ |
| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bez predznaka | | | | | | | | | $0000$ | $0001$ | $0010$ | $0011$ | $0100$ | $0101$ | $0110$ | $0111$ | $1000$ | $1001$ | $1010$ | $1011$ | $1100$ | $1101$ | $1110$ | $1111$ |
| S predznakom | | $1111$ | $1110$ | $1101$ | $1100$ | $10111$ | $1010$ | $10001$ | $0000/1000$ | $0001$ | $0010$ | $0011$ | $0100$ | $0101$ | $0110$ | $0111$ | | | | | | | | |
| Dvojni komplement | $1000$ | $1001$ | $1010$ | $1011$ | $1100$ | $1101$ | $1110$ | $1111$ | $0000$ | $0001$ | $0010$ | $0011$ | $0100$ | $0101$ | $0110$ | $0111$ | | | | | | | | |

Možemo primjetiti da brojevi bez predznaka obuhvaćaju raspon $[0, 15]$ u redovnom binarnom redoslijedu. Brojevi dvojnog komplementa obuhvaćaju raspon $[-8, 7]$. Nenegativni brojevi $[0, 7]$ dijele isto kodiranje kao i brojevi bez predznaka. Brojevi s predznakom obuhvaćaju raspon $[-7, 7]$. Pozitivni brojevi $[1, 7]$ također dijele isto kodiranje kao i brojevi bez predznaka. Negativni brojevi su simetrični, ali imaju postavljen bit predznaka. Nula je predstavljena i sa 0000 i sa 1000. Prema tome, $N$-bitni brojevi s predznakom predstavljaju samo $2^N-1$ cijelih brojeva zbog dvije reprezentacije zapisa nule.
