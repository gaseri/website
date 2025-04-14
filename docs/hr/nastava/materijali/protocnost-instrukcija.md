---
author: Patricia Hauswirth, Kristina Sobol, Matea Turalija
---

# Protočnost instrukcija

Protočnost (engl. *pipelining*) je tehnika za implementaciju oblika paralelizma ili istodobnog izvođenja više operacija na razini instrukcija unutar procesora. To se postiže rastavljanjem složenijeg zadatka na manje zadatke (podzadatke) koji se izvršavaju jedan za drugim određenim redoslijedom u dodijeljenim samostalnim jedinicama. Samostalne jedinice nazivaju se *protočnim segmentima* i istodobno su aktivne, što omogućuje preklapanje izvršavanja podzadataka. *Protočna struktura* oblikuje se povezivanjem niza međusobno povezanih protočnih segmenata, pri čemu izlaz jednog segmenta služi kao ulaz u sljedeći protočni segment. Na slici ispod prikazana je protočna struktura sastavljena on M protočnih segmenata.

``` mermaid
flowchart LR
    A( ) -- Ulaz --> 
    id1(Protočni segment 1) --> 
    id2(Protočni segment 2) --> 
    id3(...) --> 
    id4(Protočni segment M) 
    -- Izlaz --> B( )
```

## Protočna struktura mikroprocesora

Protočna struktura mikroprocesora, odnosi se na organizaciju i redoslijed zadataka ili faza unutar protočnosti za izvođenje mikroprocesora. Uključuje podjelu procesa izvršenja instrukcija u diskretne faze ili segmente, dopuštajući da se više instrukcija obrađuje istovremeno i poboljšava ukupnu izvedbu i učinkovitost. Struktura se obično sastoji od nekoliko stupnjeva, pri čemu je svaki stupanj odgovaran za određenu operaciju ili zadatak. Ove su faze osmišljene tako da se preklapaju u izvršavanju, dopuštajući da više instrukcija istovremeno napreduje.

Protočna struktura predstavlja ključni koncept za poboljšanje performansi procesora, a ključne veličine koje se koriste za evaluaciju takve strukture su vrijeme latencije i propusnost.

*Vrijeme latencije* odnosi se na vrijeme potrebno za izvršavanje jedne instrukcije, od njezinog ulaska u procesor do trenutka kada je izvršena. *Propusnost* je učestalost izvođenja instrukcija, odnosno to je broj koliko se instrukcija izvede u jedinici vremena, što može biti sekunda ili perioda.

Kod procesora koji izvršavaju samo jednu instrukciju istovremeno (bez protočne strukture) propusnost je obično obrnuto proporcionalna vremenu latencije:

$$
\text{Propusnost} = \frac{1}{\text{Vrijeme latencije}}
$$

S druge strane protočna struktura omogućava paralelno izvođenje više instrukcija, što rezultira većom propusnošću u odnosu na recipročnu vrijednost latencije jedne instrukcije:

$$
\text{Propusnost} > \frac{1}{\text{Vrijeme latencije}}
$$

Paralelno izvođenje više instrukcija rezultira povećanom propusnošću procesora. Isto tako, uobičajeno je da se vrijeme izvršavanja jedne instrukcije također povećava u odnosu na klasičnu izvedbu. To znači da će vrijeme potrebno za izvršavanje pojedinačne instrukcije biti duže nego kod procesora bez protočne strukture. Međutim, zahvaljujući paralelnom izvođenju više instrukcija istovremeno, ukupna propusnost procesora se značajno poboljšava nego kod procesora bez protočne strukture.

Dakle, iako se vremenska latencija povećava, prednost protočne strukture leži u većoj propusnosti koju pruža. Izvođenje više instrukcija paralelno omogućava bolje korištenje raspoloživih resursa procesora i smanjenje vremena čekanja između instrukcija, što dovodi do ukupno bržeg izvršavanja kompleksnih zadataka.

### Instrukcijska protočna struktura

U klasičnoj izvedbi nekog procesora možemo imati dvije faze izvršavanja instrukcija, jedno je pribavljanje i dekodiranje instrukcije, a drugo je izvršavanje instrukcije.

``` mermaid
flowchart LR
    id1((PRIBAVI)) ---> id2((IZVRŠI)) ---> id1
```

Izvođenje svake instrukcije započinje aktivnostima u fazi PRIBAVI, koje obuhvaćaju pribavljanje instrukcije iz memorije, inkrementiranje programskog brojila i dekodiranje operacijskog koda instrukcije. Nakon završetka tih aktivnosti, nastavlja se s izvršavanjem instrukcije u fazi IZVRŠI, koja uključuje dohvat operanada, izvođenje aritmetičkih ili logičkih operacija te pohranu rezultata.

To znači da se zadatak izvođenja instrukcije može rastaviti na dva slijedna podzadatka: PRIBAVI i IZVRŠI. Ako se svakom od podzadataka dodjeli samostalan sklop - protočni segment, dobiva se instrukcijska protočna struktura s M = 2 protočna segmenta

``` mermaid
flowchart LR
    A( ) --> id1(PRIBAVI) --> id2(IZVRŠI) --> B( )
```

Izvođenje slijeda instrukcija u vremenu, tj. programa u instrukcijskoj protočnoj strukturi možemo prikazati pomoću Ganttovog dijagrama. Na osi $x$ imamo vrijeme izvođenja koje je u ovom slučaju za svaki segment jednako i to je jedna perioda izvođenja, a na osi $y$ je redni broj instrukcije:

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| instrukcija i-1 | PRIBAVI | IZVRŠI | | | | |
| instrukcija i | | PRIBAVI | IZVRŠI | | | |
| instrukcija i+1 | | | PRIBAVI | IZVRŠI | | |
| instrukcija i+2 | | | | PRIBAVI | IZVRŠI | |
| instrukcija i+3 | | | | | PRIBAVI | IZVRŠI |

U takvoj instrukcijskoj protočnoj strukturi, jedna instrukcija se najprije dohvaća i dekodira, a zatim se to specifično sklopovlje koje je zaduženo za tu fazu zaustavlja i čeka dok se instrukcija izvršava. Nakon što se izvršavanje završi, sklopovlje ponovno prelazi na dohvaćanje i dekodiranje sljedeće instrukcije, dok sklopovi za izvršavanje instrukcija čekaju svoj red. Ovaj ciklus se nastavlja, omogućavajući protok instrukcija kroz različite faze i maksimizirajući iskorištenost resursa procesora.

### Instrukcijska protočna struktura za RISC procesore

U nastavku ćemo se fokusirati na tipičnu RISC protočnu strukturu s pet protočnih segmenata. Takva protočna struktura uključuje sljedeće segmente:

- IF - pribavljanje instrukcije (engl. _**I**nstruction **F**etch_),
- ID-OF - dekodiranje instrukcije i dohvat operanada (engl. _**I**nstruction **D**ecode and **O**perand **F**etch_),
- EX - izvršavanje instrukcije (engl. _**EX**ecute/compute address_),
- MEM - pristup memoriji (engl. _**MEM**ory access_) i
- WB - upis rezultata ili podataka (engl. _**W**rite **B**ack_).

Tijekom IF faze sklopovlje mikroprocesora pristupa programskom brojilu koji sadrži adresu za dohvaćanje odgovarajuće instrukcije u memoriji. Zatim koristi adresu za dohvaćanje odgovarajuće instrukcije iz podsustava memorije ili priručne memorije. Dohvaćena instrukcija obično se pohranjuje u registar instrukcija ili namjenski međuspremnik unutar mikroprocesora.

ID-OF je odgovoran za dva glavna zadatka: dekodiranje instrukcija i dohvaćanje operanada. Tijekom faze dekodiranja instrukcija, kako i samo ime kaže, dekodira se dohvaćena instrukcija iz prethodne IF faze. Dekoder dekodira polje operacijskog koda instrukcije kako bi se odredila specifična operacija ili radnja koju je potrebno izvesti. Osim dekodiranja instrukcije, odgovoran je i za dohvaćanje operanada potrebnih za izvršavanje instrukcija. Operandi su obično vrijednosti registara ili memorijske lokacije koje su potrebne za izvođenje operacije instrukcije.

EX je zaslužan za izvršavanje operacije specificirane dekodiranom instrukcijom. Izvodi izračunavanje ili manipulaciju nad podacima koje zahtijeva instrukcija. Tijekom ove faze, mikroprocesor izvodi aritmetičke i logičke operacije, prijenos podataka ili bilo koje druge specificirane operacije na temelju dekodirane instrukcije.

MEM je odgovoran za pristup memoriji. Tijekom ove faze, mikroprocesor izvodi operacije povezane s memorijom kao što je dohvaćanje podataka iz memorije, pohranjivanje podataka u memoriju ili prijenos podataka između registara mikroprocesora i memorije. Uz pristup podacima, također može upravljati drugim zadacima povezanim s memorijom, kao što su operacije priručne memorije ili upravljanje virtualnom memorijom, ovisno o arhitekturi mikroprocesora.

WB služi za pisanje rezultata instrukcije natrag na odgovarajuće odredište, obično u registar ili memorijsku lokaciju. Završava izvršenje instrukcije i ažurira stanje sutava na temelju izračunatog rezultata. Također upravlja svim potrebnim ažuriranjima statusa ili kontrolnih registara mikroprocesora. Primjerice, može ažurirati programsko brojilo da pokazuje na sljedeću instrukciju koju treba dohvatiti ili modificirati oznake statusa na temelju rezultata instrukcije.

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IF | ID | EX | MEM | WB | | | | |
| | IF | ID | EX | MEM | WB | | | |
| | | IF | ID | EX | MEM | WB | | |
| $\downarrow$ i | | | IF | ID | EX | MEM | WB | |
| $\rightarrow$ t | | | | IF | ID | EX | MEM | WB |

U ovom primjeru u prvoj periodi izvođenja prva instrukcija uđe u segment dohvaćanja instrukcije te se ta instrukcija dohvaća iz memorije i prebacuje u instrukcijski registar. U drugoj periodi prva instrukcija je dohvaćena te je sada u segmentu za dekodiranje, dok se druga instrukcija pribavlja iz memorije. U trećoj periodi prva instrukcija je u segmentu izvršavanja instrukcije, druga u segmentu dekodiranja, dok se treća počinje pribavljati iz memorije. U sljedećem koraku prva instrukcija pristupa memoriji, druga se izvršava, treća dekodira, a četvrta pribavlja. Tek nakon pete periode prva instrukcija ulazi u posljednji segment i zapisuje se rezultat izvođenja u odgovarajući registar. Istekom te periode prva instrukcija se izvršila. Istovremeno druga instukcija je u fazi pristupa memoriji, treća u fazi izvršavanja, četvrta u fazi dekodiranja, a peta se tek pribavlja.

Na ovoj shemi latencija prve instrukcije je 5 perioda jer je to vrijeme koje je potrebno da se prva instrukcija izvrši. Također, latencija druge instrukcije je 5 perioda. Međutim propusnost je sada veća od recipročne vrijednosti latencije jer se nakon pete periode druga instrukcija izvršila nakon jedne periode vremenskog vođenja. Sljedeća instrukcija se također izvršila nakon jedne periode itd. Odnosno nakon što se prva instukcija izvrši prividno je latencija svake sljedeće jedna perioda, a ne pet koliko je stvarno.

## Perioda kod procesora s protočnom strukturom

Ako je put podataka kroz procesor moguće podijeliti na protočne segmente tako da je vrijeme obrade u svakom segmentu jednako onda je ubrzanje koje je postignuto zapravo najbrže moguće. U tom slučaju vrijeme periode u protočnoj izvedbi je jednako vremenu periode u sekvencijalnoj izvedbi podjeljenim s brojem protočnih segmenata, ali povećano za vrijeme latencije registra s obzirom da između protočnih segmenata moraju postojati registri koji pamte ulaze u taj segment:

$$
TP_{\text{Protočna}} = \frac{TP_{\text{Sekvencijalna}}}{\text{Broj protočnih segmenata}} + \text{Vrijeme latencije registara}
$$

$$
TP_{\text{P}} = \frac{TP_{\text{S}}}{n_\text{PS}} + t_L
$$

!!! example "Zadatak"
    Procesor sekvencijalne izvebe (bez protočne strukture) ima trajanje periode 25 ns, a svaki registar protočnog segmenta ima vrijeme latencije 1 ns.

    1. Koliko je trajanje periode za protočnu verziju istog procesora s 5 protočnih segmenata?
    2. Koliko bi bilo da je procesor podjeljen na 50 protočnih segmenata?

??? success "Rješenje"
    1. Za 5 protočnih segmenata: $TP = \frac{25\text{ ns}}{5} + 1\text{ ns} = 6\text{ ns}$
    2. Za 50 protočnih segmenata: $TP = \frac{25\text{ ns}}{50} + 1\text{ ns} = 1,5\text{ ns}$

!!! example "Zadatak"
    Usporedi i prokomentiraj dobivene rezultate. Povećavaju li se performanse linearno s povećanjem broja protočnih segmenata?

Ako se vrijeme obrade u protočnim segmentima ne može jednoliko rasporediti, trajanje periode procesora će ovisiti o vremenu obrade najsporijeg protočnog segmenta. Pogledajmo sljedeći primjer.

!!! example "Zadatak"
    Neki procesor bez protočne strukture i trajanja periode 25 ns podjeljen je na protočne segmente s vremenima obrade 5, 7, 3, 6 i 4 ns. Ako je latencija registra protočnog segmenta 1 ns, koliko je trajanje periode novog procesora?

??? success "Rješenje"
    Najduži protočni segment ima vrijeme obrade 7 ns. Uz latenciju registra od 1 ns ukupno trajanje periode stoga je 8 ns.

## Latencija protočne strukture

Dijeljenjem u protočne segmente mijenja se zapravo i latencija jedne instrukcije. Odnosno, sada je latencija instrukcije latencija protočne strukture. Dakle, latencija protočne strukture je umnožak broja protočnih segmenata i trajanja periode:

$$L = n_{PS} \cdot T$$

!!! example "Zadatak"
    Kolika je latencija protočne strukture iz primjera 1 i 2?

??? success "Rješenje"
    Trajanje periode za procesor s 5 protočnih segmenata je 6 ns, a za 50 protočnih segmenata 1,5 ns. Vrijeme latencije u prvom slučaju je tada 30 ns, a u drugom 75 ns.

U drugom primjeru trajanje periode je 8 ns s 5 protočnih segmenata pa je latencija 40 ns.

!!! example "Zadatak"
    Usporedi i prokomentiraj dobivene rezultate. Koliko je sada realno vrijeme izvršavanja instrukcija?

## Hazardi u instrukcijskoj protočnoj strukturi

Istovremeno izvršavanje više instrukcija u protočnoj strukturi može dovesti do situacija u kojima slijedeća instrukcija ne može biti izvršena prema predviđenoj vremenskoj periodi signala vremenskog vođenja. Takvi događaji nazivaju se hazardi. Razlikujemo tri vrste hazarda:

- strukturni hazard (engl. *structural hazard*)
- podatkovni hazard (engl. *data hazard*)
- upravljački hazard (engl. *control hazard)*

### Strukturni hazardi

Strukturni hazard se javlja kada u istom trenutku dvije ili više instrukcija zahtijevaju iste resurse procesora. Na primjer kada se jedna instrukcija pribavlja iz memorija, a neka druga instrukcija čita podatke iz memorije ili piše u memoriju.
Jedno od rješenja za strukturno hazard je podjela priručne memorije na instrukcijsku i podatkovnu memoriju.

### Podatkovni hazardi

Podatkovni hazardni nastaju zbog međuzavisnosti podataka. Kada dvije ili više insturkcija, koje se nalaze u protočnoj strukturi, pristupaju istom podatku.

Postoje tri vrste takvih hazarda:

- RAW (engl. *Read After Write*) – čitanje poslije upisa
- WAW (engl. *Write After Write*) – pisanje poslije pisanja
- WAR (engl. *Write After Read*) – pisanje poslije čitanja

U jednostavnoj izvedbi s protočnom strukturom imamo problem čitanja nakon pisanja (RAW) gdje neka instrukcija ima operande koji su rezultati prethodne instrukcija. Primjerice, u ovom primjeru `add $t0, $t1, $t2` i `sub $t3, $t4, $t0` instrukcija `sub` čita kao svoj operand registar `$t0` u kojem se smješta rezultat instrukcije `add`.

U ovakvoj situaciji instukcija oduzimanja ne može napredovati dalje od faze pribavljanja operanda iz registra sve dok instrukcija `add` ne završi.

Ako imamo procesor koji ima faze dohvaćanja instrukcija, dekodiranje instrukcija, čitanja operanda iz registra, izvršavanja operacija i zapisa rezultata u registar. Onda možemo prikazati napredovanje instrukcija `add` i `sub` kroz protočnu strukturu:

|     | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IF | add | sub | | | | | | |
| ID | | add | sub | | | | | |
| RR | | | add | sub | sub | sub | | |
| EX | | | | add | n | n | sub | |
| WB | | | | | add | n | n | sub |

n - mjehurić (np-op)

U prvoj periodi instrukcija `add` ulazi u segment  dohvaćanja instrukcije. U drugoj periodi sada se operacija `sub` dohvaća, a `add` ulazi u segment dekodiranja instrukcije. Tako instrukcije ulaze slijedno kroz segmente sve do četvrte periode gdje operacija `sub` pokušava očitati svoje operande u registru. Međutim, instrukcija `add` se nalazi tek u segmentu izvršavanja. Tek po završetku pete periode operand instrukcije `sub` će biti zapisan u registar. `sub` u tom trenutku još stoji u segmentu RR, dok je `add` napredovala u sljedeću fazu, WB. U šestoj periodi instrukcija `add` je u međuvremenu zapisala rezultat u registar i `sub` sada napokon čita svoj registar i u sljedećoj periodi prelazi u fazu izvršavanja instrukcije te će u osmoj periodi biti gotovo izvršavanje ove instrukcije.

U 5, 6 i 7 periodi može se primjetiti da smo imali u nekim segmentima tzv. *No operations* instrukcije. Takve instrukcije nazivamo mjehurćima u protočnoj strukturi.

### Upravljački hazard

U protočnoj strukturi, upravljački hazard se javlja tijekom izvršavanja instrukcija koje uključuju grananje ili druge instrukcije koje mijenjaju sadržaj programskog brojila PC u fazi izvršavanja. Primjerice, to može uključivati instrukcije pozivanja potprograma ili povratka iz potprograma gdje instrukcija grananja računa adresu sljedeće instrukcije. U tom slučaju faza dohvaćanja sljedeće instrukcije ne može se izvršiti dok nije poznat ishod instrukcije grananja.

|     | 1   | 2   | 3   | 4   | 5     |
| --- | --- | --- | --- | --- | :---: |
| IF | b | n | n | n | sljedeća instrukcija |
| ID | | b | n | n | n |
| RR | | | b | n | n |
| EX | | | | b | n |
| WB | | | | | b |

b - instrukcija grananja
n - mjehurić (np-op)

U ovom primjeru segment EX je zadužen za izračun adrese sljedeće instrukcije. Stoga sljedeća instukcija koja slijedi mora čekati da instrukcija grananja izađe iz segmenta izvršavanja instrukcija.

## Vrijeme izvođenja programa

Kada nema mjehurića u protočnoj strukturi, vrijeme izvođenja programa (u periodama) jednako je zbroju protočnih segmenata i broju instrukcija umanjenog za jedan.

Primjerice ako imamo program koji ima 5 instrukcija i 4 protočna segmenta, tada se prva instrukcija izvede u 4 periode koliko imamo i protočnih segmenata. Sljedeća instrukcija se izvede nakon 5. periode, potom sljedeća nakon 6. itd. sve do 5. instrukcije koja se izvrši nakon 8. periode. Dakle imamo 4 segmenta + 5 instrukcija - 1 = 8 perioda.

U slučaju kada imamo mjehuriće u protočnoj strukturi možemo crtati dijagram protočnih segmenata ili možemo koristiti metodu razdvajanja vremena izvođenja u dva dijela. U tom slučaju vrijeme izvođenja se dijeli na latenciju protočne strukture i vrijeme izdavanja svih naredbi programa. Naredba je izdana kada je prešla iz segmenta PR u segment EX. Vrijeme izvođenja programa je u tom slučaju jednako zbroju latencije protočne strukture i vremena izdavanja svih naredbi umanjenjih za 1. Također, potrebno je znati i latenciju svake instrukcije, a to je vrijeme između izdavanja naredbe i izdavanja naredbe ovisne o njoj.

!!! example "Zadatak"
    Koliko je vrijeme izvođenja sljedećeg programa, na procesoru protočne strukture iz prethodnog primjera?

    ``` asm
    add $r1, $r2, $r3
    sub $r4, $r5, $r6
    mul $r8, $r2, $r1
    srl $r5, $r2, $r1
    or $r10, $r11, $r4
    ```

??? success "Rješenje"
    U našem primjeru vrijeme latencije protočne strukture je 5 perioda s obzirom da imamo 5 protočnih segmenata. Sada moramo izračunati vrijeme izdavanja svih ovih naredbi. Možemo pretpostaviti da je vrijeme izdavanja prve instrukcije neki trenutak $n$. Stoga je vrijeme izdavanja sljedeće instrukcije $n+1$ budući da nemamo hazarde vezane uz prvu instrukciju. Međutim instukcija `mul` ovisi o rezultatu instrukcije `add`. Sada mora isteći vrijeme latencije instrukcije `add` prije nego se može izdati naredba `mul`. U primjeru procesora kakvog promatramo nareba se može izdati 3 periode nakon naredbe o kojoj ovisi (ako se radi o instrukciji koja nije grananje). Prema tome, naredba `mul` se može izdati u $n+3$. `srl` ovisi o `add`, ali čeka na `mul` pa se izdaje u $n+4$. `or` ovisi o `sub` i najranije se može izdati 3 periode kasnije od trenutka kad je izdana naredba `sub`, a to bi onda bilo u $n+4$. Dakle, u tom trenutku `or` više ne ovidi o prethodnim pa se izdaje u $n+5$.

    Ukupno vrijeme izdavanja je sada 6 perioda, a latencija ove protočne strukture je 5 pa je ukupno vrijeme izvođenja programa 6 + 5 - 1 = 10 perioda.

!!! example "Zadatak"
    Koliko je vrijeme izvođenja sljedećeg programa (u periodama), na procesoru sa 7 protočnih segmenata i latencijom instrukcija grananja 5 perioda, a ostalih instrukcija 2 periode? Pretpostavi da uvjet grananja nije ispunjen.

    ``` asm
    bne $r4, $r0, $r5
    div $r2, $r1, $r7
    add $r8, $r9, $r10
    sub $r5, $r2, $r9
    mul $r10, $r5, $r8
    ```

??? success "Rješenje"
    Kao i u prethodnom primjeru, pretpostavimo da se prva instrukcija `bne` zadaje u trenutku $n$. Instrukcija `div` će se izdati tek nakon što istekne latencija instrukcije grananja pa će se izdati u trenutku $n+5$. `add` ne ovisi o rezultatu `div` pa se može izdati u prvoj sljedećoj periodi u trenutku $n+6$. `sub` ovisi o `div` pa se ne može izdati prije $n+7$. `mul` ovisi i o `sub` i o `add` pa se izdaje u trenutku $n+9$.

    Ukupno vrijeme izdavanja je 10 perioda pa je ukupno vrijeme izvođenja programa 7 + 10 - 1 = 16 perioda.

!!! example "Zadatak"
    Nacrtajte dijagram protočne strukture za sljedeći isječak programa:

    ``` asm
    add $r1, $r2, $r3
    sub $r4, $r5, $r6
    mul $r8, $r9, $r10
    div $r12, $r13, $r14
    ```

??? success "Rješenje"
    |     | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | IF | add | sub | mul | div | | | | |
    | ID | | add | sub | mul | div | | | |
    | RR | | | add | sub | mul | div | | |
    | EX | | | | add | sub | mul | div | |
    | WB | | | | | add | sub | mul | div |

!!! example "Zadatak"
    Nacrtajte dijagram protočne strukture s 5 segmenata iz prethodnih primjera za sljedeći isječak programa:

    ``` asm
    add $r1, $r2, $r3
    sub $r4, $r5, $r6
    mul $r8, $r9, $r4
    div $r12, $r13, $r14
    ```

??? success "Rješenje"
    |     | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | IF | add | sub | mul | div | | | | | |
    | ID | | add | sub | mul | div | div | div | | |
    | RR | | | add | sub | mul | mul | mul | div | |
    | EX | | | | add | sub | n | n | mul | div |
    | WB | | | | | add | sub | n | n | mul |

!!! example "Zadatak"
    Nacrtajte dijagram protočne strukture za sljedeći isječak programa. Pretpostavite da uvijet granjanja nije ispunjen.

    ``` asm
    add $r1, $r2, $r3
    sub $r4, $r5, $r6
    beq $r2, $r0, $r9
    div $r12, $r13, $r14
    ```

!!! example "Zadatak"
    Odredite vrijeme izvođenja sljedećeg programskog odsječka na procesoru s 5 protočnih segmenata, ako je trajanje periode 2 ns. Pretpostavite da uvjet grananja nije ispunjen.

    ``` asm
    add $r1, $r4, $r7
    beq $r2, $r0, $r1
    sub $r8, $r10, $r11
    mul $r12, $r13, $r14
    ```

??? success "Rješenje"
    `add` - n  
    `beq` - n + 3  
    `sub` - n + 7  
    `mul` - n + 8

    Ukupno vrijeme izdavanja je 9 perioda, latencija ove protočne strukture je 5 pa je ukupno vrijeme izvođenja programa 9 + 5 - 1 = 13 perioda, odnosno 13 $\cdot$ 2 ns = 26 ns.

!!! example "Zadatak"
    Odredite vrijeme izvođenja sljedećeg programskog odsječka na procesoru s 5 protočnih segmenata. Može li se smanjiti vrijeme izvođenja tog segmenta promjenom redosljeda izvršavanja instrukcija, a da se rezultat ne promjeni? Ako da, napišite preoblikovani niz instrukcija i izračunajte vrijeme izvođenja.

    ``` asm
    add $r3, $r4, $r5
    sub $r7, $r3, $r9
    mul $r8, $r9, $r10
    srl $r4, $r8, $r12
    ```

??? success "Rješenje"
    `add` - n  
    `sub` - n + 3  
    `mul` - n + 4  
    `srl` - n + 7

    Ukupno vrijeme izdavanja je 8 perioda, latencija je 5 pa je ukupno vrijeme izvođenja programa 8 + 5 - 1 = 12 perioda.

    Nakon izmjene redosljeda instrukcija:

    ``` asm
    add $r3, $r4, $r5w
    mul $r8, $r9, $r10
    sub $r7, $r3, $r9
    srl $r4, $r8, $r12
    ```

    `add` - n  
    `mul` - n + 1  
    `sub` - n + 3  
    `srl` - n + 4

    Ukupno vrijeme izdavanja je 5 perioda, latencija je 5 pa je ukupno vrijeme izvođenja programa 5 + 5 - 1 = 9 perioda.
