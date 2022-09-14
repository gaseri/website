---
author: Vedran Miletić, Matea Turalija
---

# Generiranje topologija

## Mrežne topologije

Mogućnost međusobnog povezivanja računala omogućila je ne samo razmjenu poruka ili podataka, već i stvaranje lokalnih i globalnih mreža. Razvoj takvih struktura omogućile su mrežne topologije čiji je dizajn osiguravao brzinu i sigurnost takvih veza.

Za implementaciju računalne mreže u koju želimo povezati različita računala potrebno je prethodno planiranje. Prilikom planiranja računalne mreže od velike je važnosti odabir vrste topologije, jer njezina konfiguracija određuje način rada računalne mreže. Mrežna topologija tako definira niz različitih kategorija koje se mogu koristiti za određivanje rasporeda i veza između čvorova (računala, komunikacijska oprema, ...) te putanje podataka unutar mreže.

Najčešća podjela mrežne topologije odnosi se na fizičku i logičku topologiju. Fizička topologija pokazuje na koji su način fizički povezani čvorovi mreže. Pod pojmom logičke topologije podrazumijevamo putanju koju prolazi signal od jednog računala do drugog. Ona može biti ista kao fizička topologija, ali i ne mora.

Razlikujemo više različitih topologija:

- Od točke do točke (engl. *point-to-point*)
- Sabirnička ili bus (engl. *bus*)
- Zvjezdasta (engl. *star*)
- Prstenasta (engl. *ring*)
- Stablasta (engl. *tree*) ili hijerarhijska (engl. *hierarchy*)
- Isprepletena (engl. *mesh*)
- Potpuno povezana (engl. *fully connected*)
- Hibridna (engl. *hybrid*)

### Mrežna topologija od točke do točke

Najjednostavnija je topologija s namjenskom vezom između dvije krajnje točke. Veza između čvorova može biti stalna ili dinamička. U dinamičkoj vezi uspostavlja se komunikacijski kanal prije početka razmjene podataka, kao što je telefonski poziv.

``` mermaid
graph LR;
    n1((n1)) --- n2((n2));
```

### Sabirnička mrežna topologija

U ovoj vrsti topologije, svi čvorovi su povezani preko zajedničkog vodiča na način da mogu izravno komunicirati. Budući da se topologija sabirnice sastoji od samo jedne žice, lakše ju je implementirati od drugih topologija, ali uštede su nadoknađene višim troškovima upravljanja mrežom. Osim toga, budući da mreža ovisi o jednom kabelu, svi čvorovi su isključeni u slučaju njegova prekida.

``` mermaid
graph LR;
    n1((n1)) --- b1[ ];
    n2((n2)) --- b2[ ];
    n3((n3)) --- b3[ ];
    b1[ ] --- b2[ ] --- b3[ ];
    n4((n4)) --- b1[ ];
    n5((n5)) --- b2[ ];
    n6((n6)) --- b3[ ];
```

### Zvjezdasta mrežna topologija

Zvjezdasta topologija smatra se najlakšom topologijom za projektiranje i implementaciju. U ovoj vrsti topologije postoji središnji čvor na koji su ostali čvorovi mreže izravno povezani kabelima. Ulogu središnjeg čvora uglavnom obavlja preklopnik (engl. *switch*). U ovoj mreži obično nema međusobne veze između računala jer svi podaci prolaze kroz središnji čvor.

Jedna od prednosti ove topologije je da se kvarovi mogu lako locirati te jednostavnost dodavanja dodatnih čvorova. Glavni nedostatak je da ako centralni čvor zakaže, cijela mreža pada. Kvar bilo kojeg drugog čvora u mreži, osim središnjeg čvora, ne utječe na komunikaciju ostalih čvorova.

Mreža ne mora nužno biti u obliku zvijezde da bi bila klasificirana kao zvjezdasta mreža, ali svi čvorovi mreže moraju biti povezani s jednim središnjim čvorom. Ova topologija sa svojim podvrstama je najčešći oblik povezivanja u lokalnim mrežama (LAN).

``` mermaid
graph TB;
    n1((n1)) --- n4((n4));
    n2((n2)) --- n4((n4));
    n3((n3)) --- n4((n4));
    n4((n4)) --- n5((n5));
    n4((n4)) --- n6((n6));
    n4((n4)) --- n7((n7));
```

### Prstenasta mrežna topologija

Prstenasta topologija je u osnovi sabirnička topologija u zatvorenoj petlji. Svaki čvor je povezan samo s dva susjedna čvora. Podaci se kreću kružno od jednog čvora do drugog i obično samo u jednom smjeru. Kada jedan čvor šalje podatke drugom, podaci prolaze kroz svaki međučvor u prstenu dok ne stignu do odredišta. Svaki čvor je ravnopravan; ne postoji hijerarhijski odnos između klijenata i poslužitelja.

Glavni nedostatak ove topologije je spor prijenos i mogućnost međučvorova da vide poslane pakete podataka, budući da paketi moraju proći kroz njih. Postoji i dvostruka prsetansta topologija (engl. *dual-ring topology*) s dvije veze između svaka dva čvora. Obično se koristi samo jedan prsten, dok drugi služi kao rezerva u slučaju da prvi zakaže.

``` mermaid
graph LR;
    n1((n1)) --- n2((n2)) --- n3((n3)) --- n4((n4)) --- n5((n5)) --- n1((n1));
```

### Stablasta mrežna topologija

Stablasta topologija sastoji se od korijenskog (engl. *root*) čvora koji je najviši u hijerarhijskom rasporedu čvorova i na njega spojenih čvorova koji se nalaze na sloju niže od njega. Čvorovi nižeg sloja opet mogu imati na sebe spojene čvorove još nižeg sloja, itd. Prilično je slična proširenoj topologiji zvijezde, ali njezina je temeljna razlika u tome što nema središnji čvor. Kao i u zvjezdastoj mreži, ako jedan čvor zakaže, svi čvorovi povezani s njim mogu biti izolirani od ostatka.

``` mermaid
graph TB;
    n1((n1)) --- n2((n2));
    n1((n1)) --- n3((n3));
    n2((n2)) --- n4((n4));
    n2((n2)) --- n5((n5));
    n3((n3)) --- n6((n6));
    n3((n3)) --- n7((n7));
    n4((n4)) --- n8((n8));
    n4((n4)) --- n9((n9));
    n5((n5)) --- n10((n10));
    n5((n5)) --- n11((n11));
    n6((n6)) --- n12((n12));
    n6((n6)) --- n13((n13));
    n7((n7)) --- n14((n14));
    n7((n7)) --- n15((n15));
```

### Isprepletena mrežna topologija

U isprepletenoj topologiji, čvorovi mreže mogu biti izravno povezani s nekoliko drugih čvorova. Kvar jednog čvora u mreži ne utječe na druge čvorove u mreži. Na ovakav način funkcionira i Internet. Prednost takve mreže je što se lako može proširivati i nadograđivati.

``` mermaid
graph TB;
    n7((n7)) --- n4((n4));
    n7((n7)) --- n5((n5));
    n7((n7)) --- n6((n6));
    n1((n1)) --- n2((n2));
    n1((n1)) --- n6((n6));
    n1((n1)) --- n3((n3));
    n2((n2)) --- n5((n5));
    n2((n2)) --- n4((n4));
    n1((n1)) --- n7((n7));
    n2((n2)) --- n7((n7));
    n3((n3)) --- n7((n7));
```

### Potpuno povezana mrežna topologija

U potpuno povezanoj mreži svi su čvorovi međusobno povezani. To omogućuje prijenos poruka s jednog čvora na drugi različitim rutama. Odnosno, nema prekida u komunikaciji. Cilj je osigurati neprekidnu povezanost kada je protok podataka od velike važnosti (nuklearni, istraživački centri i sl.).

Glavni nedostatak takve topologije je skupo povezivanje svih čvorova jer zahtjeva veliku količinu kabela. Također, presložena je za primjenu tako da se koristi samo tamo gdje je to krajnje neophodno i gdje nije potrebno spajati mnogo čvorova.

``` mermaid
graph LR;
    n1((n1)) --- n2((n2));
    n1((n2)) --- n3((n3));
    n1((n1)) --- n4((n4));
    n2((n4)) --- n3((n4));
    n2((n2)) --- n4((n3));
    n3((n3)) --- n4((n4));
```

### Hibridna mrežna topologija

Hibridna topologija poznata je i pod nazivom topologija mreže ili mješovita jer koristi dvije ili više topologija za njihovo povezivanje na takav način da rezultirajuća mreža nema nijednu od standardnih topologija. U praksi je to jedna od najčešće korištenih topologija. Glavna prednost je što se može jednostavno proširiti i prilagoditi zahtjevima svakog klijenta.

``` mermaid
graph TB;
    n1((n1)) --- n2((n2));
    n1((n1)) --- n3((n3));
    n2((n2)) --- n4((n4));
    n2((n2)) --- n5((n5));
    n3((n3)) --- n6((n6));
    n7((n7)) --- n6((n6));
    n8((n8)) --- n3((n3));
    n8((n8)) --- n7((n7));
```

## Postupak generiranja topologija

Kako bi CORE omogućio da se vjerno reprezentiraju mreže kakve postoje u stvarnosti, podržano je generiranje topologija mreže po različitim pravilima. U CORE-ovom izborniku pod `Tools` moguće je pronaći `Topology generator` koji nam omogućuje brzo stvaranje različitih topologija mreže. Čvorovi mogu biti nasumično postavljeni, poredani u rešetku, zvijezdu ili neki drugi od ponuđenih topoloških obrazaca; odabirom stavke izbornika `Tools/Topology generator` vidljiv je podizbornik s popisom topologija koje je moguće generirati.

Kako bismo generirali topologiju po želji, najprije je potrebno odabrati vrstu čvora od kojeg se treba sastojati topologija (u zadanim postavkama nakon pokretanja CORE-a to su čvorovi tipa `router`). Zatim je potrebno odabrati uzorak topologije koji želimo generirati; svi podržani uzorci opisani su u tablici ispod.

| Uzorak | Opis |
| ------ | ---- |
| Slučajni (`Random`) | Čvorovi su nasumično postavljeni na platno, ali nisu međusobno povezani. Ovaj uzorak se može koristiti u kombinaciji s čvorom tipa `wireless LAN` za brzo stvaranje bežične mreže. |
| Rešetka (`Grid`) | Čvorovi su smješteni u vodoravnim redovima koji počinju u gornjem lijevom kutu, ravnomjerno razmaknuti kod razmještanja prema desno. Kao i kod slučajnog uzorka, čvorovi nisu međusobno povezani. |
| Povezana rešetka (`Connected Grid`) | Čvorovi su smješteni u pravokutnu mrežu širine N i visine M i svaki je čvor povezan s čvorom iznad, dolje, lijevo i desno od sebe. |
| Lanac (`Chain`) | Čvorovi su povezani jedan za drugim u lancu. |
| Zvijezda (`Star`) | Jedan je čvor postavljen u središte s N čvorova koji ga okružuju i svaki je čvor povezan sa središnjim čvorom. |
| Ciklus (`Cycle`) | Čvorovi su raspoređeni u krug, pri čemu je svaki čvor spojen sa susjedom i zajedno tvore zatvorenu kružnu stazu. |
| Kotač (`Wheel`) | Povezuje čvorove u kombinaciji uzorka zvijezde i ciklusa. |
| Kocka (`Cube`) | Povezuje čvorove kako su povezani vrhovi (hiper)kocke. |
| Klika (`Clique`) | Povezuje čvorove u kliku (potpuni graf) u kojoj je svaki čvor povezan sa svim ostalim čvorovima. |
| Bipartitni (`Bipartite`) | Povezuje čvorove u bipartitni graf koji ima dva odvojena skupa čvorova i svaki je čvor povezan sa svim čvorovima iz drugog skupa. |

U posljednjem koraku biramo veličinu topologije koju želimo koristiti. Tu je jedino ograničenje snaga računala na kojem pokrećemo emulaciju, ali mi ćemo se iz praktičnih razloga ograničiti na topologije veličine do nekoliko desetaka čvorova.
