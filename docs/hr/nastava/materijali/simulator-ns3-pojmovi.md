---
author: Vedran Miletić, Ivan Ivakić
---

# Teorijske osnove simulacije računalnih mreža

## Pojmovi u osnovnom modelu mreže

Čvor je računalo ili prijenosnik u mreži. Kada govorimo o računalima razlikujemo dva osnovna tipa čvorova i to su **domaćini** i **prijenosnici**. Dok se na domaćinima izvode određene aplikacije, prijenosnici služe samo za povezivanje među domaćinima.

Veza je sredstvo povezivanja dvaju čvorova. U slučaju izravne povezanosti dva čvora govorimo o vezi tipa **točka-točka** (engl. *Point-to-Point*), a u slučaju povezivanja čvorova koji nisu izravno vezani govorimo o vezi tipa **s-kraja-na-kraj** (engl. *End-to-End*). Isti naziv koristimo kada govorimo o povezivanju procesa na dva čvora.

Mrežna kartica na čvoru je uređaj ili računalna komponenta zadužena za povezivanje računala u računalnu mrežu.

Paket je niz bitova određene strukture koji se računalnom mrežom prenosi kao jedna zasebna cjelina.

Adresa je jedinstveni identifikator određenog elementa računalne mreže. Razlikujemo fizičku i logičku adresu. Fizička adresa je adresa mrežne kartice svakog pojedinog čvora (primjerice MAC adresa kod mreža tipa Ethernet), a logička adresa je adresa je neovisna o fizičkim svojstvima mreže i omogućuje povezivanje različitih mreža. Primjer logičke adrese je IPv4 adresa.

Aplikacija je programski entitet koji korisniku omogućava operacije sa računalnom mrežom. Aplikacije dijelimo na **poslužitelje** (engl. *server*) i **klijente** (engl. *client*).

## Slojevi i protokoli

[OSI model](https://en.wikipedia.org/wiki/OSI_model) definira 7 slojeva računalnog sustava koji omogućava razmjenu sadržaja sa drugim računalnim sustavima:

- L1: fizički sloj,
- L2: sloj veze podataka,
- L3: mrežni sloj,
- L4: sloj kontrole prijenosa,
- L5: sloj sesije,
- L6: prezentacijski sloj i
- L7: aplikacijski sloj.

Područje koje ćemo promatrati u okviru ovih laboratorijskih vježbi većinom je sadržano u L2, L3 i L4. Aplikacije koje ćemo koristiti doživljavat ćemo kao entitete koji šalju ili primaju promet; sam sadržaj prometa nećemo razmatrati.

Fizički sloj ostvaruje prijenos signala (uglavnom elektromagnetskih ili optičkih) kojima su reprezentirani bitovi između dvaju mrežnih kartica na čvorovima.

Sloj veze podataka zadužen je za strukturiranje podataka u *okvire* sa zaglavljem koje sadrži fizičku adresu idućeg čvora.

Mrežni sloj ima za zadatak usmjeriti *pakete* od izvora do odredišta. Protokol koji vežemo za mrežni sloj je Internet Protokol (engl. *Internet Protocol*, IP). IP definira strukturu paketa podataka koji se prenose mrežom, adresni prostor i način prosljeđivanja paketa od izvora do odredišta.

Sloj kontrole prijenosa zadužen je za upravljanje prijenosom i kontrolu ispravnosti prijenosa. Protokoli koje vežemo za sloj kontrole prijenosa su TCP i UDP.

## Mjerne jedinice za količinu podataka i vrijeme

Količinu podataka mjerimo u **bitovima** i **bajtovima** pri čemu je je 1 bajt = 8 bitova. Bajte označavamo velikim slovom B i koristimo [binarne prefikse](https://en.wikipedia.org/wiki/Binary_prefix):

- Ki (kilo) -> 1 KiB = 1024 B = 2^10^ B
- Mi (mega) -> 1 MiB = 1048576 B = 2^20^ B
- Gi (giga) -> 1 GiB = 1073741824 B = 2^30^ B

Vrijeme mjerimo u sekundama (s), a prefiksi koje često koristimo su:

- n (nano) -> 1 ns = 10^-9^ s
- µ (mikro) -> 1 µs = 10^-6^$ s
- m (mili) -> 1 ms = 10^-3^ s

Širina frekventnog pojasa omjer broja bitova i vremena koju veza nominalno ostvaruje ovisno o tehničkoj realizaciji veze. Propusnost je realni (izmjereni) omjer broja bitova i vremena koje veza prenosi u danom trenutku.

Zadržavanje veze je vrijeme potrebno da se prijenos ostvari. Ukupno zadržavanje veze računamo kao sumu triju komponenata:

- vremena širenja signala (engl. *propagation*), koje se računa kao kvocijent udaljenosti i brzine signala
- vremena prijenosa podataka (engl. *transmission*), koje se računa kao kvocijent veličine paketa i propusnosti veze te
- vremena čekanja u redovima (engl. *queue*).

## Pojam simulacije i simulatora

Simulacijom nazivamo bilo koji proces kojim imitiramo ili prikazujemo realan proces u kontroliranoj okolini, obično na pojednostavljen način. Simulatorom nazivamo aplikaciju ili bilo kakav drugi mehanizam koji prikazuje realan proces.

Obzirom da je često teško ili praktički nemoguće nabaviti dovoljnu količinu dovoljno kvalitetne i moderne mrežne opreme za potrebe nastave na fakultetima, na većini njih prilikom učenja računalnih mreža koriste se simulatori.

[Simulacija mreža](https://en.wikipedia.org/wiki/Network_simulation) imitira događaje u stvarnoj računalnoj mreži na pojednostavljen način. Većina aktualnih mrežnih simulatora zasnovana je na diskretnim događajima, što znači da objekti koji sudjeluju u simulaciji zakazuju događaje na vremenskoj osi koji se zatim izvode redom kojim su zakazani. Svaki od tih događaja može prilikom izvođenja zakazati jedan ili više događaja s određenim vremenskim odmakom.

Obzirom da vrijeme u kojem se simulacija izvodi ne mora biti sinkronizirano s vremenom stvarnog svijeta, razlikovati ćemo **simulirano vrijeme**, odnosno vrijeme koje poznaju objekti unutar simulacije koja se izvodi, i **vrijeme zidnog sata**, odnosno vrijeme stvarnog svijeta. Najčešće će izvođenje simulacije trajati mnogo kraće nego što iznosi njezino simulirano vrijeme, no to će općenito ovisiti o broju i svojstvima događaja u simulaciji.

Pored simuliranog vremena, osnova simulatora je programski model dijelova mreže. Pod tim pojmom podrazumijevamo klase i funkcije koje opisuju ponašanje objekata stvarne mreže: paketa, mrežnih kartica, čvorova, aplikacija, kabela itd. Većina mrežnih simulatora implementira i prostorne koordinate, generiranje slučajnih brojeva, alate za statističku analizu rezultata simulacije, sučelja prema alatima za crtanje grafova i sl.

Simulator koji koristimo za laboratorijske vježbe je mrežni simulator [ns-3](https://en.wikipedia.org/wiki/Ns_(simulator)).

## Mrežni simulator ns-3

Mrežni simulator ns-3 nasljednik mrežnog simulatora ns-2 koji je u zadnjih dvadesetak godina izrazito popularan u akademskoj zajednici. Iako zamišljen kao poboljšana varijanta ns-2, na samom početku razvoja simulatora ns-3 odlučeno da će biti dizajniran i napisan ispočetka te biti nekompatibilan sa svojim prethodnikom. Tijekom razvoja simulatora ns-3 ideje i dijelovi koda preuzeti su iz simulatora GTNetS, yans i ns-2. Razvoj su novčano podupirali francuski Nacionalni institut za istraživanje računarske znanosti i upravljanja (Institut national de recherche en informatique et en automatique, INRIA) i američka Nacionalna zaklada za znanost (National Science Foundation, NSF).

Cilj projekta bio je stvoriti alat koji će se nastaviti razvijati od strane akademske zajednice i zainteresiranih tvrtki i nakon što prestane početno financiranje. Mnogo je uloženo u stvaranje zajednice održavatelja (engl. *maintainers*), tj. ljudi od kojih svatko održava određeni dio koda simulatora. Postavljena je infrastruktura u kojoj se svaka zainteresirana osoba može pridružiti razvoju, bilo doradom postojećih modela, bilo stvaranjem novih. Čitav kod simulatora ns-3 dostupan je pod licencom GPLv2.

Mrežni simulator ns-3 zasnovan je na diskretnim događajima. Simulirano vrijeme reprezentirano je korištenjem cjelobrojnog (integer) tipa kako bi se izbjegli problemi s prenosivosti na različite procesorske arhitekture i operacijske sustave. Ukupna veličina tipa podatka koji reprezentira vremenski trenutak je 128 bita, što omogućuje simuliranje 584 godine s nanosekundnom preciznosti.

Mrežni simulator ns-3 je implementiran u cijelosti u programskom jeziku [C++](https://en.wikipedia.org/wiki/C++) i intenzivno koristi [standardnu biblioteku predložaka](https://en.wikipedia.org/wiki/Standard_Template_Library) (engl. *Standard Template Library*, STL). Može se prevesti u izvršni kod pomoću prevoditelja [GCC](https://en.wikipedia.org/wiki/GNU_Compiler_Collection) ili [Clang](https://en.wikipedia.org/wiki/Clang) na operacijskim sustavima Linux, FreeBSD i Mac OS X te operacijskom sustavu Windows uz korištenje okoline Cygwin. Neslužbena verzija podržava i [Microsoft Visual Studio](https://en.wikipedia.org/wiki/Microsoft_Visual_Studio).

Simulator ns-3 sastoji se od više desetaka modula od kojih ćemo ovdje koristiti tek nekoliko osnovnih (u osnovnom dijelu vježbi koristiti ćemo samo pet: `core`, `network`, `point-to-point`, `internet`, `applications`). Modul je skup nekoliko modela koji imaju određeno zajedničko svojstvo (pored navedenih modula dobar primjer je `wifi` koji okuplja modele koji se tiču komponenata mreža tipa WiFi 802.11). Bez obzira na vrlo sličan izgovor, pojmovi modula i modela imaju bitno različito značenje.

!!! note
    Službena dokumentacija mrežnog simulatora ns-3 organizirana je u tri dokumenta:

    - [ns-3 Tutorial](https://www.nsnam.org/docs/tutorial/html/) opisuje kako započeti s radom i pisanjem jednostavnih simulacija i odlična je nadopuna ovih materijala,
    - [ns-3 Manual](https://www.nsnam.org/docs/manual/html/) opisuje unutarnji dizajn simulatora i može poslužiti kao dodatna literatura za studente koji žele znati više, te
    - [ns-3 Model Library](https://www.nsnam.org/docs//models/html/) opisuje dizajn modela koji su trenutno implementirani, a strukturiran je po modulima.

## Proces stvaranja simulacije

Simulacije napisane u mrežnom simulatoru ns-3 su C++ programi. Općeniti proces stvaranja simulacije može se razdvojiti u nekoliko koraka.

1. **Definicija topologije:** stvaranje osnovnih objekata i definiranje njihovih međusobnih veza; ns-3 ima sustav kontejnera (engl. *containers*) i sustav pomoćnika (engl. *helpers*) koji olakšava ovaj proces. Mrežne topologije koje ćemo stvarati bit će u početku uglavnom linearne, a u kasnijim primjerima zvjezdaste.
1. **Primjena modela:** u simulaciju se dodaju modeli koji će se koristiti (primjerice UDP, IPv4, veze tipa točka-do-točke, aplikacije); u ns-3-u se i za ovo u većini slučajeva koristi sustav pomoćnika.
1. **Konfiguracija čvorova i veza:** nakon inicijalizacije modela postavljaju se određena njihova svojstva (primjerice, veličina paketa koje aplikacija šalje ili MTU veze tipa točka-do-točke); ns-3 ovo olakšava korištenjem sustava atributa (engl. *attribute system*).
1. **Pokretanje:** objekti simulacije stvaraju događaje, bilježe se podaci koje je korisnik zatražio, vrši se provjera je li došlo do grešaka u bilo kojem od prethodnih koraka; ns-3 ima podsustav koji vrši bilježenje (engl. *logging*) i omogućuje ispis podataka o ponašanju simuliranih objekata i to tako da korisnik zatraži ispis onih koji su mu zanmljivi.
1. **Analiza izlaznih rezultata:** nakon što je simulacija izvršena podaci o stanju čvorova i veza, simuliranom prometu se stastistički obrađuju i na temelju toga se donosi zaključak (ovaj dio nećemo raditi obzirom da u trenutnom programu studija kolegij Računalne mreže prethodi kolegiju Vjerojatnost i statistika).
1. **Vizualizacija:** neobrađeni ili obrađeni podaci simulacije mogu se animirati ili prikazati grafički (primjerice, razmatranjem broja paketa koji su prošli određenom vezom u sekundi). Grafičkim prikazom izlaznih podataka simulacije ćemo se baviti kasnije u ovim vježbama. Dio koji se tiče animacije ostavljen je za samostalno istraživanje; zainteresirani su upućeni na [NetAnim](https://www.nsnam.org/wiki/NetAnim).

## Simulacijski objekti

Simulacijski objekti u mrežnom simulatoru ns-3 implemntirani su kao C++ klase. Obzirom da u trentunom programu studija kolegij Objektno orijentirano programiranje slijedi nakon Računalnih mreža u nastavku ćemo pokazati osnove rada s objektima. Kako se nećemo baviti stvaranjem novih klasa, već samo korištenjem postojećih, **smatrat ćemo da su klase ekvivalentne strukturama**. Iako općenito klase i strukture ne rade na isti način, razlike nam ovdje nisu bitne.

Unutar klase postoje funkcije, koje nazivamo **metodama** i varijable, koje nazivamo **atributima**. (Atribut objekta nešto je općenitiji pojam od ns-3 atributa objekta; svaki ns-3 atribut je atribut, ali obrat ne mora vrijediti.) Ukoliko imamo klasu `Mobitel` i u njoj metodu `ZoviBroj112()`, poziv metode se vrši na način

``` c++
Mobitel mob;
mob.ZoviBroj112();
```

Uočimo da smo prvo stvorili novi objekt tipa `Mobitel`, a zatim pozvali metodu `ZoviBroj112()` na tom objektu. Proces stvaranja objekta neke klase nazivamo **instanciranjem**. Kao i kod struktura, moguće je koristiti pokazivače, i tada poziv metode vršimo na način

``` c++
Mobitel* mob = new Mobitel();
mob->ZoviBroj112();
```

Atributima objekta se najčešće ne pristupa izravno, što je zapravo jedina suštinska razlika u odnosu na strukture. Primjerice, ako imamo atribut `brojPozivaPrema112`, postojat će i metoda `DohvatiBrojPozivaPrema112()` i evenutalno metoda `ResetirajBrojPozivaPrema112()` koje ćemo moći koristiti na opisani način. Razloge za ovu realizaciju učit ćete na kolegiju Objektno orijentirano programiranje.

ns-3 ima nekoliko istaknutih klasa koje su sadržane u gotovo svakoj simulaciji.

- `Node` je klasa koji reprezentira čvor. Pojednostavljeno rečeno čvor je računalo, ali može biti i usmjerivač, GSM bazna stanica pa čak i mobilni telefon. Svaki objekt klase `Node` može sadržavati proizvoljan broj mrežnih uređaja (klasa `NetDevice`) i aplikacija (klasa `Application`).
- `NetDevice` je klasa koja reprezntira mrežni uređaj koji je sučelje čvora prema komunikacijskom kanalu (klasa `Channel`). Pojednostavljeno rečeno mrežni uređaj je mrežna kartica, ali može biti i preklopnik, most, pa čak i koncentrator.
- `Channel` je klasa koja reprezentira vezu između dva ili više mrežna uređaja. Kod veza tipa točka-do-točke uvijek se radi o dva uređaja.
- `Packet` je klasa koja simulira paket.

Pored ovih postoji i niz drugih pomoćnih klasa. Svaka klasa predstavlja neku od fizičkih (primjerice kabel kojim putuje signal) ili softverskih komponenti računalne mreže (primjerice UDP protokol).

!!! note
    Pored već spomenute literature koja će vam pomoći, najkorisnija će vam biti [ns-3 API dokumentacija](https://www.nsnam.org/docs/doxygen/) koja sadrži opis svih objekata i metoda koje ti objekti podržavaju. Kroz vježbe će biti dane brojne poveznice na pojedine dijelove te dokumentacije.
