---
author: Vedran Miletić, Domagoj Margan
---

# Osnove rada s emulatorom računalnih mreža

## Pojam emulacije i emulatora

U proučavanju računalnih mreža koriste se dva osnovna tipa alata, tzv. **emulatori** i **simulatori**. Postoji više načina da se definira razlika između ta dva tipa alata; mi ćemo tu razliku napraviti na temelju *vremena*, i reći da emulatori rade u realnom vremenu, a simulatori u vlastitom (simuliranom) vremenu. Dakle, ako emulator treba emulirati 120 sekundi rada mreže i aplikacija, ta će emulacija zaista trajati 120 sekundi stvarnog vremena, dok simulatora simulacija 120 sekundi rada mreže može trajati mnogo više ili mnogo manje od 120 sekundi stvarnog vremena.

U ovoj vježbi i naredne dvije koristit ćemo emulator CORE.

## Osnovne informacije o alatu CORE

Alat **Common Open Research Emulator** (CORE) je [emulator računalnih mreža](https://en.wikipedia.org/wiki/Network_emulation). CORE izrađuje prikaz stvarne računalne mreže i omogućuje korisniku rad s njom u stvarnom vremenu. Takva emulacija, obzirom da radi u stvarnom vremenu, može se povezati i sa stvarnim mrežama, a u njoj se mogu koristiti i stvarne mrežne aplikacije. CORE podržava Linux i FreeBSD, a na Linuxu radi na osnovi kontejnerske virtualizacije [LinuX Containers (LXC)](https://en.wikipedia.org/wiki/LXC).

Alat CORE je razvijen od strane istraživačkog odjela [Boeinga](https://www.boeing.com/) uz podršku [istraživačkog laboratorija Američke mornarice](https://www.nrl.navy.mil/), a zasnovan je na alatu [Integrated Multiprotocol Network Emulator/Simulator (IMUNES)](http://imunes.net/) razvijenom na [Zavodu za telekomunikacije](https://www.fer.unizg.hr/ztel) [Fakulteta elektrotehnike i računarstva Sveučilišta u Zagrebu](https://www.fer.unizg.hr/).

CORE ima [vrlo bogatu dokumentaciju](https://coreemu.github.io/core/) koja je dostupna na [službenim stranicama alata](https://www.nrl.navy.mil/Our-Work/Areas-of-Research/Information-Technology/NCS/CORE/).

## Grafičko korisničko sučelje alata CORE

CORE pokrećemo naredbom `core-gui-legacy` u terminalu ili stavkom `CORE Network Emulator` u meniju.

``` shell
$ core-gui-legacy
```

Time dobivamo prazno platno na koje možemo dodavati čvorove i veze. U traci s alatima ([dokumentacija](https://coreemu.github.io/core/gui.html#toolbar)) koja se nalazi vertikalno na lijevoj strani grafičkog sučelja imamo redom

- `selection tool`, za odabir objekata emulacije,
- `start the session`, koji pokreće emulaciju,
- `link tool`, alat za stvaranje veza,
- `network-layer virtual nodes`, alat za stvaranje čvorova koji poznaju mrežni i više slojeve ([dokumentacija](https://coreemu.github.io/core/gui.html#core-nodes))

    - `router`, usmjerivač koji koristi alat Quagga za izradu tablica usmjeravanja
    - `host`, emulirano poslužiteljsko računalo koje u zadanim postavkama ima pokrenut SSH poslužitelj,
    - `pc`, emulirano račulano koje u zadanim postavkama nema pokrenutih procesa.

- `link-layer nodes`, alat za stvaranje čvorova koji rade na sloju veze podataka ([dokumentacija](https://coreemu.github.io/core/gui.html#network-nodes))

    - `ethernet hub`, Ethernet koncentrator koji primljene pakete prosljeđuje svima koji su na njega povezani,
    - `ethernet switch`, Ethernet preklopnik koji primljene pakete prosljeđuje pametnije korištenjem tablice prosljeđivanja,

- `background annotation tools`, koji služe za estetski dojam.

Čvorove postavljamo odabirom željenog čvora iz grupe `network-layer virtual nodes` ili `link-layer nodes` i klikom na odgovarajuće mjesto. Čvorove konfiguriramo odabirom opcije `Configure` u izborniku dostupnom na desni klik.

Veze postavljamo odabirom alata `link tool`, klikom na jedan od dva željena čvor i povlačenjem do drugog. Veze konfiguriramo analogno čvorovima, desnim klikom i odabirom opcije `Configure` ili dvostrukim klikom na vezu ([dokumentacija](https://coreemu.github.io/core/gui.html#wired-networks)).

U dijalogu `link configuration` možemo postaviti:

- `Bandwidth`, u bitovima po sekundi,
- `Delay`, u mikrosekundama,
- `PER`, odnosno postotak paketa koji zadobiju greške u prijenosu,
- `Duplicate`, odnosno postotak paketa koji su duplicirani u prijenosu,
- `Color`, odnosno boju crte kojom je veza nacrtana na platnu (što je osobito značajno za estetski dojam),
- `Width`, odnosno širinu crte kojom je veza nacrtana na platnu (što je također vrlo značajno za estetski dojam).

## Pokretanje emulacije

Nakon slaganja željene mreže emulaciju pokrećemo klikom na gumb `start the session` ([dokumentacija](https://coreemu.github.io/core/gui.html#editing-toolbar)), koji zatim postaje `stop the session` ([dokumentacija](https://coreemu.github.io/core/gui.html#execution-toolbar)).

Dok je emulacija pokrenuta, ostali alati u traci s alatima ([dokumentacija](https://coreemu.github.io/core/gui.html#execution-toolbar)) su:

- `observer widgets tool`, koji omogućuje odabir značajki koje će se prikazivati kod prijelaza preko čvora pokazivačem miša,
- `plot tool`, koji omogućuje da lijevim klikom na vezu nacrtamo graf propusnosti; graf se miče desnim klikom,
- `marker`, koji služi za vizualno naglašavanje pojedinih djelova emulacije kod prezentacija,
- `two-node tool`, koji omogućuje pokretanje naredbi `ping` i `traceroute` na dva čvora; čvorovi se biraju klikom na pravokutnik pored `source node`, odnosno `destination node` i zatim na odgovarajući čvor.
- `run tool`, koji omogućuje pokretanje proizvoljnih naredbi na jednom ili više čvorova.

Za vrijeme dok je emulacija pokrenuta moguće je pristupiti ljusci svakog pojedinog čvora desnim klikom na čvor i odabirom opcije `Shell window/bash` ili dvostrukim klikom na čvor.

[Službena dokumentacija grafičkog korisničkog sučelja alata CORE](https://coreemu.github.io/core/gui.html) ima više detalja o opisanoj funkcionalnosti i ostalim mogućnostima grafičkog sučelja.

## Spremanje i učitavanje emulacijskih scenarija

Scenarije za emulaciju koji smo složili moguće je spremiti kao `.imn` datoteku korištenjem `File/Save as...` i kasnije učitati korištenjem `File/Open...` ([dokumentacija](https://coreemu.github.io/core/gui.html#file-menu)). Gotovi scenariji dostupni su u direktoriju `.core/configs` unutar kućnog direktorija korisnika.

Da bi ilustrirali dostupne gotove scenarije, naredbom

``` shell
$ core-gui-legacy .core/configs/sample2-ssh.imn
```

otvorit ćemo jedan od jednostavnijih primjera. S druge strane, jedan od atraktivnijih primjera koji uključuje mobilne čvorove čije kretanje je prikazano na platnu u stvarnom vremenu, možemo otvoriti naredbom

``` shell
$ core-gui-legacy .core/configs/sample1.imn
```

Pokretanje emulacije vrši se kao i u situaciji kada slažemo vlastitu emulaciju.

!!! caution
    CORE omogućuje istovremeno pokretanje više sesija emulacije. U slučaju da vam to počne stvarati probleme (imate zaostale sesije emulacije koje ne možete uništiti korištnjem gumba `Shutdown` ili sl.), uvijek možete izvršiti ponovno pokretanje virtualne mašine, što će osigurati čišćenje svih pokrenutih sesija i riješiti problem.

## Emulacija mrežnog prometa korištenjem alata MGEN

CORE omogućuje korištenje alata Multi-Generator (MGEN) za generiranje prometa. MGEN također razvijen od strane istraživačkog laboratorija Američke mornarice, a moguće ga je koristiti i neovisno o alatu CORE. Mi se tim načinom korištenja ovdje nećemo baviti; za više informacija proučite [službenu dokumentaciju](https://github.com/USNavalResearchLaboratory/mgen/blob/master/doc/mgen.pdf) dostupnu na [službenim stranicama alata MGEN](https://www.nrl.navy.mil/Our-Work/Areas-of-Research/Information-Technology/NCS/MGEN/).

Odabirom opcije `Tools/Traffic...` ([dokumentacija](https://coreemu.github.io/core/gui.html#tools-menu)) otvara se dijalog `CORE traffic flows` u kojem je moguće klikom na gumb `new` definirati novi tok paketa. Odabir izvornog i odredišnog čvora vrši se klikom na pravokutnik pored `source node`, odnosno `destination node` i zatim na odgovarajući čvor. Pored toga moguće je konfigurirati:

- `port` za izvor i odredište,
- `protocol`, TCP ili UDP,
- `pattern`, odnosno pravilo po kojem će se paketi slati.

U dijelu `Traffic options` moguće je postaviti način na koji se pokreću podatkovni tokovi:

- kod opcije `Do not start traffic flows automatically` očekuje se da korisnik nakon pokretanja emulacije pokrene tokove ručno opcijom `Start all flows` ili `Start selected` tok po tok, dok
- kod opcije `Start traffic flows after all nodes have booted` postavlja automatsko pokretanje svih tokova nakon što se pokrenu svi čvorovi.

Stvaranje i konfiguriranje tokova moguće je prije i nakon pokretanja emulacije.

!!! caution
    Preporuča se korištenje varijante `Do not start traffic flows automatically`; tokove možete ručno pokrenuti nekoliko desetaka sekundi nakon pokretanja same emulacije kada su usmjerivači proveli proces usmjeravanja i svi čvorovi iz različitih mreža mogu međusobno komunicirati.

## Dodatak: emulacija mrežnog prometa korištenjem netcata

Za generiranje prometa možemo se puslužiti i jednostavim trikom koji uključuje funkcije ljuske `bash` i alat netcat (naredba `nc`). Kako bi emulirali protok prometa mrežom, napisati ćemo dva kratka izraza: jedan za slanje, a drugi za primanje paketa. Za naše potrebe je dovoljno ih samo znati koristiti, te nećemo ulaziti u detalje po pitanju značenja.

Na strani primatelja (servera) u ljusci upisujemo:

``` shell
$ while true; do nc -lp PORT >/dev/null; done
```

pri čemu je `PORT` je broj vrata na koje paket primamo.

Na strani pošiljatelja (klijenta) u ljusci upisujemo:

``` shell
$ while true; do echo "foo" | nc -w 1 X.Y.Z.W PORT; sleep 1; done
```

pri čemu je `X.Y.Z.W` IP adresa primatelja paketa, dok je `PORT` broj vrata na koje paket šaljemo.

Ukoliko istovremeno pokrenemo slanje paketa sa više pošiljatelja na ista vrata jednog primatelja, možemo dobiti poruku `"Connection refused"`. Tu poruku dobijemo zbog kolizije pri primanju nekih paketa na ista vrata sa više izvora, no u tom slučaju paketi koji ne dođu u koliziju putem tih vrata ipak stižu na primatelja.
