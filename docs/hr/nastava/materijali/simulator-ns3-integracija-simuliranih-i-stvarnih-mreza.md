---
author: Marina Miler, Vedran Miletić
---

# Integracija simuliranih i stvarnih mreža

!!! todo
    Napiši bolje kad makneš viškove ispod.

Već smo vidjeli da je [projekt ns-3](https://www.nsnam.org/) izgradio je robusnu jezgru simulatora, koja je uz to i dobro dokumentirana, jednostavna za korištenje i debagiranje, te pruža usluge čitavog simulacijskog tijeka, od konfiguriranja simulacije, pa sve do sakupljanja praćenih izvora (engl. *trace collection*) i analiza.

Nadalje, infrastruktura ns-3 softvera potiče razvoj simulacijskih modela koji su dovoljno realni da dopuštaju da ns-3 bude korišten kao real-time mrežni emulator povezan sa stvarnih svijetom i koji dopuštaju mnoge postojeće real-world implementacije protokola da se ponovno upotrebljuju i ns-3-om. ns-3 simulacijska jezgra dopušta istraživanja na mrežama koje su bazirane i koje nisu bazirane na IP.

Najveći broj korisnika fokusiran je na wireless/IP simulacije koje uključuju modele za Wi-Fi, WiMAX ili LTE za sloj 1 i 2, te veliki broj statičkih i dinamičkih protokola za usmjeravanje (engl. routing), kao šro su npr. OLSR i AODV za aplikacije bazirane na IP. ns-3 podržava i real-time planer (engl. scheduler) koji nudi "simulacije-u-petlji" (engl. simulation-in the-loop) use case-ove za interakciju sa stvranim mrežnim uređajima. Npr. korisnici mogu emitirati i primati pakete generirane od ns-3-a na stvarne mrežne uređaje, a ns-3 može koristiti kao okvir za povezivanje efekata između virtualnih mašina.

Još jedna značajka ovog simulatora je mogućnost ponovnog korištenja stvarnih aplikacija i kernel koda. Okviri za pokretanje nemodificiranih aplikacija ili čitavog Linux kernel mrežnog stoga unutar ns-3-a trenutno se testiraju i razvijaju.

## Povijest

!!! todo
    Stavi negdje drugdje ovo i dva sljedeća dijela.

Rad na [ns-3-u](https://en.wikipedia.org/wiki/Ns_(simulator)) započeo je 2004/2005. akademske godine. Tim kojeg je predvodio Tom Henderson sa Sveučilišta u Washingtonu, a čiji su ostali članovi George Riley sa Georgia Techa, Sally Floyd s International Computer Science Institute i Sumit Roy sa Sveučilišta u Washingtonu. Tim je uspio pridobiti finacijsku pomoć od američke [Nacionalne zaklade za znanost](https://www.nsf.gov/) (engl. *National Science Foundation*, kraće NSF), da stvori zamjenu za ns-2, pod imenom ns-3. U isto vrijeme, Planete research team na [INRIA Sophia Antipolis-u](https://www.inria.fr/), čiji su glavni članovi bili Mathieu Lacage i Walid Dabbous, počeo je istraživati zamjenu za ns-2 s početnim isticanjem IEEE 802.11 Wi-Fi modela. Lacargeov početni simulator bio je nazvan [Yet Another Network Simulator](https://hal.inria.fr/inria-00078318v2) (yans).

Dva tima su se ujedinila i započele su diskusije oko dizajna ns-3-a u veljaču 2005. Glavni ciljevi projekta uključivali su bolju implementaciju mrežnih emulacija i ponovno korištenje postojećeg koda, kako bi se alat bolje integrirao sa testbed-based istraživanjem. U procesu nastanka ns-3-a, odlučeno je da se odbaci kompatibilnost s ns-2. Razlog tome je da bi najviše napora trebalo biti uloženo u održavanje takvog sustava. Odlučeno je da će novi simulator biti stvoren skroz iz početka, te da će biti programiran u C++-u.

Razvoj ns-3-a započeo je 01. srpnja 2006. Veći dio jezgre napisao je Mathieu Lacage, posudivši pritom dio od yans simulatora (simulator kojeg je izgradio George Riley -- Georgia Tech Network Simulator(GTNetS)) i od ns-2-a. Okvir za generiranje Python povezivanja (engl. bindings) pod imenom pybindgen i korištenje sustava za izgradnju softvera Waf (engl. Waf build system) dodao je Gustavo Carneiro .

Prva verzija ns-3-a objavljena je u lipnju 2008. godine. ns-3 objavio je svoju petnaestu verziju (ns-3.15) u trećem kvartalu 2012.

Trenutni status triju verzija ns-a jest:

- ns-1 više se ne razvija niti se održava
- ns-2 samo se održava
- ns-3 se aktivno razvija

## Dizajn

ns-3 programiran je u C++-u i Pythonu. ns-3 biblioteka povezana je s pythonom pomoću pybindgen biblioteke, koja omogućava parsiranje ns-3 C++ zaglavlja u gccxml i pygccxml kako bi se automatski generirala C++ veza (engl. binding glue). Automatski generirane C++ datoteke na kraju se kompajliraju i nastaje ns-3 python modul, kako bi se korisnicima omogućila interakcija s C++ ns-3 modelima i jezgrom preko python skripti.

## Tijek simulacije

1. Definicija topologije -- kako bi se olakšalo stvaranje osnovnih sadržaja i definiranje njihovih odnosa, ns-3 ima sustav kontejnera (engl. container) i pomoćnika (engl. helper) koji omogućuju taj proces.
1. Primjena modela -- modeli se dodaju simulacijama. U većini slučajeva to se odvija pomoću pomoćnika.
1. Konfiguracija čvora i veze -- modeliraju svoje zadane vrijednosti. U većini slučajeva to se odvija pomoću sustava atributa.
1. Izvršavanje -- simulacijsko postrojenje generira događaje, odnosno podatke koje je zadao logirani korisnik.
1. Analiza performansi -- nakon što je simulacija završena i podaci su dostupni kao vremenski pečatirani tragovi događaja (engl. time-stamped event trace ). Ti se podaci zatim mogu statistički analizirati.
1. Grafička vizualizacija -- neobrađeni ili procesirani podaci prikupljeni prilikom simulacije mogu se prikazati pomoću grafa pomoću različitih alata, kao što su npr. Gnuplot, matplotlib, Xgraph.

## Interakcija stvarnih i simuliranih mreža

Primarna funkcija ns-3-a je simulacija mreža. Često se javlja potreba za kombinacijom stvarnog svijeta i simuliranih entiteta. Npr. kako bi se potvrdio mrežni model, može se simulirati određeni scenarij, pokrenuti simulirani model i prikupljati statistike. Drugi način je isti scenarij ponoviti na stvarnom hardveru. Kada se potom usporede statistike, može se zaključiti da je simulacija prilično dobro simulirala stvarni hardver. Pošto je realna mreža skupa, može se simulirati i samo mreža, a da se pritom koristi pravi hardver, odnosno realne aplikacije pomoću kojih onda možemo ostvariti virtualizaciju.

- Stvarni čvorovi i stvarne mreže: vaša računala i računalna mreža
- [Stvarni čvorovi i simulirane mreže](https://www.nsnam.org/wiki/HOWTO_make_ns-3_interact_with_the_real_world#Real_Nodes_and_Simulated_Networks): ns-3 TAP
- [Simulirani čvorovi i stvarne mreže](https://www.nsnam.org/wiki/HOWTO_make_ns-3_interact_with_the_real_world#Simulated_Nodes_and_Real_Networks): ns-3 EMU
- Simulirani čvorovi i simulirane mreže: ns-3 simulacija

### Stvarni čvorovi i simulirane mreže

Ovaj slučaj koristan je kada želimo koristiti stvarne aplikacije, ali ne želimo uspostavljati velike mreže. U ovom slučaju stvaramo virtualne čvorove i povezujemo ih simuliranim mrežama. Postoje različite mogućnosti virtualizacije. Način virtualizacije o kojemu govorimo je platformni način vizualizacije. Ovaj način dijeli se na potpunu vizualizaciju (engl. full visualization) i paravirtualizaciju (engl. paravirtualization).

Kod potpune virtualizacije, stvorena je okolina virtualne mašine koja u potpunosti simulira hardver i OS okruženje. Koristeći ovu tehnologiju jedan se sustav može prikazati kao skup sustava istog ili različitog tipa. Npr. može se pokrenuti virtualizacijski sustav na Windowsima koji daje osjećaj da je pokrenut veliki broj Windows sustava, veliki broj Linux sustava ili kombinacija ovih ili drugih okruženja. Primjer virtualizacijskih sustava su VmWare i VirtualBox. Potpuno virtualizirana okruženja smatraju se složenim (engl. heavyweight ) sustavima. To je zbog toga što svaki aspekt čvora mora biti simuliran, a zasebne virtualne mašine moraju u potpunosti biti izolirane jedna od druge.

Kod paravirtualizacije stvara se okolina virtaulne mašine, ali se hardver i OS ne simuliraju u potpunosti. Ovo može zahtjevati portanje (engl. porting) virtualiziranog operativnog sustava virtualizacijskoj okolini ili se OS koristi za stvaranje iluzije virtualizacije određenih dijelova softvera. Druga je opcija lakše izvediva i iz tog razloga prigodnija za pokretanje mnogih instanci virtualnih (simuliranih) čvorova u ns-3-u.

Primjer paravirtualizacijskih sustava su Linux Containers i OpenVz. Ovakvo rješenje dozvoljava stvaranje Linux virtualnih "guest" sustava na Linux domaćinu, ali ne dozvoljava stvaranje virtualnih Windows sustava na Linux domaćinu.

#### TAP

ns-3 ima mogućnost fukcioniranja pod potpunom virtualizacijom i paravirtualizacijom. Kako je ns-3 sustav na osnovi Linuxa (engl. Linux-based system), koristi se Linuxov mehanizam kako bi se implementirala potrebna funkcionalnost. Taj mehanizam zove se TapBridge. Naziv "Tap" dolazi od tun/tap uređaja koji je u Linuxu device driver i koristi se za povezivanje ns-3-a i Linux guest OS-a, a naziv "Bridge" dolazi od činjenice da konceptualno proširuje Linuxov most u ns-3.

### Simulirani čvorovi i stvarne mreže

Iako rjeđe dolazi do ove situacije, korištenje simuliranih čvorova sa realnim mrežama može biti bitna kod validacijskih modela. Ako koristimo simulacijski model kako bismo otkrili kako bi stvarni sustav reagirao, treba postojati osnova radi koje ćemo biti sigurni u naše rezultate, tj. da će procjena reakcije dati točni rezultat.

Svaki model ima ciljni sustav kojeg nastoji simulirati do nekog određenog detalja i točnosti. Proces nastojanja da se ponašanje modela poklapa sa ponašanjem ciljnog sustava naziva se validacija modela. Proces validacije u principu je vrlo jednostavan. Uspoređuje se ponašanje modela i stvarnog sustava. Pritom se model prilagođava kako bi se poboljšala korelacija.

Korištenje ns-3 aplikacija za pokretanje i simuliranih mreža i realnih mreža olakšava usporedbu. O tome ovdje nećemo govoriti.

## Korištenje Linux kontejnera za stvaranje virtualnih mreža

Postoji mogućnost simuliranja mreže i korištenja "pravih" domaćina. Najčešće je broj pravih domaćina relativno velik, pa bi takva solucija bila skupa. Iz tog razloga koristi se virtualizacijska tehnologija za stvaranje tzv. virtualnih mašina. Na njima se pokreće softver kao da su pravi domaćini.

Ovdje ćemo koristiti [Linux kontejnere](https://linuxcontainers.org/) (lxc) i pratiti [upute s ns-3-evog Wikija](https://www.nsnam.org/wiki/HOWTO_Use_Linux_Containers_to_set_up_virtual_networks). Da bi koristili kontejnere, najprije treba instalirati potrebne alate. To činimo pomoću sljedeće linije koda:

``` shell
$ sudo yum install lxc bridge-utils tunctl
```

Pogledajmo konfiguraciju lijevog i desnog kontejnera na putanji `~/repos/ns-3-allinone/ns-3-dev`:

``` shell
$ less src/tap-bridge/examples/lxc-left.conf
$ less src/tap-bridge/examples/lxc-right.conf
```

Prije stvaranja kontejnera, trebamo storiti mostove. Mostovi su "signalne putanje" po kojima će paketi izlaziti iz kontejnera.

``` shell
$ sudo brctl addbr br-left
$ sudo brctl addbr br-right
```

Moramo kreirati tap uređaje koje ns-3 koristi za primanje paketa sa mostova.

``` shell
$ sudo tunctl -t tap-left
$ sudo tunctl -t tap-right
```

Prije dodavanja tap uređaja, moramo podesiti njihovu IP adresu na 0.0.0.0 i postaviti ih.

``` shell
$ sudo ifconfig tap-left 0.0.0.0 promisc up
$ sudo ifconfig tap-right 0.0.0.0 promisc up
```

Sada dodajemo tap uređaje koje smo upravo stvorili njihovim mostovima, dodajemo IP adrese mostovima i postavljamo ih.

``` shell
$ sudo brctl addif br-left tap-left
$ sudo ifconfig br-left up
$ sudo brctl addif br-right tap-right
$ sudo ifconfig br-right up
```

Provjerimo jesmo li dobro postavili mostove i tap uređaje:

``` shell
$ sudo brctl show
```

Trebamo se pobrinuti da kernel ima isključeno ethernet filtriranje.

``` shell
$ cd /proc/sys/net/bridge
$ sudo -s
# for f in bridge-nf-*; do echo 0 > $f; done
# exit
$ cd -
```

Sada možemo pokrenuti kontejnere, a pritom je prvi korak stvaranje kontejnera. Najprije ćemo stvoriti lijevi, a zatim desni kontejner.

``` shell
$ sudo lxc-create -n left -f lxc-left.conf
$ sudo lxc-create -n right -f lxc-right.conf
```

Sljedećom naredbom ispisujemo stvorene kontejnere.

``` shell
$ sudo lxc-ls
left right
```

Pokrenimo kontejner za lijevog domaćina. Upišimo u terminal sljedeće:

``` shell
$ sudo lxc-start -n left /bin/bash
```

U novi terminal upisujemo naredbu:

``` shell
$ sudo lxc-start -n right /bin/bash
```

U još jedan terminal upisujemo:

``` shell
$ ./waf configure --enable-examples --enable-tests --enable-sudo
$ ./waf build
```

Pokrenimo simulaciju.

``` shell
$ ./waf --run tap-csma-virtual-machine
```

Simulacija će trajati 10 minuta, ako je prije ne prekinemo s ++control+c++. Ako u lijevi terminal upišemo:

``` shell
# ping -c 4 10.0.0.2
```

šaljemo pakete od lijevog kontejnera prema desnom, koji ima adresu 10.0.0.2, a zatim od desnog prema lijevom na adresi 10.0.0.1.

``` shell
# ping -c 4 10.0.0.1
```

Kada mijenjamo vrijednost zadržavanja paketa, te ovisno o tome šaljemo li pakete s jednog kontejnera na drugi u isto vrijeme ili nakon što završi slanje jednog kontejnera, mijenja se vrijednost vremena i odbacivanja paketa pri slanju.

Slanjem paketa prilikom različitih postavki zadržavanja paketa i različitog vremenskog razmaka između slanja sa lijevog na desni kontejner i obrnuto može se zaključiti da slanje paketa pri manjem zadržavanju rezultira manjim brojem odbaćenih paketa. Osim toga, znatno je manji broj odbačenih paketa kada drugi kontejner pošalje svoje pakete tek kada prvi završi sa slanjem.
