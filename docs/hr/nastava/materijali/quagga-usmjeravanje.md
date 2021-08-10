---
author: Domagoj Margan, Vedran Miletić
---

# Rad s alatima za usmjeravanje

## Pojam protokola usmjeravanja

S obzirom na mrežni sloj OSI modela, Internet svoj rad temelji na usmjeravanju paketa. Kada paketi putuju mrežom do svog odredišta donose se odluke o njihovom usmjeravanju. U središtu računalne mreže je usmjerivač (engl. *router*) koji odlučuje kojim putem će paket nastaviti mrežom kako bi stigao do odredišta.

Protokoli usmjeravanja omogućuju usmjerivačima da dinamički dijele informacije o udaljenim mrežama i automatski dodaju informacije u svoje tablice usmjeravanja. Oni određuju najbolji put do neke mreže koji je zatim dodan u tablicu. Jedna od najznačajnijih karakteristika dinamičkih protokola usmjeravanja je to što usmjerivači razmjenjuju informacije kada se dogodi promjena u topologiji. Ta razmjena omogućuje usmjerivačima učenje o novim mrežama i novim alternativnim putevima koje bi koristili kada bi se desio pad neke veze. Protokol usmjeravanja je u osnovi skup procesa, poruka i algoritama koji se koriste u razmjeni informacija među usmjerivačima i u popunjavanju tablice usmjeravanja.

### Svojstva protokola usmjeravanja

Glavne značajke protokola usmjeravanja su:

- otkrivanje udaljenih mreža
- održavanje informacija
- određivanje najboljeg puta do odredišne mreže
- mogućnost pronalaska alternativnog puta ukoliko trenutni put više nije dostupan

Određivanje najboljeg puta do odredišne mreže podrazumijeva procjenu različitih puteva i odabir optimalnog odnosno najkraćeg puta. Optimalan put do odredišta usmjerivač određuje na temelju vrijednosti odnosno metrike. Metrika je kvantitivna vrijednost kojom se izražava udaljenost rute pri čemu je najbolji put do udaljene mreže put s najmanjom metrikom. Ukoliko postoji više ruta sa istom metrikom, usmjerivač će koristiti sve rute ravnomjerno raspoređujući promet između njih.

Svaki protokol usmjeravanja karakterizira i administrativna udaljenost (engl. *administrative distance*) koju usmjerivač također koristi pri određivanju najboljeg puta do odredišne mreže. Administrativna udaljenost je pouzdanost određene rute izražena kao numerička vrijednost između 0 i 255. Što je ta vrijednost veća, pouzdanost rute je manja. Vrlo je korisna ukoliko usmjerivač koristi više protokola usmjeravanja te će u tom slučaju odabrati rutu s najmanjom administrativnom udaljenosti.

S obzirom na područje u kojem djeluju, protokoli usmjeravanja dijele se na unutarnje (engl. *interior gateway protocol*) i vanjske protokole (engl. *exterior gateway protocol*). Mi ćemo se fokusirati na unutarnje protokole usmjeravanja; informacije o vanjskim protokolima usmjeravanja moguće je naći na [Wikipedijinoj stranici na temu BGP-a](https://en.wikipedia.org/wiki/Border_Gateway_Protocol) i drugdje.

### Unutarnji protokoli usmjeravanja

Postoje dva tipa unutarnjih protokola usmjeravanja:

- **protokoli vektora udaljenosti** (engl. *distance vector*): [RIP](https://en.wikipedia.org/wiki/Routing_Information_Protocol), [IGRP](https://en.wikipedia.org/wiki/Interior_Gateway_Routing_Protocol), [EIGRP](https://en.wikipedia.org/wiki/Enhanced_Interior_Gateway_Routing_Protocol), te
- **protokoli stanja veze** (engl. *link state*): [OSPF](https://en.wikipedia.org/wiki/Open_Shortest_Path_First), [IS-IS](https://en.wikipedia.org/wiki/IS-IS).

Vektor udaljenosti znači da su rute odabrane na temelju udaljenosti odnosno metrike kao što je broj skokova kod RIP-a, a smjer je izlazno sučelje usmjerivača ili susjedni usmjerivač. Protokoli vektora udaljenosti podrazumijevaju periodičko slanje podataka o usmjeravanju svim susjednim usmjerivačima svakih 30 sekundi kako bi se održavale tablice usmjeravanja. Na taj način se tablice usmjeravanja redovito osvježavaju što je vrlo važno ukoliko se desi promjena u topologiji mreže poput pada veze, stvaranja nove veze, pada nekog od usmjerivača ili promjene parametara veza. Vektorom udaljenosti usmjerivači mogu sakupiti relativno dovoljan broj informacija o mreži -- metriku do udaljene mreže i rutu odnosno izlazno sučelje koje će koristiti za slanje paketa do udaljene mreže -- on ne pruža uvid u topologiju čitave mreže.

Protokoli stanja veze pružaju uvid u čitavu topologiju mreže što znači da svaki usmjerivač uči o svim putevima mreže. Cijena neke rute definirana je kao sumirana vrijednost cijena svih puteva od izvorišnog usmjerivača do odredišnog. Svaki usmjerivač popunjava tablicu usmjeravanja izračunavajući cijene ruta do svih odredišta mreže. Za razliku od protokola vektora udaljenosti, protokoli stanja veze šalju periodičke informacije samo pri procesu konvergencije i kada se desi promjena u topologiji.

Protokoli stanja veza kao što su OSPF i IS-IS općenito se teže implementiraju i rješavanje problema je kompliciranije u odnosu na protokole vektora udaljenosti kao što su RIP i EIGRP. Međutim, protokoli stanja veza nude veću skalabilnost kada se koriste u velikim i kompleksnim mrežama. Osim toga, obično se brže oporave od problema odnosno brže konvergiraju od protokola vektora udaljenosti. Protokole vektora udaljenosti lakše je implementirati, ali nisu optimalno rješenje za veće mreže obzirom da mreže sporije konvergiraju, sklonije su problemima poput beskonačnih petlji i padova veza.

## Svojstva softverskog paketa za usmjeravanje Quagga

[Quagga](https://en.wikipedia.org/wiki/Quagga_(software)) je softverski paket za implementaciju protokola usmjeravanja. Quagga pruža mogućnost implementacije protokola OSPFv2, OSPFv3, RIP V1 i V2, RIPng i BGP-4 za Linux, FreeBSD, Solaris i NetBSD platforme. Dostupna je za instalaciju kao paket na većini Linux distribucija.

Kao i većina alata s kojima radimo, Quagga jednako dobro radi na stvarnim i emuliranim mrežnim sučeljima.

## Procesi daemoni alata Quagga

Jezgru Quaggae čini centralni proces daemon nazvan `zebra`, koji se ponaša kao [sloj apstrakcije](https://en.wikipedia.org/wiki/Abstraction_layer) za sučelja jezgre operacijskog sustava Linux. Njegova je uloga da olakšava interakciju ostalih daemona s jezgrom operacijskog sustava kod zadaća kao što su promjene u tablici usmjeravanja.

Daemoni koji komuniciraju s daemonom zebra nazivaju se **Zserv** daemonima; svaki od njih je specifičan za pojedini protokol usmjeravanja. Zserv daemoni šalju informacije o dostupnim rutama daemonu `zebra` koji ih onda na odgovarajući način ujedinjava i unosi u tablicu usmjeravanja. Postojeći Zserv daemoni i pripadne implementacije su:

- `ospfd`, implementira Open Shortest Path First (OSPFv2)
- `ospf6d`, implementira Open Shortest Path First (OSPFv3) za IPv6
- `ripd`, implementira Routing Information Protocol (RIP) verziju 1 i 2;
- `ripngd`, implementira Routing Information Protocol (RIPng) za IPv6
- `isisd`, implementira Intermediate System to Intermediate System (ISIS)
- `bgpd`, implementira Border Gateway Protocol (BGPv4+), za IPv4 i IPv6
- `babeld`, implementira Babel routing protocol, za IPv4 i IPv6

Primjere konfiguracijskih datoteka za svaki daemon možemo pronaći u direktoriju `/usr/share/doc/quagga/examples`. Za naše potrebe fokusirati ćemo se samo na `ospfd` i `ripd`. Za ostale daemone informacije je moguće pronaći u [službenoj dokumentaciji](https://www.nongnu.org/quagga/docs.html).

## Sintaksa i naredbe Quagga konfiguracijskih datoteka

!!! note
    Službena dokumentacija Quagge, specifično njen [dio o parametrima mrežnih sučelja](https://www.nongnu.org/quagga/docs/docs-multi/Link-Parameters-Commands.html), koristi zapis u kojem su rasponi brojeva omeđeni šiljastim zagradama. To nije indikacija sintakse kako se pišu konfiguracijske naredbe, npr. unos u dokumentaciji `metric <0-4294967295>` znači da su ispravne konfiguracijske naredba `metric 0`, `metric 1`, ..., `metric 4294967295`.

Općenite naredbe:

- `!` i `#`

    - znakovi za komentiranje
    - ukoliko je prvi znak u liniji `!` ili `#`, ostatak linije se ignorira
    - ukoliko se `!` ili `#` nalaze na drugim pozicijama, njih se tretira kao i svaki drugi znak

- `hostname hostname`

    - postavljanje imena usmjerivača

Konfiguracija mrežnog sučelja:

- `interface ifname`

    - konfiguracija sučelja `ifname` (najčešće `eth0`, `eth1`, `eth2`, ...)

- `ip address a.b.c.d/m`

    - IPv4 adresa sučelja

- `bandwidth <1-10000000>`

    - širina frekventnog pojasa sučelja **mjerena u kilobitima po sekundi** (npr. `bandwidth 10000` znači veza širine pojasa 10Mbit/s)
    - **postavka neovisna o fizičkim svojstvima veze**; nema automatskog prepoznavanja širine frekventnog pojasa, potrebno je ručno sučelja na usmjerivaču u skladu s fizičkim karakteristikama

Konfiguriranje OSPFv2 procesa:

- `router ospf`

    - omogućavanje OSPF procesa
    - **sve** sljedeće naredbe za konfiguraciju OSPF procesa pišu se ispod ove linije, svaki novi red započinje sa jednim razmakom uvlake

- `no router ospf`

    - onemogućavanje OSPF procesa

- `ospf router-id a.b.c.d`

    - postavljanje identiteta usmjerivača (router-ID) za OSPF proces
    - identitet može biti IP adresa usmjerivača ili bilo koji 32bitni broj unikatan za OSPF domenu

- `timers throttle spf delay initial-holdtime max-holdtime`

    - postavljanje delaya, inicijalnog vremena zadržavanja i maksimalnog vremena zadržavanja
    - vrijeme se specificira u miliseknudama u rasponu od 0 do 600000

- `max-metric router-lsa [on-startup|on-shutdown] <5-86400>`
- `max-metric router-lsa administrative`

    - omogućavanje OSPF procesu da opiše udaljenosti svojih veza kao beskonačne; ostali usmjerivači na taj način izbjegavaju računanje udaljenosti preko tog usmjerivača, no i dalje mogu pristupiti ostatku mreže preko tog usmjerivača
    - može se odredit administrativno (beskonačno dugo) ili na određen broj sekundi nakon pokretanja ili prije gašenja

- `auto-cost reference-bandwidth <1-4294967>`

    - postavljanje referencnog bandwidtha za izračune cijene veze
    - zadana postavka je 100 Mbit/s (sve veze sa bandwidthom 100 Mbit/s ili više imati će cijenu 1, a svim vezama sa manjim bandwidthom će se izračunati cijena ovisno o zadanoj postavci)

- `network a.b.c.d/m area a.b.c.d`

    - omogućavanje OSPF procesa na određenom rasponu određenog mrežnog sučelja

Konfiguracije mrežnog sučelja specifične za OSPFv2:

- `ip ospf hello-interval <1-65535>`

    - vrijeme u sekundama između slanja "Hello" paketa kojim se susjedni usmjerivači otkrivaju i razmjenjuju informacije; zadana vrijednost je 10

- `ip ospf dead-interval <1-65535>`

    - vrijeme u sekundama koje će usmjerivač čekati na dolazak "Hello" paketa od susjeda, ukoliko se prekorači susjedni sumjerivač se smatra "mrtvim"; zadana vrijednost je 40

Konfiruriranje RIP procesa:

- `router rip`

    - omogućavanje RIP procesa
    - SVE iduće naredbe za konfiguraciju RIP procesa pišu se ispod ove linije, svaki novi red započinje sa jednim razmakom uvlake

- `no router rip`

    - onemogućavanje RIP procesa

- `network network`

    - omogućavanje RIP procesa za određenu mrežu

- `neighbor a.b.c.d`

    - definiranje RIP susjeda

- `no neighbor a.b.c.d`

    - onemogućavanje RIP susjeda

- `redistribute ospf`

    - prosljeđivanje informacija o OSPF usmjeravanju u RIP tablice

- `distance <1-255>`

    - određivanje RIP udaljenosti

- `distance <1-255> a.b.c.d/m`

    - određivanje RIP udaljenosti u slučaju kad izvorišta IP adresa odgovara specificiranoj vrijednosti

## Sučelje vtysh za konfiguraciju Quagga daemona

Prompt je oblika `n0#`, pri čemu je `n0` ime čvora. Neke od bitnijih naredba su:

- `n0# ?`

    - help mode -- ispis liste mogućih komandi

- `n0# show ?`

    - popis svih mogućih show komandi

- `n0# show running-config`

    - prikaz konfiguracije routera u trenutnom izvođenju

- `n0# show interface`

    - detaljne informacije o sučeljima routera

- `n0# show daemons`

    - informacije o aktivnim daemonima

- `n0# show ip route`

    - provjera postojećih zapisa u tablicama usmjeravanja

- `n0# show ip ospf` ili `n0# show ip rip`
- `n0# show ip ospf neighbor` ili `n0# show ip rip neighbor`
- `n0# configure terminal`

    - ulazak u Global configuration mode, prompt se mijenja u `n0(config)#`

- `n0(config)# interface eth0`

    - ulazak u Interaface configuration mode, prompt se mijenja u `n0(config-if)#`
    - oznaka sučelja je varijabilna (ne mora nužno biti `eth0`)

- `n0(config)# router ospf` ili `n0(config)# router rip`

    - ulazak u Router configuration mode, prompt se mijenja u `n0(config-router)#`

- `n0(config)# no router ospf` ili `n0(config)# no router rip`

    - isključivanje OSPF-a, odnosno RIP-a

- `n0(config-router)# network 10.1.1.2/24`

    - postavljanje određene IP adrese

- `n0(config-if)# exit` i `n0(config-router)# exit`

    - po izvršenju, prelazi se u "nadređeni" Global configuration mode, s promptom `n0(config)#`

- `n0(config)# exit`

    - po izvršenju, prelazi se u "nadređeni" glavni izbornik, s promptom `n0#`
