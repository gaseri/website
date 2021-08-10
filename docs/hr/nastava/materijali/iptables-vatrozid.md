---
author: Domagoj Margan, Vedran Miletić
---

# Filtriranje paketa vatrozidom

## Pojam vatrozida i proces filtriranja paketa vatrozidom

[Vatrozid](https://en.wikipedia.org/wiki/Firewall_(computing)) (engl. *firewall*) je uređaj ili aplikacija koja služi da dozvoli ili zabrani prijenos podataka mrežom na osnovu određenog skupa pravila. Najčešće se koristi za istovremeno dozvoljavanje legitimnog i onemogućavanje nelegitimnog pristupa izvana lokalnim mrežama (ili specifičnom domaćinu unutar mreže).

!!! todo
    Ovdje fali jedan paragraf objašnjenja procesa filtriranja.

U praksi najčešće korišteni vatrozidi zasnovani na filterima paketa su

- [IPFilter](https://en.wikipedia.org/wiki/IPFilter) (`ipf`), dostupan na FreeBSD-u, NetBSD-u i Solarisu, ali podržava i ostale operacijske sustave slične Unixu,
- [ipfirewall](https://en.wikipedia.org/wiki/Ipfirewall) (`ipfw`), dostupan na FreeBSD-u i Mac OS X-u,
- [NPF](https://en.wikipedia.org/wiki/NPF_(firewall)) (`npf`), dostupan na NetBSD-u,
- [Packet Filter (PF)](https://en.wikipedia.org/wiki/PF_(firewall)) (`pf`), dostupan na FreeBSD-u i ostalim BSD-ima te Mac OS X-u,
- [nftables](https://en.wikipedia.org/wiki/Nftables) (`nft`) i [iptables](https://en.wikipedia.org/wiki/Iptables) (`iptables`) zasnovani na [Netfilteru](https://en.wikipedia.org/wiki/Netfilter) i namijenjeni za Linux.

Mi ćemo se fokusirati na iptables, koji je trenutno standard na Linuxu. nftables bi trebao u bližoj budućnosti zamijeniti iptables kao standardni vatrozid na Linuxu; kada se to dogodi, osvježit ćemo ove materijale. Vrijedi spomenuti i da je [ipchains](https://en.wikipedia.org/wiki/Ipchains) (naredba `ipchains`) stariji alat zamijenjen iptablesom od verzije Linux jezgre 2.4.

## Način rada alata iptables

Alat iptables koristimo kako bi postavili, mijenjali i pregledavali tablice pravila za fliteriranje IPv4 paketa i prevođenje mrežnih adresa u Linux operacijskom sustavu. Moguće je definirati više različitih tablica. Svaka tablica sadrži određen broj različitih lanaca (engl. *chains*), a svaki je lanac lista pravila kojima se određuje što sustav čini sa mrežnim paketima koje to pravilo obuhvaća.

Alatu pristupamo putem ljuske naredbom `iptables`. U ljusci upisujemo `iptables` konstrukte narebi i parametara, baš kao što to radimo i sa već poznatim naredbama i alatima unutar komandne linije. Kao i druge naredbe koje se bave konfiguracijom mreže, jednako dobro radi na stvarnim i emuliranim čvorovima. Vrijedi napomenuti da se kod ponovnog pokretanja stvarnog ili emuliranog računala postavke brišu.

### Tablice, lanci pravila i ciljevi

Alat iptables za ostvarivanje različitih akcija nad paketima koristi iduće tablice (engl. *tables*):

- (`raw`),
- `mangle` -- postavlja TOS ili DSCP,
- `nat` -- vrši NAT,
- `filter` -- vrši filtriranje paketa.

iptables svrstava mrežni paket prema tome gdje je nastao, odnosno za koji domaćin je namijenjen. S obzirom na to, postoje idući lanci (engl. *chains*) pravila:

- `PREROUTING` -- paketi pristigli s mreže; u ovom lancu se još ne zna jesu li ili ne za ovog domaćina,
- `POSTROUTING` -- paketi poslani na mrežu; mogu nastati na domaćinu ili biti prosljeđeni,
- `INPUT` -- paketi pristigli mreže namijenjeni za ovog domaćina,
- `FORWARD` -- paketi pristigli s mreže namijenjeni za prosljeđivanje,
- `OUTPUT` -- paketi nastali na domaćinu namijenjeni za slanje.

Unutar lanca pravila, paketi mogu biti povezani sa praćenim vezama u takozvanim stanjima (engl. *states*):

- `NEW` -- stanje nove veze koja još nije poznata na mreži,
- `ESTABLISHED` -- stanje poznate veze,
- `RELATED` -- stanje nove veze koja je povezana sa već poznatom vezom,
- `INVALID` -- stanje u kojem paketi (promet) ne mogu biti identificirani.

Način baratanja paketom, u `iptables` terminologiji nazivaju se ciljevima (engl. *targets*):

- `DNAT` -- vrši Destination NAT na paketu (najčešće se koristi za prosljeđivanje vrata),
- `SNAT` -- vrši Source NAT na paketu (1:1 NAT "prema van"),
- `MASQUERADE` -- vrši IP maškaradu na paketu (M:1 NAT "prema van"),
- `ACCEPT` -- prihvaća paket,
- `REJECT` -- odbija paket, pošiljatelju paketa šalje ICMP poruku tip 3 kod 3 Destination port unreachable,
- `DROP` -- odbacuje paket, pošiljatelju paketa ne šalje ništa,
- `LOG` -- bilježi prolaz paketa, ne mijenja način baratanja s njim (najčešće se koristi uz ostale opcije).

### Pozivanje naredbe

Sintaksa alata `iptables` je:

- `iptables [-t table]  {-A|-C|-D} chain rule-specification`

    - rule-specification = `[matches...] [target]`
    - match = `-m matchname [per-match-options]`
    - target = `-j targetname [per-target-options]`

- `iptables [-t table]  {-L|-F} [chain]`

Neke od najvažnijih naredbi u alatu `iptables`, kojima određujemo uvjete za izvađanje željenih akcija:

- `-A`, `--append chain rule-specification`

    - dodavanje jednog ili više pravila na kraj odabranog lanca

- `-D`, `--delete chain rule-specification`

    - brisanje pravila iz odabranog lanca

- `-I`, `--insert chain rule-specification`

    - dodavanje jednog ili više pravila na početak odabranog lanca

- `-L`, `--list [chain]`

    - izlistavanje svih pravila odabranog lanca
    - ako lanac nije naveden, izlistavaju se sva pravila

- `-F`, `--flush [chain]`

    - brisanje cijelog lanca pravila
    - ukoliko lanac nije naveden, brišu se svi lanci

Neki od najvažnijih općenitih parametara u alatu `iptables`:

- `-t table`

    - pozivanje određene tablice
    - zadana `iptables` tablica je `filter`; ukoliko se ne navede druga tablica za korištenje, koristi se tablica `flter`.

- `-j target`

    - određivanje cilja

- `-p`, `--protocol protocol`

    - određivanje protokola
    - dozvoljene vrijednosti: `tcp`, `udp`, `udplite`, `icmp`, `esp`, `ah`, `sctp`

- `-m state --state state`

    - određivanje stanja

## Način filtriranja paketa alatom iptables

Lanci koje iptables koristi za filtriranje paketa su:

- `INPUT`,
- `FORWARD`,
- `OUTPUT`.

Ciljevi koje iptables koristi za filtriranje paketa su:

- `ACCEPT`,
- `REJECT`,
- `DROP`,
- `LOG`.

Elementi tablice filter svakog lanca su pravila koja pridružuju cilj paketima koji imaju određenu izvorišnu adresu, odredišnu adresu, protokol, izvorišna vrata, odredišna vrata i/ili ostalo. Poredak elemenata tablice filter u svakom lancu je značajan, odnosno kod obrade paketa uzima se prvo navedeno pravilo koje odgovara i ne gledaju se preostala. Ukoliko nijedno pravilo ne odgovara uzima se zadani cilj za taj lanac.

Neki od najvažnijih parametara u alatu `iptables` koje koristimo pri filtriranju paketa:

- `--source-port`, `--sport port[:port]` -- paketi sa određenim izvorišnim vratima
- `-m multiport --sports port[,port|,port:port]` -- paketi sa jednim od navedenih izvorišnih vratiju
- `--destination-port`, `--dport port[:port]` -- paketi sa određenim odredišnim vratima
- `-m multiport --dports port[,port|,port:port]` -- paketi sa jednim od navedenih odredišnih vratiju
- `--src-range from[-to]` -- određivanje raspona izvorišnih adresa
- `--dst-range from[-to]` -- određivanje raspona odredišnih adresa
- `-i`, `--in-interface name` -- određivanje mrežnog sučelja preko kojeg dolazi paket, radi samo u lancima INPUT, FORWARD i PREROUTING
- `-o`, `--out-interface name` -- određivanje mrežnog sučelja preko kojeg se paketi šalju, radi samo u lancima FORWARD, OUTPUT i POSTROUTING

## Primjeri filtriranja paketa vatrozidom

Na mreži na slici računala n1, n2 i n3 ostvaruju vezu sa domaćinima n7 i n10 i računalom n9 putem usmjerivača n5, n6, n8 i preklopnika n4. Mrežna sučelja usmjerivača n5 su eth0 prema n6 i eth1 prema n4. Adresa usmjerivača n5 na sučelju eth0 je 10.0.3.1, a na sučelju eth1 je 192.168.4.1. Adrese n1, n2 i n3 su redom 192.168.4.20, 192.168.4.21 i 192.168.4.22, a adrese n7, n10 i n9 su redom 10.0.2.10, 10.0.4.10 i 10.0.0.20. Za potrebe zadatka pretpostavimo da je na usmjerivaču n5 ispravno postavljen NAT te da je međusobna komunikacija između svih čvorova moguća. Vatrozid postoji na usmjerivaču n5 i na računalu n1.

```
    n1             n7   n10
     \             /    /
      \           /    /
n2----n4----n5---n6---n8
      /                \
     /                  \
    n3                  n9
```

Za svaki navedeni primjer generiranjem odgovarajućeg prometa (korištenjem MGEN-a kroz CORE-ov dijalog `CORE traffic flows` dostupan pod `Tools/Traffic` ili korištenjem netcata (naredba `nc`) u ljusci čvora) i hvatanjem prometa putem alata Wireshark ispitajmo stanje veze prije i nakon unosa `iptables` naredbe.

### Primjer 1

Želimo da računalo n1 ima vatrozid koji odbija sve veze prema vlastitim HTTP vratima (80) osim onih sa n10 (HTTP koristi TCP).

Generiranjem odgovarajućeg prometa (dva toka prema TCP vratima 80 na n1, od kojih je jedan s n10, a drugi s, primjerice, n7) i hvatanjem prometa putem alata Wireshark (na mrežnom sučelju na n1) ispitajmo stanje veze. Uočimo kako n1 trenutno može uspostaviti HTTP vezu sa svim preostalim čvorovima u mreži. Pokrenemo ljusku na n1 te upišemo:

``` shell
# iptables -A INPUT -p tcp -s 10.0.4.10 --dport 80 -m state --state NEW,ESTABLISHED -j ACCEPT
```

Zatim ponovno upišemo u ljusku na n1:

``` shell
# iptables -A INPUT -p tcp -s 0.0.0.0/0 --dport 80 -j REJECT
```

Generiranjem odgovarajućeg prometa i hvatanjem prometa putem alata Wireshark kao iznad ispitajmo stanje veze. Uočimo kako sada n1 može uspostaviti HTTP vezu samo sa n10.

### Primjer 2

Pretpostavimo da je vatrozidu usmjerivača n5 na FORWARD lancu zadan cilj DROP. Želimo da usmjerivač n5 ima vatrozid koji odbacuje sve pakete za prosljeđivanje osim onih namijenjenih za računalo n3.

Postavimo prvo zadani cilj DROP na lancu FORWARD. Na n5 pokrenimo:

``` shell
# iptables -P FORWARD DROP
```

Generiranjem odgovarajućeg prometa i hvatanjem prometa putem alata Wireshark ispitajmo stanje veze. Uočimo kako se preko n5 trenutno ne može uspostaviti veza između nikoja dva čvora u mrežama spojenim na suprotna sučelja usmjerivača n5. Pokrenemo ljusku na usmjerivaču n5 te upišemo:

``` shell
# iptables -A FORWARD -s 0.0.0.0/0 -d 192.168.4.22 -j ACCEPT
```

Generiranjem odgovarajućeg prometa i hvatanjem prometa putem alata Wireshark ispitajmo stanje veze. Uočimo da sada možemo poslati pakete računalu n3 sa bilo kojeg domaćina ili računala u mreži. Također uočimo da na pošiljatelje paketa za n3 ne može stići nikakav povratni paket sa n3.

### Primjer 3

Želimo da usmjerivač n5 ima vatrozid koji odbacuje sve pakete namijenjene za mrežu 192.168.4.0/24, a da pritom pošiljatelju ne šalje nikakvu obavijest.

Generiranjem odgovarajućeg prometa i hvatanjem prometa putem alata Wireshark ispitajmo stanje veze. Uočimo da možemo uspostaviti komunikaciju sa postojećim čvorovima iz 192.168.4.0/24 od strane ostatka mreže. Pokrenemo ljusku na usmjerivaču n5 te upišemo:

``` shell
# iptables -A FORWARD -d 192.168.4.0/24 -j DROP
```

Generiranjem odgovarajućeg prometa i hvatanjem prometa putem alata Wireshark ispitajmo stanje veze. Uočimo da sada više ne možemo slati pakete prema 192.168.4.0/24 od strane ostatka mreže. Također, uočimo da računala n1, n2 i n3 mogu slati pakete prema ostatku mreže, no ne mogu primiti nikakav povratni paket.

### Primjer 4

Želimo da računalo n1 ima vatrozid koji zaustavlja sav odlazni TCP promet namijenjen za domaćina n7.

Generiranjem odgovarajućeg prometa i hvatanjem prometa putem alata Wireshark ispitajmo stanje veze. Uočimo da je za n1 moguće poslati TCP pakete domaćinu n7, kao i bilo kojem drugom čvoru na mreži. Pokrenemo ljusku na n1 te upišemo:

``` shell
# iptables -A OUTPUT -p tcp -d 10.0.2.10 -j DROP
```

Generiranjem odgovarajućeg prometa i hvatanjem prometa putem alata Wireshark ispitajmo stanje veze. Uočimo kako sada više računalo n1 ne može poslati TCP pakete domaćinu n7, no može bilo kojem drugom čvoru. Računalo n1 od domaćina n7 dobiva poruku `"Connection refused"`.

### Primjer 5

Želimo da računalo n1 ima vatrozid koji čini to da čvorovi u mreži ne mogu znati je li računalo n1 *živo* ili *mrtvo* slanjem ICMP poruka (tj. *pinganjem* n1).

Uporabom alata `ping` i hvatanjem prometa putem alata Wireshark ispitajmo stanje veze. Uočimo kako računalo n1 može primiti ICMP poruku od bilo kojeg čvora u mreži, a bilo koji čvor u mreži može dobiti odgovor na ICMP poruku poslanu n1. Pokrenemo ljusku na n1 te upišemo:

``` shell
# iptables -A INPUT -p icmp --icmp-type echo-request -j DROP
```

Uporabom alata `ping` i hvatanjem prometa putem alata Wireshark ispitajmo stanje veze. Uočimo kako n1 i dalje prima ICMP poruke, no čvorovi koji su poslali ICMP poruku više ne dobivaju odgovor od n1.

### Primjer 6

Želimo da računalo n1 ima vatrozid koji onemogućava slanje elektronske pošte (vrata 25).

Generiranjem odgovarajućeg prometa i hvatanjem prometa putem alata Wireshark ispitajmo stanje veze. Uočimo kako n1 može poslati poruku s odredišnim vratima 25 na bilo koji čvor u mreži (za primanje poruke, čvor primatelj mora imati otvorena vrata 25). Pokrenemo ljusku na n1 te upišemo:

``` shell
# iptables -A OUTPUT -p tcp --dport 25 -j DROP
```

Generiranjem odgovarajućeg prometa i hvatanjem prometa putem alata Wireshark ispitajmo stanje veze. Uočimo kako poruke s odredišnim vratima 25 računalo n1 više ne može poslati niti jednom čvoru u mreži te na svaki poslani paket dobiva odgovor `"Connection timed out"`.
