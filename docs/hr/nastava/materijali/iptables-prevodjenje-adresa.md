---
author: Domagoj Margan, Vedran Miletić
---

# Prevođenje mrežnih adresa

## Pojam prevođenja mrežnih adresa

[Prevođenje mrežnih adresa](https://en.wikipedia.org/wiki/Network_address_translation) (engl. *Network Address Translation*, NAT) je proces promjene IP adresa u zaglavlju IP paketa unutar usmjerivača. NAT je uveden kako bi se riješio problem nestanka IPv4 adresa kada je broj domaćina povezanih na internet počeo naglo rasti. NAT čini da više domaćina dijele istu IP adresu, odnosno preciznije rečeno, da postoje domaćini koji mogu pristupati internetu bez da imaju adresu koja ima globalno jedinstveno značenje. NAT tako omogućuje domaćinima pristup internetu iz privatne mreže, što se obilato koristi kod kućnog pristupa internetu putem ADSL-a, kabela ili optičke veze gdje će korisnik (ili čak više njih) dobiti na internetu jednu adresu, a iz kućanstva će na internet bez problema moći povezati veći broj uređaja.

Postoje dvije vrste prevođenja mrežnih adresa:

- 1:1 NAT ili osnovni NAT, koji mijenja samo izvorišnu i/ili odredišnu IP adresu,
- M:1 NAT ili NAPT (Network Address and Port Translation), poznat i pod nazivom IP maškarada (engl. *IP masquerade*), koji pored IP adresa mijenja i broj TCP/UDP vrata.

Prosljeđivanje vrata (engl. *port forwarding*) je tehnika koja se koristi da se omogući dolazne konekcije prema domaćinima koji se nalaze "iza NAT-a". Vrata se otvaraju na vanjskoj adresi kojoj domaćini na Internetu mogu pristupiti, a zatim se paketi prosljeđuju na odgovarajuća vrata domaćina "iza NAT-a". To se postiže kombiniranjem:

- osnovnog NAT-a ili NAPT-a,
- filtriranja unutar vatrozida,
- prosljeđivanja paketa prema tablicama prosljeđivanja usmjerivača.

Mi ćemo se opet fokusirati na iptables.

## Način prevođenja adresa alatom iptables

Lanci koje iptables koristi za prevođenje mrežnih adresa su:

- `PREROUTING`,
- `POSTROUTING`.

Ciljevi koje iptables koristi za prevođenje mrežnih adresa su:

- `DNAT` -- moguće izvesti samo sa PREROUTING lancem,
- `SNAT` -- moguće izvesti samo sa POSTROUTING lancem,
- `MASQUERADE` -- moguće izvesti samo sa POSTROUTING lancem.

Elementi tablice `nat` svakog lanca su pravila koja određenom mrežnom sučelju pridružuju cilj i specifikaciju pretvorbe izvorišne ili odredišne adrese.

Neki od najvažnijih parametara u alatu `iptables` koje koristimo pri prevođenju adresa:

- `-s`, `--source address[/mask][,...]` -- određivanje izvora
- `--to-source [ipaddr[-ipaddr]][:port[-port]]` -- vrijedi samo za cilj SNAT, služi za određivanje nove izvorišne adrese i određivanje izvorišnih mrežnih vrata (uz opciju `-p tcp` ili `-p udp`)
- `-d`, `--destination address[/mask][,...]` -- određivanje odredišta
- `--to-destination [ipaddr[-ipaddr]][:port[-port]]` -- vrijedi samo za cilj DNAT, služi za određivanje nove odredišne adrese i određivanje odredišnih mrežnih vrata (uz opciju `-p tcp` ili `-p udp`)
- `-i`, `--in-interface name` -- služi za određivanje mrežnog sučelja preko kojeg dolazi paket, odnosi se samo na lance INPUT, FORWARD i PREROUTING
- `-o`, `--out-interface name` -- služi za određivanje mrežnog sučelja preko kojeg se paketi šalju, vrijedi samo na lance FORWARD, OUTPUT i POSTROUTING

## Primjeri prevođenja adresa

Na mreži na slici računala n1, n2 i n3 ostvaruju vezu sa domaćinom n7 i računalom n9 putem usmjerivača n5, n6, n8 i preklopnika n4. Mrežna sučelja usmjerivača n5 su eth0 prema n6 i eth1 prema n4. Adresa usmjerivača n5 na sučelju eth0 je 10.0.3.1, a na sučelju eth1 je 192.168.4.1. Adrese n1, n2 i n3 su redom 192.168.4.20, 192.168.4.21 i 192.168.4.22, a adrese n7 i n9 su redom 10.0.2.10 i 10.0.0.20.

```
    n1             n7
     \             /
      \           /
n2----n4----n5---n6---n8
      /                \
     /                  \
    n3                  n9

```

### Primjer 1

Pokrenemo li na n1 `ping` čvora n9, uočit ćemo da čvorovi iz različitih podmreža mogu međusobno komunicirati izravno. U toj situaciji emulacija zapravo odstupa od stvarnosti (kako bi nam olakšala korištenje); u stvarnosti se usmjeravanje ne vrši na privatne raspone adresa (između ostalog, to je zato što velik broj kućnih korisnika ima svoju kućnu mrežu na rasponu adresi 192.168.5.0/24 pa je pitanje kojem od njih bi paketi trebali stići).

Želimo omogućiti da n1, n2 i n3 mogu komunicirati s ostatkom mreže (n7 i n9) preko vanjske adrese korištenjem prevođenja adresa. Tada će se prilikom prolaska paketa poslanih s n1, n2 i n3 promijeniti izvorišna adresa u vanjsku adresu od n5, a kod primanja odgovora će se odredišna adresa od n5 promijeniti u adresu od n1, n2 ili n3, ovisno o korištenim odredišnim vratima.

Uočimo da se izvodi prevođenje izvorišne adrese što ne utječe na usmjeravanje, te možemo pravila naizgled dodati ili u PREROUTING ili u POSTROUTING. Međutim, kako bi pravila filtriranja vatrozida zadana u terminima privatnih adresa bila ispravno primijenjena, dodajemo ih u lanac POSTROUTING. Koristimo IP maškaradu naredbom na usmjerivaču n5

``` shell
# iptables -t nat -A POSTROUTING -o eth1 -j MASQUERADE
```

Ova naredba je specifična za ovu mrežu iznad jer je eth1 mrežno sučelje usmjerivača n5 prema van (iz perspektive mreže u kojoj su n1, n2, n3 i n5), odnosno prema n6. U nekoj drugoj mreži to može biti eth0, eth2 ili bilo koja treće mrežno sučelje, ovisno o načinu na koji su domaćini u mreži povezani.

Ponovo provjerimo stanje veze alatom `ping`. Uočimo da sada možemo slati ICMP poruke između čvorova u različitim podmrežama. Ova postavka obuhvaća n1, n2 i n3.

SNAT ne možemo koristiti jer imamo više od jednog računala te bi moglo doći do kolizije vrata s vanjske strane u slučaju da operacijski sustavi na oba računala otvore ista kratkoživuća vrata u istom trenutku.

### Primjer 2

Za potrebe idućeg primjera potrebno je uključiti SSH servis na čvoru n1, kako bi se na njemu pokrenuo SSH poslužitelj na vratima 22. Ako se radi o emuliranom čvoru alatu CORE, to možemo napraviti desnim klikom na čvor -> Configure -> Services -> označimo SSH. Želimo omogućiti pristup n1 SSH klijentima iz ostatka mreže.

Koristit ćemo DNAT. Prevođenje je potrebno napraviti prije usmjeravanja, te u lanac PREROUTING na čvoru n5 uključujemo DNAT naredbom

``` shell
# iptables -t nat -A PREROUTING -p tcp --dport 2222 -d 10.0.3.1 -j DNAT --to-destination 192.168.4.20:22
```

Uočimo da smo mogli umjesto vrata 2222 iskoristiti proizvoljna neiskorištena vrata (npr. 4587 ili 23851), no odabrali smo ova samo zbog lakšeg pamćenja.

### Primjer 3

Za potrebe idućeg primjera potrebno je odstraniti čvorove n1 i n2. Dakle, sada je n3 jedino računalo na eth1 sučelju usmjerivača n5. Na n5 iskoristiti ćemo SNAT za n3.

Pokrenemo li na n3 `ping` čvora n7, uočit ćemo da ne možemo slati ICMP poruke između n3 i ostatka mreže.

Kako sada imamo samo jedno računalo, možemo koristiti SNAT jer se sva vrata dostupna na vanjskoj adresi mogu izravno preslikati u vrata računala n1. Ukoliko to želimo, u POSTROUTING na usmjerivaču n5 ćemo uključiti SNAT naredbom

``` shell
# iptables -t nat -A POSTROUTING -o eth1 -j SNAT -s 10.0.3.1 --to-source 192.168.4.22
```

Ova naredba je također specifična za ovu mrežu iznad zbog navedenog izlaznog mrežnog sučelja i adrese domaćina.

Ponovo provjerimo stanje veze alatom `ping`. Uočimo da sada možemo slati ICMP poruke između čvora n3 i ostatka mreže.
