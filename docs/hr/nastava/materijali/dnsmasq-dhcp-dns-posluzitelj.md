---
author: Kristian Skender, Vedran Miletić
---

# DHCP i DNS poslužitelj dnsmasq

[Dnsmasq](https://thekelleys.org.uk/dnsmasq/doc.html) je vrlo popularno besplatno programsko rješenje s otvorenim kodom namijenjeno za DHCP i DNS poslužitelje. Dizajniran je tako da može posluživati imena lokalnih mašina koja nisu u globalnom DNS-u. DHCP poslužitelj integriran u DNS poslužitelj omogućava mašinama sa DHCP alociranom adresom pojavu u samom DNS-u sa nazivom konfiguriranim ili posebno na svakom domaćinu ili konfiguriranom u centralnoj konfiguracijskoj datoteci. Jedna od najvećih prednosti ovoga alata je njegova jednostavna konfiguracija koja će biti objašnjena u ovom radu, te njegova potreba za vrlo malim sistemskim resursima. Takodjer, u širokoj upotrebi je na pametnim telefonima kod kojih se koristi za razna dijeljenja i u radu prijenosnih hotspotova. Dnsmasq kao DNS prikuplja DNS upite koje sprema te na taj način ostvaruje bržu vezu sa stranicama koje su ranije posjećivane, a kao DHCP poslužitelj koristi se za predviđanje unutarnje IP adrese i puta do računala u lokalnoj mreži, LAN-u. Podržane platforme su Linux, Android i Mac OS X.

## Instalacija

Na samome početku, dnsmasq moramo instalirati na operativni sustav, u našem slučaju Linux Ubuntu, to izvršavamo naredbom:

``` shell
# apt-get install dnsmasq
```

Dnsmasq je automatski konfiguriran i pokrenut nakon instalacije. Da bi testirali jednostavni DNS server upisujemo iduću naredbu za bilo koju željenu web stranicu:

``` shell
$ dig 9gag.com @localhost
```

U "Answer section" možemo pronaći IP adresu na kojoj se tražena stranica nalazi.

```
; <<>> DiG 9.11.4-4-Debian <<>> 9gag.com @localhost
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 57886
;; flags: qr rd ra; QUERY: 1, ANSWER: 4, AUTHORITY: 4, ADDITIONAL: 9

;; OPT PSEUDOSECTION:
; EDNS: version: 0, flags:; udp: 4096
;; QUESTION SECTION:
;9gag.com.                      IN      A

;; ANSWER SECTION:
9gag.com.               289     IN      A       151.101.194.133
9gag.com.               289     IN      A       151.101.66.133
9gag.com.               289     IN      A       151.101.2.133
9gag.com.               289     IN      A       151.101.130.133

;; AUTHORITY SECTION:
9gag.com.               251     IN      NS      ns-1294.awsdns-33.org.
9gag.com.               251     IN      NS      ns-1673.awsdns-17.co.uk.
9gag.com.               251     IN      NS      ns-629.awsdns-14.net.
9gag.com.               251     IN      NS      ns-408.awsdns-51.com.

;; ADDITIONAL SECTION:
ns-408.awsdns-51.com.   170089  IN      A       205.251.193.152
ns-408.awsdns-51.com.   164278  IN      AAAA    2600:9000:5301:9800::1
ns-629.awsdns-14.net.   133726  IN      A       205.251.194.117
ns-629.awsdns-14.net.   133726  IN      AAAA    2600:9000:5302:7500::1
ns-1294.awsdns-33.org.  133750  IN      A       205.251.197.14
ns-1294.awsdns-33.org.  133750  IN      AAAA    2600:9000:5305:e00::1
ns-1673.awsdns-17.co.uk. 133874 IN      A       205.251.198.137
ns-1673.awsdns-17.co.uk. 133874 IN      AAAA    2600:9000:5306:8900::1

;; Query time: 33 msec
;; SERVER: 87.98.175.85#53(87.98.175.85)
;; WHEN: ned kol 12 22:54:26 CEST 2018
;; MSG SIZE  rcvd: 414
```

## Konfiguracija

Nakon instalacije paketa, potrebno je odraditi konfiguraciju dnsmasqa. Konfiguracijska datoteka nalazi se na putanji `/etc/dnsmasq.conf`. Nakon što smo otvorili datoteku, moramo pronaći komentirani redak u kojem piše `listen-adress` te ga odkomentirati. Da bi konfigurirali dnsmasq kao DNS instancu na računalu upisujemo lokalnu IP adresu:

``` ini
listen-adress=127.0.0.1
```

Ako želimo da nam računalo sluša ostala računala na lokalnoj IP adresi upisujemo proizvoljnu IP adresu:

``` ini
listen-adress=192.168.1.1
```

U istoj datoteci radimo slijedeće izmijene radi odabiranja sučelja i odabira ethernet sučelja koje će ili neće dnsmasq slušati. Komentirani red `#interface`, odkomentiramo te dodamo:

``` ini
interface=eth1
interface=eth2
```

Da ovi redovi nisu dodani, dnsmasq bi također slušao na `eth0`, odnosno na našoj internet konekciji. Ovo nikako nije preporučljivo jer dodaje hakerima veću mogućnost za provalu u vaše računalo.

Po defaultu, DHCP je ugašen i to je vrlo dobra stvar jer ako smo neoprezni, postoji mogućnost da srušimo cijelu mrežu. Da bi ga aktivirali u istoj datoteci pronalazimo redak:

``` ini
#dhcp-range=192.168.0.50,192.168.0.150,12h
```

Da bi aktivacija bila uspješna moramo dodijeliti opseg IP adresa koje će biti obuhvaćene. U gornjem retku biti će obuhvaćeno 101 adresa, počevši od 192.168.0.50 i završavajući na 192.168.0.150. Kako imamo dvije različite mreže kojima je potreban DHCP, prethodnu liniju ćemo zamijeniti ovima:

``` ini
dhcp-range=eth1,192.168.100.100,192.168.100.199,4h
dhcp-range=eth2,192.168.200.100,192.168.200.199,4h
```

Nazivi `eth1` i `eth2` nisu potrebni, ali ako radimo mnogo napredniju konfiguraciju tada je potrebno radi boljeg snalaženja pisati imena, da se ne bih zabunili u opsezima (range).

Nakon konfiguriranja dnsmasq.conf datoteke, iduće što moramo napraviti je da DHCP klijent nadoda lokalnu adresu poznatoj DNS adresi u datoteci `/etc/resolv.conf`. U navedenu datoteku upisujemo:

```
nameserver 127.0.0.1
```

Ovo omogućuje da svi upiti budu poslani na dnsmasq prije nego li budu riješeni na vanjskom DNS-u. Nakon ove rekonfiguracije mrežu je potrebno restartati da bi se nove postavke prihvatile. Restartamo je preko naredbe:

``` shell
# /etc/init.d/dnsmasq restart
```

DHCPCD teži za pisanje preko datoteke `/etc/resolv.conf` po defaultnim postavkama, stoga ako koristimo DHCP dobra ideja je zaštiti `/etc/resolv.conf` datoteku. Da bi ju zaštitili, u datoteku `/etc/dhcpcd.conf` nadodajemo:

```
nohook resolv.conf
```
