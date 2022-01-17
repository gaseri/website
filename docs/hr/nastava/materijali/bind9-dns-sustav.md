---
author: Ivona Bjelobradić, Vedran Miletić
---

# DNS sustav BIND9

!!! todo
    Ovaj dio je vrlo sirov i treba ga temeljito pročistiti i dopuniti.

[Domenski sustav imena](https://hr.wikipedia.org/wiki/Sustav_domenskih_imena) (engl. Domain Name System, kraće DNS) je distribuirani hijerarhijski sustav Internet poslužitelja u kojem se nalaze informacije povezane s nazivima domena, tj. o povezanosti IP adresa i njihovih logičkih (simboličkih) imena. Važnost DNS sustava se očituje u tome što ne moramo pamtiti IP adrese web stranica nego koristimo logička imena, npr. umjesto IP adrese `108.61.208.98` pisat ćemo `example.group.miletic.net`.

[Berkeley Internet Name Domain](https://www.isc.org/bind/) (BIND) je open source softver koji implementira DNS protokole na Internetu. DNS sustav je baza podataka u kojoj su upisana sva imena i odgovarajuće IP adrese pojedinih računala i biblioteka funkcija koja omogućuje prevođenje istih. U nastavku će biti opisana konfiguracija i popratni alati, te način definiranja zone koji će biti popraćeni primjerima. DNS protokoli su dio jezgre internestkih standarda. Oni specificiraju proces prema kojem jedno računalo može pronaći drugo na temelju njegovog imena. Implementacija DNS protokola podrazumijeva da naša distribucija softvera sadrži sav softver potreban za postavljanje i odgovaranje na pitanja imena usluga. BIND je referentna je implementacija tih prokolola ali je također softver proizvodnog stupnja koji je prikladan za korištenje u aplikacijama visoke pouzdanosti i opsega.

BIND je daleko najkorišteniji DNS softver na Internetu, koji pruža robustnu i stabilnu platformu na kojoj organizacije mogu izgraditi distribuirane računalne sustave sa spoznajom da su ti sustavi u potpunosti u skladu s objavljenim DNS standardima.

## Dijelovi BIND-a

BIND softver se sastoji od 3 dijela:

- DNS servera: program koji se zove 'named' koji je skraćenica od Name Daemon, odgovara na sva primljena pitanja slijedeći pravila specificirana u standardima DNS protokola. Možete dobaviti DNS usluge na Internetu instalirajući ovaj softver na serveru i davajući mu ispravne informacije o imenima domena.
- DNS zbirke razlagača (engl. DNS resolver library): razlagač je program koji razlaže pitanja o imenima šaljući ta imena na odgovarajuće servere i koji propisno odgovara na zahtjeve servera. Zbirka razlgača je skup softverskih komponenti koje programer može dodati u softver koji se razvija, a koji će dati softveru mogućnost da razlaže imena. Npr. programer koji programira novi web preglednik ne mora isprogramirati dio koji će tražiti imena u DNS-u već može priljučiti zbirku razlagača i slati pitanja u knjižnicu softverskih komponenti. Ovaj način štedi vrijeme i osigurava da novi preglednik ispravno slijedi DNS standarde.
- Alata za testiranje servera: ovo su alati koje uključujemo u distribuciju da bi nam asistirali u dijagnozama. Kada instaliramo neki operacijski sustav na računalo, to računalo će sadržavati koju god zbirku razlagača je odabrao njegov programer. Kada ste postavili računalo poslužitelja, njegov prodavač obično je postavio neki DNS sustav tako da će server raditi kad je isporučen. Zato što BIND vjerno implementira DNS protokole nema potrebe da razlagač i server pokreću isti softver.

## Konfiguracija

!!! todo
    Ovdje treba citirati [upute na Server Worldu za Fedoru 28](https://www.server-world.info/en/note?os=Fedora_28&p=dns&f=1) ili osvježiti za novije verzije Fedore i citirati odgovarajući izvor.

Konfiguracija se nalazi u datoteci `/etc/named.conf` sadržaja

``` nginx
//
// named.conf
//
// Provided by Red Hat bind package to configure the ISC BIND named(8) DNS
// server as a caching only nameserver (as a localhost DNS resolver only).
//
// See /usr/share/doc/bind*/sample/ for example named configuration files.
//
options {
  listen-on port 53 { 127.0.0.1; };
  listen-on-v6 port 53 { ::1; };
  directory "/var/named";
  dump-file "/var/named/data/cache_dump.db";
  statistics-file "/var/named/data/named_stats.txt";
  memstatistics-file "/var/named/data/named_mem_stats.txt";
  allow-query { localhost; };
  /*
   - If you are building an AUTHORITATIVE DNS server, do NOT enable recursion.
   - If you are building a RECURSIVE (caching) DNS server, you need to enable
     recursion.
   - If your recursive DNS server has a public IP address, you MUST enable access
     control to limit queries to your legitimate users. Failing to do so will
     cause your server to become part of large scale DNS amplification
     attacks. Implementing BCP38 within your network would greatly
     reduce such attack surface
   */
  recursion yes;

  dnssec-enable yes;
  dnssec-validation yes;
  dnssec-lookaside auto;

  /* Path to ISC DLV key */
  bindkeys-file "/etc/named.iscdlv.key";

  managed-keys-directory "/var/named/dynamic";
  pid-file "/run/named/named.pid";
  session-keyfile "/run/named/session.key";
};

logging {
  channel default_debug {
    file "data/named.run";
    severity dynamic;
  };
};

zone "." IN {
  type hint;
  file "named.ca";
};

include "/etc/named.rfc1912.zones";
include "/etc/named.root.key";
```

Datoteku ćemo izmijeniti na sljedeći način (nakon promjene potrebno je ponovno pokrenuti uslugu `named`)

``` nginx
options {
  // listen-on port 53 { 127.0.0.1; };
  listen-on-v6 { none; };
  directory "/var/named";
  dump-file "/var/named/data/cache_dump.db";
  statistics-file "/var/named/data/named_stats.txt";
  memstatistics-file "/var/named/data/named_mem_stats.txt";
  allow-query {
    localhost;
    192.168.1.238/24;
  };
  /*
   - If you are building an AUTHORITATIVE DNS server, do NOT enable recursion.
   - If you are building a RECURSIVE (caching) DNS server, you need to enable
     recursion.
   - If your recursive DNS server has a public IP address, you MUST enable access
     control to limit queries to your legitimate users. Failing to do so will
     cause your server to become part of large scale DNS amplification
     attacks. Implementing BCP38 within your network would greatly
     reduce such attack surface
   */
  recursion yes;

  dnssec-enable yes;
  dnssec-validation yes;
  dnssec-lookaside auto;

  /* Path to ISC DLV key */
  bindkeys-file "/etc/named.iscdlv.key";

  managed-keys-directory "/var/named/dynamic";
  pid-file "/run/named/named.pid";
  session-keyfile "/run/named/session.key";
};

view "internal" {
  match-clients {
    localhost;
    192.168.1.238/24;
  };
  zone "." IN {
    type hint;
    file "named.ca";
  };
  zone "server.world" IN {
    type master;
    file "server.world.lan";
    allow-transfer { 192.168.1.238/24; };
  };
  zone "238.1.168.192.in-addr.arpa" IN {
    type master;
    file "238.1.168.192.db";
  };
  include "/etc/named.rfc1912.zones";
  include "/etc/named.root.key";
};
```

Opis korištenih funkcionalnosti:

- `allow-query`: raspon koji dozvoljavamo
- `allow-transfer`: raspon koji dozvoljavamo za prijenos informacija o zoni
- `recursion`: dozvoljava rekurzivno pretraživanje
- `view "internal" { ... };`: unutarnje definicije

Za obrnuto razlaganje mrežne adrese se pišu obrnutim redoslijedom, primjerice za adresu u zapisu CIDR 192.168.1.238/24 imamo:

- mrežna adresa: 192.168.1.238
- raspon mreža (podmreža): 192.168.1.0 -- 192.168.1.238.255
- podmreža u zapisu rDNS: 238.1.168.192.in-addr.arpa

## Način definiranja zone

Definiranje datoteke zone pomoću kojih server razlaže IP adrese preko imena domene za unutarnju zonu vrši se unutar datoteke `/var/named/server.world.lan` sadržaja

```
$ORIGIN server.world
$TTL 86400
@ IN SOA dns.server.world. root.server.world. (
    2001062501 ; serial
    21600      ; refresh after 6 hours
    3600       ; retry after 1 hour
    604800     ; expire after 1 week
    86400 )    ; minimum TTL of 1 day

  IN NS  dns.server.world.

  IN A   192.168.1.238

  IN MX  mail.server.world

dns IN A   192.168.1.238
```

Postavljanje zona za obrnuto razlaganje vrši se unutar datoteke `/var/named/238.1.168.192.db` sadržaja

```
$TTL 86400
@ IN SOA dns.server.world. root.server.world. {
    2013121801 ; serial
    21600      ; refresh
    3600       ; retry
    604800     ; expire
    86400 )    ; minimum TTL

// definiranje imenskog servera

IN NS dns.server.world.

// definiranje raspona koji je uključen u domenu imena

IN PTR server.world.
IN A   255.255.255.0

// definiranje IP adrese i imenskog servera

238 IN PTR dns.server.world
```

## Pomoćni alat `dig`

Naredba `dig` dopušta izvođenje bilo kojeg valjanog DNS upita od kojih su najčešće korišteni

- `A`: IP adrese,
- `TXT`: tekstualne bilješke,
- `MX`: e-mail poslužitelji te
- `NS`: imena servera.

Provjerit ćemo ako server može 'razlagati' domenu imena i IP adrese na slijedeći način

``` shell
$ dig dns.server.world

; <<>> DiG 9.11.3-RedHat-9.11.3-6.fc28 <<>> dns.server.world
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NXDOMAIN, id: 56466
;; flags: qr aa rd ra; QUERY: 1, ANSWER: 0, AUTHORITY: 0, ADDITIONAL: 0

;; QUESTION SECTION:
;dns.server.world.      IN  A

;; Query time: 2 msec
;; SERVER: 192.168.1.1#53(192.168.1.1)
;; WHEN: čet lip 28 18:32:10 CEST 2018
;; MSG SIZE  rcvd: 34

$ dig -x 192.168.1.238

; <<>> DiG 9.11.3-RedHat-9.11.3-6.fc28 <<>> dns.server.world
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 2514
;; flags: qr rd ra; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 0

;; QUESTION SECTION:
;238.1.168.192.in-addr.arpa.    IN  PTR

;; ANSWER SECTION:
238.1.168.192.in-addr.arpa.  900  IN  PTR  localhost.dummy.porta.siemens.net.

;; Query time: 5 msec
;; SERVER: 192.168.1.1#53(192.168.1.1)
;; WHEN: čet lip 28 18:32:14 CEST 2018
;; MSG SIZE  rcvd: 91
```

Sljedeći primjer pokazuje upit na DNS server za određene zapise; u ovom primjeru to predstavlja zapis MX

``` shell
$ dig MX wikimedia.org @ns0.wikimedia.org

; <<>> DiG 9.11.3-RedHat-9.11.3-6.fc28 <<>> MX wikimedia.org @ns0.wikimedia.org
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 58783
;; flags: qr aa rd; QUERY: 1, ANSWER: 2, AUTHORITY: 0, ADDITIONAL: 5
;; WARNING: recursion requested but not available

;; OPT PSEUDOSECTION:
; EDNS: version: 0, flags:; udp: 1024
;; QUESTION SECTION:
;wikimedia.org.      IN  MX

;; ANSWER SECTION:
wikimedia.org.    3600  IN  MX  10 mx1001.wikimedia.org.
wikimedia.org.    3600  IN  MX  50 mx2001.wikimedia.org.

;; ADDITIONAL SECTION:
mx1001.wikimedia.org.  3600  IN  A  208.80.154.76
mx1001.wikimedia.org.  3600  IN  AAAA  2620:0:861:3:208:80:154:76
mx2001.wikimedia.org.  3600  IN  A  208.80.153.45
mx2001.wikimedia.org.  3600  IN  AAAA  2620:0:860:2:208:80:153:45

;; Query time: 110 msec
;; SERVER: 208.80.154.238#53(208.80.154.238)
;; WHEN: čet lip 28 18:42:18 CEST 2018
;; MSG SIZE  rcvd: 176
```

Sljedeći primjer prikazuje isključivanje pojedinih dijelova ispisa. U ovom slučaju isključili smo ispise o komentarima i o statistikama

``` shell
$ dig redhat.com

; <<>> DiG 9.11.3-RedHat-9.11.3-6.fc28 <<>> redhat.com
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 8390
;; flags: qr rd ra; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 1

;; OPT PSEUDOSECTION:
; EDNS: version: 0, flags:; udp: 4000
; COOKIE: 6a8cead90047c547 (echoed)
;; QUESTION SECTION:
;redhat.com.      IN  A

;; ANSWER SECTION:
redhat.com.    971  IN  A  209.132.183.105

;; Query time: 1 msec
;; SERVER: 10.1.5.3#53(10.1.5.3)
;; WHEN: čet lip 28 18:49:49 CEST 2018
;; MSG SIZE  rcvd: 67

$ dig redhat.com +nocomments +nostats

; <<>> DiG 9.11.3-RedHat-9.11.3-6.fc28 <<>> redhat.com +nocomments +nostats
;; global options: +cmd
;redhat.com.      IN  A
redhat.com.    925  IN  A  209.132.183.105
```

Dijelovi ispisa su:

- preambula: ispis tehničkih detalja o odgovoru primljenom od DNS servera,
- `QUESTION SECTION`:  ispis upita poslanog DNS serveru,
- `ANSWER SECTION`: ispis odgovora na zahtjev poslan DNS serveru,
- kraj: ispis statističkih podataka zahtjeva.

## Pomoćni alat `host`

Naredba `host` je jednostavna naredba za obavljanje DNS dohvata. Obično se koristi za pretvaranje imena u IP adrese i obrnuto. Sljedeći primjer pokazuje korištenje naredbe host korištenjem imena, a zatim IP adrese

``` shell
$ host example.group.miletic.net
example.group.miletic.net is an alias for reaction.miletic.net.
reaction.miletic.net has address 108.61.208.98
reaction.miletic.net has IPv6 address 2001:19f0:6801:4e:5400:ff:fe5d:b9a6
$ host 108.61.208.98
98.208.61.108.in-addr.arpa domain name pointer reaction.miletic.net.
```

Sljedeći primjer prikazuje otkrivanje mail servera domene

``` shell
$ host -t mx apache.org
apache.org mail is handled by 10 mx1-lw-us.apache.org.
apache.org mail is handled by 10 mx1-lw-eu.apache.org.
```

Dobivanje TTL informacija

``` shell
$ host -v -t a mesa3d.org
Trying "mesa3d.org"
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 6792
;; flags: qr rd ra; QUERY: 1, ANSWER: 1, AUTHORITY: 0, ADDITIONAL: 0

;; QUESTION SECTION:
;mesa3d.org.      IN  A

;; ANSWER SECTION:
mesa3d.org.    10800  IN  A  131.252.210.176

Received 44 bytes from 10.1.5.3#53 in 47 ms
```

Više o alatima `host` i `dig` moguće je pronaći u nixCraftovom članku [Linux / UNIX: DNS Lookup Command](https://www.cyberciti.biz/faq/unix-linux-dns-lookup-command/).

!!! todo
    Ovdje treba dodati ili barem spomenuti `arpaname`, `delv`, `nslookup` i `nsupdate`.

## Datoteka `resolv.conf`

Ukoliko u datoteci `/etc/resolv.conf` promijenimo IP adresu nameservera na neku adresu koja nije valjana, proces rezolucije imena neće raditi (što možemo testirati ranije spomenutom naredbom `host`).

U sljedećem primjeru smo ubacili adresu lokalnog DNS servera (127.0.0.1), te zakomentirali vanjski DNS server (192.168.1.1)

```
domain dummy.porta.siemens.net
search dummy.porta.siemens.net
# nameserver 192.168.1.1
nameserver 127.0.0.1
```

Nakon toga rezolucija domene `example.group.miletic.net` postaje nemoguća, što možemo provjeriti naredbama `host` i `dig`

``` shell
$ host example.group.miletic.net
;; connection timed out; no servers could be reached
$ dig example.group.miletic.net

; <<>> DiG 9.11.3-RedHat-9.11.3-6.fc28 <<>> example.group.miletic.net
;; global options: +cmd
;; connection timed out; no servers could be reached
```
