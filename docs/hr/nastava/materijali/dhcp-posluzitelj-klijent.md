---
author: Nikola Barković, Vedran Miletić
---

# Dinamičko dodjeljivanje adresa domaćinima

[Dynamic Host Configuration Protocol](https://en.wikipedia.org/wiki/Dynamic_Host_Configuration_Protocol) (kraće DHCP) je mrežni protokol koji se koristi za konfiguraciju mrežnih uređaja kako bi mogli komunicirati u IP mreži.

DHCP je prvi put standardiziran 1993. godine ([RFC 1541](https://datatracker.ietf.org/doc/html/rfc1541)) kao dodatak protokolu [Bootstrap Protocol](https://en.wikipedia.org/wiki/Bootstrap_Protocol) (kraće BOOTP). Motivacija za nadogradnju BOOTP-a je njegova potreba za ručnim unosom konfiguracijskih informacija za svaki od klijenata i nedostatak mehanizma za uzimanje nekorištenih IP adresa. DHCP je postao popularan i od 1997. godine ([RFC 2131](https://datatracker.ietf.org/doc/html/rfc2131)) postaje samostalni protokol, ali nastavlja koristiti vrata koja je IANA dodjelila za BOOTP: UDP vrata 67 za slanje podataka poslužitelju i UDP vrata 68 za slanje podataka klijentu. Danas je standard na gotovo svim uređajima koji koriste IP.

DHCP se koristi za IPv4 i IPv6; iako obje verzije služe istoj svrsi, detalji protokola su dovoljno različiti da bi bili različiti protokoli. DHCPv6 (DHCP za IPv6) opisan je u [RFC-u 8415](https://datatracker.ietf.org/doc/html/rfc8415). Mi ćemo se u nastavku ograničiti na DHCP za IPv4.

## Uloge DHCP klijenta i poslužitelja

DHCP automatizira dodjeljivanje mrežnih parametara mrežnim uređajima s jednog ili više DHCP poslužitelja. I kod malih mreža DHCP je koristan jer omogućuje jednostavno dodavanje novih uređaja na mrežu.

DHCP klijent koristi protokol DHCP da bi zatražio mrežnu konfiguraciju, u koju spadaju IP adresa, zadana ruta, jedna ili više adresa DNS poslužitelja i druge. DHCP klijent se obično inicijalizira nakon pokretanja operacijskog sustava, a upit koji traži mrežnu konfiguraciju se šalje po raspoznavanju postojanja mrežne veze na fizičkom i veznom sloju (npr. kabel i lampica koja svijetli u slučaju Ethernet mreže i povezanost na pristupnu točku u slučaju Wi-Fi mreže). DHCP klijent će po dobitku mrežne konfiguracije koristiti te informacije kako konfigurirao mrežne postavke na svom domaćinu i ostvario komunikaciju s ostalim domaćinima u mreži te, potencijalno, komunikaciju s drugim mrežama i internetom.

DHCP poslužitelj održava popis slobodnih IP adresa i ima konfiguracijska pravila koja utječu na njihovo dodjeljivanje. Kada zaprimi zahtjev od klijenta, DHCP poslužitelj alocira IP adresu koja je prikladna klijentu i šalje konfiguracijske informacije klijentu. Budući da je protokol DHCP mora raditi ispravno prije nego su DHCP klijenti konfigurirani, DHCP poslužitelj i DHCP klijent moraju biti spojeni na istu mrežu kako bi se mogli doseći putem adrese mreže za broadcast slanje. DHCP poslužitelj obično dodjeljuje IP adresu klijentu na određeno vrijeme pa su DHCP klijenti odgovorni za obnavljanje IP adrese prije nego to vrijeme istekne. U slučaju da vrijeme istekne i adresa nije obnovljena, DHCP klijent mora prestati koristiti dobivenu IP adresu.

Konkretno, kada se DHCP klijent spoji na mrežu, on šalje DHCP upit na adresu za broadcast slanje i zatražuje potrebne informacije od DHCP poslužitelja. Kada zaprimi ispravam upit poslužitelj dodjeljuje klijentu IP adresu i ostale konfiguracijske parametre. Nakon sto se klijent odluči isključi s mreže, poruku o tome poslat će DHCP poslužitelju koji će dotadašnju klijentovu IP adresu označiti kao slobodnu za ponovno korištenje od strane drugih klijenata.

Ovisno o konfiguraciji, DHCP poslužitelj ima tri načina alokacije IP adrese:

- **Dinamička alokacija**: mrežni administrator dodjeli niz IP adresa DHCP poslužitelju, a svaki klijent u lokalnoj mreži je konfiguriran tako da potražuje IP adresu od DHCP poslužitelja nakon raspoznavanja veze na fizičkom i veznom sloju. DHCP poslužitelj koristi koncept iznajmljivanja IP adrese s kontroliranim vremenom kako bi mogao adresu dodjeliti nekom drugom klijentu ako prvi ne zatraži produljenje. Ovaj pristup je koristan kod mreža koje imaju veliku fluktuaciju domaćina (npr. [Free Wi-Fi Rijeka](https://www.rijeka.hr/servisne-informacije/free-wi-fi-rijeka/), [eduroam](https://www.eduroam.org/)).
- **Automatska alokacija**: DHCP poslužitelj trajno dodjeljuje slobodnu IP adresu klijentu koji je potražuje iz niza koji je definirao administrator, kao i kod dinamičke alokacije, ali u ovom slučaju DHCP poslužitelj pohranjuje u tablicu dodjelu adrese kako bi klijent mogao opet dobiti istu adresu ako je zatraži. Ovaj pristup je koristan kod mreža koje imaju periodičku, ali rijetku promjenu domaćina koji se na mreži nalaze (npr. mreže računalnih učionica na Odjelu za informatiku, mreže u tvrtkama).
- **Statička alokacija**: DHCP poslužitelj alocira IP adrese na temelju tablice s parovima MAC adresa i IP adresa koji su ručno uneseni od strane administratora. Samo klijentima s MAC adresom koja je u tablici će biti dodjeljena IP adresa. Ova opcija nije podržana od strane svih DHCP poslužitelja. Ovaj pristup je koristan, primjerice, za konfiguraciju domaćina koji služe kao web i mail poslužitelji pa njihova adresa mora biti predvidiva, a nalaze se u mrežama koje koriste protokol DHCP.

## Proces konfiguracije

Proces konfiguracije korištenjem DHCP-a ima četri osnovne faze: otkriće (engl. *discovery*), ponuda (engl. *offer*), zahtjev (engl. *request*) i prihvat ponude (engl. *acknowledgment*).

### Otkriće

Klijent šalje poruku DHCP otkriće na broadcast adresu podmreže kako bi otkrio moguće DHCP poslužitelje. Mrežni administrator može konfigurirati lokalni usmjerivač da prosljeđuje DHCP pakete DHCP poslužitelju koji se nalazi u drugoj podmreži; u potonjem slučaju, stvara se UDP paket s odredišnom adresom 255.255.255.255 ili sa specifičnom podmrežnom adresom za broadcast slanje.

DHCP klijent također može tražiti adresu koja mu je bila dodjeljena prethodni put. Ako je klijent spojen na mrežu za koju mu je dodjeljena valjana IP adresa, poslužitelj mu može odobriti zahtjev. U suprotnom, poslužitelj, ovisno o implementaciji i konfiguraciji, može:

- zanemariti zahtjev pa, nakon što prođe određeno vrijeme u kojem poslužitelj ne odgovara, klijent odustaje od takvog zahtjeva i traži novu adresu, ili
- odbiti zahtjev pa klijent odmah traži novu adresu.

### Ponuda

Kada DHCP poslužitelj primi zahtjev za IP adresom od DHCP klijenta, tada poslužitelj rezervira tu adresu za klijenta i šalje poruku DHCP ponude klijentu. Ova poruka sadrži MAC adresu, IP adresu koju poslužitelj nudi, podmrežnu masku, duljinu trajanja valjanosti adrese i IP adresu DHCP poslužitelja koju je poslao ponudu.

### Zahtjev

Odgovor DHCP klijenta na DHCP ponudu je DHCP zahtjev: klijent odgovara porukom koja se šalje DHCP poslužitelju tražeći ponuđenu adresu. Klijent može primiti više DHCP ponuda, ali će prihvatiti samo jednu. DHCP zahtjev se može slati na samo jedan poslužitelj (onaj čiji je zahtjev prihvaćen) ili na sve poslužitelje. Kada DHCP poslužitelji prime poruku da njihova ponuda nije prihvaćena, ponuđenu adresu vraćaju u skup adresa za dodjelu.

### Prihvat ponude

Kada DHCP poslužitelj primi zahtjev od DHCP klijenta, proces konfiguracije ulazi u posljednju fazu. U ovoj fazi se klijentu šalje DHCP paket koji sadrži sve potrebne informacije koje je klijent tražio i IP konfiguracija je onda gotova.

## ISC DHCP

Mi ćemo u nastavku koristiti [ISC DHCP](https://www.isc.org/dhcp/), DHCP poslužitelj i klijent čiji je autor [Internet Systems Consortium](https://www.isc.org/) (kraće ISC). ISC DHCP podržava IPv4 i IPv6, skalabilan je i dostupan pod licencom slobodnog softvera otvorenog koda [MPL 2.0](https://www.mozilla.org/en-US/MPL/2.0/). Očekuje se da će u budućnosti ISC-ov DHCP poslužitelj postepeno biti zamijenjen novijim ISC-ovim DHCP poslužiteljem imena [Kea](https://www.isc.org/kea/).

### Klijentska strana

ISC DHCP klijent naziva se `dhclient`. Korištenje klijenta je vrlo jednostavno, naredbom

``` shell
# dhclient
```

pokreće se proces konfiguracije korištenjem DHCP-a na svim mrežnim sučeljima, a naredbom

``` shell
# dhclient eth0
```

samo na sučelju `eth0`. Ostali parametri naredbenog retka opisani su u man stranici `dhclient(8)` (naredba `man 8 dhclient`). Specijalno, parametar `-cf` omogućuje navođenje konfiguracijske datoteke na način

``` shell
# dhclient -cf mydhclient.conf eth0
```

unutar koje se mogu nalaziti konfiguracijske naredbe isteka vremena (`timeout`), vremena prije ponovnog pokušaja (`retry`) i druge. Opis pojedinih konfiguracijskih naredbi dan je u man stranici `dhclient.conf(5)` (naredba `man 5 dhclient.conf`).

!!! caution
    Moguće je da `dhclient` kod pokretanja javi pogrešku oblika:

    ```
    System has not been booted with systemd as init system (PID 1). Can't operate.
    Failed to connect to bus: Host is down
    ```

    Suvremeni operacijski sustavi temeljeni na Linuxu koriste [systemd](https://systemd.io/) (izvedenica od **system** **d**aemon) za pokretanje sustavskih procesa, ali čvorovi u CORE-u to ne rade jer ima je cilj zauzeti što manje memorije (i time omogućiti pokretanje većih emulacija). To objašnjava prvi redak pogreške.

    Što se drugog retka, za međuprocesnu komunikaciju koristi se [D-Bus](https://www.freedesktop.org/wiki/Software/dbus/) koji također nije pokrenut (iz istih motiva kao i systemd) pa se proces na njega ne može povezati.

    Procesi koje koristimo u emuliranim čvorovima, uključujući i `dhclient`, uredno rade bez obzira na ove pogreške pa ih možemo ignorirati.

### Poslužiteljska strana

ISC DHCP poslužitelj naziva se `dhcpd`; slovo `d` na kraju imena oznaka je da se radi o [daemonu](https://en.wikipedia.org/wiki/Daemon_(computing)). Poslužitelj se pokreće naredbom

``` shell
# dhcpd -cf mydhcpd.conf
```

jer će, za razliku od klijenta, bez konfiguracije vrlo teško raditi ispravno. Ostale parametre naredbenog retka moguće je naći u man stranici `dhcpd(8)`. Tipična konfiguracijska datoteke `mydhcpd.conf` sadrži sljedeće konfiguracijske naredbe

``` nginx
option domain-name "core-emulation.rm.miletic.net";
authoritative;
```

Prva naredba postavlja ime domene DHCP poslužitelja, a druga kaže da je to glavni DHCP poslužitelj na toj domeni. Zatim naredbama

``` nginx
default-lease-time 86400;
max-lease-time     172800;
```

postavljamo zadano vrijeme iznajmljivanja IP adrese na 24 sata (86400 sekundi), a maksimalno dozvoljeno na 48 sati (172800 sekundi). Također navodimo putanju datoteke gdje su spremljene iznajmljene adrese

``` nginx
lease-file-name    "/tmp/dhcpd.leases";
```

Za svaku od podmreža možemo definirati posebne postavke. Uzmimo da nam je, kao tipičnom kućnom ADSL/VDSL korisniku, dana podmreža 192.168.5.0/24 i da DHCP poslužitelj ima adresu 192.168.5.1 i služi kao izlaz iz mreže. Konfiguracijske naredbe su tada

``` nginx
subnet 192.168.5.0 netmask 255.255.255.0 {
    option routers           192.168.5.1;
    option subnet-mask       255.255.255.0;
    option broadcast-address 192.168.5.255;
    range  192.168.5.101     192.168.5.200;
}
```

DHCP klijenti koji zatraže IP adresu od poslužitelja dobit će adresu u rasponu navedenom naredbom `range`. Detaljniji opis pojedinih konfiguracijskih naredbi dan je u man stranici `dhcpd.conf(5)`.

## Primjer korištenja ISC DHCP-a unutar CORE-a

Stvorimo jednostavnu mrežu u kojoj se nalaze dva računala n1 i n2 s DHCP klijentima i domaćin n4 s DHCP poslužiteljem vezani preklopnikom n3. Mreža ima raspon adresa 172.16.25.0/24 i DHCP poslužitelj će imati adresu 172.16.25.1, a klijentima će dinamički dodjeljivati adrese od 172.16.25.20 do 172.16.249. Mreža ima topologiju oblika

```
n1
 \
  \
  n3 ---- n4
  /
 /
n2
```

Radi potpunosti, isključit ćemo sve usluge (desni klik na čvor, `Configure` pa `Services`) na n1, n2 i n4 pa ćemo ručno pokretati DHCP poslužitelj i klijente. Osim toga, izbrisat ćemo IPv6 adrese na n1, n2 i n4 te IPv4 adrese na n1 i n2. Na n4 postavit ćemo IPv4 adresu na 172.16.25.1.

Nakon pokretanja emulacije, u ljusci čvora n4 korištenjem bilo kojeg uređivačem teksta stvaramo datoteku `mydhcpd.conf` oblika

``` nginx
option domain-name "core-emulation.rm.miletic.net";
authoritative;

default-lease-time 86400;
max-lease-time     172800;

lease-file-name    "/tmp/dhcpd.leases";

subnet 172.16.25.0 netmask 255.255.255.0 {
    option routers           172.16.25.1;
    option subnet-mask       255.255.255.0;
    option broadcast-address 172.16.25.255;
    range  172.16.25.20      172.16.25.249;
}
```

Kako bismo mogli pokrenuti `dhcpd`, treba nam i datoteka `/tmp/dhcpd.leases` pa ćemo je stvoriti (praznu) naredbom

``` shell
# touch /tmp/dhcpd.leases
```

Poslužitelj se sada pokreće naredbom

``` shell
# dhcpd -cf mydhcpd.conf
Internet Systems Consortium DHCP Server 4.4.1
Copyright 2004-2018 Internet Systems Consortium.
All rights reserved.
For info, please visit https://www.isc.org/software/dhcp/
Not searching LDAP since ldap-server, ldap-port and ldap-base-dn were not specified in the config file
Config file: mydhcpd.conf
Database file: /tmp/dhcpd.leases
PID file: /var/run/dhcpd.pid
Source compiled to use binary-leases
Wrote 0 leases to leases file.
Listening on LPF/eth0/00:00:00:aa:00:02/172.16.25.0/24
Sending on   LPF/eth0/00:00:00:aa:00:02/172.16.25.0/24
Sending on   Socket/fallback/fallback-net
```

U ljusci čvora n1 pokrenimo naredbu

``` shell
# dhclient
```

Uvjerimo se da je `dhclient` uspješno dobio IPv4 adresu

``` shell
# ifconfig
eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 172.16.25.20  netmask 255.255.255.0  broadcast 172.16.25.255
        inet6 fe80::200:ff:feaa:0  prefixlen 64  scopeid 0x20<link>
        ether 00:00:00:aa:00:00  txqueuelen 1000  (Ethernet)
        RX packets 113  bytes 12804 (12.5 KiB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 26  bytes 2508 (2.4 KiB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```

Isti postupak ponovimo na čvoru n2. Sada je komunikacija omogućena kao da u situaciji kad se koriste statičke adrese.

## Dodatak: sigurnost

U osnovi DHCP protokol ne nudi niti jedan mehanizam za autentifikaciju. Zbog toga je podložan različitim vrstama napada koji se mogu podjelit u tri glavne kategorije:

- Neautorizirani DHCP poslužitelj daje pogrešne podatke klijentima.
- Neautorizirani klijent ima pristup resursima.
- Napadi koji zagušuju resurse.

Budući da za klijenta nema načina da provjeri identitet DHCP poslužitelja, neautorizirani DHCP poslužitelji mogu raditi na bilo kojoj mreži. Ovo može biti napad koji brani pristup klijenta mreži (engl. *denial-of-service attack*) ili napad prisluškivanjem (engl. *man-in-the-middle attack*).

Budući da DHCP poslužitelj nema sigurnog mehanizma za provjeru klijenata, klijenti mogu neautorizirano prestupiti IP adresama prezentirajući se kao drugi DHCP klijenti. Ovo omogućava da se iskoriste sve dostupne IP adrese i onemogući pristup pravim korisnicima.

DHCP ipak ima neke mehanizme za rješavanje ovih problema. The Relay Agent Information Option protokol omogućuje mrežnim operaterima da dodaju potpis na DHCP poruke. Ovaj potpis se koristi kao provjera kako bi se kontrolirao pristup mrežama od strane klijenata.

Jos jedna mogućnost je Authentication for DHCP Messages koja omogućuje provjeru DHCP poruka. Problem kod ove metode je spremanje ključeva za velik broj DHCP klijenata.
