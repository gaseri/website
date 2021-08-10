---
author: Kristijan Lenković, Vedran Miletić
---

# Tuneliranje alatom tinc

[Virtualna privatna mreža](https://searchenterprisewan.techtarget.com/definition/tunneling) (engl. *virtual private network*, kraće VPN) je tehnologija koja omogućava sigurno povezivanje računala u virtualne privatne mreže preko dijeljene ili javne mrežne infrastrukture. Korištenjem VPN-a moguće je povezivanje geografski odvojenih korisnika, kupaca ili poslovnih partnera. VPN podrazumijeva korištenje istih sigurnosnih i upravljačkih pravila koja se primjenjuju unutar lokalnih mreža. Također, VPN veze mogu se uspostaviti preko različitih komunikacijskih kanala; preko interneta, preko komunikacijske infrastrukture davatelja internet usluga, ATM mreža itd.

Za razliku od privatnih mreža koje koriste iznajmljene linije za slanje podataka, virtualna privatna mreža preko javne mreže stvara sigurni kanal između dviju krajnjih točaka.

U nastavku ćemo pokazati Tinc, popularan alat za VPN koji se odlikuje jednostavnošću instalacije i konfiguracije.

## Virtualna privatna mreža

Virtualna privatna mreža je privatna komunikacijska mreža koja se koristi za komunikaciju u okviru javne mreže.

Transport VPN paketa podataka odvija se u sklopu javne mreže (npr. internet) korištenjem standardnih komunikacijskih protokola te tako omogućava korisnicima na udaljenim lokacijama da preko javne mreže jednostavno održavaju zaštićenu komunikaciju.

Virtualna privatna mreža omogućava korisnicima razmjenu podataka vezom koja je emulirana kao direktna veza (engl. *point-to-point link*, kraće PPP) između klijenta i servera. PPP emulacija dobiva se učahurivanjem podataka zaglavljem koje omogućava rutiranje tj. usmjeravanje kroz javnu mrežu do odredišta koje je dio privatne mreže. Podaci su šifrirani, pa se presreteni paketi u okviru javne ili dijeljene mreže ne mogu pročitati bez ključa za dešifriranje. Infrastruktura javne mreže je nebitna jer korisnik vidi samo svoj privatni link, odnosno nalazi se u lokalnoj mreži, iako je od drugih korisnika razdvojen javnom mrežom.

Korištenje VPN-a u današnje doba je strahovito popularno. Koristi ga izrazito puno državnih institucija, kompanija, društva i organizacija. Upravo radi svoje pouzdanosti i visokog stupnja sigurnosti dosegao je visok stupanj popularnosti upravo iz razloga što se na jednostavan način, klijenti i serveri mogu komunicirati bez posrednika. Korištenje VPN-a ima izrazito puno svrha, bilo da se prijenos podataka mora dodatno zaštiti ili da se računala međusobno jednostavno "vide" i konstantno komuniciraju, simulirajući lokalnu mrežu (LAN) što preko interneta (WAN) nije jednostavno moguće.

## Tuneliranje

Tuneliranje je najvažnija komponenta tehnologije virtualnih privatnih mreža i predstavlja [prijenos paketa podataka namjenjenih privatnoj mreži preko javne mreže](https://searchenterprisewan.techtarget.com/definition/tunneling). Ruteri javne mreže nisu svjesni da prenose pakete koji pripadaju privatnoj mreži i VPN pakete tretiraju kao dio normalnog prometa.

Tuneliranje ili enkapsulacija tj. učahurivanje je metoda pri kojoj se koristi infrastruktura jednog protokola za prijenos paketa podataka drugog protokola. Umjesto da se šalju originalni paketi, oni su učahureni dodatnim zaglavljem. Dodatno zaglavlje sadrži informacije potrebne za rutiranje, odnosno usmjeravanje paketa kroz mrežu, tako da novodobiveni paket može slobodno putovati transportnom mrežom.

Tunel predstavlja logičnu putanju paketa kojom se on usmjerava preko mreže. Učahureni podaci se usmjeravaju transportnom mrežom s jednog kraja tunela na drugi. Pojam tunel uvodise jer su podaci koju putuju tunelom razumljivi samo onima koji se nalaze na njegovom izvorištu i odredištu. Ovi paketi se na mreži rutiraju kao svi ostali paketi.

Početak i kraj tunela nalaze se u VPN mrežama. Kada učahureni tj. enkapsulirani paket stigne na odredište vrši se deenkapsulacija i proslijeđivanje na konačno odredište. Cijeli proces enkapsulacije, transporta i deenkapsulacije paketa naziva se tuneliranje.

### Osobine tehnologije tuneliranja

Tehnologija tuneliranja ima osobine čije prednosti značajno doprinose njezinoj upotrebi, od kojih su najvažnije:

- Sigurnost -- bez obzira što tunel ide kroz nesigurnu javnu mrežu, pristup podacima koji su tunelirani nije dozvoljen neautoriziranim korisnicima što transport čini relativno sigurnim.
- Niska cijena -- obzirom se koriste javne mreže troškovi su dosta niski u usporedbi s troškovima potrebnim za iznajmljivanje privatnih linija ili implementaciju privatnih intranet mreža.
- Lakoća implementacije -- nema potrebe za promjenom postojeće infrastrukture javnih mreža, pa se VPN implementira samo na strani korisnika
- Univerzalnost -- zbog enkapsulacije moguće je koristiti i podatke koji pripadaju nerutabilnim protokolima. Takođe se štedi i na broju globalnih IP adresa koje kompanija mora posjedovati, što opet smanjuje cijenu implementacije virtualnih privatnih mreža.

### Protokoli koji se koriste pri tuneliranju

Tehnologija tuneliranja koristi tri vrste protokola:

- Protokol nosač -- ovi protokoli služe za usmjeravanje paketa po mreži ka njihovom odredištu. Tunelirani paketi imaju enkapsulaciju ovih protokola. Za usmjeravanje paketa po internetu koristi se IP protokol.
- Protokol za enkapsulaciju -- ovi protokoli služe za učahurivanje originalnih podataka, i koriste se za stvaranje, održavanje i zatvaranje tunela. Najčešće korišteni su PPTP i L2TP protokoli.
- Transportni protokol -- enkapsulira originalne podatke za transport kroz tunel.

Najpoznatiji su PPP i SLIP protokol.

### Upravljanje VPN-om

Govoreći o upravljanju, postoje dva pristupa virtualnim privatnim mrežama. Razlikujemo VPN kojima upravljaju korisnici, i VPN kojima upravljaju pružatelji mrežnih usluga (npr. Internet Service Provider -- ISP). Virtualne privatne mreže kojima upravljaju pružatelji mrežnih usluga dijele se na osnovu toga gdje se nalazi oprema koja implementira VPN:

- na strani pružatelja (PE -- provider edge),
- na strani korisnika (CE -- customer edge).

### Sigurnost VPN-a

Sigurnost je integralni dio VPN usluge. Postoji veliki broj prijetnji VPN mrežama:

- Neovlašteni pristup VPN prometu
- Izmjena sadržaja VPN prometa
- Ubacivanje neovlaštenog prometa u VPN (spoofing)
- Brisanje VPN prometa
- DoS (denial of service) napadi
- Napadi na infrastrukturu mreže preko softvera za upravljanje mrežom
- Izmjene konfiguracije VPN mreže
- Napadi na VPN protokole

Obrana od VPN napada realizira se i na korisničkom i na nivou poslužitelja VPN usluga:

- Kriptozaštita paketa
- Kriptozaštita kontrolnog prometa
- Filteri
- Firewall
- Kontrola pristupa
- Izolacija

VPN mreže koje koriste internet ili druge nezaštićene mreže obično koriste razne metode kriptozaštite. Korisnici VPN mreža s posebnim zahtjevima za sigurnost, na primjer banke, obično implementiraju i dodatnu infrastrukturu za zaštitu podataka.

## Tinc

[Tinc](https://www.tinc-vpn.org/) je open-source, samousmjeravajući protokol mesh mreža, koji se koristi za komprimiranje i kriptiranje virtualnih privatnih mreža. Projekt su započeli 1998. Guus Sliepen, Ivo Timmermans, i Wessel Dankers, a objavljen je kao GPL licencirani projekt.

Tinc je aplikacija koja se koristi za tuneliranje i šifriranje tj. stvaranje sigurne VPN virtualne privatne mreže između korisnika tj. hostova na internetu. Tinc je Free Software aplikacija, licencirana kao GNU General Public License verzija 2 ili novija. Budući da se VPN pojavljuje na IP razini mrežnog koda kao normalan mrežni uređaj, nema potrebe prilagođavati bilo koji postojeći softver. To omogućava VPN korisnicima sigurnu međusobnu razmjenu informacija putem interneta, bez opasnosti izlaganja privatnih podataka neovlaštenim osobama.

Osim toga, TINC ima sljedeće značajke:

- **Šifriranje, autentifikacija i kompresija.** Sav promet se opcionalno komprimira pomoću zlib ili LZO, a OpenSSL se koristi za enkripciju prometa te njegovu zaštitu od neovlaštenih izmjena pomoću poruka s autentifikacijskim kodovima i brojevnim sekvencama.
- **Automatsko potpuno usmjeravanje u mesh mrežama (engl. *full mesh routing*).** Bez obzira na postavke TINC aplikacija kako će se međusobno povezivati, VPN promet se uvijek (ako je to moguće) šalje izravno na odredište, bez nepotrebnih prijelaza preko dodatnih, posrednih točaka.
- **Jednostavno proširivanje vlastite VPN mreže.** Kada želite dodati nove čvorove svojoj VPN mreži, sve što morate učiniti je dodati novu konfiguracijsku datoteku, nema potrebe za novim aplikacijama tj. demonima, a ne treba ni stvarati ni konfigurirati nove uređaje ili pak mrežna sučelja.
- **Sposobnost premošćivanja Ethernet segmenata.** Možete povezati više mrežnih segmenata zajedno da rade kao jedan segment, omogućujući tako pokretanje aplikacija i igrara koje obično rade samo na LAN preko interneta.
- **Funkcionira na mnogim operativnim sustavima i podržava IPv6.** Trenutno su podržane Linux, FreeBSD, OpenBSD, NetBSD, MacOS / X, Solaris, Windows 2000, XP, Vista i Windows 7 i 8 platforme. TINC također podržava u potpunosti IPv6, pružajući mogućnost tuneliranja IPv6 prometa svojim tunelima kao i stvaranja tunela u već postojećim IPv6 mrežama

### Instalacija i konfiguracija

Kao praktični primjer, na vlastiti laptop (`narciss`) i server (`lines`) postavit ću VPN pomoću Tinc-a, prateći pritom Tinc-ovu [službenu dokumentaciju](https://www.tinc-vpn.org/documentation/). Oba računala koriste operativni sustav Debian GNU/Linux:

``` shell
$ uname -a
Linux narciss 3.16.0-4-amd64 #1 SMP Debian 3.16.7-ckt20-1+deb8u3 (2016-01-17) x86_64 GNU/Linux
$ uname -a
Linux lines 3.16.0-4-amd64 #1 SMP Debian 3.16.7-ckt11-1+deb8u5 (2015-10-09) x86_64 GNU/Linux
```

Za početak je na oba računala potrebno instalirati aplikaciju tinc koja je dostupna u Debian službenom repozitoriju:

``` shell
$ sudo apt-get install tinc
```

Na serveru (`lines`) kreirat ćemo direktorije za konfiguraciju pomoću komande:

``` shell
$ sudo mkdir -p /etc/tinc/linesvpn/hosts
```

Zatim je potrebno kreirati i konfigurirati Tinc postavke u datoteci `tinc.conf` na slijedeći način:

``` shell
$ sudo nano /etc/tinc/linesvpn/tinc.conf
```

Postavke moraju sadržavati:

``` ini
Name = lines
AddressFamily = ipv4
Interface = tun0
```

U gornjem primjeru, direktorij `linesvpn` je ime VPN mreže koja će se međusobno uspostaviti između računala narciss i lines. VPN ime može imati bilo koji alfanumerički znak, ali ne smije sadržavati znak '-'.

Zatim, kreiramo datoteku s postavkama za server (lines) s detaljnim informacijama:

``` shell
$ sudo nano /etc/tinc/linesvpn/hosts/lines
```

Datoteka sadrži:

``` ini
Address = 81.169.198.47
Subnet = 10.0.0.1/32
```

Ime datoteke mora odgovarati imenu specificiranom u datoteci `tinc.conf`. Polje `Address` predstavlja javnu IP adresu servera lines. Ovo je polje ujedno obavezno za najmanje jedno računalo u VPN mreži kako bi se ostala računala mogla spojiti na njega.

Zatim je potrebno generirati privatni i javni ključ:

``` shell
$ sudo tincd -n myvpn -K4096
```

Ova komanda generira 4096-bitni privatni i javni ključ za server lines. Privatni ključ biti će spremljen u `/etc/tinc/linesvpn/rsa_key.priv`, a javni nadodan u datoteku `/etc/tinc/linesvpn/hosts/lines`.

Na kraju, potrebno je konfigurirati skripte koje se pokreću kada se servis Tinc pali i gasi:

``` shell
$ sudo nano /etc/tinc/linesvpn/tinc-up
```

Sadržaj datoteke `tinc-up`:

``` bash
#!/bin/sh
ifconfig $INTERFACE 10.0.0.1 netmask 255.255.255.0
```

``` shell
$ sudo nano /etc/tinc/linesvpn/tinc-down
```

Sadržaj datoteke `tinc-down`:

``` bash
#!/bin/sh
ifconfig $INTERFACE down
```

Izvršimo iduću komandu da se skripte mogu pokretati:

``` shell
$ sudo chmod 755 /etc/tinc/linesvpn/tinc-*
```

Na sličan način konfigurira se i klijent (`narciss`):

``` shell
$ sudo mkdir -p /etc/tinc/linesvpn
$ sudo nano /etc/tinc/linesvpn/tinc.conf
```

``` ini
Name = narciss
AddressFamily = ipv4
Interface = tun0
ConnectTo = lines
```

Za razliku od servera lines, nadodaje se polje `ConnectTo` obzirom da će klijent `narciss` inicirati VPN konekciju na server lines.

``` shell
$ sudo nano /etc/tinc/linesvpn/hosts/narciss
```

``` ini
Subnet = 10.0.0.2/32
```

``` shell
$ sudo tincd -n myvpn -K4096
```

Kao i na serveru `lines`, `narciss`-ov privatni ključ spremljen je u datoteci `/etc/tinc/linesvpn/rsa_key.priv`, a `narciss`-ov javni ključ je dodan konfiguracijskoj datoteki `/etc/tinc/linesvpn/hosts/narciss`.

``` shell
$ sudo nano /etc/tinc/myvpn/tinc-up
```

``` bash
#!/bin/sh
ifconfig $INTERFACE 10.0.0.2 netmask 255.255.255.0
```

``` shell
$ sudo nano /etc/tinc/linesvpn/tinc-down
```

```
#!/bin/sh
ifconfig $INTERFACE down
```

``` shell
$ sudo chmod 755 /etc/tinc/linesvpn/tinc-*
```

Po završetku, datoteke unutar direktorija `/etc/tinc/linesvpn/hosts` potrebno je međusobno kopirati među računalima. Dakle datoteku `/etc/tinc/linesvpn/hosts/narciss` na klijentu `narciss` potrebno je kopirati u direktorij `/etc/tinc/linesvpn/hosts` na serveru lines, a datoteku `/etc/tinc/linesvpn/hosts/lines` sa servera `lines`, potrebno je kopirati u direktorij `/etc/tinc/linesvpn/hosts` na klijentu `narciss`.

Na kraju, pokrenemo tinc prvo na serveru `lines`, a zatim na klijentu `narciss`:

``` shell
$ sudo tincd -n linesvpn
```

Nakon izvršene komande, računala narciss i lines bi trebala moći normalno međusobno komunicirati preko VPN adresa koje su im dodijeljene (10.0.0.1 -- `lines` i 10.0.0.2 -- `narciss`).

Alat Tinc, poslužio je kao jednostavan primjer konfiguriranja i korištenja VPN-a. Vrlo je jednostavan za naučiti, nema kompleksnih postavki i svakako ga preporučujem svakome tko želi uspostaviti svoju vlastitu virtualnu privatnu mrežu.
