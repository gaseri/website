---
author: Ingrid Beloša, Vedran Miletić
---

# Raspoznavanje upada u sustav alatom Bro

Bro je alat otvorenog koda koji služi za detekciju upada u sustav i nadgledanje mrežnog prometa u stvarnom vremenu (engl. *network intrusion detection system*) s ciljem otkrivanja malicioznog sadržaja. Namijenjen je instalaciji na Linux/Unix operacijskim sustavima. Naziv dolazi od simpatičnog termina "Big Brother" za koji možemo reći da predstavlja samu ideju alata i njegovih mogućnosti. Razvijen je od strane Verna Paxsona korištenjem libpcap biblioteke. Uz pomoć libpcap programske biblioteke koju je unaprijed potrebno imati instaliranu na sustavu, odvija se praćenje i generiranje podataka o mrežnom prometu. Osim što analizira mrežni promet, Bro omogućuje mjerenje performansi i uklanjanje grešaka. Nema grafičko sučelje, već se naredbe izvršavaju preko komandne linije.

Upad podrazumijeva aktivnosti koje su usmjerene na narušavanje integriteta, povjerljivosti ili dostupnosti mrežnih resursa, a uloga sustava za detekciju upada je da nadgleda takve aktivnosti i izvještava o njihovoj pojavi. Drugim riječima, takav sustav automatizirano prati mrežne i sistemske događaje u svrhu detekcije narušavanja sigurnosne politike, ali ne i sprječavanja upada i nedozvoljenih radnji. Kako bi otkrio nestandardne radnje i upade, Bro koristi skup pravila otvorenog koda koja se mogu mijenjati i dodavati otkrivanjem novih oblika mrežnih napada što ima za cilj bolje nadziranje mreže. Sve nepravilnosti na koje naiđe, Bro može zapisati u posebne datoteke -- log zapise. Log zapisi su automatski kreirani podaci o radu sustava koji omogućuju olakšan nadzor tog sustava odnosno mreže te pronalaženje i otklanjanje grešaka. Također, otkrije li kakvo odstupanje, može stvarati obavijest u stvarnom vremenu ili pokrenuti izvođenje određene naredbe.

## Mogućnosti alata

Bro pruža potpuno pasivnu analizu (pasivna analiza mrežnog prometa znači da implementacijom Bro osluškuje i analizira mrežni promet bez ikakvog utjecaja na funkcionalnost mreže)mrežne aktivnosti usporedbom "snimljenog" stanja ili stvarnog stanja mreže i unaprijed definiranih pravila skriptnim jezikom Bro -- programski jezik pokretan događajima (engl. *event driven*). Napredniji korisnici imaju mogućnost definirati svoja pravila Bro jezikom te tako proširiti funkcionalnost alata. Također, Bro može pročitati podatke i analizirati promet uhvaćen alatima kao što su Wireshark, Snort i tcpdump.

BroControl je interaktivna ljuska (engl. *shell*) za jednostavno upravljanje alatom Bro odnosno nadziranje jednog sustava ili klastera.

Log datoteke pružaju sveobuhvatnu evidenciju svake konekcije na mreži. Rad s uhvaćenim prometom podrazumijeva uvid u strukturirane dokumente i dohvaćanje pojedinih atributa odnosno pretraživanje prikupljenih podataka naredbama u komandnoj liniji, mogućnost filtriranja i sortiranja paketa prema različitim kriterijima. Pakete je, primjerice, moguće filtrirati s obzirom na protokol, a konekcije možemo sortirati prema trajanju.

Još neke značajne mogućnosti:

- implementacija u klaster pri čemu snima performanse svih strojeva klastera i nadzire mrežni promet
- potpuna integracija u mrežu
- nadzor mreže s IPv6
- detekcija i analiza tunela (logički put kroz koji enkapsulirani podaci prolaze kroz javnu mrežu, npr. Ayiya, Teredo, GTPv1)

## Arhitektura

Bro ima [tri glavne komponente](https://docs.zeek.org/en/current/about.html#architecture) pomoću kojih nadzire mrežu: libpcap biblioteku za hvatanje mrežnog prometa, generator događaja (engl. *event engine*) koji analizira mrežni promet u stvarnom vremenu ili snimljeni promet i pretvara u događaje (engl. *events*) kao što su `connection_attempt`, `http_reply`, `user_logged_in` i interpreter sigurnosnih pravila. Događaji odražavaju stanje mreže, a uspoređuju se s pravilima odnosno skriptama u kojima su pravila definirana. Skripte zapravo opisuju sigurnosnu politiku sustava koji će se nadzirati odnosno aktivnosti koje se poduzimaju ukoliko se uoči neka nepravilnost. Primjerice, svaki HTTP zahtjev ima odgovarajući http_request događaj koji sadrži IP adrese ishodišta i odredišta paketa, portove, traženi URI. No, taj događaj ne nosi informaciju o možebitnom malicioznom sadržaju tražene stranice ili informaciju da li se radi o zloćudnoj stranici. Interpreter sigurnosnih pravila izvodi upravljače događajima (engl. *event handlers*) pisane u Bro skriptnom jeziku, analizira događaje i sadržaj stranice te poduzima određenu aktivnost kao što je automatski zapis u log datoteku, slanje alarmnih poruka u realnom vremenu, slanje maila ili pozivanje druge Bro skripte. Interpreter sigurnosnih pravila je zapravo najvažniji dio alata Bro jer otkriva maliciozne događaje odnosno pakete korištenjem unaprijed definiranih pravila. Ukoliko se neki događaj poklapa s određenim pravilom, obavljaju se određene radnje nad paketom ili se on odbacuje.

``` dot
digraph G {
   node [shape = box];
   mreža -> libpcap [label = paketi];
   libpcap -> "generator događaja";
   "generator događaja" -> "interpreter sigurnosnih pravila" [label = događaji];
   "interpreter sigurnosnih pravila" -> "korisnik" [label = logovi];
   "interpreter sigurnosnih pravila" -> "korisnik" [label = notifikacije];
}
```

## Instalacija i konfiguracija alata

Instalacija alata ovisi o operativnom sustavu odnosno zahtijeva Unix-bazirani OS. Prije instalacije alata Bro, potrebno je instalirati sljedeće biblioteke i alate:

- Libpcap
- biblioteka OpenSSL-a
- biblioteka BIND-a 9
- Libmagic
- Libz
- Bash (za BroControl)
- Python (za BroControl)

odnosno za build sa sourcea:

- CMake
- Make
- C/C++ compiler
- SWIG Bison (GNU Parser Generator)
- Flex (Fast Lexical Analyzer)
- Libpcap headers
- OpenSSL headers
- libmagic headers
- zlib headers
- Perl

Navedene biblioteke i alate instaliramo naredbom:

``` shell
$ sudo yum install cmake make gcc gcc-c++ flex bison libpcap-devel openssl-devel python-devel swig zlib-devel file-devel
```

za RPM/RedHat-bazirani Linux odnosno naredbom

``` shell
$ sudo apt-get install cmake make gcc g++ flex bison libpcap-dev libssl-dev python-dev swig zlib1g-dev libmagic-dev
```

za DEB/Debian-bazirani Linux.

Bro je moguće preuzeti u obliku unaprijed kompajliranog paketa ili u obliku [izvornog koda](https://zeek.org/get-zeek/). Naredbe za instalaciju paketa iz izvornog koda ili sa git repozitorija su sljedeće:

``` shell
$ git clone --recursive https://github.com/zeek/zeek.git
$ ./configure
$ make
$ make install
```

Nakon instalacije, potrebno je postaviti putanju do Bro-a:

``` shell
# export PATH=/usr/local/bro/bin:$PATH
# bro -v
bro version 2.2-75
```

Za upravljanje jednom instancom Bro-a lokalnog hosta potrebno je izmijeniti sadržaj datoteke `node.cfg`:

- u `/usr/local/bro/etc/node.cfg` postaviti odgovarajuće mrežno sučelje čiji promet će se nadzirati:

    ``` ini
    [bro]
    type=standalone
    host=localhost
    interface=eno167777736
    ```

Pokretanje BroControl ljuske (kako bi mogao pratiti promet, korisnik mora pokrenuti BroControl kao root):

``` shell
# broctl

Welcome to BroControl 1.2-3

Type "help" for help.

[BroControl] >
```

Kod prvog korištenja BroControl ljuske, naredbom install će se izvesti inicijalna konfiguracija:

```
[BroControl] > install
removing old policies in /usr/share/bro/.site ... done.
creating policy directories ... done.
installing site policies ... done.
generating broctl-layout.bro ... done.
generating analysis-policy.bro ... done.
generating local-networks.bro ... done.
updating nodes ... done.
[BroControl] >
```

Ukoliko nakon toga dođe do modificiranja BroControla kao što je promjena konfiguracijskih datoteka ili unaprijed definiranih skripti sigurnosne politike, naredbom install se ponovno uspostavlja inicijalna konfiguracija. Dakle, ukoliko promijenimo sadržaj unaprijed definiranih skripti ili izbrišemo datoteke koje dolaze sa BroControlom, lako se možemo vratiti na početnu konfiguraciju.

Bro instanca se pokreće naredbom start, a zaustavlja sa stop:

```
[BroControl] > start
starting bro ...
[BroControl] > status
Name       Type       Host       Status          Pid      Peers   Started
bro        standalone localhost  running         5809     0       30 Dec 11:57:50
[BroControl] > start
stopping bro ...
```

Pokretanjem Bro instance započinje snimanje prometa odnosno zapis u log datoteke koje se nalaze u direktoriju `/usr/local/bro/logs`. Direktorij logs sadrži poddirektorije čiji je naziv datum snimanja prometa, a sadrži log datoteke -- `conn.log`, `dhcp.log`, `dns.log`, `http.log`, `smtp.log`, `ftp.log`... -- koje se stvaraju po zaustavljanju praćenja prometa. Drugim riječima, sve što je Bro uhvatio za vrijeme rada (status: running), po zaustavljanju sortira s obzirom na protokol ili pojedinu aktivnost i kopira u odgovarajuće datoteke.

## Demonstracija sposobnosti alata

Bro sažima svaku TCP i UDP konekciju u jednu liniju u datoteku `conn.log`. Obzirom da su ti sažeci prilično detaljni, moguće je doći do [mnoštva korisnih informacija i statistika](https://docs.zeek.org/en/current/logs/index.html). Valja spomenuti i datoteke `http.log`, `conn-summary.log`, `known_services.log`, `ssh.log` i `weird.log`. Analiza spomenutih datoteka slijedi u nastavku.

### Analiza snimljenog prometa

Sve konekcije koje traju dulje od jedne minute:

``` shell
$ awk 'NR > 4 && $9 > 60' conn.log
```

!!! todo
    Potrebno je dodati nekakav izlaz ovdje i osvježiti dio ispod u skladu s dodanim.

IP adrese svih web servera koji šalju više od 1 MB podataka klijentu:

``` shell
$ bro-cut service resp_bytes id.resp_h < conn.log | awk '$1 == "http" && $2 > 1000000 { print $3 }' | sort -u
130.59.10.36
137.226.34.227
151.207.243.129
193.1.193.64
198.189.255.73
198.189.255.74
198.189.255.82
208.111.128.122
208.111.129.48
65.54.95.201
65.54.95.209
65.54.95.7
```

Prvo izvlačimo relevantna polja iz datoteke `conn.log`, a to su `id.resp_h`, `service`, i `resp_bytes`. Ideja je filtriranje svih konekcija označenih s `HTTP` kod kojih server koji odgovara na zahtjev šalje više od 1000000 bajtova podataka. Uvjet filtriranja izvodi naredba `awk` koja će pročitati redak, izvesti analizu podataka datoteke `conn.log` i ispisati rezultat. Naredbom `sort -u` izbacuju se duplicirane IP adrese.

Postoje li web serveri na nestandardnim vratima (80, 8080)?

``` shell
$ bro-cut service id.resp_p id.resp_h < conn.log | awk '$1 == "http" && ! ($2 == 80 || $2 == 8080) { print $3 }' | sort -u
```

Ukoliko je izlaz prazan, Bro nije pronašao web servere na nestandardnim vratima.

Korisnici na mreži koji stvaraju najviše prometa:

``` shell
$ bro-cut id.orig_h orig_bytes < conn.log \
  | sort \
  | awk '{ if (host != $1) {
    if (size != 0)
    print $1, size;
    host=$1;
    size=0
    } else
    size += $2
    }
    END {
    if (size != 0)
    print $1, size
  }' \
  | sort -k 2 \
  | head -n 10

192.168.1.6 14100
192.168.1.7 2406
fe80::5dd:9f97:fe64:5cf2 2952
fe80::510e:a847:25fe:6cbd 3352
192.168.1.5 36410
fe80::5dd:9f97:fe64:5cf2 38913
fe80::2ccf:1f9a:246:fbea 660
```

Stranice prema kojima je poslan najveći broj zahtjeva:

``` shell
$ bro-cut host < http.log | sort | uniq -c | sort -n | tail -n 3
231 safebrowsing-cache.google.com
259 scores.espn.go.com
297 download.windowsupdate.com
```

!!! todo
    Dodaj nekakve primjere sadržaja datoteka koje se spominju.

Otvaranjem stranice konfiguratora usmjerivača Mikrotik koji se nalazi na adresi 192.168.13.254 i unosom korisničkih podataka, stvara se HTTP promet odnosno HTTP zahtjevi (npr. `GET`) i odgovori (npr. `200 OK`) koji stižu u toj komunikaciji, a uhvatio ih je Bro i zapisao u `http.log`. Iz niza zapisa u log datoteci, vidljivo je učitavanje početne stranice konfiguratora usmjerivača, odnosno datoteka koje čine tu stranicu.

Svakako je zanimljiv i sadržaj datoteke conn-summary.log koji prikazuje sažetak, odnosno informacije o porukama koje je Bro "uhvatio", agregirane po najznačajnijim kriterijima (npr. vratima, mrežama, dolaznom i odlaznom prometu i sl.). Iz loga je vidljivo da je Bro na mojoj mreži "uhvatio" najviše poruka DNS zahtjeva prema mom lokalnom DNS poslužitelju, koji se nalazi na usmjerivaču na adresi 192.168.13.254. Nakon toga je uhvaćeno dosta HTTP poruka, kao i SSH sesije koje sam pokrenula (`ssh -p 2222 stud530@example.group.miletic.net`) te poruke prema vratima koje koriste neke od aplikacija na ostalim računalima u mreži.

Datoteka `known_services.log` prikazuje koje mrežne servise koji od poslužitelja na mreži nudi. U mom primjeru, Bro je uhvatio da usmjerivač nudi SSH i HTTP servise, ovisno o mojim prethodno ostvarenim konekcijama.

Datoteka ssh.log prikazuje podatke o uhvaćenim SSH sesijama. Za probu sam se spojila SSH-om na usmjerivač (192.168.13.254) iz virtualnog stroja te (neuspješno) na poslužitelj example.group.miletic.net sa svog laptopa. Dobila sam informacije o klijentu i poslužitelju na koji se spajam (npr. da server example.group.miletic.net koristi Debian distribuciju Linuxa te da je veza odlazna). Spajanje na poslužitelj example.group.miletic.net je bilo neuspješno, što se vidi iz sadržaja datoteke -- vrijednost polja status je `failure`.

Datoteka weird.log sadrži sve što je Bro smatrao "čudnim", odnosno greške koje je uhvatio snimanjem mrežnog prometa. Na primjer, zapisao je da se za moje SSH konekcije koristi postojeća konekcija i to označio kao `active_connection_reuse`, dok je za paket s IP adrese 37.252.248.71 zapisao da je došlo do pojave neispravnog TCP-ovog kontrolnog zbroja (`bad_TCP_checksum`) odnosno da paket najvjerojatnije treba ponovno slati.

### Analiza prometa uhvaćenog alatom Wireshark

!!! todo
    Cross-linkaj na Wireshark.

Kao što je već spomenuto, Bro može analizirati promet uhvaćen alatom Wireshark odnosno čitati .pcap datoteke.
HTTP protokol sadrži vlastite mehanizme za autentifikaciju korisnika:

- Basic je jednostavan autentifikacijski mehanizam koji sa svakom porukom šalje korisničke podatke kao Base64 kriptiran niz unutar zaglavlja zahtjeva (podaci se prenose preko mreže kao običan tekst).
- Digest je "challenge-response" mehanizam koji koristi MD5.

[Unaprijed definirana HTTP skripta Bro-a](https://old.zeek.org/current/solutions/incident-response/index.html) provjerava postojanje Authorization zaglavlja u HTTP zahtjevima, dekodira zaglavlje Basic autentifikacije i kopira vjerodajnice u datoteci `http.log` na mjesto polja username i password. Slijedi analiza prometa uhvaćenog alatom Wireshark s ciljem otkrivanja web servera koji koristi slabu autentifikaciju.

``` shell
$ bro-cut id.orig_h id.resp_h username password < http.log | awk '$3 != "-"'
192.168.121.175 192.168.121.176 hbdairye4 -
192.168.121.175 192.168.121.176 hbdairye4 -
```

Web server na adresi 192.168.121.176 koristi slabu autentifikaciju (korisničko ime je `hbdairye4`). Bro ne kopira lozinke u datoteku `http.log` sve dok to ne omogućimo naredbom:

``` shell
$ bro -r illauth.pcap "HTTP::default_capture_password=T"
```

Ponovno pokrećemo naredbu za "izvlačenje" vjerodajnica:

``` shell
$ bro-cut id.orig_h id.resp_h username password < http.log | awk '$3 != "-"'
192.168.121.175 192.168.121.176 hbdairye4 cheesecake
192.168.121.175 192.168.121.176 hbdairye4 cheesecake
```

Korisničko ime i lozinka za spajanje na web server na adresi 192.168.121.176 su `hbdairye4` i `cheesecake`.
