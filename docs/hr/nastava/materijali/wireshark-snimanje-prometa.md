---
author: Domagoj Margan, Vedran Miletić
---

# Snimanje prometa aplikacija

Prilikom analize rada računalne mreže često se služimo alatima za hvatanje mrežnog prometa i analizu sadržaja paketa. Dva najpopularnija alata ove namjene su [tcpdump](https://www.tcpdump.org/) ([Wikipedia](https://en.wikipedia.org/wiki/Tcpdump)) i [Wireshark](https://www.wireshark.org/) ([Wikipedia](https://en.wikipedia.org/wiki/Wireshark)). Način rada s potonjim opisujemo u nastavku.

Wireshark je slobodni softver otvorenog koda za analizu paketa koji se često koristi za dijagnostiku problema u mrežama, analizu rada komunikacijskih protokola te učenje o računalnim mrežama i protokolima. Dostupan je za korištenje pod licencom [GNU GPLv2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html). Može se pokrenuti na Linuxu, BSD-ima, macOS-u, Solarisu i drugim operacijskim sustavima sličnim Unixu te na Windowsima. Osim grafičkog sučelja razvijenog korištenjem [Qt](https://www.qt.io/)-a, ima i sučelje naredbenog retka [TShark](https://www.wireshark.org/docs/man-pages/tshark.html) (naredba `tshark`, man stranica `tshark(1)`).

Razvoj Wiresharka započeo je Gerald Combs 1998. godine s ciljem razvoja softvera za analizu mrežnih protokola koji se mogu pokrenuti na Linuxu i Solarisu. [Prvotno se zvao Ethereal](https://www.wireshark.org/faq.html#_what_is_wireshark) i, iako se radilo o slobodnom softveru, njegovo je ime bilo pod zaštitnim znakom (engl. *trademark*). Kada je 2006. godine autor promijenio kompaniju u kojoj radi i koja sponzorira razvoj softvera, [ime je moralo biti promijenjeno pa je odabrano ime Wireshark](https://www.wireshark.org/faq.html#_whats_up_with_the_name_change_is_wireshark_a_fork).

Osim sa stvarnim čvorovima i mrežnim sučeljima, Wireshark radi i sa emuliranim. Unutar alata CORE dostupan je desnim klikom na čvor i odabirom opcije `Wireshark` te željenog mrežnog sučelja. Kako bi pritom lakše otkrili koje vam mrežno sučelje treba, u slučaju kad ih čvor ima više, možete pod `View/Show` uključiti `Interface Names`.

## Upoznavanje sa sučeljem Wiresharka

!!! note
    Studenti koji žele znati više, ili iz bilo kojeg razloga preferiraju video lekcije pred klasičnim tekstualnim, mogu materijale naći u [službenoj Wireshark dokumentaciji](https://www.wireshark.org/docs/) koja uključuje i nešto video materijala.

Nakon pokretanja Wiresharka, pojaviti će se grafičko sučelje s nizom korisničkih opcija i mogućnosti. Grafičko sučelje sastoji se od pet glavnih djelova:

- Izbornik naredbi (engl. *command menu*)
- Prozor za prikaz uhvaćenih paketa
- Prozor za prikaz detalja paketnih zaglavlja
- Prozor za prikaz sadržaja paketa
- Polje filtra kod prikaza (engl. *display filters*)

**Izbornik naredbi** je standardna alatna traka locirana na vrhu sučelja. Od posebne su važnosti za rad izbornici `File` i `Capture`. Izbornik `File` omogućava spremanje uhvaćenih podataka, kao i otvaranje prethodno uhvaćenih podataka s mreže. Također, u izborniku `File` nude se opcije za import i eksport paketa. Putem izbornika `Capture` možemo namjestiti bitne postavke za hvatanje mrežnog prometa, te pokrenuti i prekinuti hvatanje prometa.

**Prozor za prikaz uhvaćenih paketa** u jednoj liniji prikazuje sažetak svakog uhvaćenog paketa: redni broj paketa, vrijeme hvatanja paketa, izvorišnu i odredišnu adresu paketa te informacije o protokolu.

**Prozor za prikaz detalja paketnih zaglavlja** prikazuje detalje zaglavlja uhvaćenog paketa.

**Prozor za prikaz sadržaja paketa** prikazuje cjelokupni sadržaj uhvaćenog okvira, u ASCII i hexadecimalnom formatu.

**Polje filtra kod prikaza** je polje za unos određenog filtra za prikaz, na temelju kojeg se filtriraju informacije na prozoru za prikaz uhvaćenih paketa (te na prozoru za prikaz paketnih zaglavlja i sadržaja paketa).

## Početak i završetak hvatanja prometa

Odabir mrežnog sučelja na kojem će se vršiti hvatanje prometa radi se putem opcije `Interfaces...` u izborniku `Capture`. Hvatanje prometa može biti započeto odabirom opcije `Start` u izborniku `Capture` nakon odabira mrežnog sučelja.

Hvatanje prometa može, potpuno analogno, biti zaustavljeno odabirom opcije `Stop` u izborniku `Capture`.

Nakon zaustavljanja uhvaćeni promet moguće je spremiti u formatima [pcap](https://en.wikipedia.org/wiki/Pcap) (**p**acket **cap**ture), [pcapng](https://wiki.wireshark.org/Development/PcapNg) (**p**acket **cap**ture **n**ext **g**eneration) i [brojnim drugim](https://wiki.wireshark.org/FileFormatReference). Za osnovne potrebe spremanja uhvaćenog prometa formati pcap i pcapng su jednako dobri.

!!! note
    Format pcap je izvorno razvijen od strane autora tcpdumpa i podržan od strane Wiresharka, a pcapng ga proširuje s dodatnim informacijama o paketima (detaljne informacije o oba formata moguće je pronaći [u repozitoriju pcapng/pcapng na GitHubu](https://github.com/pcapng/pcapng)).

## Kopiranje podataka o uhvaćenih paketima

Osim spremanja uhvaćenih paketa u datoteku, moguće je prikaz paketa kopirati korištenjem opcije `Copy` u izborniku `Edit`:

- u obliku čistog teksta (`As Plain Text`):

    ```
    No. Time            Source              Destination         Protocol    Length  Info
    11  10.121072728    00:00:00_aa:00:00   Broadcast           ARP         42      Who has 10.0.0.1? Tell 10.0.0.20
    12  10.121098296    00:00:00_aa:00:01   00:00:00_aa:00:00   ARP         4       10.0.0.1 is at 00:00:00:aa:00:01
    13  10.121099859    10.0.0.20           10.0.1.10           ICMP        98      Echo (ping) request  id=0x1299, seq=1/256, ttl=64 (reply in 14)
    14  10.121171135    10.0.1.10           10.0.0.20           ICMP        98      Echo (ping) reply    id=0x1299, seq=1/256, ttl=63 (request in 13)
    ```

- u obliku [vrijednosti odvojenih zarezom](https://en.wikipedia.org/wiki/Comma-separated_values) (engl. *comma-separated values*) (`As CSV`):

    ``` csv
    "No.","Time","Source","Destination","Protocol","Length","Info"
    "11","10.121072728","00:00:00_aa:00:00","Broadcast","ARP","42","Who has 10.0.0.1? Tell 10.0.0.20"
    "12","10.121098296","00:00:00_aa:00:01","00:00:00_aa:00:00","ARP","42","10.0.0.1 is at 00:00:00:aa:00:01"
    "13","10.121099859","10.0.0.20","10.0.1.10","ICMP","98","Echo (ping) request  id=0x1299, seq=1/256, ttl=64 (reply in 14)"
    "14","10.121171135","10.0.1.10","10.0.0.20","ICMP","98","Echo (ping) reply    id=0x1299, seq=1/256, ttl=63 (request in 13)"
    ```

- u obliku za serijalizaciju podataka [YAML](https://yaml.org/) (`As YAML`):

    ``` yaml
    ----
    # Packet 10 from /tmp/wireshark_veth1.0.ddK7QIF1.pcapng
    - 11
    - 10.121072728
    - 00:00:00_aa:00:00
    - Broadcast
    - ARP
    - 42
    - Who has 10.0.0.1? Tell 10.0.0.20

    ----
    # Packet 11 from /tmp/wireshark_veth1.0.ddK7QIF1.pcapng
    - 12
    - 10.121098296
    - 00:00:00_aa:00:01
    - 00:00:00_aa:00:00
    - ARP
    - 42
    - 10.0.0.1 is at 00:00:00:aa:00:01

    ----
    # Packet 12 from /tmp/wireshark_veth1.0.ddK7QIF1.pcapng
    - 13
    - 10.121099859
    - 10.0.0.20
    - 10.0.1.10
    - ICMP
    - 98
    - Echo (ping) request  id=0x1299, seq=1/256, ttl=64 (reply in 14)

    ----
    # Packet 13 from /tmp/wireshark_veth1.0.ddK7QIF1.pcapng
    - 14
    - 10.121171135
    - 10.0.1.10
    - 10.0.0.20
    - ICMP
    - 98
    - Echo (ping) reply    id=0x1299, seq=1/256, ttl=63 (request in 13)
    ```

## Razmatranje strukture i sadržaja paketa

Najčešće korišteni protokol podatkovnog sloja na lokalnim mrežama je Ethernet. Svaki okvir koji je upućen u mrežu Ethernet dospijeva na svaku mrežnu karticu te mreže. U standardnom načinu rada, okvir preuzima (kopira) samo ona mrežna kartica na čiju je adresu taj okvir upućen. Eternet zaglavlja sadrže podatke potrebne za ispravanu komunikaciju lokalnom mrežom, tj adresu mrežne kartice odredišnog čvora i adresu mrežne kartice izvorišnog čvora.

TCP i IP paketi sastoje se od dva osnovna dijela: zaglavlja i tijela. Zaglavlje sadrži brojeve portova i druge upravljačke sadržaje, dok se u tijelo upisuje podatkovni sadržaj kojeg paket prenosi. S obzirom na to da je IP protokol niže razine od TCP-a, unutar IP paketa sadržan je cjelokupni TCP segment koji se također sastoji od zasebnih dijelova. Možemo reći da IP zaglavlje okružuje TCP segment (sa zaglavljem) unutar kojeg se nalaze podaci.

TCP i IP zaglavlja su uglavnom velika po 20 bajtova, no mogu imati i opcionalni dio varijabilne dužine. Primjerice, u IP paketu ukupne dužine 1500 bajtova, prenosi se TCP segment ukupne dužine 1480 bajtova, u kojem se prenosi 1460 bajtova korisnog tereta, odnosno sadržaja s aplikacijske razine.

TCP protokol promatra sadržaj kojeg prima od aplikacije kao niz bajtova te tvori TCP segmente iz tog niza bajtova; pritom svakom segmentu dodjeljuje kao slijedni broj onu vrijednost koja je jednaka poziciji (rednom broju) prvog bajta iz tog paketa u (većem) nizu bajtova koje proizvodi aplikacija i koje njena TCP veza prenosi na odredište. Svaki segment (uključujući i TCP zaglavlje) mora stati u 65535 bajtova IP paketa. Ako je segment prevelik, usmjerivač vrši fragmentaciju u više manjih segmenata od kojih svaki dobiva svoje IP zaglavlje.

Kako bi razumjeli strukturu Ethernet, IP i TCP zaglavlja, moramo poznavati polja koja postoje u njihovim zaglavljima i značenje vrijednosti koje ona sadrže.

### Struktura zaglavlja Ethernet, IP i TCP paketa

Ethernet zaglavlja sadrže tri bitna polja:

- **Adresa odredišta (Destination)** -- MAC adresa odredišta (mrežne kartice odredišta), može biti unicast, broadcast ili multicast adresa
- **Adresa izvora (Source)** -- MAC adresa izvorišta (mrežne kartice izvorišta)
- **Tip Ethernet paketa (Ethertype)** -- tip paketa koji je sadržan u Ethernet okviru

Značajna polja IP zaglavlja su:

- **Version** -- Verzija IP protokola, određuje format zaglavlja
- **Internet Header Length (IHL)** -- Duljina IP zaglavlja u 32-bitnim riječima, omogućava određivanje početka podataka
- **Type of Service** -- Tip usluge, omogućava usmjernicima različit tretman pojedinih paketa u cilju postizanja zadovoljavajuće kvalitete usluge (QoS), a s obzirom na dopušteno kašnjenje, količinu prometa i zahtijevanu pouzdanost
- **Total Length** -- Ukupna duljina IP paketa u oktetima, uključujući IP zaglavlje i podatke; najveća duljina paketa je 65 535 okteta (s obzirom na 16-bitno polje TL)
- **Identification** -- Identifikator paketa, važan je pri povezivanju svih fragmenata u paket
- **Flags** -- Kontrolne zastavice, definiraju je li fragmentacija dopuštena i ako jest, ima li još fragmenata istog paketa
- **Fragment Offset** -- Definira mjesto fragmenta u originalnom paketu, mjereno u jedinicama od 8 okteta (64 bita); odstupanje prvog fragmenta je nula
- **Time to Live (TTL)** -- Maksimalno vrijeme života paketa u mreži, nakon čega se neisporučeni paket odbacuje; mjeri se u sekundama, čvor koji obrađuje paket umanjuje vrijednost za najmanje 1, a ako je vrijednost nula paket se odbacuje
- **Protocol** -- Označava protokol više razine kojem se podaci prosljeđuju
- **Header Checksum** -- Kontrolni zbroj zaglavlja; ponovno se obračunava i provjerava pri svakoj promjeni podataka u zaglavlju
- **Source Address** -- IP adresa predajnika paketa
- **Destination Address** -- IP adresa prijemnika paketa
- **Options** -- Varijabilne duljine, opcionalno; sadrži kontrolne informacije o usmjeravanju, sigurnosne parametre itd.
- **Padding** -- Varijabilne duljine, dopuna polja opcija do 32 bita; popunjava se nulama

Značajna polja TCP zaglavlja su:

- **Source Port** -- Broj priključne točke usluge izvorišta
- **Destination Port** -- Broj priključne točke usluge odredišta
- **Sequence Number** -- Redni broj prvog okteta podataka u tom segmentu; ako je postavljena zastavica S (SYN), onda je to početni redni broj (ISN** -- Initial Sequence Number), a prvi oktet podataka ima broj ISN+1
- **Acknowledgment Number** -- Broj potvrde; ako je postavljen A (ACK) bit, polje sadrži redni broj sljedećeg okteta kojeg primatelj očekuje
- **Offset** -- Pomak podataka, pokazuje na početak podataka u TCP segmentu, izraženo u 32-bitnim riječima (TCP zaglavlje je uvijek višekratnik 32-bitne riječi)
- **Reserved** -- Polje je rezervirano za buduće potrebe; popunjeno je nulama
- **Kontrolni bitovi (zastavice)**

    - **URG** -- Indikator hitnih podataka
    - **ACK** -- Indikator paketa potvrde
    - **PSH** -- Inicira prosljeđivanje svih do tada neproslijeđenih podataka korisniku
    - **RST** -- Ponovna inicijalizacija veze
    - **SYN** -- Sinkronizacija rednih brojeva
    - **FIN** -- Izvorište više nema podataka za slanje

- **Window** -- Prozor, označava koliko je okteta prijemnik spreman primiti
- **Checksum** -- Kontrolni zbroj; računa se kao 16-bitni komplement jedinice komplementa zbroja svih 16-bitnih riječi u zaglavlju i podacima; pokriva i 96 bitova pseudozaglavlja koje sadrži izvorišnu i odredišnu adresu, protokol i duljinu TCP zaglavlja i podataka
- **Urgent Pointer** -- Pokazivač na redni broj okteta gdje se nalaze hitni podaci; polje se gleda jedino ako je postavljena zastavica URG
- **Options + Padding** -- Options mogu, a ne moraju biti uključene; ako postoje, veličine su x×8 bita, Padding je dopuna nulama do 32 bita
- **Data** -- Podaci aplikacijske razine

### Uvid u strukturu i sadržaj paketa korištenjem Wiresharka

Wireshark nam omogućuje direktan uvid u strukturu i sadržaj paketa. Odabirom paketa iz prozora za prikaz uhvaćenih paketa možemo vidjeti njegovu strukturu, tj. informacije o njegovim zaglavljima i korisnom teretu (tijelu).

U prozoru za prikaz detalja paketnih zaglavlja nalaze se svi podaci o zaglavljima odabranog paketa. Ovisno o tome koje zaglavlje paketa odaberemo za pregled, vidimo različite odgovarajuće informacije. Tako ćemo, primjerice, za odabrani TCP segment moći pregledati informacije o Ethernet, TCP i IP zaglavljima, kao i o cjelokupnom okviru paketa. Ukoliko je riječ o HTTP prijenosu podataka, imati ćemo uvid i u informacije o tom segmentu paketa (uz Ethernet, TCP i IP).

Klikom na neki od dijelova zaglavlja u prozoru za prikaz detalja zaglavlja interaktivno se *zatamnjuje* povezani dio paketa u prozoru za prikaz sadržaja paketa. Na taj način možemo vidjeti odnos zaglavlja sa samom strukturom čitavog paketa; vidimo *poziciju* svakog podatka zaglavlja. Isto vrijedi i obrnuto -- klikom na bilo koji dio sadržaja paketa *zatamnjuje* se i prikazuje odgovarajući dio zaglavlja. Pogledom na detalje zaglavlja paketa možemo utvrditi koliko u stvarnosti svako pojedino zaglavlje zauzima byteova te usporediti tu veličinu sa veličinom paketnog okvira i veličinom korisnog tereta.
