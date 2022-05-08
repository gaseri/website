---
author: Ivan Hrastinski, Vedran Miletić
---

# OpenWrt

[OpenWrt](https://openwrt.org/) je distribucija GNU/Linuxa za uređaje kao što su kućni ruteri. Ova distribucija ima veliku mogućnost proširivanja. Ovaj operacijski sustav je građen od temelja s ciljem jednostavne modifikacije i dostupnosti svih potrebnih mogućnosti. U praksi to znači da možemo imati sva potrebne mogućnosti bez nepotrebnog softvera.

Umjesto pokušaja stvaranja jedinog i statičnog firmware-a, OpenWrt omogućuje datotečni sustav s mogućnošću zapisivanja i upotrebu upravitelja paketa koji nam omogućuje da instaliramo na uređaj pakete koje želimo. To nas oslobađa od restrikcija korištenja odabranih aplikacija i konfiguracija koje je uvjetovao proizvođač, odnosno možemo uređaj prilagoditi kako nam u kojoj situaciji odgovara.

Za programere OpenWrt je framework koji omogućava razvoj aplikacija bez potrebe za stvaranjem cijelog firmware-a i okoline. Korisnicima OpenWrt omogućuje potpunu slobodu prilagođavanja, korisnik može podesiti uređaj i koristiti na način koji proizvođač nije nikad predvidio.

OpenWrt može običan kućni ruter pretvariti u uređaj raznolikih mogućnosti. Ovisno o ruteru neke od tih mogućnosti su: dsl modem, usb podrška, podrška za 3G ili 4G usb modeme, torrent klijent, server za pisač, apache server itd. Konfiguraciju svih tih mogućnosti možemo pronaći u dokumentaciji, a neke ćemo pokazati u nastavku. OpenWrt nije zamišljen kao distribucija od koje možemo očekivati da kad ju instaliramo na ruter da će sve odmah raditi, umjesto toga potrebno je malo proučiti dokumentaciju i sami si podesiti uređaj prema svojim potrebama.

## Značajke

### Besplatan i open source

Ovaj projekt je u potpunosti besplatan i opensource, licenciran pod licencom GNU General Public Licence (GPL). Projekt nastoji biti stalno dostupan preko dostupne web stranice s potpunim izvornim kodom.

### Jednostavan i besplatan pristup

Projekt će uvijek biti dostupan za nove suradnike i ima niski prag za sudjelovanje. Svako ima mogućnost doprinosa projektu. Trenutni programeri će dati svakome pravo pisanja ako se netko zanima za projekt i odgovorno se ponaša.

### Vođen zajednicom

Svi programeri dolaze zajedno raditi i surađivati kako bi se postigao cilj.

## Povijest

Projekt OpenWrt je počeo u siječnju 2004. godine. Prve verzije OpenWrta su se temeljile na Linksysovom izvornom kodu za WRT 54G izdanom pod GPL-om i buildroot-u iz uclibc projekta. Ova verzija je bila poznata kao OpenWrt "stable release" i bila je široko primijenjena. Još uvijek postoje aplikacije koje se temelje na ovoj verziji OpenWrt-a, neke od tih aplikacija su [Freifunk-Firmware](https://wiki.freifunk.net/Freifunk-Firmware) i Sip@Home. U početku 2005. godine timu su se pridružili novi programeri. Nakon nekoliko mjeseci zatvorenog razvoja, tim je odlučio izdati prve eksperimentalne verzije.

Eksperimentalna verzija koristi prilagođeni sistemski build koji se temelji na buildroot2 iz uclibc projekta. OpenWrt koristi službeni GNU/Linux kernel i dodaje zakrpe za sustave na čipu i driver-e za mrežna sučelja.

| Naziv verzije | Broj verzije | Datum izdavanja |
| ------------- | ------------ | --------------- |
| n/a | 18.06 | Srpanj 2018. |
| Reboot | 17.01 | Veljača 2017. |
| Chaos Calmer | 15.05 | Rujan 2015. |
| Barrier Breaker | 14.07 | Listopad 2014. |
| Attitude Adjustment | 12.09 | Travanj 2013. |
| BackFire | 10.03.1 | Prosinac 2011. |
| BackFire | 10.03 | Travanj 2010. |
| Kamikaze | 8.09.2 | Siječanj 2010. |
| Kamikaze | 8.09.1 | Lipanj 2009. |
| Kamikaze | 8.09 | Rujan 2008. |
| Kamikaze | 7.09 | Rujan 2007. |
| Kamikaze | 7.07 | Srpanj 2007. |
| Kamikaze | 7.06 | Lipanj 2007. |
| White Russian | 0.9 | Siječanj 2007. |

Verzija 17.01 (Reboot) izdana je u sklopu projekta [LEDE](https://openwrt.org/about).

## Ciljevi

- OpenWrt nikada neće biti proizvod, ali je nešto što omogućuje jednostavnu izradu proizvoda
- OpenWrt nikada neće biti specifičan, nego će uvijek ostati generički
- OpenWrt nikada neće biti završen, ali uvijek će pratiti napredak u tehnologiji
- Wireless freedom

## Projekti temeljeni na OpenWrtu

- Gargoyle -- pomno prati razvoj OpenWrt-a, posjeduje vlastito web sučelje pomoću kojeg je moguće detaljno konfigurirati napredne mogućnosti.
- DD-WRT -- u prošlosti je upotrijebljen OpenWrt kernel.
- CoovaAP -- firmware za bežične pristupne točke, temeljen na CoovaChilli.
- FON -- Wifi pristupne točke dostupne kupcima uređaja od određenog proizvođača.
- ROOter -- besplatan firmware temeljen na OpenWrt-u koji pretvara klasične rutere s USB portovima u 3G/4G/LTE modeme/rutere.
- Doodle3D -- omogućuje jednostavnije 3D ispisivanje.
- [Turris OS](https://www.turris.com/en/turris-os/) -- fork (kopija repozitorija) OpenWrt-a prilagođenog za [ruter Turris Omnia](https://www.turris.com/en/omnia/overview/).

Turris je neprofitni istraživački projekt upravitelja Češke domene `.cz`, [CZ.NIC](https://www.nic.cz/).

## Upravitelj paketa

Jedna od važnijih komponenta OpenWrt-a je upravitelj paketa (engl. *package manager*) `opkg` pomoću kojeg preuzimamo i instaliramo OpenWrt pakete iz lokalnih repozitorija ili onih koji su smješteni na internetu. Uređaji na koje se instalira OpenWrt su slabih hardverskih mogućnosti, zato je mogućnost instaliranja paketa koji su nam potrebni velika prednost. Opkg nam isto tako omogućuje dodavanje upravljačkih programa i kernel modula. Korisnici Linux-a su upoznati s alatima apt-get, aptitude, pacman, yum i sličnima, primijetiti će sličnost s opkg-om. Opkg je ponekad nazivan Entware, zbog toga jer se uglavnom referira na Entware repozitorij.

Naredbe za manipulaciju paketima: `update`, `upgrade <pkgs>`, `install <pkgs|FQDN>`, `configure <pkgs>`, `remove <pkgs|globp>` i `flag <flag> <pkgs>`.

- `update` -- ažurira listu dostupnih paketa.
- `upgrade <pkgs>` -- ažurira određeni paket ili grupu paketa
- `install <pkgs>` -- instalira paket, paket se može instalirati na 3 načina:

    - ime: `opkg install hiawatha`
    - URL: `opkg install http://downloads.openwrt.org/snapshots/trunk/ar71xx/packages/hiawatha_7.7-2_ar71xx.ipk`
    - putanja datoteke: `opkg install /tmp/hiawatha 7.7.-2 ar71xx.ipk`

- `configure <pkgs>` -- konfiguracija neotpakiranih paketa
- `remove <pkgs|globp>` -- uklanjanje paketa
- `flag <flag> <pkgs>` -- označavanje paketa ili grupe paketa, moguće oznake su: `hold`, `noprune`, `user`, `ok`, `installed`, `unpacked`

Naredbe za informacije o paketima su: `list`, `list-installed`, `list-upgradable`, `list-changed-conffiles`, `files`, `search`, `info`, `status`, `download`, `compare-versions`, `print-architecture`, `whatdepends`, `whatdependsrec`, `whatprovides`, `whatconflicts`, `whatreplaces`.

## Web sučelje LuCi

[LuCi](https://openwrt.org/docs/guide-user/luci/start) je web sučelje za upravljanje i podešavanje OpenWrta. Pristupamo mu putem web preglednika tako da upišemo IP adresu na kojoj se nalazi. Web korisničko sučelje omogućava lakše korištenje korisnicima kojima nije poznato komadno sučelje. LuCi je osnovan u ožujku 2008. godine kao FFLuCI, koji je bio dio nastojanja za stvaranje port-a Freifunk-Firmware iz OpenWrt branch-a WhiteRussian na njegovog nasljednika Kamikaze branch-a. Glavni razlog pokretanja ovog projekta je nedostatak besplatnog, čistog, proširivog web korisničkog sučelja koje se može jednostavno održavati za kućne rutere. Dok većina sličnih sučelja su teška za korištenje zbog Shell-scripting jezika, LuCi koristi Lua programski jezik i dijeli sučelje na logičke dijelove kao što su modeli i pogledi koristeći objektno orijentirane predloške. Sve to osigurava bolje performanse, smanjenje veličine instalacije i jednostavno održavanje. LuCi je postao službeni dio OpenWrt-a od izdanja Kamikaze 8.09.

## Dokumentacija

Dokumentacija vrlo detaljna te postoji dio za korisnike i dio za programere. Sama struktura dokumentacije je podijeljena u nekoliko glavnih cjelina. Cjeline su: O OpenWrt-u, hardver, često postavljena pitanja, početnički vodič, vodič za instalaciju, vodič za konfiguraciju, vodiči i kako uraditi, stvaranje OpenWrt-a, razvijanje i zakrpe. O OpenWrt-u se nalaze informacije o projektu, povijesti, vijestima i novostima o izdavanju verzija OpenWrt-a. U cjelini vodiči i kako uraditi su objašnjeni postupci kako instalirati i podesiti ruter da bi dobili željenu mogućnost npr. web poslužitelj Apache.

## Instalacija

Instalacija OpenWrta ovisi o proizvođaču i modelu rutera na koji želimo instalirati. Popis uređaja na koje je moguće instalirati OpenWrt možemo naći na službenim stranicama projekta pod naslovom [Supported Devices](https://openwrt.org/supported_devices).

### Siemens Gigaset SX763

Uređaj na koji sam instalirao OpenWrt je Siemens Gigaset SX763. Specifikacije uređaja se nalaze u tablici koja slijedi.

| Ključ | Vrijednost |
| ----- | ---------- |
| Vrsta uređaja | Modem |
| Proizvođač | Siemens |
| Brend | Gigaset |
| Model | SX763 |
| Platforma | Lantiq XWAY Danube |
| Brzina procesora | 333 MHz |
| Flash memorija | 8 MB |
| Radna memorija | 32 MB |
| LAN priključci | 4 |
| VLAN | n/a |
| WLAN hardver | Atheros AR2414A (na matičnoj ploči) |
| WLAN 2.4 GHz | b/g |
| WLAN 5 GHz | n/a |
| USB priključci 2.0 | 1 |
| USB priključci 3.0 | n/a |
| SATA priključci | n/a |

Nastavak teksta slijedi [upute s Bugovog foruma](https://forum.bug.hr/forum/topic/ostale-mrezne-teme/openwrt-dd-wrt-firmware-tutorial-gigaset-sx76/136525.aspx) i [upute s PC Ekspert Foruma](https://forum.pcekspert.com/showthread.php?t=227002).

### Postupak instalacije

Tvornički firmware koristi dva bootloader-a koji preko checksuma provjeravaju dali ih je netko promijenio. Prvi korak u postupku instalacije je zamjena drugog bootloader-a.

Ubacuje se modificirani drugi bootlader koji omogućuje instalaciju OpenWrt-a. Modificirani bootloader ubacujemo tako da u tvorničkom web sučelju otiđemo do sekcije nadogradnja firmware-a, te ga tu ubacimo kao i svaki drugi firmware.

Nakon što smo ubacili modificirani bootloader trebamo si podesiti mrežne postavke na računalu s kojim pristupamo ruteru. Mrežne postavke su:

- IP adresa: 192.168.1.16
- subnet maska: 255.255.255.0
- default gateway: 192.168.1.1
- DNS server: 192.168.1.1

Nakon podešavanja mrežnih postavki i nakon što se ruter resetira, automatski će se upaliti web server na adresi 192.168.1.1 pomoću kojeg ubacujemo željeni OpenWrt firmware u ruter.

Nakon što smo ubacili željeni firmware treba proći nekoliko minuta da se ruter konfigurira. Nakon toga možemo pristupiti web korisničkom sučelju LuCi ili se možemo spojiti ssh-om, te dalje upravljati i podešavati ruter.
