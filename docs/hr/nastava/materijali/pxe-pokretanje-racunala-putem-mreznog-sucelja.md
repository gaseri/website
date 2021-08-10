---
author: Nikola Barković, Vedran Miletić
---

# Pokretanje računala putem mrežnog sučelja

!!! note
    Sastavljeno prema [stranici PXEInstallServer](https://help.ubuntu.com/community/PXEInstallServer) s Ubuntuovog wikija.

Preboot eXecution Environment (PXE, [Wikipedia](https://en.wikipedia.org/wiki/Preboot_Execution_Environment)) je okruženje za pokretanje računala pomoću mrežnog sučelja neovisno od uređaja za pohranu podataka (poput tvrdog diska) ili instaliranim operacijskim sustavima.

PXE objavljen 1999. od strane Systemsofta i Intela, nastao je kao dio Wired for Management okvira (framework). Koristi nekoliko mrežnih protokola kao što su: IP, UDP, DHCP, TFTP i td.

## Lanac

Firmware na klijentu pokušava pronaći uslugu PXE preusmjeravanja na mreži (proxy DHCP) kako bi dobilo informacije o dostupnim PXE boot poslužiteljima. Nakon raščlanjivanja odgovora, firmware će tražiti odgovarajući boot server za put do datoteke od network bootstrap program (NBP), preuzima datoteku u memoriju računala s izravnim pristupom (RAM) pomoću TFTP-a i pokreće je. Ako se samo jedan NBP koristi među svim PXE klijentima to bi moglo biti specificirano korištenjem BOOTP-a bez potrebe za proxy DHCP, ali TFTP poslužitelj je još uvijek potreban.

## Protokol

PXE protokol je neka vrsta kombinacije između DHCP-a i TFTP-a s malim modifikacijama. DHCP se koristi kako bi se locirao boot poslužitelj, a TFTP se koristi za skidanje (download) početnog bootstrap programa i dodatnih datoteka.

Da bi se inicijalizirao PXE bootstra sesija PXE firmware šalje DHCP otkrića paket s dodatnim PXE opcijama na UDP port 67. PXE opcije prikazuju da je firmware sposoban za PXE, a standardni DHCP serveri će ih ignorirat.

## Proxy DHCP

Ako PXE servis za preusmjeravanje prima proširenu poruku DHCP otkrića, odgovara sa produljenom porukom DHCP ponude klijentu na UDP port 68. Proširena poruka DHCP ponude se sastoji od:

- PXE Discovery Control polja koji preporučava način komunikacije sa PXE boot serverom (multicast, broadcast ili unicast).
- Liste IP adresa za svaki mogući PXE Boot Server tip.
- PXE Boot meni gdje svaki unos predstavlja PXE Boot Server tip.
- PXE Boot Prompt koji govori korisniku koju tipku treba stisniut kako bi se vidio boot meni.
- Odbrojavanje do pokretanja prvog zapisa u boot meniju.

Proxy DHCP servis može bit raditi na istom posluživaču kao i standardni DHCP servis. Budući da ne mogu oba koristit UDP port 67, Proxy DHCP onda koristi UDP port 4011 i očekuje da proširena poruka DHCP otkrića od strane PXE klijena bude DHCP zahtjev. Standardni DHCP servis mora poslati određenu kombinaciju PXE opcija u DHCP ponudi kako bi PXE klijent znao da se Proxy DHCP nalazi na istom poslužitelju, ali na UDP portu 4011.

## Boot server contact

Da bi kontaktirao PXE Boot Server sistem koji zatražuje pokretanje moram imati IP adresu. Nakon toga šalje poruku DHCP zahtjeva proširenu s PXE opcijama na UDP port 4011 ili 67. Ovaj paket sadrži PXE Boot Server tip i PXE Boot Layer, koji omogućuje pokretanje više Boot Servera s jednog DEAMON-a.

Server odgovara s porukom DHCP prihvat ponude koja sadrži:

- Kompletan put do datoteke koja se treba skinuti sa TFTP-om.
- PXE Boot Server tip i PXE Boot Layer.
- TFTP konfiguraciju.

Od verzije 2.1 PXE Boot Server podržava provjeru integriteta datoteke skinute pomoći TFTP-a koristeći checksum datoteku.

Nakon primanja poruke DHCP prihvat ponude, NBP se sprema u RAM nakon čega se verificira ako je potrebno i pokreće.

## Integracija

PXE Klijent/Server Protokol je napravljen kako bi mogao:

- Koristiti se u istoj mreži u kojoj već postoji DHCP okolina bez sučelja.
- Integrirati se u postojeće standardne DHCP serivisa.
- Lako se nadogradit na najvažnijim djelovima.
- Svaka usluga (DHCP, Proxy DHCP, Boot Sever) se može samostalno inplementirati ili u kombinacijama.

## PXE Install server

PXE konfiguraciju ću prikazat na praktičnom primjeru koristeći Ubuntu 12.04 server instaliran kao virtualna mašina te klijent računalo također kao virtualna mašina unutar interne mreže pomoću VirtualBox programa. Iako je ovaj postupak prikazan na radu između dvije virtualne mašine isproban je i radi i na dva različita računala.

### Instalacija paketa

Pokrenuti terminal te upisati sljedeće naredbe kako bi instaliri potrebne pakete.

```
sudo apt-get install isc-dhcp-server
sudo apt-get install tftpd-hpa
sudo apt-get install apache2
```

isc-dhcp-server je DHCP poslužiteljski paket. DHCP protokol je detaljno opisan u poglavlju 2.

tftpd-hpa je TFTP poslužiteljski paket. Uloga TFTP protokola je preuzimanje PXE datoteke sa servera.

apache2 peket u ovom slučaju se koristi za instalaciju OS-a na klijent računalu.

### Konfiguracija poslužitelja DHCP-a

Da bi naš Ubuntu server radio kao DHCP server potrebno je konfigurirati DHCP protokol te pokrenuti servis. Konfiguracijska datoteka je `/etc/dhcp/dhcpd.conf`. U datoteci već postoje određeni zapisi koji su uglavnom komentari, možete ih ili ostavit i dodat sljedeće ili samo dodat sljedeće:

```
allow bootp;
allow booting;
default-lease-time 600;
max-lease-time 7200;
subnet 192.168.0.0 netmask 255.255.255.0 {
        range 192.168.0.100 192.168.0.200;
        filename "pxelinux.0";
}
```

S ovim omogućujemo da DHCP dodjeljuje adrese s podmreže 192.168.0.0 i s net maskom 255.255.255.0, skup adresa za dodjelu je od 192.168.0.100 do 192.168.0.200, te vrijeme dodjele adrese koje je početno 600 ms te maximalno 7200 ms. Filename datoteka nam govori koju će datoteku klijent treba preuzeti nakon što dobije IP adresu.

Također potrebno je podesiti koji mrežni uređaj će se koristit za primanje podataka za DHCP server, u datoteci `/etc/default/isc-dhcp-server` potrebno je upisati mrežnu komponentu u INTERFACE rubriku:

```
INTERFACE="eth0"
```

Da bi servisi radili s novim postavkama potrebno je napraviti restart servisa.

``` shell
# service isc-dhcp-server restart
*isc-dhcp-server start/running, process 7231
# service dhcpd-hpa restart
*dhcpd-hpa start/running, process 7242
# service apache2 restart
*[OK]
```

### Priprema datoteka za instalaciju

Budući da koristimo virtualnu mašinu moguće je u operativnom sustavu u kojem je pokrenut VirtualBox napravit mount .iso datoteke te u virtualnoj mašini to prikazat kao cd. U slučaju da želimo instalirati sustav tako da se direktno skine s interneta, ovaj korak nije potreban.

1. Prvi korak: kopiranje s cd-a

```
cp -r /media/"Ubuntu 12.04.1 LTS i386"/* /var/www/ubuntu/
```

1. Drugi korak: priprema Packet.gz datoteke

```
cd /var/www/ubuntu/dists/precise/restricted/binary-i386/
gunzip -c Packages.gz > Package
```

### Netboot

Da bi sustav znao koje datoteke mora instalirati i gdje se nalaze potrebno je pripremiti datoteke koje će klijent preuzeti prilikom pokretanja s TFTP protokolom.

```
cp -r /var/www/ubuntu/install/netboot/* /var/lib/tftpboot/
```

U slučaju da instaliramo sustav s interneta potrebno je s interneta preuzet `netboot.tar.gz` datoteku i staviti u `/var/lib/tftpboot/` folder i nakon toga je raspakirati.

### Kickstart

Ubuntu PXE instalacija moguća je pomoću datoteka sa server računala ili preuzimanjem s interneta. U ovom slučaju radimo instalaciju pomoću datoteka sa server računala. Pa je potrebno urediti kickstart datoteku koja će uputiti instalaciju na lokaciju na serveru. Za instalaciju prilikom koje se sustav instalira s interneta ovaj korak se može preskočit. U datoteku `/var/www/ks.cfg` je potrebno dodati:

```
install
url --url http://192.168.0.1/ubuntu
```

### Priprema konfiguracije instalacije

Da bi sustav uputili na lokalnu instalaciju također je potrebno PXE konfiguraciju podesit tako da je traži `ks.cfg` datoteku prilikom pokretanja. Ova datoteka nam omogućuje da kad u izborniku za instalaciju odaberemo Install sustav pokreće instalaciju sa server računala.

U datoteci `/var/lib/tftpboot/ubuntu-installer/i386/boot-screens/txt.cfg` protrebno je promjeniti append red:

```
default install
label install
        menu label ^Install
        menu default
        kernel ubuntu-installer/i386/linux
        append ks=http://192.168.0.1/ks.cfg vga=788 initrd=ubuntu-
installer/i386/initrd.gz -- quiet
label cli
        menu label ^Command-line install
        kernel ubuntu-installer/i386/linux
        append tasks=standard pkgsel/language-pack-patterns=
pkgsel/install-language-support=false vga=788 initrd=ubuntu-
installer/i386/initrd.gz -- quiet
```

Ponekad treba promijeniti datoteku `/var/lib/tftpboot/pxelinux.cfg/default`.
