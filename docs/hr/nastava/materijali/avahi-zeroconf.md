---
author: Igor Kos, Vedran Miletić
---

# Zerconf NSS/mDNS sustav Avahi

[Avahi](https://www.avahi.org/) je slobodni softver otvorenog koda licenciran pod GNU LGPL koji implementira [Zero Configuration Networking (Zeroconf)](http://www.zeroconf.org/). Zeroconf uključuje sustav za otkrivanje multicast DNS i DNS-SD uslugu otkrivanja.

Avahi je sustav koji omogućuje programima da objave i pretraže usluge i domaćine pokrenute unutar lokalne mreže. Avahi je nastao kao zamjena za Bonjour (Apple zeroconf sustav) jer je on koristio GPL nekompatibilnu licencu. Apple je promijenio te dijelove licence no Avahi je već postao standardna implementacija za mDNS/DNS-SD na besplatnim operativnim sustavima kao što su GNU/Linux.

Avahi performanse su nalik onima koje nudi [Bonjour](https://developer.apple.com/bonjour/), u nekim slučajevima ga i nadmašuje.

Općenito zero-konfiguracija služi za lakše, brže i jednostavnije postavljanje računalne mreže bazirane na TCP/IP protokolima. Zeroconf automatski postavlja sve parametre mreže i servise kako bi korisniku olakšao korištenje mreže i svih ponuđenih usluga na mreži, odnosno kako sam korisnika ne bih morao ručno postavljati mrežu koja na kraju možda i ne bi funkcionirala.

Avahi mDNS responder danas ima kompletnu provedbu svih "MORA" i većinu "TREBALO BI" od mDNS/DNS-SD RFC-a. Prolazi sve testove u Apple Bonjour testu sukladnosti. Kao dodatak podržava točnu mDNS reflekciju preko cijelog LAN segmenta. mDNS responder je pisan i implementiran u C-u zbog čega ga je moguće ubaciti u mnoge druge aplikacije.

Značajke Avahija:

- Licenciran pod LGPL
- Podrška za IPv4 i IPv6
- D-Bus sučelje
- Odbacuje sve privilegije i radi kao korisnik "avahi"
- Chroot() podrška
- Ugradivi mDNS stop (npr. mDNS stog dostupan kao biblioteka)
- Podrška za učitavanje statičke definicije usluga iz XML fragmenata
- Sučelje za GLIBC NSS koristeći nss-mdns
- Sposobnost da odražava mDNS promet između više podmreža
- Sposobnost konfiguriranja unicast DNS poslužitelja automatski od poslužitelja podataka objavljenih na LAN
- Širokopojasna podrška DNS-SD
- Kompatibilnost biblioteka sa [HOWL](https://0pointer.de/blog/projects/howl.html) i Apple Bonjour

Avahi je osmišljen da bude vrlo jednostavan za korištenje u postavljanju i upravljanju mrežama. Vrlo lako se kreiraju novi serveri, a još lakše se pronalaze postojeći unutar mreže.

## Instalacija

Avahi je dio većine poznatih distribucija GNU/Linuxa, štoviše Avahi je već instaliran na većini desktop distribucija te nije potrebna posebna instalacija.

Avahi se pokreće prilikom podizanja Linuxa, da bi se uvjerili u to upisujemo u treminal:

``` shell
$ sudo update-rc.d avahi-daemon defaults
```

## Oglašavanje usluge

Dalje, kreiramo datoteku `/etc/avahi/services/afpd.service` koje će specificirati informacije o AppleShare serveru pokrenutom na našem sustavu te kopiramo sljedeći XML kod u tu datoteku:

``` xml
<?xml version="1.0" standalone='no'?><!--*-nxml-*-->
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
<name replace-wildcards="yes">%h</name>
<service>
<type>_afpovertcp._tcp</type>
<port>548</port>
</service>
</service-group>
```

Nakon svega, ponovno pokrenemo Avahi:

``` shell
$ sudo /etc/init.d/avahi-daemon restart
```

Nakon svih ovih koraka, prilikom pokretanja aplikacije Avahi Zeroconf browser (naredba `avahi-discover`) instalirane unutar sustava, dobijemo sljedeće.

Vidimo da je naš AppleShare server aktivan na našem sustavu. Vidimo sve nama zanimljive i potrebne podatke o tom serveru.

- Tip servisa: `_afpovertcp._tcp`
- Ime servisa: `mymachine`
- Domena: `local`
- Sučelje: `eth0` IPv4
- Adresa: `mymachine.local/192.168.142.128:548`
- TXT dana: prazno

Također postoje i još dvije verzije avahi-browsera:

- Avahi SSH server browser, naredba `bssh`
- Avahi VNC server browser, naredba `bvnc`

Iz navedenog primjera možemo vidjeti kako vrlo jednostavno funkcionira Avahi i kako s lakoćom pronalazi sve DNS pristupe na mreži na kojoj se nalazimo i mi.

## Dijeljenje podataka

Također možemo postaviti i dijeljenje podataka putem Avahi-a. Ako koristimo Network File System (NFS) postavke, možemo koristiti Avahi za automatsko pokretanje unutar preglednika datoteka Konqueror.

Kreiramo novu `.service` datoteku unutar mape `/etc/avahi/services`, primjerice imena `/etc/avahi/services/nfs_Zephyrus_Music.service`, te u nju upišemo sljedeće:

``` xml
<?xml version="1.0" standalone='no'?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
  <name replace-wildcards="yes">NFS Music Share on %h</name>
  <service>
    <type>_nfs._tcp</type>
    <port>2049</port>
    <txt-record>path=/data/shared/Music</txt-record>
  </service>
</service-group>
```

Dijeljenje datoteka i podataka putem Sambe je vrlo jednostavno ako je Avahi pokrenut i na serveru i na klijentu jer menadžer datoteka na klijentu automatski prepoznaje i pronalazi server.
