---
author: Domagoj Margan, Vedran Miletić
---

# Osnovni alati za konfiguraciju računalne mreže

## Alat ifconfig

`ifconfig` je moćan alat za konfiguriranje mrežnih sučelja, te za dobivanje informacija o istima. Pokreće se u ljusci, a osim sa stvarnim mrežnim sučeljima ifconfig radi i sa emuliranima.

Ukoliko ga koristimo bez argumenata, ifconfig ispisuje informacije o trenutno aktivnim mrežnim sučeljima, baš kao i `netstat -ie`. Ako želimo ispis informacija o točno određenom mrežnom sučelju, kao argument navodimo naziv tog sučelja. Primjerice, prikaz informacija za mrežno sučelje `wlan0` (bežična mrežna kartica) radimo na način:

``` shell
# ifconfig wlan0
wlan0     Link encap:Ethernet  HWaddr 00:24:d6:0b:61:26
          inet addr:192.168.1.2  Bcast:192.168.1.255  Mask:255.255.255.0
          inet6 addr: fe80::224:d6ff:fe0b:6126/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:1101738 errors:0 dropped:0 overruns:0 frame:0
          TX packets:1024191 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000
          RX bytes:407548170 (388.6 MiB)  TX bytes:146331082 (139.5 MiB)
```

Ifconfig se koristi za manipulaciju mrežnim sučeljima možemo određeno sučelje omogućiti ili onemogućiti za rad, te možemo određenom sučelju dodijeliti željenu IP adresu.

!!! note
    Alat `ifconfig` zahtijeva administratorske privilegije za manipuliranje postavkama mrežnih sučelja, odnosno ta funkcionalnost dostupna je samo `root` korisniku.

Ukoliko želimo neaktivno sučelje omogućiti za rad (staviti sučelje u aktivno stanje), navodimo naziv sučelja iza kojeg slijedi `up`:

``` shell
# ifconfig eth0 up
```

Prisjetimo se da se sučelje `eth0` odnosi na Ethernet karticu. Analogno tome, aktivo sučelje možemo staviti u neaktivno stanje, navođenjem argumenta `down`:

``` shell
# ifconfig eth0 down
```

Sučeljima možemo dodjeljivati IP adrese, navođenjem naziva sučelja, te željene adrese. Primjerice, sučelju wlan0 dodjeljujemo adresu 192.168.1.10 naredbom

``` shell
# ifconfig wlan0 192.168.1.10
```

Također, možemo mijenjati i maske podmreže, navođenjem naziva sučelja, argumenta `netmask` i željene maske. Primjerice, sučelju eth0 stavljamo masku 255.255.255.0 naredbom

``` shell
# ifconfig eth0 netmask 255.255.255.0
```

Svakom sučelju možemo argumentom `mtu` odrediti i MTU (ukoliko postavimo valjanu vrijednost). Za ethernet sučelja vrijednost je 1500; želimo li recimo, sučelju eth0 promijeniti vrijednost MTU na 1000, to ćemo učiniti naredbom

``` shell
# ifconfig eth0 mtu 1000
```

Kada sučelje (mrežna kartica) primi paket, provjerava je taj paket namjenjen namjenjen njemu. Ukoliko nije, sučelje odbacuje paket. Argumentom `promisc` moguće je postaviti sučelje u 'promiskuitetan' način rada, tj. odrediti da sučelje prima sve pakete bez obzira jesu li paketi namjenjeni njemu.

``` shell
# ifconfig eth0 promisc
```

Za vraćanje sućelja u normalan način rada, koristimo `-promisc`:

``` shell
# ifconfig eth0 -promisc
```

Pregled parametara za alat ifconfig:

- `up` : postavljanje mrežnog sučelja u aktivan način za rad
- `down` : deaktiviranje mrežnog sučelja
- `netmask` : određivanje maske podmreže
- `mtu` : određivanje MTU vrijednosti
- `promisc` : postavljanje sučelja u promiskuitetan način rada
- `-promisc` : vraćanje sučelja u normalan način rada


## Dodatak: alat ip

!!! todo
    Ovaj dio još nije napisan.

## Dodatak: alat iwconfig

!!! todo
    Ovaj dio još nije napisan.

## Dodatak: alat iw

!!! todo
    Ovaj dio još nije napisan.
