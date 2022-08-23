---
author: Luka Ljubojević, Vedran Miletić
---

# Konfiguracija virtualne privatne mreže alatom WireGuard

[Virtualna privatna mreža](https://en.wikipedia.org/wiki/Virtual_private_network) (engl. *virtual private netvork*, kraće VPN) pruža usluge privatne mreže korištenjem neke javne mreže kao što je internet. Pomoću VPN-a možemo slati i primati podatke preko javne mreže, a istovremeno koristiti konfiguraciju privatne lokalne mreže. VPN koristi različite sigurnosne mehanizme koji osiguravaju tajnost i autentičnost podaka koje šaljemo. Virtualnu privatnu mrežu moguće je stvoriti brojnim protokolima, npr. [L2TP](https://en.wikipedia.org/wiki/Layer_2_Tunneling_Protocol) na veznom sloju (koji implementira [xl2tpd](https://github.com/xelerance/xl2tpd)), [IPsec](https://en.wikipedia.org/wiki/IPsec) na mrežnom sloju (koji implementira [strongSwan](https://www.strongswan.org/)) i [SSTP](https://en.wikipedia.org/wiki/Secure_Socket_Tunneling_Protocol) na transportnom sloju (koji implementira [SoftEther VPN](https://www.softether.org/)).

[WireGuard](https://www.wireguard.com/) je vrlo jednostavan, ali moderan VPN poslužitelj i klijent koji cilja biti brži, jednostavniji, pouzdaniji i korisniji od IPsec-a, a također je i ozbiljna konkurencija [OpenVPN](https://openvpn.net/community/)-u. Inicijalno je izdan za operacijske sustave zasnovane na jezgri Linux, no sada je postao višeplatformsko rješenje za uspostavu VPN mreže (podržava Windowse, macOS, BSD-e, iOS i Android). Razvoj WireGuarda još uvijek je u toku, iako se već sada smatra jednim od najsigurnijih dostupnih VPN rješenja. WireGuard koristi suvremene kriptografske tehnologije poput:

- [Noise protocol framework](https://noiseprotocol.org/),
- [Curve25519](https://en.wikipedia.org/wiki/Curve25519),
- [ChaCha20](https://www.cryptopp.com/wiki/ChaCha20),
- [Poly1305](https://en.wikipedia.org/wiki/Poly1305),
- [BLAKE2](https://en.wikipedia.org/wiki/BLAKE_(hash_function)#BLAKE2),
- [SipHash24](https://en.wikipedia.org/wiki/SipHash),
- [HKDF](https://en.wikipedia.org/wiki/HKDF).

WireGuard je sastavni dio jezgre Linuxa [od verzije 5.6](https://www.phoronix.com/scan.php?page=article&item=linux-56-features&num=1). Kako su derivati službene verzije jezgre Linuxa ono što koristi većina distribucija, time WireGuard postaje dostupan širem skupu korisnika.

U nastavku koristimo Docker sliku [linuxserver/wireguard](https://hub.docker.com/r/linuxserver/wireguard), autora [LinuxServer.io](https://www.linuxserver.io/). Njen izvorni kod moguće je pronaći [na GitHubu](https://github.com/linuxserver/docker-wireguard). Nakon povezivanja WireGuard klijenta s WireGuard poslužiteljem (koje opisujemo u nastavku), stvara se tunel kroz koji se onda šalju podaci. Podaci koji prolaze tunelom su šifirirani kako ih treće strane na internetu ne bi mogle čitati, a dodatno se mogu koristiti i razne vrste kompresije kako bi presenesna količina podataka bila manja.

## Stvaranje mreže

U većini slučajeva nam nije naročito važno koje će adrese koristiti Docker kontejneri. Međutim, kod WireGuarda nam to je važno zato što postoji poslužiteljska strana koja mora imati predvidivu adresu kako bi se klijentska strana mogla na nju povezati. Stvorimo mrežu koju ćemo koristiti:

``` shell
$ docker network create wg-net --subnet=172.28.0.0/16
```

U ovoj podmreži je 172.28.0.1 gateway prema domaćinskom računalu na kojem radi Docker daemon, a prvi pokrenuti kontejner (u našem slučaju WireGuard poslužitelj) dobit će adresu 172.28.0.2.

## Podešavanje WireGuard poslužitelja

Podešavanje WireGuard poslužitelja i klijenta započet ćemo stvaranjem direktorija za konfiguraciju:

``` shell
$ mkdir -p wireguard/server/config
```

Pokretanje poslužitelja izvršit ćemo naredbom `docker run`:

``` yaml
$ docker run -d \
  --name=wireguard-server \
  --network=wg-net \
  --cap-add=NET_ADMIN \
  --cap-add=SYS_MODULE \
  -e PUID=1000 \
  -e PGID=1000 \
  -e TZ=Europe/London \
  -e SERVERURL=172.28.0.2 \
  -e SERVERPORT=51820 \
  -e PEERS=2 \
  -e PEERDNS=auto \
  -e INTERNAL_SUBNET=10.31.31.0/24 \
  -e ALLOWEDIPS=0.0.0.0/0 \
  -p 51820:51820/udp \
  -v /home/vedranm/wireguard/server/config:/config \
  -v /lib/modules:/lib/modules \
  --sysctl="net.ipv4.conf.all.src_valid_mark=1" \
  --restart unless-stopped \
  linuxserver/wireguard
```

Ova naredba je dosta složena pa je razmotrimo dio po dio:

- `--cap_add`: pruža kontejneru povišena dopuštenja na domaćinskom računalu, konkretno
    - `NET_ADMIN` omogućuje mu interakciju s mrežnim sučeljima domaćina
    - `SYS_MODULE` omogućuje rad s modulima jezgre
- `-e` dodaje varijablu okoline s danom vrijednosti
    - `PUID=1000` i `PGID=1000` su varijable koje definiraju korisnika i grupu na korisnika računala
    - `TZ=Europe/Zagreb` je vremenska zona kontejnera, potrebno je definirati da vrijeme bude točno
    - `SERVERURL=172.28.0.2` je IP ili adresa poslužitelja, koristi se kod stvaranja konfiguracije za klijente
    - `SERVERPORT=51820` su vrata na kojima se poslužitelj otvara, koristi se kod stvaranja konfiguracije za klijente
    - `PEERS=2` navodi broj klijenata u VPN mreži za koje će konfiguracija biti zapisana, u našem slučaju 2
    - `PEERDNS=auto` uključuje automatsko postavljanje DNS-a
    - `INTERNAL_SUBNET=10.31.31.0/24` je interna podmreža koja se koristi za VPN
    - `ALLOWEDIPS=0.0.0.0/0` navodi dozvoljene adrese koje se mogu povezati na WireGuard poslužitelj
- `-p 51820:51820/udp` prosljeđuje UDP vrata 51820 domaćina na vrata 51280 Docker kontejnera
- `-v` montira direktorij na datečnom sustavu domaćina na direktorij unutar Docker kontejnera
    - `/home/vedranm/wireguard/server/config:/config` montira `/home/vedranm/wireguard/server/config` na `/config`
    - `/lib/modules:/lib/modules` montira `/lib/modules` na `/lib/modules`

Za provjera ispravnosti postupka razmotrimo zapisnike pokretanja poslužitelja naredbom `docker logs`:

``` shell
$ docker logs wireguard-server
[s6-init] making user provided files available at /var/run/s6/etc...exited 0.
[s6-init] ensuring user provided files have correct perms...exited 0.
[fix-attrs.d] applying ownership & permissions fixes...
[fix-attrs.d] done.
[cont-init.d] executing container initialization scripts...
[cont-init.d] 01-envfile: executing...
[cont-init.d] 01-envfile: exited 0.
[cont-init.d] 01-migrations: executing...
[migrations] started
[migrations] no migrations found
[cont-init.d] 01-migrations: exited 0.
[cont-init.d] 02-tamper-check: executing...
[cont-init.d] 02-tamper-check: exited 0.
[cont-init.d] 10-adduser: executing...

-------------------------------------
          _         ()
         | |  ___   _    __
         | | / __| | |  /  \
         | | \__ \ | | | () |
         |_| |___/ |_|  \__/


Brought to you by linuxserver.io
-------------------------------------

To support the app dev(s) visit:
WireGuard: https://www.wireguard.com/donations/

To support LSIO projects visit:
https://www.linuxserver.io/donate/
-------------------------------------
GID/UID
-------------------------------------

User uid:    1000
User gid:    1000
-------------------------------------

[cont-init.d] 10-adduser: exited 0.
[cont-init.d] 30-module: executing...
Uname info: Linux 293ae5c89924 5.16.16-zen1-1-zen #1 ZEN SMP PREEMPT Mon, 21 Mar 2022 22:59:42 +0000 x86_64 x86_64 x86_64 GNU/Linux
**** It seems the wireguard module is already active. Skipping kernel header install and module compilation. ****
[cont-init.d] 30-module: exited 0.
[cont-init.d] 40-confs: executing...
**** Server mode is selected ****
**** External server address is set to 172.28.0.2 ****
**** External server port is set to 51820. Make sure that port is properly forwarded to port 51820 inside this container ****
**** Internal subnet is set to 10.31.31.0/24 ****
**** AllowedIPs for peers 0.0.0.0/0 ****
**** PEERDNS var is either not set or is set to "auto", setting peer DNS to 10.31.31.1 to use wireguard docker host's DNS. ****
**** No wg0.conf found (maybe an initial install), generating 1 server and 2 peer/client confs ****
grep: /config/peer*/*.conf: No such file or directory
PEER 1 QR code:
█████████████████████████████████████████████████████████████████
█████████████████████████████████████████████████████████████████
████ ▄▄▄▄▄ █ ▄▀  ▄  █ ▀▄ ▀█▄▄▀▄█▀▀▄▀  ▀ █▀▄▀▀▄ █▄█▄ ██ ▄▄▄▄▄ ████
████ █   █ █   █▀ █▄▀ ██▄█ ▄█▀█▄ ██ ▄  ██▄▄█ ██ ▄ ▄ ██ █   █ ████
████ █▄▄▄█ █▀  █ ▄█▀██▄▀█▀█▀   ▄▄▄ ▀ █▀█ ▀▄▄█▄▄▀▀ ▀▄██ █▄▄▄█ ████
████▄▄▄▄▄▄▄█▄█ █ █ ▀ ▀ ▀ ▀ ▀▄█ █▄█ █▄▀▄█▄█ ▀▄█▄▀ ▀ ▀▄█▄▄▄▄▄▄▄████
████▄ ▀█  ▄▀██▄█▄█▄▀▄▄   ▄▀ ▀▀ ▄▄    ██▀▄  █▄█▀ ▀▀ ▀▀█▀▄█▄▄ ▄████
████ ▀▄▄█▄▄█ ▄▀ ▄▄ ▀▄▄█ ▀▀██▄██▀▄▄█▀ ▀ ██▄▄▀▄▀▄▀█▀▄▄ █▀▀▄█▄██████
████▄ ▀  ▀▄  ▀  █▀▀▀▄▀ ▀ ▄▄▄▄██▀█▄ ▄ ▄▄ ▄ ▄▄▀▄▀▀ ▀▄█ █ ▄▀█▄  ████
████▀▄▄▄ ▄▄▄▄ ██ ▄█  ▀█▀▀▄██▄▀▄ ▀▄█     ▀█▀▀  ▄██▄▄█▀█▄▄▄██▀█████
████ █▀▀█▀▄▄▄  ▀█ ▄ ▄▄▄█ ▄█▄▀█ ▄ ▀ ▄ ▀▄▄ █ ▄▀  █▄▀██▀▄▀████▀█████
█████ ██▄█▄  ▄█▀ ▀█ █ █▄ ▄ ▀▄▀█▀█  ▄▀▀█ █▄  ▀█ ▀▀█▄▄ ▀█▀▄▄▄▄▀████
█████   █▀▄█▄█▀ █▄ ▀▀▄ ▀  █▀▄▄▄▄▀█ ▀▄ ▄█ █▀ █▀ █ ▄██▀▄▀▄█▄ ▄▀████
████ ██▀▀▀▄▄▀▄▀██ ▀▄███▄█▄ ▄ ▀   █  █▄▀███▀▀▄▄▄▄▀▄▀▀▀▀▄██▀  █████
████ █▀█ ▄▄▀▀ ▄▀▄  ▀▄▄▄▀▀▄▀  ▄▄▄▀▄▀▄█▄▄█ ▀▄█▀█▀▄ ██▀ █▀ ▀▄▄▀▄████
██████▄█ ▄▄▄ █ █ ▀ ██▀   ▄ █▄▀ ▄▄▄ ▄█▀▄ ▄▄▀▄▄▀▄ █  ▀ ▄▄▄ ██▀█████
████  █▀ █▄█ ▀▄ ▀ █ █▄▀█ ▀█▄   █▄█ ▀ ▀▄▄ ▄▄ ██ █  █▄ █▄█ ▀██▀████
████ ▄▄▄  ▄   █▀█▄▀▀██ ▄█▀▄ █▄▄ ▄▄   █ ▄▄█▄▀ █▄▀▀▄▀▀ ▄ ▄ █▄▄▀████
████  ▀▄ ▄▄▄█ ▀   ▀▀█▄▄▄▀▀▀▄▀▀▄ ▄█   ██▄▄█▀▀█▀ ▄▄▄▄██▄▄ ▄▄   ████
████ ▀▄ ▄ ▄▀▀█▄█▀ ▀ ▀▀▄ ▄   █▄██  █▀▀▄▀▀   █▀▄ ▄▀ ▀ █▄██ ▄█ █████
████▄   █▄▄▄ ▀▀ ▄▄▀▄██ ▄▄█▄▄ ▀█▄█  █ ▀██ █ ▀ ▀▀▀▄▀█▀▀  ▀▄ ▀█ ████
████▀▄▀▀▄▄▄ ███▄▀▀▀▀ ▀ ▀ ▄█▀██▄▀ ▀█▀▄▀██▄▄██▄█▄██ ▀▄ █ ██████████
██████▄██ ▄▄ ▄█  ▀ ▀▄▄▄▄▀ █▀▀ ▄▄▄█▄  ███   █ ▄█  █▄██▄ ▀▄▀▄▄▄████
████ ▀▀▄▄ ▄▀██ ▄  █  ▄█▄▄▀█▀▄█▀▀█▄▀█ ▀ ▄█ ▀▄ █ ▄▄▄  ▀▄▀ ▀▄▄██████
████ ██▀▄▀▄█ ███ ▄▄ █▀  █▄▄█▀  ███▄▀▄▀█ ▄ ▄ ▀▄ ▀ ▀█▀ █ ▀▄ ▀▄▄████
████ ▀ ▀▀▄▄ ▄▀ ▀▄ ▀ ███▀▀ ▀█▀█▄  ▀██ ▀▀ ▀▀▄█▄▄ ▀▀▄██▀ ▀ ▄▄█ █████
██████████▄█ ▄▄▀▄▄  █ ▄█ ▄█    ▄▄▄ ▄ ██  ▄ ▀█▀▀▀ ▄▄▀ ▄▄▄ ▀ █▄████
████ ▄▄▄▄▄ █▀▄▄█  █▄▀  ▀  ▀█ █ █▄█ ▄ ▀▀██▀▄█ ██ ▀██▀ █▄█ ██▄▀████
████ █   █ █▄█ █▀▄▄▀▀█ ▀ ▄ ▀▄▄▄▄  ▄▀▄▀▄  ▄ ██  ▀  ▄█  ▄▄ ██ ▀████
████ █▄▄▄█ █▀ ▄██ ▄▄█▀█▄█  ▄ ▄██▄▄ ▀█▄▀▄█▀ ▄▄▀▀▄█▀▀▀█ ▄▀ █▀██████
████▄▄▄▄▄▄▄█▄▄█▄██▄▄▄▄▄█▄▄█▄▄█▄█▄█▄▄▄▄▄██▄██▄▄█▄▄████▄█▄████▄████
█████████████████████████████████████████████████████████████████
█████████████████████████████████████████████████████████████████
PEER 2 QR code:
█████████████████████████████████████████████████████████████████
█████████████████████████████████████████████████████████████████
████ ▄▄▄▄▄ █ ▄▀  █▀ █▀▀█▄ ▄█▄▀▄█▀▀▄▀  ▀ █▀▄▀▀    █▄ ██ ▄▄▄▄▄ ████
████ █   █ █   █▀ ██▀ █▄██ █▄▀█▄ ██ ▄  ██▄▄█ ██ ▄ ▄ ██ █   █ ████
████ █▄▄▄█ █▀ ▄▀▄▄█▀██▄▀█▀█▀   ▄▄▄ ▀ █▀█ ▀▄▄█▄▄▀▀ ▀▄██ █▄▄▄█ ████
████▄▄▄▄▄▄▄█▄█ █ █ ▀ ▀ ▀ ▀ ▀▄█ █▄█ █▄▀▄█▄█ ▀▄█▄▀ ▀ ▀▄█▄▄▄▄▄▄▄████
████▄ ▀█  ▄████▀▄█▄▀▄▄   ▄▀ ▀▀ ▄▄    ██▀▄  █▄█▀ ▀▀ ▀▀▀▀▄▀▄▄ ▄████
████ ▀▄▄█▄▄█▀█  ▄▄ ▀▄▄█ ▀▀██▄██▀▄▄█▀ ▀ ██▄▄▀▄▀▄▀█▀▄▄ ▄ ▀▄█▄██████
████▄  █ ▀▄▄▀   █▀▀▀▄▀ ▀ ▄▄▄▄██▀█▄ ▄ ▄▄ ▄ ▄▄▀▄▀▀ ▀▄█ █ ▄▀▀▄  ████
████▄▄ ▀ ▄▄█▄ ██ ▄█    ▀▀▀ █▄▀▄ ▀▄█     ▀█▀▀ ▄▄▀▀▄▄█▀█▄▄▄ █▀█████
████ ▀▄▄█▀▄▄▄  ▀█ ▄▀▄▀▄▀▀ ▀▀ █ ▄ ▀ ▄ ▀▄▄ █ ▄▀ ▀▄▄ ▄█▀▄▀██▄█▀█████
████▀▀█▄ █▄  ▄█▀ ▀█▄  █▄▄▄ ▄▄▀█▀█  ▄▀▀█ █▄  ▀▄▀ ▀▄█▄ ▀█▀▄▄▄▄█████
████    ▄█▄█▄█▀▀█▄ ▀▀▄ ▀  █▀▄▄▄▄▀█ ▀▄ ▄█ █▀ █▀ █ ▄██▀▄▀▄█▄ █ ████
████ ██▀▄ ▄▄▀▄█▄▀▄▄▄███▄█▄ ▄ ▀   █  █▄▀███▀▀▄▄▄▄▀▄▀▀▀█▄██▀  █████
████ █▀█ █▄▄ █ ▀██▀▀▄▄▄▀▀▄▀  ▄▄▄▀▄▀▄█▄▄█ ▀▄█▀█▀▄ ██▀ █▀▄█▄▄▀▄████
██████▄█ ▄▄▄  ▀  ▀▀██▀   ▄ █▄▀ ▄▄▄ ▄█▀▄ ▄▄▀▄▄▀▄ █  ▄ ▄▄▄ ██▀█████
████  ▀█ █▄█ ▄  ▀█▄ █▄▀█ ▀█▄   █▄█ ▀ ▀▄▄ ▄▄ ██ █  █▄ █▄█ █▀█▀████
█████    ▄▄   █▀█▄▀▀█▀█▄█▄▄ █▄▄ ▄▄   █ ▄▄█▄▀ ▀ ▀▀ ▀▀ ▄ ▄▄█▄▄▀████
████▄▀▄█ ▄▄█▀ ▀   ▀ █▄▄█▀▄▀  ▀▄ ▄█   ██▄▄█▀▀█ ▀▀▄▀▄██▄▄ ▄█▀▄ ████
█████▀ █▄ ▄ ▄█▄█▀ ▀ ▀▀▄▄   █ ▄██  █▀▀▄▀█   █▀▄ █▀▀  █▄██ ▄█ █████
████▄▄  █▄▄▄ ▀▀▀▄▄▀▄██ ▄▄█▄▄ ▀█▄█  █ ▀██ █ ▀ ▀▀▀▄▀█▀▀  ▀▄ ▀▄▀████
████▀▄▀▀▄▄▄ █ ▄█ ▀█▀ ▀ ▀ ▄█▀██▄▀ ▀█▀▄▀██▄▄██▄█▄██ ▀▄ █ ██████████
██████▄█  ▄▄ ▄▄▄▀▄ ▀▄▄▄▄▀ █▀▀ ▄▄▄█▄  ███   █ ▄█  █▄██▄ ▀▄▀▄▄▄████
████ ▀▀▄██▄▀█▄   ▄█  ▄█▄▄▀█▀▄█▀▀█▄▀█ ▀ ▄█ ▀▄ █ ▄▄▄ ▄▀▄▀ ▀▄▄██████
████ ██▄ █▄▀▄█▀█    █▀  █▄▄█▀  ███▄▀▄▀█ ▄ ▄ ▀▄ ▀ ▀█▀▀█▄▀▄▄█▄▄████
████ ▀ ▀▀▄▄▀▄▀ ▀▄ ▀ █▀▄▀▀▄▀█▀▄   ▀██ ▀▀ ▀▀▄█▄▄ █▀▄▀▄▀ █▀▀▄█ █████
██████████▄▄ ▄▄▀▄▄ ▄▄▄ ▄▀▄█▀   ▄▄▄ ▄ ██  ▄ ▀█▀▀█▄ ▄▀ ▄▄▄  ▀█▄████
████ ▄▄▄▄▄ █▀▄▄█  ███  ▄▄ ▀█▄█ █▄█ ▄ ▀▀██▀▄█▀▄█▀ ▄█▀ █▄█ ████████
████ █   █ █▄█ █▀▄▄▀▀█ ▀ ▄ ▀▄▄▄▄  ▄▀▄▀▄  ▄ ██  ▀  ▄█  ▄▄ ██ ▀████
████ █▄▄▄█ █▀▄▀██▄█▄█▀█▄█  ▄ ▄██▄▄ ▀█▄▀▄█▀ ▄▄▀▀▄█▀▀▀█ ▄▀ █▀██████
████▄▄▄▄▄▄▄█▄▄▄▄██▄▄▄▄▄█▄▄█▄▄█▄█▄█▄▄▄▄▄██▄██▄▄█▄▄████▄█▄████▄████
█████████████████████████████████████████████████████████████████
█████████████████████████████████████████████████████████████████
[cont-init.d] 40-confs: exited 0.
[cont-init.d] 90-custom-folders: executing...
[cont-init.d] 90-custom-folders: exited 0.
[cont-init.d] 99-custom-scripts: executing...
[custom-init] no custom files found exiting...
[cont-init.d] 99-custom-scripts: exited 0.
[cont-init.d] done.
[services.d] starting services
[services.d] done.
[#] ip link add wg0 type wireguard
[#] wg setconf wg0 /dev/fd/63
[#] ip -4 address add 10.31.31.1 dev wg0
[#] ip link set mtu 1420 up dev wg0
[#] ip -4 route add 10.31.31.3/32 dev wg0
[#] ip -4 route add 10.31.31.2/32 dev wg0
[#] iptables -A FORWARD -i wg0 -j ACCEPT; iptables -A FORWARD -o wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
.:53
CoreDNS-1.9.1
linux/amd64, go1.17.8, 4b597f8ž
```

Poslužitelj se ispravno pokrenuo. Iako ih nećemo koristiti, možemo u ispisu uočiti QR kodove pojedinih klijenata u kojima je zapisana njihova konfiguracija za lakše korištenje od strane mobilnih uređaja. Mi ćemo koristiti obične konfiguracijske datoteke pa provjerimo jesu li stvorene:

``` shell
$ ls -la wireguard/server/config
sveukupno 8
drwxr-xr-x 1 vedranm vedranm 186 ožu  28 12:28 .
drwxr-xr-x 1 vedranm vedranm  12 ožu  28 12:27 ..
drwxr-xr-x 1 vedranm vedranm  16 ožu  28 12:28 coredns
drwxr-xr-x 1 root    root      0 ožu  28 12:28 custom-cont-init.d
drwxr-xr-x 1 root    root      0 ožu  28 12:28 custom-services.d
-rw------- 1 vedranm vedranm 147 ožu  28 12:28 .donoteditthisfile
drwx------ 1 vedranm vedranm 100 ožu  28 12:28 peer1
drwx------ 1 vedranm vedranm 100 ožu  28 12:28 peer2
drwxr-xr-x 1 vedranm vedranm  66 ožu  28 12:28 server
drwxr-xr-x 1 vedranm vedranm  40 ožu  28 12:28 templates
-rw------- 1 vedranm vedranm 585 ožu  28 12:28 wg0.conf
```

Specijalno, sadržaj konfiguracije poslužitelja `wireguard/server/config/wg0.conf` je:

``` ini
[Interface]
Address = 10.31.31.1
ListenPort = 51820
PrivateKey = mCRIucO9HQOLLBqtLD0eb/Tt/1ooa9RybHelrBnfq3U=
PostUp = iptables -A FORWARD -i %i -j ACCEPT; iptables -A FORWARD -o %i -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i %i -j ACCEPT; iptables -D FORWARD -o %i -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

[Peer]
# peer1
PublicKey = CUh/ZZCPud7+0iZ688otYVr0/B3+OqzCDGcKF+Fol04=
AllowedIPs = 10.31.31.2/32

[Peer]
# peer2
PublicKey = h7/C7D+V18ByOgwNuLXL6KzMm+dDH6sn5hEdOepDcxM=
AllowedIPs = 10.31.31.3/32
```

## Podešavanje WireGuard klijenta

Stvorimo direktorij, a zatim u njega kopirajmo konfiguraciju prvog klijenta koju je poslužitelj stvorio:

``` shell
$ mkdir -p wireguard/peer1/config
$ cp wireguard/server/config/peer1/peer1.conf wireguard/peer1/config/wg0.conf
```

Datoteka `wireguard/peer1/config/wg0.conf` ima sadržaj:

``` ini
[Interface]
Address = 10.31.31.2
PrivateKey = OP32KF8NWkY4oevRJyJuJioTHgtGezENlW/4w3MAtnU=
ListenPort = 51820
DNS = 10.31.31.1

[Peer]
PublicKey = opAPUlnuWUvlkMTcXCP+drjlM5O2K2hdGDSgWtakPSo=
Endpoint = 172.28.0.2:51820
AllowedIPs = 0.0.0.0/0
```

Uočimo kako je klijentu druga strana poslužitelj na adresi 172.28.0.2 i vratima 51820.

Pokrenimo klijent naredbom `docker run`:

``` shell
$ docker run -d \
  --name=wireguard-peer1 \
  --network my-net \
  --cap-add=NET_ADMIN \
  --cap-add=SYS_MODULE \
  -e PUID=1000 \
  -e PGID=1000 \
  -e TZ=Europe/London \
  -e PEERDNS=auto \
  -e INTERNAL_SUBNET=10.31.31.0/24 \
  -e ALLOWEDIPS=0.0.0.0/0 \
  -v /home/vedranm/wireguard/peer1/config:/config \
  -v /lib/modules:/lib/modules \
  --sysctl="net.ipv4.conf.all.src_valid_mark=1" \
  --restart unless-stopped \
  linuxserver/wireguard
```

Ponovno možemo iskoristiti naredbu `docker logs` da se uvjerimo kako je pokretanje bilo uspješno:

``` shell
$ docker logs wireguard-server
[s6-init] making user provided files available at /var/run/s6/etc...exited 0.
[s6-init] ensuring user provided files have correct perms...exited 0.
[fix-attrs.d] applying ownership & permissions fixes...
[fix-attrs.d] done.
[cont-init.d] executing container initialization scripts...
[cont-init.d] 01-envfile: executing...
[cont-init.d] 01-envfile: exited 0.
[cont-init.d] 01-migrations: executing...
[migrations] started
[migrations] no migrations found
[cont-init.d] 01-migrations: exited 0.
[cont-init.d] 02-tamper-check: executing...
[cont-init.d] 02-tamper-check: exited 0.
[cont-init.d] 10-adduser: executing...

-------------------------------------
          _         ()
         | |  ___   _    __
         | | / __| | |  /  \
         | | \__ \ | | | () |
         |_| |___/ |_|  \__/


Brought to you by linuxserver.io
-------------------------------------

To support the app dev(s) visit:
WireGuard: https://www.wireguard.com/donations/

To support LSIO projects visit:
https://www.linuxserver.io/donate/
-------------------------------------
GID/UID
-------------------------------------

User uid:    1000
User gid:    1000
-------------------------------------

[cont-init.d] 10-adduser: exited 0.
[cont-init.d] 30-module: executing...
Uname info: Linux 9e20b9c1077c 5.16.16-zen1-1-zen #1 ZEN SMP PREEMPT Mon, 21 Mar 2022 22:59:42 +0000 x86_64 x86_64 x86_64 GNU/Linux
**** It seems the wireguard module is already active. Skipping kernel header install and module compilation. ****
[cont-init.d] 30-module: exited 0.
[cont-init.d] 40-confs: executing...
**** Client mode selected. ****
**** Disabling CoreDNS ****
[cont-init.d] 40-confs: exited 0.
[cont-init.d] 90-custom-folders: executing...
[cont-init.d] 90-custom-folders: exited 0.
[cont-init.d] 99-custom-scripts: executing...
[custom-init] no custom files found exiting...
[cont-init.d] 99-custom-scripts: exited 0.
[cont-init.d] done.
[services.d] starting services
[services.d] done.
[#] ip link add wg0 type wireguard
[#] wg setconf wg0 /dev/fd/63
[#] ip -4 address add 10.31.31.2 dev wg0
[#] ip link set mtu 1420 up dev wg0
[#] resolvconf -a wg0 -m 0 -x
[#] wg set wg0 fwmark 51820
[#] ip -4 route add 0.0.0.0/0 dev wg0 table 51820
[#] ip -4 rule add not fwmark 51820 table 51820
[#] ip -4 rule add table main suppress_prefixlength 0
[#] sysctl -q net.ipv4.conf.all.src_valid_mark=1
sysctl: setting key "net.ipv4.conf.all.src_valid_mark": Read-only file system
[#] iptables-restore -n
```

Na isti način možemo pokrenuti i drugi klijent, koristeći konfiguraciju u direktoriju `peer2`.

## Testiranje virtualne privatne mreže

Isprobajmo naredbu `ping` klijenta na poslužitelju:

``` shell
docker exec wireguard-server ping 10.31.31.2
PING 10.31.31.2 (10.31.31.2) 56(84) bytes of data.
64 bytes from 10.31.31.2: icmp_seq=1 ttl=64 time=0.187 ms
64 bytes from 10.31.31.2: icmp_seq=2 ttl=64 time=0.546 ms
64 bytes from 10.31.31.2: icmp_seq=3 ttl=64 time=0.486 ms
64 bytes from 10.31.31.2: icmp_seq=4 ttl=64 time=0.255 ms
^C
```

Isprobajmo naredbu `ping` poslužitelja na klijentu:

``` shell
docker exec wireguard-peer1 ping 10.31.31.1
PING 10.31.31.1 (10.31.31.1) 56(84) bytes of data.
64 bytes from 10.31.31.1: icmp_seq=1 ttl=64 time=0.325 ms
64 bytes from 10.31.31.1: icmp_seq=2 ttl=64 time=0.268 ms
64 bytes from 10.31.31.1: icmp_seq=3 ttl=64 time=0.422 ms
64 bytes from 10.31.31.1: icmp_seq=4 ttl=64 time=0.124 ms
^C
```

Uvjerili smo se da paketi `ping`-a uredno prolaze u oba smjera te pokazali da je VPN uspješno uspostavljen.
