---
title: Konfiguracija virtualne privatne mreže alatom WireGuard
author: Luka Ljubojević, Vedran Miletić, Tin Švagelj
---

# Konfiguracija virtualne privatne mreže alatom WireGuard

[Virtualna privatna mreža](https://en.wikipedia.org/wiki/Virtual_private_network) (engl. _virtual private network_, kraće VPN) pruža usluge privatne mreže korištenjem neke javne mreže kao što je internet. Pomoću VPN-a možemo slati i primati podatke preko javne mreže, a istovremeno koristiti konfiguraciju privatne lokalne mreže. VPN koristi različite sigurnosne mehanizme koji osiguravaju tajnost i autentičnost podataka koje šaljemo. Virtualnu privatnu mrežu moguće je stvoriti brojnim protokolima, npr. [L2TP](https://en.wikipedia.org/wiki/Layer_2_Tunneling_Protocol) na veznom sloju (koji implementira [xl2tpd](https://github.com/xelerance/xl2tpd)), [IPsec](https://en.wikipedia.org/wiki/IPsec) na mrežnom sloju (koji implementira [strongSwan](https://www.strongswan.org/)) i [SSTP](https://en.wikipedia.org/wiki/Secure_Socket_Tunneling_Protocol) na transportnom sloju (koji implementira [SoftEther VPN](https://www.softether.org/)).

[WireGuard](https://www.wireguard.com/) je vrlo jednostavan, ali moderno VPN (_peer-to-peer_) rješenje koje nastoji biti brže, jednostavnije, pouzdanije i korisnije od IPsec-a, a također je i ozbiljna konkurencija [OpenVPN-u](https://openvpn.net/community/). Inicijalno je izdan za operacijske sustave temeljene na jezgri Linux, no sada je postao višeplatformsko rješenje za uspostavu VPN mreže (podržava Windows, macOS, BSD-e, iOS i Android). Razvoj WireGuarda još uvijek je u tijeku, iako se već sada smatra jednim od najsigurnijih dostupnih VPN rješenja. WireGuard koristi suvremene kriptografske tehnologije poput:

- [Noise protocol framework](https://noiseprotocol.org/),
- [Curve25519](https://en.wikipedia.org/wiki/Curve25519),
- [ChaCha20](https://www.cryptopp.com/wiki/ChaCha20),
- [Poly1305](https://en.wikipedia.org/wiki/Poly1305),
- [BLAKE2](https://en.wikipedia.org/wiki/BLAKE_(hash_function)#BLAKE2),
- [SipHash24](https://en.wikipedia.org/wiki/SipHash),
- [HKDF](https://en.wikipedia.org/wiki/HKDF).

WireGuard je dio jezgre Linuxa [od verzije 5.6](https://www.phoronix.com/scan.php?page=article&item=linux-56-features&num=1). Kako su derivati službene verzije jezgre Linuxa ono što koristi većina distribucija, time WireGuard postaje dostupan širem skupu korisnika.

U nastavku izravno koristimo WireGuard alate na razini operacijskog sustava, kao što se često radi u stvarnim sustavima. Mrežnu izolaciju između poslužitelja i klijenata ostvarujemo [mrežnim imenskim prostorima](<https://en.wikipedia.org/wiki/Linux_namespaces#Network_(net)>) (engl. _network namespaces_), koji su mehanizam jezgre za potpunu izolaciju mrežnih sučelja, adresa i tablica usmjeravanja. Povezivanje imenskih prostora izvodimo [virtualnim Ethernet uređajima](https://developers.redhat.com/blog/2018/10/22/introduction-to-linux-interfaces-for-virtual-networking#veth) (engl. _veth pairs_) i mrežnim mostom (engl. _bridge_). Ovakav pristup je prenosiv između različitih Linux okruženja (WSL2, nativni Linux, virtualna računala) jer se oslanja na mogućnosti Linux jezgre, bez dodatnih pozadinskih servisa.

Mrežni imenski prostori omogućuju izradu ugniježđene mrežne topologije unutar jednog Linux sustava, pri čemu svaki imenski prostor može imati vlastita mrežna sučelja, adrese, rute i pravila.

## Instalacija potrebnih alata

Za rad su nam potrebni alati `wg` (WireGuard) i `ip` (iproute2). Na većini distribucija `ip` je već instaliran, a WireGuard alate instaliramo na sljedeće načine:

=== "Arch Linux / Manjaro / Garuda"

    ```shell
    sudo pacman -S wireguard-tools
    ```

=== "Debian / Ubuntu / WSL2 (Ubuntu)"

    ```shell
    sudo apt install wireguard-tools
    ```

=== "Fedora"

    ```shell
    sudo dnf install wireguard-tools
    ```

=== "Alpine"

    ```shell
    sudo apk add wireguard-tools
    ```

=== "openSUSE"

    ```shell
    sudo zypper install wireguard-tools
    ```

Provjerimo je li WireGuard dostupan u jezgri i po potrebi učitajmo modul:

```shell
lsmod | grep wireguard || modprobe wireguard
```

```shell-session
wireguard            122880  0
...
```

Od Linuxa 5.6 WireGuard je sastavni dio službene jezgre. Ako je iz nekog razloga učitan kao modul, naredba `modprobe` ga učitava.

### Pokretanje imenskog prostora

!!! warning
    Ovaj odjeljak se odnosi isključivo na Linux. Može se preskočiti u WSL okruženju.

Kako bi se izbjegla potreba za root pristupom, za ove vježbe je na Linuxu potrebno instalirati GNU Screen:

=== "Arch Linux / Manjaro / Garuda"

    ```shell
    sudo pacman -S screen
    ```

=== "Debian / Ubuntu"

    ```shell
    sudo apt install screen
    ```

=== "Fedora"

    ```shell
    sudo dnf install screen
    ```

=== "Alpine"

    ```shell
    sudo apk add screen
    ```

=== "openSUSE"

    ```shell
    sudo zypper install screen
    ```

Prije početka vježbe pokrenite:

```shell
screen -S wg-lab unshare --user --map-root-user --net --mount --fork bash -c '
set -e

mkdir -p /run/netns
mount -t tmpfs tmpfs /run/netns

cat > /tmp/wg-lab-bashrc <<'"'"'EOF'"'"'
alias sudo=""
export PS1="(wg-lab) \u@\h:\w\$ "

echo "U ovoj sesiji je sudo ignoriran."
echo "Izlazak (privemeni) iz sesije: [Ctrl] + [A],  [Ctrl] + [D]."
echo "Nastavite sesiju sa:           screen -r wg-lab"
echo "Terminirajte sesiju sa:        [Ctrl] + [D]    (GUBITAK RADA!)"
EOF

exec bash --rcfile /tmp/wg-lab-bashrc
'
```

U slučaju da slučajno zatvorite emulator terminala, možete nastaviti od prethodne točke sa `screen -r wg-lab` naredbom.

Skripta otvara imenski prostor (engl. _namespace_) za vježbu. Radi se o tehničkom detalju koji **nije bitan za sadržaj vježbe**, samo dozvoljava pokretanje `ip` naredbi bez root ovlasti (ignorira `sudo`), i nastavak rada u slučaju zatvaranja emulatora terminala.

!!! info
    Stvoreni imenski prostor posjeduje sadržaj `/run/netns` direktorija i sve postavke primjenjene pomoći `ip` naredbe. Izvan imenskog prostora ti resursi su nedostupni.

## Stvaranje mrežnih imenskih prostora

Mrežni imenski prostor (engl. _network namespace_) izolira vlastita mrežna sučelja, IP adrese, tablice usmjeravanja i pravila vatrozida. Svaki imenski prostor vidi vlastitu "kopiju" mrežnog stoga.

Stvaramo tri mrežna imenska prostora:

```shell
sudo ip netns add wg-server
sudo ip netns add wg-peer1
sudo ip netns add wg-peer2
```

Naredbe u imenskom prostoru izvršavamo s `ip netns exec <ime> <naredba>`. Te naredbe imaju normalan pristup datotečnom sustavu domaćina, samo su mrežne postavke i sučelja koje im jezgra Linuxa predstavlja drugačija od domaćinskih.

## Povezivanje sučelja i imenskih prostora

Za međusobno povezivanje imenskih prostora koristimo mrežni most (_bridge_) i virtualne ethernet parove (_veth_).
Most se ponaša kao virtualni mrežni preklopnik (engl. _switch_).
_Veth_ par djeluje poput virtualnog Ethernet kabela koji direktno povezuje:

- dva mrežna imenska prostora, ili
- most s mrežnim imenskim prostorom.

Možete koristiti `man ip-link` za pregled svih vrsta mrežnih sučelja/uređaja koje Linux podržava.

U ovom primjeru jedan kraj ide u most na domaćinu, drugi u imenski prostor. Stvaramo most na domaćinu:

```shell
sudo ip link add br0 type bridge
sudo ip addr add 172.28.0.1/16 dev br0
sudo ip link set br0 up
```

!!! info "Spajanje na fizičko sučelje"
    U stvarnim uvjetima bi se na `br0` spojilo fizičko Ethernet sučelje kako bi preusmjerili stvaran promet kroz ovako uređenu mrežu. U tom slučaju se uklanja IP adresa s fizičkog sučelja te se ona dodjeljuje mostu `br0`. Wi-Fi sučelja uglavnom nisu pogodna za ovakvo podešavanje jer u klijentskom načinu ne prosljeđuju okvire s proizvoljnim izvornim MAC adresama drugih uređaja.

Zatim stvaramo _veth_ parove i spajamo ih:

```shell
sudo ip link add veth-s     type veth peer name veth-s-br
sudo ip link set veth-s-br  master br0
sudo ip link set veth-s-br  up
sudo ip link set veth-s     netns wg-server

sudo ip link add veth-p1    type veth peer name veth-p1-br
sudo ip link set veth-p1-br master br0
sudo ip link set veth-p1-br up
sudo ip link set veth-p1    netns wg-peer1

sudo ip link add veth-p2    type veth peer name veth-p2-br
sudo ip link set veth-p2-br master br0
sudo ip link set veth-p2-br up
sudo ip link set veth-p2    netns wg-peer2
```

!!! info "Nazivi imenskih prostora i veza"
    Svaki imenski prostor i mrežno sučelje imaju svoj naziv. U ovoj vježbi nazivi nisu nužni za razumijevanje topologije, no inače dopuštaju individualno uklanjanje i podešavanje svake komponente. Poželjno je koristiti ime koje opisuje namjenu.

Dodjeljujemo adrese sučeljima unutar svakog imenskog prostora:

```shell
sudo ip netns exec wg-server ip addr add 172.28.0.2/16 dev veth-s
sudo ip netns exec wg-server ip link set veth-s up
sudo ip netns exec wg-server ip link set lo up

sudo ip netns exec wg-peer1  ip addr add 172.28.0.3/16 dev veth-p1
sudo ip netns exec wg-peer1  ip link set veth-p1 up
sudo ip netns exec wg-peer1  ip link set lo up

sudo ip netns exec wg-peer2  ip addr add 172.28.0.4/16 dev veth-p2
sudo ip netns exec wg-peer2  ip link set veth-p2 up
sudo ip netns exec wg-peer2  ip link set lo up
```

Provjerimo povezanost prije podizanja WireGuarda:

```shell
sudo ip netns exec wg-server ping -c 2 172.28.0.3
```

```shell-session
PING 172.28.0.3 (172.28.0.3) 56(84) bytes of data.
64 bytes from 172.28.0.3: icmp_seq=1 ttl=64 time=0.052 ms
64 bytes from 172.28.0.3: icmp_seq=2 ttl=64 time=0.038 ms

--- 172.28.0.3 ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 1024ms
```

Tri imenska prostora dijele podmrežu `172.28.0.0/16` putem mosta. To simulira javnu mrežu (internet) kroz koju će prolaziti šifrirani VPN promet.

### Usporedba s Docker kontejnerima

Docker se interno oslanja na iste mehanizme (`ip netns`, _veth_ parove, `iptables`), ali dodaje vlastiti format slika, mrežni driver, _overlay_ datotečni sustav i demona koji mora raditi u pozadini. Cilj ove vježbe je demonstrirati WireGuard, a ne orkestraciju kontejnera pa je izravan rad s jezgrenim primitivima jednostavniji.

U praksi, Docker ima smisla koristiti samo ako je potrebna replikacija postavki. Ako se radi o serveru koji se jednokratno postavlja, izravno podešavanje jezgre troši manje resursa sustava.

## Generiranje kriptografskih ključeva

WireGuard koristi par javnog i privatnog ključa (Curve25519) za svaki čvor te dijeljeni ključ (engl. _preshared key_, PSK) za bolju otpornost u slučaju slabljenja kriptografije (npr. napadi kvantnim računalima). Ključeve stvaramo alatom `wg`. Operacije `wg genkey`, `wg pubkey` i `wg genpsk` su isključivo kriptografske i ne mijenjaju mrežne postavke, pa ih možemo izvršiti izravno na domaćinu:

```shell
mkdir -p ~/wg-ime-prezime/{server,peer1,peer2}đ
chmod -R 600 ~/wg-ime-prezime    # wg genkey ne radi bez ispravnih ovlasti

wg genkey > ~/wg-ime-prezime/server/private
cat         ~/wg-ime-prezime/server/private | wg pubkey > ~/wg-ime-prezime/server/public

wg genkey > ~/wg-ime-prezime/peer1/private
cat         ~/wg-ime-prezime/peer1/private | wg pubkey > ~/wg-ime-prezime/peer1/public

wg genkey > ~/wg-ime-prezime/peer2/private
cat         ~/wg-ime-prezime/peer2/private | wg pubkey > ~/wg-ime-prezime/peer2/public

wg genpsk > ~/wg-ime-prezime/peer1/psk
wg genpsk > ~/wg-ime-prezime/peer2/psk
```

Pregledajmo stvorene javne ključeve:

```shell
cat ~/wg-ime-prezime/server/public
cat ~/wg-ime-prezime/peer1/public
cat ~/wg-ime-prezime/peer2/public
```

```shell-session
Server public key:  oBxUQm5VYT0txhd3BwxvMjA2Rkv06d1T2RhPjyfyyks=
Peer1  public key:  HzE3fRq0S7YAHbaCVBAcFGLfkkahFPZmAUFA4hOcT00=
Peer2  public key:  z/9Dw4fWOejOGhtdzOs8EEDb2OZZPy8JZ1N1H1d6OS4=
```

## Stvaranje WireGuard konfiguracija

WireGuard konfiguracijska datoteka definirana je [INI formatom](https://en.wikipedia.org/wiki/INI_file) i sastoji se od odjeljka `[Interface]` za lokalne postavke te `[Peer]` odjeljaka za druge čvorove.

U odjeljku `[Interface]` definiramo adresu vlastitog WireGuard sučelja i privatni ključ. Poslužitelj dodatno navodi vrata na kojima osluškuje (`ListenPort`). U odjeljku `[Peer]` definiramo javni ključ druge strane, dijeljeni ključ (`PresharedKey`) te raspon adresa koje se usmjeravaju kroz tunel (`AllowedIPs`). Klijent dodatno navodi javnu adresu i vrata/port poslužitelja u `Endpoint` polju.

### Podešavanje poslužitelja

Konfiguracijske datoteke pišemo izravno na domaćinu u `~/wg-ime-prezime/`:

```shell
SERVER_PRIV=$(cat ~/wg-ime-prezime/server/private)
PEER1_PUB=$(  cat ~/wg-ime-prezime/peer1/public  )
PEER1_PSK=$(  cat ~/wg-ime-prezime/peer1/psk     )
PEER2_PUB=$(  cat ~/wg-ime-prezime/peer2/public  )
PEER2_PSK=$(  cat ~/wg-ime-prezime/peer2/psk     )

tee ~/wg-ime-prezime/server/wg0.conf << KRAJ
[Interface]
ListenPort = 51820
PrivateKey = ${SERVER_PRIV}

[Peer]
# peer1
PublicKey    = ${PEER1_PUB}
PresharedKey = ${PEER1_PSK}
AllowedIPs   = 10.31.31.2/32

[Peer]
# peer2
PublicKey    = ${PEER2_PUB}
PresharedKey = ${PEER2_PSK}
AllowedIPs   = 10.31.31.3/32
KRAJ
```

!!! info
    Mrežni imenski prostori ne izoliraju datotečni sustav, tj. datoteke su im vidljive jednako kao i domačinu.

Pregledajmo stvorenu konfiguraciju:

```shell
cat ~/wg-ime-prezime/server/wg0.conf
```

```ini
[Interface]
ListenPort = 51820
PrivateKey = oBxUQm5VYT0txhd3BwxvMjA2Rkv06d1T2RhPjyfyyks=

[Peer]
# peer1
PublicKey    = HzE3fRq0S7YAHbaCVBAcFGLfkkahFPZmAUFA4hOcT00=
PresharedKey = OKrMyn4TV8JPVuflYJYdbCFcgiYq6Xlc0Sxsrjrx0c8=
AllowedIPs   = 10.31.31.2/32

[Peer]
# peer2
PublicKey    = z/9Dw4fWOejOGhtdzOs8EEDb2OZZPy8JZ1N1H1d6OS4=
PresharedKey = Eer8v6Q6KSc9/iy1Li++ofr4M1EvmPEyD2IYF2NflHg=
AllowedIPs   = 10.31.31.3/32
```

### Podešavanje klijenta 1

```shell
SERVER_PUB=$(cat ~/wg-ime-prezime/server/public)
PEER1_PRIV=$(cat ~/wg-ime-prezime/peer1/private)

tee ~/wg-ime-prezime/peer1/wg0.conf << KRAJ
[Interface]
PrivateKey = ${PEER1_PRIV}

[Peer]
PublicKey    = ${SERVER_PUB}
PresharedKey = ${PEER1_PSK}
Endpoint     = 172.28.0.2:51820
AllowedIPs   = 10.31.31.0/24
KRAJ
```

```shell
cat ~/wg-ime-prezime/peer1/wg0.conf
```

```ini
[Interface]
PrivateKey = gOGqN6sm6QMM+r0nQoNpktnvujwP9QyfgOEK7F0RVHU=

[Peer]
PublicKey    = oBxUQm5VYT0txhd3BwxvMjA2Rkv06d1T2RhPjyfyyks=
PresharedKey = OKrMyn4TV8JPVuflYJYdbCFcgiYq6Xlc0Sxsrjrx0c8=
Endpoint     = 172.28.0.3:51820
AllowedIPs   = 10.31.31.0/24
```

### Podešavanje klijenta 2

```shell
PEER2_PRIV=$(cat ~/wg-ime-prezime/peer2/private)
PEER2_PSK=$( cat ~/wg-ime-prezime/peer2/psk    )

tee ~/wg-ime-prezime/peer2/wg0.conf << KRAJ
[Interface]
PrivateKey = ${PEER2_PRIV}

[Peer]
PublicKey    = ${SERVER_PUB}
PresharedKey = ${PEER2_PSK}
Endpoint     = 172.28.0.2:51820
AllowedIPs   = 10.31.31.0/24
KRAJ

# Konfiguracije također sadrže privatne ključeve
chmod a-rwx,u+rw ~/wg-ime-prezime/*/wg0.conf
# U produkciji: /etc/wireguard/, vlasništvo root:root, dozvole 600
```

<!-- chmod nije potreban, ali je dobra praksa -->

## Podizanje WireGuard sučeljâ

WireGuard sučelje podižemo izravno naredbama `ip` i `wg`:

```shell
# Poslužitelj
sudo ip netns exec wg-server ip link add dev wg0 type wireguard
sudo ip netns exec wg-server ip addr add 10.31.31.1/24 dev wg0
sudo ip netns exec wg-server wg setconf wg0 ~/wg-ime-prezime/server/wg0.conf
sudo ip netns exec wg-server ip link set wg0 up

# Klijent 1
sudo ip netns exec wg-peer1 ip link add dev wg0 type wireguard
sudo ip netns exec wg-peer1 ip addr add 10.31.31.2/32 dev wg0
sudo ip netns exec wg-peer1 wg setconf wg0 ~/wg-ime-prezime/peer1/wg0.conf
sudo ip netns exec wg-peer1 ip link set wg0 up
sudo ip netns exec wg-peer1 ip route add 10.31.31.0/24 dev wg0

# Klijent 2
sudo ip netns exec wg-peer2 ip link add dev wg0 type wireguard
sudo ip netns exec wg-peer2 ip addr add 10.31.31.3/32 dev wg0
sudo ip netns exec wg-peer2 wg setconf wg0 ~/wg-ime-prezime/peer2/wg0.conf
sudo ip netns exec wg-peer2 ip link set wg0 up
sudo ip netns exec wg-peer2 ip route add 10.31.31.0/24 dev wg0
```

Naredba `ip link add type wireguard` traži od jezgre da stvori WireGuard sučelje. Zatim `ip addr add` dodjeljuje adresu iz VPN podmreže, `wg setconf` učitava kriptografske ključeve i podatke o klijentima, a `ip link set up` aktivira sučelje. Na kraju `ip route add` usmjerava promet za VPN podmrežu kroz `wg0` — bez ove rute jezgra ne zna kamo poslati pakete namijenjene VPN adresama. Poslužitelju ruta nije potrebna jer `ip addr add .../24` stvara povezanu rutu za cijelu podmrežu. WireGuard potom kriptografskim usmjeravanjem odlučuje kojem čvoru proslijediti pojedini paket.

!!! info
    Cijeli gornji blok može se zamijeniti `wg-quick up` naredbom po imenskom prostoru:

    ``` shell
    sudo ip netns exec wg-server wg-quick up ~/wg-ime-prezime/server/wg0.conf
    sudo ip netns exec wg-peer1  wg-quick up ~/wg-ime-prezime/peer1/wg0.conf
    sudo ip netns exec wg-peer2  wg-quick up ~/wg-ime-prezime/peer2/wg0.conf
    ```

    `wg-quick` interno provodi `ip link add type wireguard`, `ip addr add`, `wg setconf` i `ip link set up`. Zahtijeva da se konfiguracijska datoteka zove jednako kao sučelje (`wg0.conf` → sučelje `wg0`), zbog čega su konfiguracije organizirane u poddirektorije po čvoru. U čistom mrežnom imenskom prostoru pravila vatrozida koja `wg-quick` dodaje nemaju nikakav učinak (prazan *ruleset*, politika `ACCEPT`), pa ga možemo koristiti i u okruženjima poput WSLa gdje jezgreni moduli nisu dostupni. Gore navedene ručne naredbe su sigurnija putanja jer izvođenje jedne za drugom dopušta uvid u uzrok problema; `wg-quick` je zgodan, ali može prikriti probleme — preporučeno je da naredbe unosite pojedinačno prilikom vježbe.

!!! warning "wg-quick zahtjeva Address"
    Kada se koristi `wg-quick`, onda je bitno navesti `Address` polje u `[Interface]` bloku. `Address` parametar uzrokuje pogrešku kada se koristi `wg` direktno pa nije uključen u danim primjerima postavki. Prilikom ručnog postavljanja je [CIDR](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing) postavljen `ip addr add <CIDR> dev wg0` naredbom.

Provjerimo stanje na poslužitelju:

```shell
sudo ip netns exec wg-server wg show
```

```shell-session
interface: wg0
  public key: oBxUQm5VYT0txhd3BwxvMjA2Rkv06d1T2RhPjyfyyks=
  private key: (hidden)
  listening port: 51820

peer: HzE3fRq0S7YAHbaCVBAcFGLfkkahFPZmAUFA4hOcT00=
  preshared key: (hidden)
  allowed ips: 10.31.31.2/32

peer: z/9Dw4fWOejOGhtdzOs8EEDb2OZZPy8JZ1N1H1d6OS4=
  preshared key: (hidden)
  allowed ips: 10.31.31.3/32
```

Na klijentu 1:

```shell
sudo ip netns exec wg-peer1 wg show
```

```shell-session
interface: wg0
  public key: HzE3fRq0S7YAHbaCVBAcFGLfkkahFPZmAUFA4hOcT00=
  private key: (hidden)
  listening port: 34512

peer: oBxUQm5VYT0txhd3BwxvMjA2Rkv06d1T2RhPjyfyyks=
  endpoint: 172.28.0.2:51820
  allowed ips: 10.31.31.0/24
```

Poslužitelj zna za oba klijenta, svaki klijent zna za poslužitelja. WireGuard je protokol bez zasebnog koraka uspostave veze (engl. _connectionless protocol_). Zbog toga se rukovanje (engl. _handshake_) ne vidi u `wg show` prije ikakvog prometa.

## Testiranje virtualne privatne mreže

S klijenta na poslužitelja kroz VPN tunel:

```shell
sudo ip netns exec wg-peer1 ping -c 3 10.31.31.1
```

```shell-session
PING 10.31.31.1 (10.31.31.1) 56(84) bytes of data.
64 bytes from 10.31.31.1: icmp_seq=1 ttl=64 time=0.089 ms
64 bytes from 10.31.31.1: icmp_seq=2 ttl=64 time=0.065 ms
64 bytes from 10.31.31.1: icmp_seq=3 ttl=64 time=0.071 ms

--- 10.31.31.1 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2051ms
```

S poslužitelja na klijenta:

```shell
sudo ip netns exec wg-server ping -c 3 10.31.31.2
```

```shell-session
PING 10.31.31.2 (10.31.31.2) 56(84) bytes of data.
64 bytes from 10.31.31.2: icmp_seq=1 ttl=64 time=0.071 ms
64 bytes from 10.31.31.2: icmp_seq=2 ttl=64 time=0.059 ms
64 bytes from 10.31.31.2: icmp_seq=3 ttl=64 time=0.062 ms

--- 10.31.31.2 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2038ms
```

Između klijenata (promet ide kroz poslužitelja):

```shell
sudo ip netns exec wg-peer1 ping -c 3 10.31.31.3
```

```shell-session
PING 10.31.31.3 (10.31.31.3) 56(84) bytes of data.
64 bytes from 10.31.31.3: icmp_seq=1 ttl=64 time=0.095 ms
64 bytes from 10.31.31.3: icmp_seq=2 ttl=64 time=0.071 ms
64 bytes from 10.31.31.3: icmp_seq=3 ttl=64 time=0.068 ms

--- 10.31.31.3 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2044ms
```

Nakon prometa `wg show` prikazuje i _handshake_ i prenesene podatke:

```shell
sudo ip netns exec wg-server wg show
```

```shell-session
interface: wg0
  public key: oBxUQm5VYT0txhd3BwxvMjA2Rkv06d1T2RhPjyfyyks=
  private key: (hidden)
  listening port: 51820

peer: HzE3fRq0S7YAHbaCVBAcFGLfkkahFPZmAUFA4hOcT00=
  preshared key: (hidden)
  endpoint: 172.28.0.3:34512
  allowed ips: 10.31.31.2/32
  latest handshake: 4 seconds ago
  transfer: 5.14 KiB received, 3.56 KiB sent

peer: z/9Dw4fWOejOGhtdzOs8EEDb2OZZPy8JZ1N1H1d6OS4=
  preshared key: (hidden)
  endpoint: 172.28.0.4:48291
  allowed ips: 10.31.31.3/32
  latest handshake: 2 seconds ago
  transfer: 3.12 KiB received, 2.84 KiB sent
```

Uočimo u izlazu: `latest handshake` (vrijeme zadnjeg rukovanja), `transfer` (preneseni podaci po smjeru), `endpoint` (stvarna adresa klijenta iz koje je poslužitelj primio paket).

Uvjerili smo se da paketi uredno prolaze u svim smjerovima te pokazali da je VPN uspješno uspostavljen.

## Čišćenje

Mrežni imenski prostori čine zasebnu cjelinu. Brisanjem imenskog prostora uklanjaju se njegovi mrežni resursi, koji uključuju:

- sva mrežna sučelja unutar prostora (uključujući WireGuard `wg0`),
- sve rute,
- sve WireGuard veze i pridruženo kriptografsko stanje, te
- sve procese čiji je mrežni prostor bio taj prostor.

!!! info
    Ovo je isti jezgreni mehanizam koji koristi i Docker. Razlika je u tome što ovdje mrežu podešavamo ručno naredbom `ip`, dok Docker iste mrežne primitive podešava u pozadini, komunikacijom s jezgrom preko netlink sučelja.

```shell
sudo ip netns del wg-server wg-peer1 wg-peer2
sudo ip link del br0
```

Most `br0` pripada domaćinskom prostoru pa se briše zasebno; `ip link del br0` automatski uklanja i sve _veth_ veze koje su na njega bile spojene.
