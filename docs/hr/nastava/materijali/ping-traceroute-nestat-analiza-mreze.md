---
author: Domagoj Margan, Vedran Miletić
---

# Osnovni alati za analizu računalne mreže

Svi alati koje opisujemo ovdje pokreću se u ljusci, a rade jednako dobro na stvarnim i na emuliranim čvorovima.

## Alat ping

`ping` je osnovni alat za slanje ICMP ECHO_REQUEST domaćinima na mreži. Koristimo ga kada želimo provjeriti je li neki domaćin/gateway dostupan na mreži.

Ping je vrlo jednostavno za koristiti. Najjednostavniji oblik korištenja je korištenje bez dodatnih parametara. Ako je domaćin dostupan, te ako se paketi mogu slati bez poteškoća, dobiti ćemo povratnu informaciju o uspješnosti, te informaciju o tome kako nema izgubljenih paketa ("0% packet loss").

Ukoliko želimo provjeriti je li domaćin example dostupan na mreži group.miletic.net jednostavno iza naredbe pišemo ime domaćina i mrežu kojoj pripada:

``` shell
$ ping example.group.miletic.net
PING example.group.miletic.net (193.198.209.42) 56(84) bytes of data.
64 bytes from 193.198.209.42: icmp_seq=1 ttl=56 time=3.48 ms
64 bytes from 193.198.209.42: icmp_seq=2 ttl=56 time=3.44 ms
64 bytes from 193.198.209.42: icmp_seq=3 ttl=56 time=3.49 ms
64 bytes from 193.198.209.42: icmp_seq=4 ttl=56 time=3.49 ms
64 bytes from 193.198.209.42: icmp_seq=5 ttl=56 time=3.46 ms
^C
--- example.group.miletic.net ping statistics ---
5 packets transmitted, 5 received, 0% packet loss, time 4006ms
rtt min/avg/max/mdev = 3.445/3.476/3.497/0.077 ms
```

Ping bez dodatnih argumenata šalje ICMP zahtjeve u beskonačnost, pa je slanje nužno prekinuti kombinacijom tipki ++control+c++.

Ukoliko je potrebno, umjesto imena domaćina možemo pisati njegovu IP adresu:

``` shell
$ ping 193.198.209.42
PING 193.198.209.42 (193.198.209.42) 56(84) bytes of data.
64 bytes from 193.198.209.42: icmp_seq=1 ttl=56 time=3.48 ms
64 bytes from 193.198.209.42: icmp_seq=2 ttl=56 time=3.61 ms
64 bytes from 193.198.209.42: icmp_seq=3 ttl=56 time=4.13 ms
^C
--- 193.198.209.42 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2003ms
rtt min/avg/max/mdev = 3.488/3.746/4.135/0.288 ms
```

Uvođenjem dodatnih parametara, možemo manipulirati načinom i učestalosti slanja ICMP zahtjeva. Parametre možemo međusobno kombinirati, osim u slučajevima kada to izaziva kontradikciju.

Za slanje u određenom intervalu (svakih n sekundi), koristimo parametar `-i`, te broj sekundi. Ping bez argumenata zahtjev šalje u intervalima od 1 sekunde. Moguće je smanjiti taj interval, npr. na pola sekunde:

``` shell
$ ping -i 0.5 example.group.miletic.net
```

Želimo li poslati određen broj zahtjeva (umjesto da sami prekidamo slanje kombinacijom tipki ++control+c++), koristimo parametar `-c` i broj paketa. Primjerice, ako želimo poslati 3 zahtjeva:

``` shell
$ ping -c 3 example.group.miletic.net
PING example.group.miletic.net (193.198.209.42) 56(84) bytes of data.
64 bytes from 193.198.209.42: icmp_seq=1 ttl=56 time=3.82 ms
64 bytes from 193.198.209.42: icmp_seq=2 ttl=56 time=3.45 ms
64 bytes from 193.198.209.42: icmp_seq=3 ttl=56 time=6.35 ms

--- example.group.miletic.net ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2003ms
rtt min/avg/max/mdev = 3.454/4.545/6.354/1.290 ms
```

Također, parametrom `-w` možemo odrediti ukupno vrijeme u kojem želimo da se zahtjevi šalju. Nakon određenog broja sekundi slanje se prekida (u našem primjeru 3 sekunde).

``` shell
$ ping -w 3 example.group.miletic.net
```

Ako ne želimo vidjeti informacije o svakom poslanom zahtjevu posebno, možemo uključiti 'tihi' način rada parametrom `-q`; u tom slučaju prikazat će se samo završni izvještaj.

``` shell
$ ping -q -c 10 example.group.miletic.net
PING example.group.miletic.net (193.198.209.42) 56(84) bytes of data.

--- example.group.miletic.net ping statistics ---
10 packets transmitted, 10 received, 0% packet loss, time 9014ms
rtt min/avg/max/mdev = 3.472/3.931/5.105/0.540 ms
```

Parametrom `-s` možemo mijenjati veličinu paketa kojeg šaljemo. Zadana veličina paketa je 56 bajta. Veličini koju odredimo se dodaje još 28 byteova, zbog veličine zaglavlja paketa (u nastavku kolegija naučit ćemo zašto baš 28 bajta). U primjeru šaljemo pakete čije je tijelo veličine 200 bajta, što zajedno sa zaglavljem čini ukupno 228 bajta.

``` shell
$ ping -s 200 example.group.miletic.net
PING example.group.miletic.net (193.198.209.42) 200(228) bytes of data.
208 bytes from 193.198.209.42: icmp_seq=1 ttl=56 time=3.49 ms
208 bytes from 193.198.209.42: icmp_seq=2 ttl=56 time=3.75 ms
^C
--- example.group.miletic.net ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 1001ms
rtt min/avg/max/mdev = 3.493/3.626/3.759/0.133 ms
```

Pregled parametara za alat ping:

- `-i` : slanje paketa u određenom vremenskom intervalu
- `-c` : slanje određenog broja paketa
- `-w` : određivanje timeouta
- `-q` : tihi način rada
- `-s` : mijenjanje veličine paketa za slanje

Za informacije o ostalim mogućnostima proučite `man ping`.

## Alati traceroute i tracepath

Alati `traceroute` i `tracepath` prate putanju kretanja paketa od slanja do odredišta. Iako postoje netrivijalne razlike između alata `traceroute` i `tracepath`, za naše potrebe oni su jednako dobri. U nastavku opisujemo korištenje naredbe `tracepath`; `traceroute` se koristi na vrlo sličan način.

`tracepath` koristimo navođenjem imena ili IP adrese čvora u mreži. Primjerice, želimo li pratiti putanju kretanja paketa s privatnog domaćina (T-Com ISP) do poslužitelja example:

``` shell
$ tracepath example.group.miletic.net
 1:  host.lan                                         0.191ms pmtu 1500
 1:  dsldevice.lan                                        84.268ms
 1:  dsldevice.lan                                        95.075ms
 2:  dsldevice.lan                                        98.775ms pmtu 1492
 2:  no reply
 3:  172.29.49.245                                        56.365ms
 4:  gtr11-hdr01.ip.t-com.hr                              53.908ms
 5:  193.192.15.65                                        80.398ms
 6:  CN-Srce-03-RO.core.carnet.hr                         59.637ms asymm 12
 7:  CN-Riteh-01-RO.core.carnet.hr                        66.910ms asymm 12
 8:  CN-Riteh.01-ES.core.carnet.hr                        67.485ms
 9:  kbf03-ro.access.carnet.hr                            72.179ms
10:  example.group.miletic.net                                        78.724ms
11:  example.group.miletic.net                                        82.123ms reached
     Resume: pmtu 1492 hops 11 back 54
```

Ukoliko želimo vidjeti putanju kretanja isključivo po IP adresama, koristimo parametar `-n`:

``` shell
$ tracepath -n example.group.miletic.net
 1:  192.168.1.2                                           0.209ms pmtu 1500
 1:  192.168.1.1                                          14.326ms
 1:  192.168.1.1                                         300.088ms
 2:  192.168.1.1                                         217.613ms pmtu 1492
 2:  no reply
 3:  172.29.49.245                                        54.245ms
 4:  195.29.240.210                                       66.783ms
 5:  193.192.15.65                                        58.056ms
 6:  193.198.228.154                                      57.958ms asymm 12
 7:  193.198.237.54                                       56.744ms asymm 12
 8:  193.198.235.2                                        56.589ms
 9:  193.198.235.174                                      57.391ms
10:  161.53.45.42                                         58.736ms
11:  161.53.45.42                                         60.637ms reached
     Resume: pmtu 1492 hops 11 back 54
```

Također, ako želimo vidjeti ispis i imena domaćina/gatewaya i njihovih IP adresa, koristimo parametar `-b`:

``` shell
$ tracepath -b example.group.miletic.net
```

Pregled parametara za alat `tracepath`:

- `-n` : pregled putanje po IP adresama
- `-b` : pregled putanje po IP adresama i nazivima domaćina

Za informacije o ostali parametrima proučite `man tracepath` (ili `man traceroute`).

## Alat netstat

Alat `netstat` pruža nam raznolike informacije o konfiguraciji i aktivnostima na mreži.

Kako bi vidjeli informacije i statistike vezane uz pojednino mrežno sučelje na našem računalu, koristimo parametar `-i`:

``` shell
$ netstat -i
Kernel Interface table
Iface   MTU Met   RX-OK RX-ERR RX-DRP RX-OVR    TX-OK TX-ERR TX-DRP TX-OVR Flg
eth0       1500 0         0      0      0 0             0      0      0      0 BMU
lo        16436 0       931      0      0 0           931      0      0      0 LRU
wlan0      1500 0   1074438      0      0 0       1000130      0      0      0 BMRU
```

U ovoj tablici stupci koje možemo sa sadašnjim znanjem čitati su redom:

- `Iface`: ime mrežnog sučelja (`eth*` su obično Ethernet mrežne kartice, `wlan*` su WiFi mrežne kartice)
- `RX*`: informacije o primljenim paketima
- `TX*`: informacije o poslanim paketima

Ukoliko želimo detaljniji prikaz informacija o sučeljima, potrebno je nadodati parametar `-e`:

``` shell
$ netstat -ie
Kernel Interface table
eth0      Link encap:Ethernet  HWaddr 00:26:9e:78:3e:36
          UP BROADCAST MULTICAST  MTU:1500  Metric:1
          RX packets:0 errors:0 dropped:0 overruns:0 frame:0
          TX packets:0 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000
          RX bytes:0 (0.0 B)  TX bytes:0 (0.0 B)
          Interrupt:30 Base address:0xc000

lo        Link encap:Local Loopback
          inet addr:127.0.0.1  Mask:255.0.0.0
          inet6 addr: ::1/128 Scope:Host
          UP LOOPBACK RUNNING  MTU:16436  Metric:1
          RX packets:946 errors:0 dropped:0 overruns:0 frame:0
          TX packets:946 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:0
          RX bytes:55873 (54.5 KiB)  TX bytes:55873 (54.5 KiB)

wlan0     Link encap:Ethernet  HWaddr 00:24:d6:0b:61:26
          inet addr:192.168.1.2  Bcast:192.168.1.255  Mask:255.255.255.0
          inet6 addr: fe80::224:d6ff:fe0b:6126/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:1074857 errors:0 dropped:0 overruns:0 frame:0
          TX packets:1000600 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000
          RX bytes:383343384 (365.5 MiB)  TX bytes:143158985 (136.5 MiB)
```

Za prikaz routing tablice, koristimo parametar `-r`:

``` shell
$ netstat -r
Kernel IP routing table
Destination     Gateway         Genmask         Flags   MSS Window  irtt Iface
192.168.1.0     *               255.255.255.0   U         0 0          0 wlan0
link-local      *               255.255.0.0     U         0 0          0 wlan0
default         dsldevice.lan   0.0.0.0         UG        0 0          0 wlan0
```

Dodamo li i parametar `-n`, dobit ćemo prikaz IP adresa umjesto imena domaćina i mreža:

``` shell
$ netstat -rn
Kernel IP routing table
Destination     Gateway         Genmask         Flags   MSS Window  irtt Iface
192.168.1.0     *               255.255.255.0   U         0 0          0 wlan0
192.168.0.0     *               255.255.0.0     U         0 0          0 wlan0
0.0.0.0         192.168.1.1     0.0.0.0         UG        0 0          0 wlan0
```

Parametrom `-t` možemo pregledati aktivne TCP konekcije (otvorene utičnice). Dodatkom parametra `-a` (all), prikazujemo i poslužitelje:

``` shell
$ netstat -ta
Active Internet connections (servers and established)
Proto Recv-Q Send-Q Local Address           Foreign Address         State
tcp        0      0 *:http                  *:*                     LISTEN
tcp        0      0 *:domain                *:*                     LISTEN
tcp        0      0 *:ssh                   *:*                     LISTEN
tcp        0      0 localhost:postgres      *:*                     LISTEN
tcp        0      0 *:smtp                  *:*                     LISTEN
tcp        0      0 *:https                 *:*                     LISTEN
tcp        0      0 10.11.22.238:49921      93.152.160.101:6697     ESTABLISHED
tcp        0      0 10.11.22.238:48270      HUBBARD.CLUB.CC.CM:6697 ESTABLISHED
tcp        0    368 10.11.22.238:ssh        78-2-77-250.adsl.:54528 ESTABLISHED
tcp6       0      0 [::]:domain             [::]:*                  LISTEN
tcp6       0      0 [::]:ssh                [::]:*                  LISTEN
tcp6       0      0 [::]:ipp                [::]:*                  LISTEN
tcp6       0      0 localhost:postgres      [::]:*                  LISTEN
```

Također, moguće je vidjeti i UDP konekcije (otvorene utičnice), parametrom `-u`:

``` shell
$ netstat -ua
Active Internet connections (servers and established)
Proto Recv-Q Send-Q Local Address           Foreign Address         State
udp        0      0 *:domain                *:*
udp        0      0 10.11.22.238:ntp        *:*
udp        0      0 localhost:ntp           *:*
udp        0      0 *:ntp                   *:*
udp        0      0 *:38069                 *:*
udp        0      0 *:mdns                  *:*
udp        0      0 *:ipp                   *:*
udp6       0      0 [::]:domain             [::]:*
udp6       0      0 fe80::218:f3ff:fe6c:ntp [::]:*
udp6       0      0 localhost:ntp           [::]:*
udp6       0      0 [::]:ntp                [::]:*
udp6       0      0 localhost:58156         localhost:58156         ESTABLISHED
```

Za detaljan prikaz svih otvorenih utičnica, koristimo kombinaciju parametara `-utae`:

``` shell
$ netstat -utae
Active Internet connections (servers and established)
Proto Recv-Q Send-Q Local Address           Foreign Address         State       User       Inode
tcp        0      0 *:http                  *:*                     LISTEN      root       14299
tcp        0      0 *:domain                *:*                     LISTEN      root       13609
tcp        0      0 *:ssh                   *:*                     LISTEN      root       14062
tcp        0      0 localhost:postgres      *:*                     LISTEN      postgres   15239
tcp        0      0 *:smtp                  *:*                     LISTEN      root       14435
tcp        0      0 *:https                 *:*                     LISTEN      root       14306
tcp        0      0 10.11.22.238:49921      93.152.160.101:6697     ESTABLISHED lukav      18851
tcp        0      0 10.11.22.238:48270      HUBBARD.CLUB.CC.CM:6697 ESTABLISHED vedranm    68450
tcp        0    384 10.11.22.238:ssh        78-2-77-250.adsl.:54528 ESTABLISHED root       202243
tcp6       0      0 [::]:domain             [::]:*                  LISTEN      root       13611
tcp6       0      0 [::]:ssh                [::]:*                  LISTEN      root       14070
tcp6       0      0 [::]:ipp                [::]:*                  LISTEN      root       11021
tcp6       0      0 localhost:postgres      [::]:*                  LISTEN      postgres   15238
udp        0      0 *:domain                *:*                                 root       13608
udp        0      0 10.11.22.238:ntp        *:*                                 ntp        14078
udp        0      0 localhost:ntp           *:*                                 root       12545
udp        0      0 *:ntp                   *:*                                 root       12406
udp        0      0 *:38069                 *:*                                 avahi      12946
udp        0      0 *:mdns                  *:*                                 avahi      12945
udp        0      0 *:ipp                   *:*                                 root       11022
udp6       0      0 [::]:domain             [::]:*                              root       13610
udp6       0      0 fe80::218:f3ff:fe6c:ntp [::]:*                              ntp        14079
udp6       0      0 localhost:ntp           [::]:*                              root       12546
udp6       0      0 [::]:ntp                [::]:*                              root       12407
udp6       0      0 localhost:58156         localhost:58156         ESTABLISHED postgres   15246
```

Otvorene utičnice možemo parametrom `-l` filtrirati samo na one koje slušaju, tj. na one koje su u stanju `LISTEN`:

``` shell
$ netstat -l
Active Internet connections (only servers)
Proto Recv-Q Send-Q Local Address           Foreign Address         State
tcp        0      0 *:http                  *:*                     LISTEN
tcp        0      0 *:domain                *:*                     LISTEN
tcp        0      0 *:ssh                   *:*                     LISTEN
tcp        0      0 localhost:postgres      *:*                     LISTEN
tcp        0      0 *:smtp                  *:*                     LISTEN
tcp        0      0 *:https                 *:*                     LISTEN
tcp6       0      0 [::]:domain             [::]:*                  LISTEN
tcp6       0      0 [::]:ssh                [::]:*                  LISTEN
tcp6       0      0 [::]:ipp                [::]:*                  LISTEN
tcp6       0      0 localhost:postgres      [::]:*                  LISTEN
udp        0      0 *:domain                *:*
udp        0      0 10.11.22.238:ntp        *:*
udp        0      0 localhost:ntp           *:*
udp        0      0 *:ntp                   *:*
udp        0      0 *:38069                 *:*
udp        0      0 *:mdns                  *:*
udp        0      0 *:ipp                   *:*
udp6       0      0 [::]:domain             [::]:*
udp6       0      0 fe80::218:f3ff:fe6c:ntp [::]:*
udp6       0      0 localhost:ntp           [::]:*
udp6       0      0 [::]:ntp                [::]:*
Active UNIX domain sockets (only servers)
Proto RefCnt Flags       Type       State         I-Node   Path
unix  2      [ ACC ]     STREAM     LISTENING     14510    private/error
unix  2      [ ACC ]     STREAM     LISTENING     11017    /var/run/cups/cups.sock
unix  2      [ ACC ]     STREAM     LISTENING     14513    private/retry
unix  2      [ ACC ]     STREAM     LISTENING     15240    /tmp/.s.PGSQL.5432
unix  2      [ ACC ]     STREAM     LISTENING     12302    /var/run/acpid.socket
unix  2      [ ACC ]     STREAM     LISTENING     11029    /var/run/avahi-daemon/socket
unix  2      [ ACC ]     STREAM     LISTENING     18721    /var/run/screen/S-lukav/1998.pts-0.ares
unix  2      [ ACC ]     STREAM     LISTENING     11041    /var/run/dbus/system_bus_socket
unix  2      [ ACC ]     STREAM     LISTENING     14472    private/defer
unix  2      [ ACC ]     STREAM     LISTENING     14516    private/discard
unix  2      [ ACC ]     STREAM     LISTENING     14466    private/bounce
unix  2      [ ACC ]     STREAM     LISTENING     14443    public/cleanup
unix  2      [ ACC ]     STREAM     LISTENING     7494     /run/systemd/private
unix  2      [ ACC ]     STREAM     LISTENING     14478    private/verify
unix  2      [ ACC ]     STREAM     LISTENING     14475    private/trace
unix  2      [ ACC ]     STREAM     LISTENING     14451    /var/run/wsgi.729.0.1.sock
unix  2      [ ACC ]     STREAM     LISTENING     14519    private/local
unix  2      [ ACC ]     STREAM     LISTENING     14525    private/lmtp
unix  2      [ ACC ]     STREAM     LISTENING     14458    /var/run/wsgi.729.0.2.sock
unix  2      [ ACC ]     STREAM     LISTENING     14467    /var/run/wsgi.729.0.3.sock
unix  2      [ ACC ]     SEQPACKET  LISTENING     7563     /run/udev/control
```

Parametrom `-s` prikazuje se skup statistika za pojedine protokole:

``` shell
$ netstat -s
Ip:
    228437 total packets received
    8393 with invalid addresses
    0 forwarded
    ...
Icmp:
    188 ICMP messages received
    0 input ICMP message failed.
    ...
IcmpMsg:
        InType3: 38
        InType8: 99
        InType13: 51
        OutType0: 99
        OutType3: 12
        OutType14: 51
Tcp:
    688 active connections openings
    4374 passive connection openings
    8 failed connection attempts
    ...
Udp:
    4932 packets received
    12 packets to unknown port received.
    0 packet receive errors
    ...
TcpExt:
    4 invalid SYN cookies received
    8 resets received for embryonic SYN_RECV sockets
    2191 TCP sockets finished time wait in fast timer
    ...
IpExt:
    InMcastPkts: 504
    OutMcastPkts: 23
    InBcastPkts: 21972
    ...
```

Pregled parametara za alat netstat:

- `-i` : ispis informacija o mrežnim sučeljima
- `-e` : detaljniji ispis
- `-r` : prikaz routing tablice
- `-n` : prikaz IP adresa umjesto imena domaćina i mreža
- `-t` : pregled tcp konekcija (utičnica)
- `-u` : pregled udp konekcija (utičnica)
- `-a` : prikaz svih (poslužiteljskih) konekcija
- `-l` : filtriranje utičnica u stanju slušanja
- `-s` : prikaz statistika za pojedine protokole

Za više informacija proučite `man netstat`.

## Dodatak: alat clockdiff

Alatom `clockdiff` možemo mjeriti satnu razliku između domaćina na mreži, koristeći ICMP TIMESTAMP pakete. `clockdiff` koristimo navođenjem imena ili IP adrese domaćina.

``` shell
$ clockdiff example.group.miletic.net
...................................................
host=example.group.miletic.net rtt=15(0)ms/15ms delta=-2844ms/-2843ms Wed Jun 13 20:21:57 2012
```

Za informacije o ostalim parametrima proučite `man clockdiff`.

## Dodatak: alat arping

`arping` je alat za slanje ARP REQUEST susjedinim domaćinima na mreži. `arping` koristimo navođenjem mrežnog sučelja (parametrom `-I`), te IP adrese domaćina kojem želimo poslati ARP zahtjev.

!!! note
    Alat `arping` zahtijeva administratorske privilegije, odnosno dostupan je samo `root` korisniku.

Primjerice, želimo li poslati ARP zahtjev domaćinu na adresi 192.168.1.4 s mrežnog sučelja `wlan0`, radimo to na idući način:

``` shell
# arping -I wlan0 192.168.1.4
ARPING 192.168.1.4 from 192.168.1.2 wlan0
Unicast reply from 192.168.1.4 [00:0F:EA:54:B7:2C]  2.306ms
Unicast reply from 192.168.1.4 [00:0F:EA:54:B7:2C]  1.591ms
Unicast reply from 192.168.1.4 [00:0F:EA:54:B7:2C]  1.568ms
^CSent 3 probes (1 broadcast(s))
Received 3 response(s)
```

Parametrom `-c` određujemo količinu poslanih ARP zahtjeva. Nakon određenog broj poslanih zahtjeva, slanje se prekida:

``` shell
# arping -c 3 -I wlan0 192.168.1.4
ARPING 192.168.1.4 from 192.168.1.2 wlan0
Unicast reply from 192.168.1.4 [00:0F:EA:54:B7:2C]  2.176ms
Unicast reply from 192.168.1.4 [00:0F:EA:54:B7:2C]  1.476ms
Unicast reply from 192.168.1.4 [00:0F:EA:54:B7:2C]  2.026ms
Sent 3 probes (1 broadcast(s))
Received 3 response(s)
```

Pregled parametara za alat arping:

- `-I` : određivanje mrežnog sučelja
- `-c` : slanje određenog broja paketa

Za više informacija o dostupnoj funkcionalnosti proučite `man arping`.
