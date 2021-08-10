---
author: Domagoj Margan, Vedran Miletić
---

# Filtriranje paketa

Pri pokretanju Wiresharka sa zadanim postavkama, postoji problem prevelikog broja informacija koje se prikazuju. Time je otežan pronalazak tražane informacije/podatka, te zbog toga moramo primjeniti filtre. Filtri su bitni jer omogućuju jednostavno lociranje podataka na temelju danih opcija.

Razlikujemo dva tipa filtara: **filtri kod snimanja** (engl. *capture filters*) i **filtri kod prikaza** (engl. *display filters*). Filtrima kod snimanja se bavimo sada; filtrima kod prikaza nešto kasnije.

## Filtri kod snimanja

Capture filtre koristimo za određivanje podataka koje ćemo spremiti i analizirati. Njihova glavna primjena je smanjenje količine podataka za hvatanje. Capture filtri moraju biti postavljeni prije pokretanja Wireshrark hvatanja, što nije slučaj sa display filtrima koji mogu biti modificirani u bilo kojem trenutku tijekom hvatanja.

Popis predefiniranih capture filtara, koji služe kao primjeri kod izrade vlastitih, dostupan je pod `Capture/Capture Filters...` u izborniku.

Da bi uključili korištenje capture filtra kod snimanja prometa, odaberite opciju `Capture Options` na početnom ekranu ili stavku `Capture/Options` u izborniku. U popisu su navedena mrežna sučelja; zadnji stupac navodi aktivne capture filtre, i prazan je ukoliko nijedan nije aktivan. Dvoklikom na sučelje na kojem snimamo pakete dobiva se mogućnost unosa u polje `Capture Filter:` koje odmah po unosu provjerava valjanost definiranih filtara, i u skladu s tim ih boji u crvenu ili zelenu boju. Klikom na gumb `Capture Filters:` dobiva se ranije navedeni popis predefiniranih capture filtera, s kojima možete isprobavati po želji.

### Način rada filtera kod snimanja

Sintaksa filtra kod snimanja uključuje: protokol, smjer, domaćin(e), vrijednost, logičku operaciju, ostalo. Razmotrimo te elemente redom.

**Protokol** dozvoljava vrijednosti `ether`, `fddi`, `ip`, `arp`, `rarp`, `decnet`, `lat`, `sca`, `moprc`, `mopdl`, `tcp` i `udp`. Ako nije određen protokol, pretpostavljaju se svi protokoli.

**Smjer** dozvoljava vrijednosti: `src`, `dst`, `src and dst`, `src or dst`. Ako nije specificiran izvor ili odredište, izraz `src or dst` je primjenjen. (Primjerice, `host 192.168.1.2` je ekvivalentno `src or dst host 192.168.1.2`).

**Domaćin(i)** dozvoljava vrijednosti: `net`, `port`, `host`, `portrange`. Ako domaćin nije specificiran, koristi se izraz `host`. (Primjerice, `dst 10.2.2.2` je jednako `dst host 10.2.2.2`.)

**Logičke operacije** dozvoljavaju vrijednosti: `not`, `and` i `or`.  Negacija (`not`) ima najviši prioritet (prednost). Operatori `or` i `and` imaju jednak prioritet. Primjerice:

- `not tcp port 118 and tcp port 530` je ekvivalentno `(not tcp port 118) and tcp port 530`,
- `not tcp port 118 and tcp port 530` **nije** ekvivalentno `not (tcp port 118 and tcp port 530)`.

### Primjeri filtera kod snimanja

Za hvatanje prometa sa ili od jednog određenog domaćina, koristimo `host` i IP adresu domaćina:

```
host 192.168.7.2
```

Umjesto IP adrese, možemo upisati ime domaćina:

```
host www.primjer.com
```

Za hvatanje prometa sa ili od određenog raspona IP adresa, koristimo `net` i IP raspon:

```
net 192.168.0.0/24
```

ili, ekvivalentno, koristeći izraz `mask`:

```
src net 192.168.0.0 mask 255.255.255.0
```

Za hvatanje prometa samo sa određenog porta, koristimo `port` i broj porta:

```
port 21
```

Možemo filtrirati i po protokolu, pa za hvatanje prometa s određenog protokola navodimo taj protokol:

```
ip
```

Prikaz paketa sa odredištem TCP port 3128:

```
tcp dst port 3128
```

Prikaz paketa sa izvornom IP adresom 10.1.1.1:

```
ip src host 10.1.1.1
```

Prikaz paketa sa izvornom ili odredišnom IP adresom 10.1.2.3

```
src or dst host 10.1.2.3
```

Prikaz paketa sa izvorom UDP ili TCP portom u rasponu od 2000 do 2500.

```
src portrange 2000-2500
```

Prikaz svega osim icmp paketa:

```
not icmp
```

Prikaz paketa sa izvornom IP adresom 10.7.2.12, isključujući sve pakete sa odredišnom adresom 10.200.0.0/16

```
src host 10.7.2.12 and not dst net 10.200.0.0/16
```

Prikaz paketa sa izvornom IP adresa 19.4.1.12 ili izvornom mrežom 10.6.0.0/16, koji imaju odredišni TCP port u rasponu od 200 do 10000, u mreži 10.0.0/8:

```
src host 10.4.1.12 or src net 10.6.0.0/16) and tcp dst portrange 200-10000 and dst net 10.0.0.0/8
```

## Filtri kod prikaza

Display filtre koristimo kako bi filtrirali i tražili specifične podatke unutar podataka koje smo prikupili Capture filtrom. Za razliku od Capture filtra, ne moramo ponovo započinjati sesiju prikupljanja podataka u slučaju kada želimo promijeniti filter. Display filtri vidljivi su osnovnom sučelju Wiresharka; unose se u polje `Filter:`.

Wireshark pruža moćan jezik za filtriranje pomoću kojeg možemo stvoriti kompleksne izraze za napredno pretraživanje i analiziranje prikupljenog ('uhvaćenog') mrežnog prometa.

### Način rada filtera kod prikaza

Sintaksa filtra kod prikaza uključuje: protokol, izraz1, izraz2, operator usporedbe, vrijednost, logičku operaciju i ostalo.

**Protokol** ima kao dozvoljenu vrijednost sve protokole locirane između drugog i sedmog sloja OSI modela. Podržane protokole moguće je pronaći pod opcijom `Supported Protocols` u izborniku `Internals` (Mi ćemo se ovdje ograničiti na nekoliko najčešće korištenih protokola.). Pregled dozvoljenih vrijednosti i detaljan opis mogućnost filtara kod prikaza moguće je naći u [referentnom priručniku za filtre kod prikaza](https://www.wireshark.org/docs/dfref/). Primjer protokola je `ftp`, `ip`, `ssh`.

**Izraz1**, **izraz2** su opcionalne kategorije pod-protokola (kategorije unutar protokola). Primjerice, za protokol TCP postoje pod-kategorije povezane s poljima zaglavlja: `tcp.port`, `tcp.ack`, `tcp.flags`, itd.

**Operator usporedbe** ima dozvoljene vrijednosti: `eq`, `ne`, `qt`, `lt`, `ge` i `le` (`==`, `!=`, `>`, `<`, `>=`, `<=`).

| Riječ | Simbol | Značenje |
| ----- | ------ | -------- |
| `eq` | `==` | jednako |
| `ne` | `!=` | nije jednako |
| `gt` | `>` | veće od |
| `lt` | `<` | manje od |
| `ge` | `>=` | veće ili jednako |
| `le` | `<=` | manje ili jednak |

**Logička operacija** ima dozvoljene vrijednosti: `and`, `or`, `xor`, `not` (`&&`, `||`, `^^`, `!`).

| Riječ | Simbol | Značenje |
| ----- | ------ | -------- |
| `and` | `&&` | logičko AN |
| `or` | `||` | logičko OR |
| `xor` | `^^` | logičko XO |
| `not` | `!` | logičko NO |

### Primjeri filtara kod prikaza

Za pregled paketa sa određenog protokola (npr. ICMP), navodimo ime tog protokola:

```
icmp
```

Također, umjesto navođenja imena protokola (npr. SMTP), možemo navesti `tcp.port`, operator usporedbe, te broj željenog porta:

```
tcp.port == 25
```

Za pregled paketa s određenom izvorišnom ili odredišnom IP adresom, navodimo `ip.addr`, operator usporedbe i željenu adresu:

```
ip.addr == 10.0.0.6
```

Treba naglasiti da je `ip.addr == 10.0.0.6` ekvivalentno:

```
ip.src == 10.0.0.6 or ip.dst == 10.0.0.6
```

Operatorima usporedbe moguće je definirati različite zahtjeve, poput npr. paketa s određenom veličinom okvira:

```
frame.len < 128
```

Prikaz FTP ili SSH ili Telnet prometa:

```
ftp || ssh || telnet
```

Prikaz paketa sa izvornom ili odredišnom IP adresom 192.168.1.5:

```
ip.addr == 192.168.1.5
```

Prikaz paketa sa izvorišnom IP adresom 192.168.1.2 i odredišnom IP adresom različitom od 192.168.1.8:

```
ip.src == 192.168.1.2 and ip.dst != 192.168.1.8
```

Prikaz paketa koji nemaju izvorišnu adresu 10.1.1.2:

```
ip.src != 10.1.1.2
```

Prikaz paketa koji su imaju TCP izvorni ili odredišni port 22:

```
tcp.port == 22
```

Prikaz paketa sa TCP zastavicama:

```
tcp.flags
```

Prikaz paketa koji imaju TCP veličinu prozora jednaku 0:

```
tcp.window_size == 0
```

Prikaz paketa s određenog računala (tj. s određene mrežne kartice):

```
eth.addr == 00:26:9e:78:3e:36
```
