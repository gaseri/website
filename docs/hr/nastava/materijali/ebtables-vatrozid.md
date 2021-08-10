---
author: Vedran Miletić, Domagoj Margan
---

# Filtriranje okvira vatrozidom na veznom sloju

## Pojam vatrozida i proces filtriranja okvira vatrozidom

[Vatrozid](https://en.wikipedia.org/wiki/Firewall_(computing)) (engl. *firewall*) je uređaj ili aplikacija koja služi da dozvoli ili zabrani prijenos podataka mrežom na osnovu određenog skupa pravila. Na veznom sloju koristi se za istovremeno dozvoljavanje legitimnog i onemogućavanje nelegitimnog pristupa unutar lokalnih mreža.

Pravila koja se koriste kategoriziraju okvire na temelju njihovih svojstava. Primjerice, unutar lokalne mreže tipa Ethernet može se zabraniti komunikacija svim protokolima mrežnog sloja osim ARP-a i IPv4 ili se može na preklopniku (engl. *switch*) bilježiti sve okvire proslijeđene određenom čvoru.

Mi ćemo se fokusirati na [ebtables](https://ebtables.netfilter.org/), koji je, uz njegovu nešto moderniju alternativu [nftables](https://netfilter.org/projects/nftables/), najkorišteniji vatrozid na veznom sloju na Linuxu. Oba se razvijaju u okviru projekta [netfilter](https://netfilter.org/) koji je dio [jezgre Linuxa](https://www.kernel.org/) od verzije 2.4 nadalje.

## Način rada alata ebtables

!!! caution
    Ebtables radi samo na premoštenim mrežnim adapterima (engl. *bridged network adapter*). Unutar CORE-a, ebtables radi samo na čvorovima tipa `router` i složenijim jer čvorovi tipa `switch` nemaju ljusku. Iskoristit ćemo čvorove tipa `router` da stvorimo svojevrsni preklopnik (engl. *switch*) koji ima ljusku. Prvo ćemo u konfiguraciji čvora (desni klik na čvor pa `Configure`) pod `Services` isključiti Quaggine usluge `OSPFv2`, `OSPFv3` i `zebra` te `IPForward`. Zatim ćemo prilagoditi sve adrese da budu dio iste mreže.

    Primjerice, stvorili smo emulaciju mreže oblika `n1 -- n2 -- n3` gdje su `n1` i `n3` čvorovi tipa `pc`, a n2 je čvor tipa `router` s isključenim navedenim uslugama. CORE će stvoriti dvije mreže (n1 -- n2 s adresama 10.0.0.0/24 i n2 -- n3 s adresama 10.0.1.0/24), a mi ćemo prilagoditi postavke n2 i n3 tako da i oni budu dio mreže 10.0.0.0/24. Konkretno, postavit ćemo da `n2` na sučelju prema `n3` ima adresu 10.0.0.2, a `n3` ima adresu 10.0.0.11.

    Naposlijetku, iskoristit ćemo naredbu `brctl` za dodavanje mrežnih adaptera u most, a zatim ćemo mu naredbom `ifconfig` dodijeliti dotad nekorištenu mrežnu adresu koja je unutar mreže unutar koje se nalaze oba adaptera.

    Primjerice, ako su na usmjerivaču na kojem radimo mrežni adapteri `eth0` i `eth1` oni na kojima želimo uvesti filtriranje prometa ebtablesom i oni imaju dodijeljene IP adrese 10.0.0.1 i 10.0.0.2, tada ćemo stvoriti most `br0` s adresom 10.0.0.3 i maskom podmreže 255.255.255.0, odnosno /24 nizom naredbi:

    ``` shell
    # brctl addbr br0
    # brctl addif br0 eth0
    # brctl addif br0 eth1
    # ifconfig br0 10.0.0.3 255.255.255.0
    ```

    Ukoliko posljednja naredba vrati grešku `SIOCSIFADDR: Invalid argument`, napišimo je u ekvivalentnom obliku:

    ``` shell
    # ifconfig br0 10.0.0.3/24
    ```

    Već poznatom naredbom `netstat -ie` možemo se uvjeriti da smo dobro dodijelili adresu.

    Analogno je moguće ovaj postupak provesti za više od dvije mreže.

Alat ebtables koristimo kako bi postavili, mijenjali i pregledavali tablice pravila za filtriranje Ethernet okvira. Moguće je definirati više različitih tablica. Svaka tablica sadrži određen broj različitih lanaca (engl. *chains*), a svaki je lanac lista pravila kojima se određuje što sustav čini sa okvirima koje to pravilo obuhvaća.

Alatu pristupamo putem ljuske naredbom `ebtables`. U ljusci upisujemo `ebtables` konstrukte narebi i parametara, baš kao što to radimo i sa već poznatim naredbama i alatima unutar komandne linije. Kao i druge naredbe koje se bave konfiguracijom mreže, jednako dobro radi na stvarnim i emuliranim čvorovima. Vrijedi napomenuti da se kod ponovnog pokretanja stvarnog ili emuliranog računala postavke brišu.

### Tablice, lanci pravila i ciljevi

Alat ebtables za ostvarivanje različitih akcija nad okvirima koristi iduće tablice (engl. *tables*):

- (`raw`),
- `filter` -- vrši filtriranje okvira,
- `nat` -- vrši prevođenje adresa,
- `broute` -- vrši premošćenje (engl. *bridging*) i usmjeravanje (engl. *routing*).

Ebtables svrstava mrežni okvir prema tome gdje je nastao, odnosno za koji domaćin je namijenjen. S obzirom na to, postoje idući lanci (engl. *chains*) pravila:

- `INPUT` -- okviri pristigli mreže namijenjeni za ovaj čvor,
- `OUTPUT` -- okviri nastali na čvoru namijenjeni za slanje,
- `FORWARD` -- okviri pristigli s mreže namijenjeni za prosljeđivanje.

Načini baratanja okvirom nazivaju se ciljevima (engl. *targets*):

- `ACCEPT` -- prihvaća okvir,
- `DROP` -- odbacuje okvir,
- `CONTINUE` -- provjerava iduće pravilo (korisno za izvedbu složenih pravila),
- `RETURN` -- prestaje provjeravati pravila koja slijede u lancu.

### Pozivanje naredbe

Sintaksa alata `ebtables` je `ebtables -[ADI] chain rule-specification [options]`. Neke od najvažnijih naredbi u alatu `ebtables`, kojima određujemo uvjete za izvađanje željenih akcija:

- `-A`, `--append chain rule-specification`

    - dodavanje jednog ili više pravila na kraj odabranog lanca

- `-D`, `--delete chain rule-specification`

    - brisanje pravila iz odabranog lanca

- `-I`, `--insert chain rule-specification`

    - dodavanje jednog ili više pravila na početak odabranog lanca

- `-L`, `--list [chain]`

    - izlistavanje svih pravila odabranog lanca
    - ako lanac nije naveden, izlistavaju se sva pravila

- `-F`, `--flush [chain]`

    - brisanje cijelog lanca pravila
    - ukoliko lanac nije naveden, brišu se svi lanci

Neki od najvažnijih općenitih parametara u alatu `ebtables`:

- `-t table`

    - pozivanje određene tablice
    - zadana tablica je `filter`; ukoliko se ne navede druga tablica za korištenje, koristi se tablica `filter`.

- `-j target`

    - određivanje cilja

- `-p`, `--protocol protocol`

    - određivanje protokola mrežnog sloja
    - dozvoljene vrijednosti: [EtherType](https://en.wikipedia.org/wiki/EtherType) = 16-bitni broj u heksadekadskom zapisu (npr. `0x0800` za IPv4, `0x0806` za ARP), ime protokola (npr. `IPv4`, `ARP`), ili `LENGTH` za okvire koji u polju EtherType imaju navedenu duljinu

## Način korištenja značajki filtriranja okvira

Lanci koje ebtables koristi za filtriranje okvira su:

- `INPUT`,
- `OUTPUT`,
- `FORWARD`.

Ciljevi koje ebtables koristi za filtriranje okvira su:

- `ACCEPT`,
- `DROP`.

Elementi tablice filter svakog lanca su pravila koja pridružuju cilj okvirima koji imaju određenu izvorišnu IPv4 ili MAC adresu, odredišnu IPv4 ili MAC adresu, protokol i/ili nešto drugo. Poredak elemenata tablice filter u svakom lancu je značajan, odnosno kod obrade okvira uzima se prvo navedeno pravilo koje odgovara i ne gledaju se preostala. Ukoliko nijedno pravilo ne odgovara uzima se zadani cilj za taj lanac.

Neki od najvažnijih parametara u alatu `ebtables` koje koristimo pri filtriranju okvira:

- `-s`, `--src address[/mask]` -- izvorišna MAC adresa s opcionalnom maskom
- `-d`, `--dst address[/mask]` -- odredišna MAC adresa s opcionalnom maskom
- `--ip-src`, `--ip-source address[/mask]` -- izvorišna IP adresa
- `--ip-dst`, `--ip-destination address[/mask]` -- odredišna IP adresa
- `--arp-ip-src address[/mask]` -- izvorišna IP adresa u ARP paketu
- `--arp-ip-dst address[/mask]` -- odredišna IP adresa u ARP paketu
- `--arp-mac-src address[/mask]` -- izvorišna MAC adresa u ARP paketu
- `--arp-mac-dst address[/mask]` -- odredišna MAC adresa u ARP paketu
- `--arp-opcode opcode` -- tip ARP paketa (1 = Request, 2 = Reply)

## Primjeri filtriranja okvira vatrozidom

Na mreži na slici računala n1, n2 i n3 ostvaruju vezu sa domaćinima n7 i n10 i računalom n9 putem usmjerivača n5 i n6 te preklopnika n4 i n8. Adresa usmjerivača n5 na sučelju prema n4 je 10.0.0.1, a na sučelju prema n6 je 10.0.1.1. IPv4 adrese n1, n2 i n3 su redom 10.0.0.20, 10.0.0.21 i 10.0.0.22, a adrese n7, n10 i n9 su redom 10.0.2.20, 10.0.3.20 i 10.0.3.21. Za potrebe zadatka pretpostavimo da je međusobna komunikacija između svih čvorova moguća. Vatrozid postoji na usmjerivaču n5 i na računalu n1.

```
    n1             n7   n10
     \             /    /
      \           /    /
n2----n4----n5---n6---n8
      /                \
     /                  \
    n3                  n9
```

Za generiranje prometa koristit ćemo MGEN na isti način kao i do sada. Za svaki navedeni primjer generiranjem odgovarajućeg prometa i hvatanjem prometa putem alata Wireshark ispitajmo preneseni promet prije i nakon unosa `ebtables` naredbe.

### Primjer 1

Želimo da računalo n1 ima vatrozid koji odbacuje sve okvire koji imaju EtherType (tip) AppleTalk (vrijednost 0x809B). Vrijednosti različitih tipova možemo po potrebi pronaći na [IANA-inim stranicama o standardu IEEE 802](https://www.iana.org/assignments/ieee-802-numbers/ieee-802-numbers.xhtml#ieee-802-numbers-1) ili u datoteci `/etc/ethertypes`.

``` shell
$ cat /etc/ethertypes
#
# Ethernet frame types
#     This file describes some of the various Ethernet
#     protocol types that are used on Ethernet networks.
#
# This list could be found on:
#         http://www.iana.org/assignments/ethernet-numbers
#         http://www.iana.org/assignments/ieee-802-numbers
#
# <name>    <hexnumber> <alias1>...<alias35> #Comment
#
IPv4        0800    ip ip4      # Internet IP (IPv4)
X25         0805
(...)
```

Pravilo odbacivanja dolaznih okvira dodat ćemo naredbom ljuske na n1:

``` shell
# ebtables -A INPUT -p 0x809B -j DROP
```

Alternativno mogli smo protokol navesti riječju. Naredba je tada:

``` shell
# ebtables -A INPUT -p ATALK -j DROP
```

### Primjer 2

Uvjerimo se da pravila uistinu rade. Stvorimo bilo kakav promet prema n1 od bilo kuda i uvjerimo se Wiresharkom da promet stiže do n1. Zatim uvedimo odbacivanje svih dolaznih okvira na n1 naredbom ljuske na n1:

``` shell
# ebtables -P INPUT DROP
```

Provjerimo primljeni promet u Wiresharku. Zatim vratimo zadani cilj na prihvaćanje dolaznih okvira naredbom:

``` shell
# ebtables -P INPUT ACCEPT
```

Ponovno provjerimo primljeni promet u Wiresharku.

### Primjer 3

Želimo da usmjerivač n5 odbacuje sve IPv4 okvire namijenjene za n1. Stvorimo bilo kakav promet prema n1 od bilo kuda i uvjerimo se Wiresharkom da promet stiže do n1.

Pokrenemo ljusku na usmjerivaču n5 te upišemo:

``` shell
# ebtables -A FORWARD -p IPv4 --ip-dst 10.0.0.20 -j DROP
```

Provjerimo primljeni promet u Wiresharku. Stvorimo sad bilo kakav promet prema n2 od bilo kuda i uvjerimo se da, za razliku od prometa prema n1, taj promet zaista stiže do n2.

Alternativno, promet prema n1 mogli smo blokirati prema njegovoj MAC adresi umjesto prema IPv4 adresi. Primjerice, ako je MAC adresa n1 00:11:22:33:44:55, naredba bi bila:

``` shell
# ebtables -A FORWARD --dst 00:11:22:33:44:55 -j DROP
```

Uočimo kako blokiranje prometa prema IPv4 adresi blokira samo promet koji koristi IPv4, dok blokiranje prometa prema MAC adresi blokira sav promet (npr. ARP i IPv6).

### Primjer 4

Želimo da usmjerivač odbacuje sve IPv4 pakete koje n3 šalje prema n9. Stvorimo odgovarajući promet i uvjerimo se Wiresharkom da promet stiže do n9.

U ljusku na n5 upisujemo:

``` shell
# ebtables -A FORWARD -p IPv4 --ip-src 10.0.0.22 --ip-dst 10.0.3.21 -j DROP
```

Provjerimo primljeni promet u Wiresharku. Stvorimo sad bilo kakav promet od n3 do n10 i uvjerimo se da, za razliku od prometa prema n9, taj promet zaista stiže do n10.

### Primjer 5

Vrlo stara računala koja implementiraju ranu verziju Etherneta koriste polje EtherType kao duljinu okvira. U modernim Ethernet mrežama takvih okvira ne bi trebalo biti te, ako se javljaju, uglavnom se radi o zlonamjernim okvirima. Blokirajmo ih na n1 naredbom ljuske:

``` shell
# ebtables -A INPUT -p LENGTH -j DROP
```

### Primjer 6

Recimo da n1 ne želi primati ARP zahtjeve. Blokirat ćemo ih na n1 naredbom ljuske:

``` shell
# ebtables -A INPUT -p ARP --arp-opcode 1 -j DROP
```

Ako pak n1 ne želi slati odgovore na ARP zahtjeve, naredba ljuske je:

``` shell
# ebtables -A OUTPUT -p ARP --arp-opcode 2 -j DROP
```
