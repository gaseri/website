---
author: Vedran Miletić
---

# Emulacija bežičnih mreža

Emulator CORE nudi emulaciju bežičnih mreža koje rade po standardu [IEEE 802.11](https://en.wikipedia.org/wiki/IEEE_802.11), poznatijih pod komercijalnim nazivom [Wi-Fi](https://en.wikipedia.org/wiki/Wi-Fi). Za tu svrhu među čvorovima veznog sloja (gumb `link-layer nodes` u alatnoj traci s lijeve strane) moguće je odabrati `wireless LAN` ([dokumentacija](https://coreemu.github.io/core/gui.html#network-nodes)).

## Konfiguracija čvora bežične mreže

Nakon dodavanja čvora tipa `wireless LAN` na platno stvara se novi čvor imena `wlanN`, pri čemu je `N` redni broj čvora ([dokumentacija](https://coreemu.github.io/core/gui.html#wireless-networks)). Desnim klikom na čvor nudi nam se mogućnost konfiguracije (`Configure`). U konfiguraciji čvora na kartici `Basic` možemo postaviti sljedeće:

- Doseg (`Range`), izražen u pikselima (pretvorbu piksela i metara te veličinu platna na kojem crtamo mrežu moguće je konfigurirati pod stavkom izbornika `Canvas` pa onda `Size/scale`)
- Širinu pojasa (`Bandwidth`), izraženu u bitovima po sekundi
- Zadržavanje (`Delay`), izraženo u mikrosekundama
- Postotak gubitaka (`Loss`)
- Podrhtavanje zadržavanja (`Jitter`), izraženo u mikrosekundama

Kartica `EMANE` omogućuje nam omogućuje integraciju CORE-a s emulatorom [Extendable Mobile Ad-hoc Network Emulator (EMANE)](https://www.nrl.navy.mil/Our-Work/Areas-of-Research/Information-Technology/NCS/EMANE/). Time se nećemo baviti.

Također možemo postaviti IPv4 podmrežu (`IPv4 subnet`) i IPv6 podmrežu (`IPv6 subnet`).

!!! warning
    Kod dodavanja čvora bežične mreže CORE će dodijeliti IPv4 podmrežu s duljinom prefiksa 32 (npr. 10.0.0.0/32). To je pogrešno (i vjerojatno je [bug u CORE-u](https://github.com/coreemu/core/issues)) jer u toj podmreži postoji samo jedna adresa pa nema mjesta za adrese domaćina, računala i usmjerivača. Svaki put kad emuliramo Wi-Fi morat ćemo urediti ovu postavku svake bežične mreže koju koristimo i postaviti podmrežu koju već želimo koristiti.

Naposlijetku, imamo još tri opcije:

- Korištenje skripti mobilnosti [mrežnog simulatora ns-2](https://www.isi.edu/nsnam/ns/) (`ns-2 mobility script`), čime ćemo se baviti nešto kasnije
- Povezivanje sa svim usmjerivačima u emulaciji (`Link to all routers`)
- Odabir članova bežične mreže (`Choose WLAN members`)

## Primjer bežične mreže

Složimo jednostavnu bežičnu mrežu `wlan1` u kojoj se nalaze domaćin `n2` i računalo `n3`. Postavimo doseg čvorova na 300 piksela (450 metara) i IPv4 podmrežu na 192.168.10.0/24. Topologija mreže je oblika

```
n2 ----- wlan1 ----- n3
```

Čvorove `n2` i `n3` povežimo na `wlan1` i postavimo ih na platnu negdje blizu te mreže. Uočimo kako su povezivanjem na bežičnu mrežu `n2` i `n3` dobili po jednu antenu, što je oznaka bežične veze. Nakon pokretanja emulacije čvorove možemo možemo se ping-anjem uvjeriti da bežična veza između čvorova radi.

Za vrijeme izvođenja emulacije čvorove je moguće pomicati po želji pa ćemo, ako dovoljno odmaknemo čvorove, nakon nekog vremena vidjeti izlaz oblika

``` shell
# ping 192.168.10.20
PING 192.168.10.20 (192.168.10.20) 56(84) bytes of data.
64 bytes from 192.168.10.20: icmp_seq=1 ttl=64 time=0.118 ms
64 bytes from 192.168.10.20: icmp_seq=2 ttl=64 time=0.077 ms
64 bytes from 192.168.10.20: icmp_seq=3 ttl=64 time=0.071 ms
64 bytes from 192.168.10.20: icmp_seq=4 ttl=64 time=0.056 ms
64 bytes from 192.168.10.20: icmp_seq=5 ttl=64 time=0.073 ms
64 bytes from 192.168.10.20: icmp_seq=6 ttl=64 time=0.071 ms
64 bytes from 192.168.10.20: icmp_seq=7 ttl=64 time=0.073 ms
64 bytes from 192.168.10.20: icmp_seq=8 ttl=64 time=0.061 ms
From 192.168.10.10 icmp_seq=39 Destination Host Unreachable
From 192.168.10.10 icmp_seq=40 Destination Host Unreachable
From 192.168.10.10 icmp_seq=41 Destination Host Unreachable
From 192.168.10.10 icmp_seq=42 Destination Host Unreachable
```

Nakon vraćanja čvorova u međusobni doseg izlaz postaje

```
From 192.168.10.10 icmp_seq=312 Destination Host Unreachable
From 192.168.10.10 icmp_seq=313 Destination Host Unreachable
From 192.168.10.10 icmp_seq=314 Destination Host Unreachable
From 192.168.10.10 icmp_seq=315 Destination Host Unreachable
64 bytes from 192.168.10.20: icmp_seq=316 ttl=64 time=2080 ms
64 bytes from 192.168.10.20: icmp_seq=317 ttl=64 time=1040 ms
64 bytes from 192.168.10.20: icmp_seq=318 ttl=64 time=0.119 ms
64 bytes from 192.168.10.20: icmp_seq=319 ttl=64 time=0.074 ms
```

## Međusobno povezivanje bežičnih mreža

Usmjerivači povezani na više bežičnih (ili žičnih) mreža omogućit će komunikaciju među tim mrežama. Specijalno, unutar CORE-a takvi će usmjerivači na platnu biti prikazani s više antena kada su spojeni na više bežičnih mreža. Iako postoje specijalizirani algoritmi usmjeravanja za bežične mreže, dosad korišteni algoritmi usmjeravanja će uredno raditi bez obzira radi li se na veznom i fizičkom sloju o žičnim ili bežičnim mrežama.
