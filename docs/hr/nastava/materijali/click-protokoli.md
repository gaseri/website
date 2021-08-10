---
author: Vedran Miletić
---

# Korištenje mrežnih protokola u modularnom usmjerivaču Click

[Modularni usmjerivač Click](https://github.com/kohler/click) omogućuje nam korištenje brojnih elemenata koji implementiraju često korištene mrežne protokole. Navedimo redom te elemente za pojedine protokole.

## Ethernet

- `EnsureEther` ([dokumentacija](https://github.com/kohler/click/wiki/EnsureEther)) -- osigurava da su (IP) paketi unutar Ethernet okvira
- `EtherEncap` ([dokumentacija](https://github.com/kohler/click/wiki/EtherEncap)) -- stavlja (IP) paket u Ethernet okvir
- `EtherMirror` ([dokumentacija](https://github.com/kohler/click/wiki/EtherMirror)) -- zamjenjuje međusobno izvorišnu i odredišnu adresu Ethernet okvira
- `EtherRewrite` ([dokumentacija](https://github.com/kohler/click/wiki/EtherRewrite)) -- prepisuje izvorišnu i odredišnu adresu Ethernet okvira novim vrijednostima
- `EtherSwitch` ([dokumentacija](https://github.com/kohler/click/wiki/EtherSwitch)) -- Ethernet preklopnik

### Primjer korištenja Etherneta

!!! todo
    Ovdje nedostaje primjer.

## ARP

- `ARPFaker` ([dokumentacija](https://github.com/kohler/click/wiki/ARPFaker)) -- periodički šalje lažni ARP odgovor
- `ARPPrint` ([dokumentacija](https://github.com/kohler/click/wiki/ARPPrint)) -- ispis podataka o ARP paketu
- `ARPQuerier` ([dokumentacija](https://github.com/kohler/click/wiki/ARPQuerier)) -- stavlja IP paket u Ethernet okvir gdje je odredišna adresa određena pomoću ARP zahtjeva
- `ARPResponder` ([dokumentacija](https://github.com/kohler/click/wiki/ARPResponder)) -- generira odgovor na ARP zahtjev
- `CheckARPHeader` ([dokumentacija](https://github.com/kohler/click/wiki/CheckARPHeader)) -- provjerava ispravnost ARP zaglavlja okvira

## IPv4

- `CheckIPHeader` ([dokumentacija](https://github.com/kohler/click/wiki/CheckIPHeader)) -- provjerava IP zaglavlje
- `DecIPTTL` ([dokumentacija](https://github.com/kohler/click/wiki/DecIPTTL)) -- smanjuje TTL
- `IPClassifier` ([dokumentacija](https://github.com/kohler/click/wiki/IPClassifier)) -- klasificira IP pakete po sadržaju, slična pravila kao Wiresharkovi filtri kod snimanja paketa
- `IPFilter` ([dokumentacija](https://github.com/kohler/click/wiki/IPFilter)) -- filtrira IP pakete po sadržaju, slična pravila kao Wiresharkovi filtri kod snimanja paketa
- `IPFragmenter` ([dokumentacija](https://github.com/kohler/click/wiki/IPFragmenter)) -- fragmentira velike IP pakete
- `IPMirror` ([dokumentacija](https://github.com/kohler/click/wiki/IPMirror)) -- zamjenjuje izvorišnu i odredišnu adresu IP paketa
- `IPPrint` ([dokumentacija](https://github.com/kohler/click/wiki/IPPrint)) -- ispis podataka o IP paketu
- `IPReassembler` ([dokumentacija](https://github.com/kohler/click/wiki/IPReassembler)) -- ponovno sastavlja fragmentirane IP pakete

### Primjer korištenja IPv4

Želimo da usmjerivač provjerava zaglavlje primljenih paketa, sastavlja fragmente IPv4 paketa, filtrira među njima ICMP pakete koje ping koristi (`echo request` i `echo reply`) i ispisuje podatke o njima. Za testiranje iskoristite `ipv4frags.pcap` sa [SampleCaptures na Wiresharkovom Wikiju](https://wiki.wireshark.org/SampleCaptures).

### Rješenje primjera korištenja IPv4

!!! todo
    Ovdje nedostaje rješenje primjera.

## ICMP

- `ICMPPingResponder` ([dokumentacija](https://github.com/kohler/click/wiki/ICMPPingResponder)) -- odgovara na ICMP echo zahtjev
- `ICMPPingSource` ([dokumentacija](https://github.com/kohler/click/wiki/ICMPPingSource)) -- periodički šalje ICMP echo zahtjev

## NAT

- `ICMPPingRewriter` ([dokumentacija](https://github.com/kohler/click/wiki/ICMPPingRewriter)) -- rewrites ICMP echo requests and replies
- `IPAddrPairRewriter` ([dokumentacija](https://github.com/kohler/click/wiki/IPAddrPairRewriter)) -- rewrites IP packets' addresses by address pair
- `IPAddrRewriter` ([dokumentacija](https://github.com/kohler/click/wiki/IPAddrRewriter)) -- rewrites IP packets' addresses
- `IPRewriter` ([dokumentacija](https://github.com/kohler/click/wiki/IPRewriter)) -- rewrites TCP/UDP packets' addresses and ports

## TCP

!!! todo
    Ovaj dio treba napisati.

## UDP

!!! todo
    Ovaj dio treba napisati.
