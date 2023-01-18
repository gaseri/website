---
marp: true
author: Vedran Miletić
title: Prošlost, sadašnjost i budućnost weba kao platforme za razvoj aplikacija i usluga
description: Razvoj web aplikacija i usluga
keywords: razvoj web aplikacija usluga
theme: default
class: _invert
paginate: true
---

# Prošlost, sadašnjost i budućnost weba kao platforme za razvoj aplikacija i usluga

## doc. dr. sc. Vedran Miletić, vmiletic@inf.uniri.hr, [vedran.miletic.net](https://vedran.miletic.net/)

### Fakultet informatike i digitalnih tehnologija Sveučilišta u Rijeci, akademska 2021./2022. godina

---

## Povijest interneta

- [The Living Internet](https://www.livinginternet.com/)
- [Serverless Architectures Review, Future Trend and the Solutions to Open Problems](http://pubs.sciepub.com/ajse/6/1/1/index.html)
    - [Figure 1. Evolution from Bare metal to Serverless (adapted from \[1\])](http://pubs.sciepub.com/ajse/6/1/1/bigimage/fig1.png)

---

## Važne epizode u povijesti weba

- [A Brief History of Web Development](https://devdojo.com/tnylea/a-brief-history-of-web-development)
- [Ratovi web preglednika](https://youtu.be/pz73gD1H-s4)
- [Microsoft is dead](http://www.paulgraham.com/microsoft.html)

---

## Budućnost razvoja softvera

- [GitHub Copilot](https://github.com/features/copilot/) ([proširenje za Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot))

---

## Web 1.0

- statičke stranice, namijenjene samo za čitanje
- stranice kompanija i pojedinaca, npr. slike prostora proizvodne linije
- reklame u obliku bannera oko sadržaja
- sadržaj organiziran u direktorije
- web forme (koje npr. šalju e-mail vlasniku stranice) omogućuju interakciju
    - realizirane korištenjem Common Gateway Interfacea (CGI), prvotno u jezicima C i Perl, kasnije u jeziku PHP
- stranice namijenjene za pregledavanje od strane korisnika za računalom
- vlastita infrastruktura

---

## Web 2.0

- stranice za čitanje i pisanje, korisnici koji konzumiraju sadržaj mogu ga i stvarati
- stranice orijentirane na stvaranje zajednice, npr. blog ili wiki
- interaktivne reklame, npr. story na Instagramu, objava na FB
- sadržaj organiziran po (hash)tagovima
- web aplikacije umjesto web formi
    - jezici i okviri za razvoj postaju puno jednostavniji za korištenje
- stranice namijenjene za pregledavanje od strane korisnika na brojnim uređajima, ali i botova (npr. tražilica)
- oblak

---

## Web 3.0

- semantički web: razumijevanje riječi
- osim korisnika, sadržaj generira umjetna inteligencija prema potrebama korisnika
- decentralizacija, npr. kroz *edge computing*
- interoperabilnost
- kriptovalute, *blockchain*
- 3D grafika (VR/AR)
- svepristuan web: pristup webu kroz sve uređaje

---

## Primjeri web 3.0 aplikacija

- [Siri](https://www.apple.com/siri/)
- [Wolfram Alpha](https://www.wolframalpha.com/)
- [Sapien](https://www.sapien.network/) ([Ethereum](https://ethereum.org/))
- [Steemit](https://steemit.com/) ([Steem](https://steem.com/))
- [IDEX](https://idex.io/)
- [Obsidian](https://obsidian.md/)
- [ySign](https://ysign.app/)
- [Filecoin](https://filecoin.io/)
- [Storj](https://www.storj.io/)
- [Sia](https://sia.tech/)
- [IPFS](https://ipfs.io/)

---

## Primjeri web 3.0 aplikacija (nast.)

- [Cashaa](https://cashaa.com/)
- [Everledger](https://everledger.io/)
- [LivePeer](https://livepeer.com/)
- [LBRY](https://lbry.com/); implementacija: [odysee](https://odysee.com/)
- [Invidious](https://invidious.io/)
- [CryptoTask](https://www.cryptotask.org/)
- [Atlas.Work](https://www.atlas.work/)
- [Sapien](https://www.sapien.network/)
- [Brave](https://brave.com/)
- [Beaker Browser](https://beakerbrowser.com/)

---

## Zaključak

- Web 1.0 -> Web 2.0 -> Web 3.0
- Popularnost kriptovaluta
- Distribuirani sustavi bez jednog mjesta gdje se sustav može uništiti
- Evolucija umjesto revolucije

---

## Epilog

> Dinamičke web aplikacije 2 su sintetski predmet: intenzivno se koriste znanja iz predmeta *Objektno orijentirano programiranje*, *Uvod u baze podataka*, *Operacijski sustavi 1*, *Operacijski sustavi 2* i *Računalne mreže 2*, a srodne teme obrađuju na predmetima *Objektno orijentirano modeliranje* i *Uvod u programsko inženjerstvo*

Na predavanjima smo obradili:

- Razvoj stražnjeg dijela web aplikacije (monolit, mikroservisi)
- Korištenje objektno orijentiranog modeliranja i programiranja na webu
- Povezivanje s bazom podataka, pretvorba objektnog u relacijski model
- Faktori razvoja koji olakšavaju postavljanje i održavanje aplikacije
- Testiranje i automatizacija testiranja u sustavu kontinuirane integracije
- Poboljšanje performansi i izazovi sigurnosti aplikacije
