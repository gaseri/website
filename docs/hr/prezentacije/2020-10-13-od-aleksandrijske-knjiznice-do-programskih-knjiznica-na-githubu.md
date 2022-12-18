---
marp: true
author: Vedran Miletić
title: Od Aleksandrijske knjižnice do programskih knjižnica na GitHubu
description: Predavanje na Otvorenom danu sveučilišnih odjela 2020
keywords: alexandria, github, archive program
theme: default
class: _invert
paginate: true
abstract: |
  Aleksandrijska knjižnica bila je jedna od najvećih i najvažnijih knjižnica antičkog svijeta. Točan broj svitaka papirusa koje je knjižnica imala nije poznat, ali procjenjuje se na desetke pa čak i stotine tisuća u doba kad je knjižnični fond bio najveći. Uništenje Aleksandrijske knjižnice često se pogrešno doživljava kao iznenadan događaj, dok je zapravo u pitanju bilo višestoljetno polagano propadanje. Već za dinastije Ptolemejevića provedena je čistka intelektualaca iz Aleksandrije, a zatim pod vlašću Rima opada financiranje i podrška knjižnici te s vremenom i brojnost njenih članova. Uništena je i naposlijetku srušena odredbom pape Teofila I. Aleksandrijskog 391. godine, no današnja Bibliotheca Alexandrina služi istovremeno kao spomen i kao pokušaj obnove. Stoljeće nakon uništenja Aleksandrijske knjižnice slijedi pad Zapadnog Rimskog Carstva i pogoršanje stanja u Zapadnoj Europi u demografskim, kulturološkim i ekonomskim terminima, doba koje Francesco Petrarca naziva mračnim, nasuprot svijetlom dobu klasične antike.

  Današnje knjižnice pohranjuju podatke u digitalnom obliku pa, makar naizgled, postoji neograničena mogućnost pohrane rezervnih kopija i time manja bojazan da će pohranjeni podaci biti uništeni. Ipak, digitalizacija nosi svoje probleme: kroz vrijeme se mijenjaju formati pohranjenih dokumenata, softver koji te formate čita i zapisuje te hardver na kojem se taj softver izvodi, a promjene nisu uvijek takve da zadržavaju kompatibilnost s tehnološkim nasljeđem formata, softvera i hardvera. Pored toga, sami podaci se mogu korumpirati, bilo zbog nenamjernih oštećenja medija na kojem su zapisani ili grešaka softvera koji na njima radi, bilo zbog ciljanog djelovanja zlonamjernih aktera. Uzimajući u obzir te probleme digitalizacije, teoretičari digitalnog knjižničarstva uvode pojam digitalnog mračnog doba koje bi uslijedilo u slučaju gubitka kulturološkog nasljeđa koje je digitalizirano. Prilikom govora o podacima pohranjenim u knjižnicama često se ograničavamo na multimedijski sadržaj: tekst, slike, zvuk i video. Međutim, softver koji svakodnevno koristimo također je skup podataka, konkretno podataka u kojima zapisani postupci kao što su obrada slika u digitalnom obliku, prikaz web stranica, emitiranje komprimiranog audiviozualnog sadržaja visoke rezolucije, izračun energije vezanja molekula, predviđanje putanje planeta oko Sunca, abecedno nizanje popisa učenika i slično.

  Štoviše, u računalnim se znanostima datoteke u koje se spremaju implementacije tih različitih postupaka nazivaju (programskim) knjižnicama. Zahvaljujući uspjehu slobodnog softvera otvorenog kôda posljednjih desetljeća, izvorni je kôd velikog broja računalnih programa i programskih knjižnica dostupan bilo kome za proučavanje i izmjenu, a razvijen je i komercijalno uspješan sustav za pohranu, dijeljenje, označavanje i pretragu izvornog kôda poznat pod nazivom GitHub. Kako znanost teži biti otvorena, značajan dio tih programa su programi koje znanstvenici koriste u istraživačkom radu i onda dijele s ostalima kako bi primili povratnu informaciju i na temelju nje u budućnosti poboljšali postupke koje koriste. GitHub tako sprema znanje o poznatim postupcima u raznim domenama znanosti i zbog toga služi kao knjižnica suvremenog doba.

  Predavanje će govoriti o načinima pohrane znanja o tehnologiji koja podupire suvremenu kulturu i civilizaciju te rizicima koje takva pohrana povlači, kao i postupcima koji bi mogli dovesti do digitalnog mračnog doba.
curriculum-vitae: |
  Vedran Miletić radi kao viši predavač na Odjelu za informatiku Sveučilišta u Rijeci, gdje je nositelj više kolegija iz područja računalnih mreža. Član je Povjerenstva za superračunalne resurse Sveučilišta u Rijeci. Doktorirao je računarstvo na FER-u u Zagrebu, a poslijedoktorsko usavršavanje u području računalne biokemije proveo je na Heidelberškom institutu za teorijske studije u Heidelbergu, Njemačka. Doprinio je nekolicini slobodnosoftverskih projekata razvojem otvorenog koda, a voditelj je razvoja slobodnog softvera otvorenog koda za pristajanje, visokoprotočni virtualni probir i predviđanje načina vezivanja biomolekula RxDock.
---

# Od Aleksandrijske knjižnice do programskih knjižnica na GitHubu

## dr. sc. Vedran Miletić, Odjel za informatiku Sveučilišta u Rijeci

Otvoreni dan sveučilišnih odjela, 13. listopada 2020.

---

## Pad Zapadnog Rimskog Carstva (1/2)

- 285--305. n.e. Diokecijan dijeli Carstvo i dodaje četiri cara pomoćnika, građanski rat nakon abdikacije
- 306--337. n.e. Konstantin pobjeđuje istočnog cara i postaje vladar čitavog Carstva, uspostavlja Konstantinopol, uvodi kršćanstvo
- 360--363. n.e. Julijan pokušava preokrenuti prijelaz na kršćanstvo, ali ne uspijeva i umire u bitci s Partima na istoku

---

## Pad Zapadnog Rimskog Carstva (2/2)

- 379--395. n.e. Teodozije nakratko ujedinjuje Carstvo, ali se opet dijeli nakon njegove smrti među njegovim sinovima
- 401--476. n.e. napadi Vizigota, Vandala, Huna i drugih prvo oslabljuju, a naposlijetku i ruše Zapadno Rimsko Carstvo

---

![bg left 95%](https://upload.wikimedia.org/wikipedia/commons/6/64/Ancientlibraryalex.jpg)

## Knjižnica u Aleksandriji

- Jedna od najvećih i najvažnijih knjižnica antičkog svijeta
- Točan broj svitaka papirusa nije poznat
    - Procjenjuje se na stotine tisuća kad je fond bio najveći

---

![bg right 80%](https://upload.wikimedia.org/wikipedia/commons/6/62/Retrato_de_Julio_C%C3%A9sar_%2826724093101%29_%28cropped%29.jpg)

## Uništenje knjižnice (1/3)

- Postoji teorija da ju je spalio Julije Cezar 48. p.n.e. za vrijeme građanskog rata
    - Spalio je brodove u pristaništu i vatra se proširila na obližnje dijelove

---

## Uništenje knjižnice (2/3)

- Višestoljetno polagano propadanje
    - U vrijeme dinastije Ptolemejevića provedena je čistka intelektualaca iz Aleksandrije
    - Pod vlašću Rima opada financiranje i podrška knjižnici te s vremenom i brojnost njenih članova
    - Ratovi Palmirskog Carstva i Rimskog Carstva za Aleksandriju 270--271. n.e.

---

![bg right 95%](https://upload.wikimedia.org/wikipedia/commons/8/86/Alexandria_-_Pompey%27s_Pillar_-_view_of_ruins.JPG)

## Uništenje knjižnice (3/3)

- Hram Serapejon naposlijetku srušen 391. n.e. odredbom pape Teofila I. Aleksandrijskog

---

## Zapisano znanje je temelj za održavanje civilizacije

- Stoljeće nakon uništenja Aleksandrijske knjižnice slijedi pad Zapadnog Rimskog Carstva
- Pogoršanje stanja u Zapadnoj Europi u terminima:
    - demografije
    - kulture
    - ekonomije
- Francesco Petrarca doba nakon pada Carstva naziva *mračnim*, nasuprot *svijetlom* dobu klasične antike

---

![bg right 95%](https://upload.wikimedia.org/wikipedia/commons/5/5e/Alexandria%27s_Bibliotheca.jpg)

## Bibliotheca Alexandrina

- Služi istovremeno kao spomen i kao pokušaj obnove

---

## Digitalno arhiviranje sadržaja

- Postoji neograničena mogućnost pohrane rezervnih kopija
    - Manja bojazan da će pohranjeni podaci biti uništeni
- Napretkom tehnologije mijenjaju se:
    - Formati pohranjenih dokumenata
    - Programska podrška koja te formate čita i zapisuje
    - Uređaji na kojima se programska podrška izvodi
- Promjene (pre)često uvode nekompatibilnost

---

![bg left 80%](https://upload.wikimedia.org/wikipedia/commons/5/51/Domesday-book-1804x972.jpg)

## Domesday Book

- Inventar engleskih zemalja sastavljen od strane normanskih svećenika 1086. godine
- 900 godina kasnije BBC izdao digitalnu verziju
    - Već 2001. godine postala nečitljiva (trajala 15 godina)

---

![bg right 70%](https://upload.wikimedia.org/wikipedia/commons/6/63/Data_loss_of_image_file.JPG)

## Uništenje arhiviranih podataka

- Podaci se mogu korumpirati
    - Oštećenja fizičkog medija na kojem su zapisani (diskovi, trake i dr.)
    - Greške u programskoj podršci koja s podacima radi
- Ciljano djelovanje zlonamjernih aktera

---

## "Digitalno mračno doba"

- Uslijedilo bi u slučaju gubitka kulturološkog nasljeđa i drugog znanja koje je digitalizirano
- Često se ograničavamo na multimedijski sadržaj: tekst, slike, zvuk i video
    - Programska podrška koju svakodnevno koristimo također je skup podataka (algoritam ~= recept)

---

![bg right 50%](https://upload.wikimedia.org/wikipedia/commons/thumb/d/db/Euclid_flowchart.svg/267px-Euclid_flowchart.svg.png)

## Programska poodrška

- Ima zapisane postupke (algoritme) kao što su:
    - Obrada slika u digitalnom obliku
    - Prikaz web stranica
    - Emitiranje ovog predavanja
    - Izračun energije vezanja molekula
    - Predviđanje putanje planeta oko Sunca
    - Abecedno nizanje popisa učenika

---

![bg left 75%](https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Ogg_vorbis_libs_and_application_dia.svg/800px-Ogg_vorbis_libs_and_application_dia.svg.png)

## Programska knjižnica

U računarstvu i informatici datoteke koje sadrže implementacije algoritama nazivamo programskim knjižnicama (engl. *software library*).

---

## Slobodna programska podrška i otvoreni kôd

- Slobodna programska podrška i otvoreni kôd postali uspješni u posljednjih 20--30 godina
- Izvorni kôd velikog broja računalnih programa i programskih knjižnica dostupan bilo kome za proučavanje i izmjenu

---

![bg right 70%](https://upload.wikimedia.org/wikipedia/commons/9/95/Font_Awesome_5_brands_github.svg)

## GitHub

- Komercijalno uspješan sustav za pohranu, dijeljenje, označavanje i pretragu povijesti promjena izvornog kôda programske podrške

---

## Programska podrška na GitHubu

- operacijski sustavi GNU/Linux i FreeBSD (Netflix, Sony PlayStation)
- web preglednici Chromium (temelj za Google Chrome, Brave, Operu, Microsoft Edge i druge) i Mozilla Firefox
- prevoditelji programskih jezika C/C++ (većina suvremenog znanstvenog softvera), Fortran (većina starijeg znanstvenog softvera), C# (velik broj aplikacija za Windows)
- interpreteri programskih jezika PHP (Facebook, Wikipedia, WordPress), Python (YouTube, Instagram, Disqus, Spotify, Dropbox), Ruby (GitHub, Airbnb, Soundcloud), JavaScript/V8/Node.js (PayPal, LinkedIn, Medium) i drugih
- skup kriptografskih alata OpenSSL
- skup alata za dizajn web stranica Bootstrap
- alati za rad s kriptovalutom Bitcoin

---

## Ariviranje programske podrške u okviru programa GitHub Archive Program

### [archiveprogram.github.com](https://archiveprogram.github.com/)

---

## Znanstvena programska podrška na GitHubu

- Programska podrška koju znanstvenici koriste u istraživačkom radu uglavnom se dijele s ostalim znanstvenicima kao otvoreni kôd, često baš na GitHubu
    - Mogućnost javne provjere implementacije znanstvenih metoda, slično kao objavljeni znanstveni radovi
    - Poboljšanja algoritama i sučelja na temelju povratne informacije korisnika (uglavnom drugih znanstvenika)
- GitHub tako sprema znanje o poznatim postupcima u raznim domenama znanosti i zbog toga služi kao knjižnica suvremenog doba

---

## Zaključak

- Računarstvo i informatika su ugrađeni u temelje suvremenog visokotehnološkog društva
- Gubitak programske podrške
