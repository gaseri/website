---
marp: true
author: Vedran Miletić
title: Zašto i kako izraditi web sjedište istraživačke grupe
description: Predavanje na Mutimiru 2023 u organizaciji Udruge Penkala
keywords: web site, markdown, mkdocs, material for mkdocs
theme: default
class: _invert
paginate: true
abstract: |
  Postali ste novi voditelj istraživačke grupe, ili, zašto ne, (poslije)doktorand ste i njezin entuzijastični član, i želite poboljšati vidljivost vaše grupe među kolegama u području. Institucijska web sjedišta su često ograničena u formi i sadržaju, dok izrada vlastitog web sjedišta omogućuje oblikovanjei organizaciju sadržaja prema potrebi te postavljanje materijala po želji.
---

# Zašto i kako izraditi web sjedište istraživačke grupe

## Dr. [Vedran](https://vedran.miletic.net/) [Miletić](https://www.miletic.net/), Računarsko i podatkovno postrojenje Max Plancka; Fakultet informatike i digitalnih tehnologija, Sveučilište u Rijeci

![GASERI logo](../../images/gaseri-logo-text.png)

### Mutimir 2023 (Udruga Penkala), Hotel Matija Gubec, 28. prosinca 2023.

---

![MPG logo h:250px](https://upload.wikimedia.org/wikipedia/en/9/9a/Max_Planck_Society_logo.svg)

Izvor slike: [Wikimedia Commons: File:Max Planck Society logo.svg](https://commons.wikimedia.org/wiki/File:Max_Planck_Society_logo.svg)

![FIDIT logo h:250px](https://upload.wikimedia.org/wikipedia/commons/1/14/FIDIT-logo.svg)

Izvor slike: [Wikimedia Commons File:FIDIT-logo.svg](https://commons.wikimedia.org/wiki/File:FIDIT-logo.svg)

---

## Predstavljanje predavača

* docent i voditelj [grupe](../index.md), [Fakultet informatike i digitalnih tehnologija](https://www.inf.uniri.hr/), [Sveučilište u Rijeci](https://uniri.hr/)
* na dopustu od godinu dana zbog poslijedoktorskog usavršavanja na [Max Planck Computing and Data Facility](https://www.mpcdf.mpg.de/) (nekad: [Rechenzentrum Garching](https://www.mpg.de/mpcdf-de))
    * razvoj znanstvenog softvera za superračunala i računalne oblake
    * simulacija molekulske dinamike biomolekula

![MPCDF logo](https://www.mpcdf.mpg.de/assets/institutes/headers/mpcdf-desktop-en-bc7963e480b5b24ed4797d156e680a45658f3ec11384599af70c07eca1002285.svg)

Izvor slike: [Max Planck Computing and Data Facility](https://www.mpcdf.mpg.de/assets/institutes/headers/mpcdf-desktop-en-bc7963e480b5b24ed4797d156e680a45658f3ec11384599af70c07eca1002285.svg)

---

Izvor slike: [Wikimedia Commons File:110716031-TUM.JPG](https://commons.wikimedia.org/wiki/File%3A110716031-TUM.JPG)

![TUM Campus bg right:70%](https://upload.wikimedia.org/wikipedia/commons/2/2e/110716031-TUM.JPG)

---

## Znate li izraditi web sjedište i u kojoj tehnologiji?

![Ključni slojevi interneta bg left:60% 110%](https://upload.wikimedia.org/wikipedia/commons/3/39/Internet_Key_Layers.png)

Izvor slike: [Wikimedia Commons File:Internet Key Layers.png](https://commons.wikimedia.org/wiki/File:Internet_Key_Layers.png)

---

## Izrada web sjedišta

* najčešći posao kojim se bave programski inženjeri u suvremeno doba
    * budžet po web sjedištu kreće se od nekoliko stotina do milijuna EUR
    * česta zabluda: to što je netko u području računarstva ne znači da se bavi web tehnologijama
* može i "uradi sam": [Wix.com](https://www.wix.com/), [Weebly](https://www.weebly.com/), [Squarespace](https://www.squarespace.com/), [WordPress](https://wordpress.com/), [Webflow](https://webflow.com/)...
* stotine pristupa, tehnologija, platformi, rješenja, novi dolaze svaki dan

![Proces razvoja softvera bg right:40% 95%](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Three_software_development_patterns_mashed_together.svg/1081px-Three_software_development_patterns_mashed_together.svg.png)

Izvor slike: [Wikimedia Commons File:Three software development patterns (...).svg](https://commons.wikimedia.org/wiki/File:Three_software_development_patterns_mashed_together.svg)

---

## Zašto izraditi web sjedište istraživačke grupe? (1/2)

* nema li svaka istraživačka grupa već svoje mjesto na web sjedištu fakulteta/istituta/odjela/centra?
    * primjeri: [FIDIT](https://www.inf.uniri.hr/), [HITS](https://www.h-its.org/), [MPI-NAT](https://www.mpinat.mpg.de/) i [KTH](https://www.kth.se/)
* nisu li društvene mreže korisnije za promociju svojeg znanstvenog rada?
    * "tko danas gleda web, zabava je na Instagramu i TikToku, posao je na LinkedInu, a ni Facebook nije za baciti"
* isplati li se uopće ulagati ograničeno i dragocjeno vrijeme koje imamo za znanost u marketing?

![Web sjedište Antarktičkog programa bg left:30% 90%](https://upload.wikimedia.org/wikipedia/commons/8/87/United_States_Antarctic_Program_website_from_2018_02_22.png)

Izvor slike: [Wikimedia Commons File:United States Antarctic Program website from 2018 02 22.png](https://commons.wikimedia.org/wiki/File:United_States_Antarctic_Program_website_from_2018_02_22.png)

---

## Zašto izraditi web sjedište istraživačke grupe? (2/2)

* ograničenja po pitanju oblikovanja
    * zadani dizajn i organizacija sadržaja web sjedišta institucije
    * identitet grupe ~= identitet institucije
* ograničenja po pitanju sadržaja
    * možete li staviti slike/video/animacije, npr. kao dodatne materijale kod objave preprinta ili rada?
    * postoji li ugrađen blog za objavu vijesti/promišljanja, idealno s podrškom za više autora?
    * imate li prostor za postavljanje nastavnih materijala?
    * je li broj stranica sadržaja ograničen (uglavnom nije tehnički problem, već pitanje odluke institucije)?

---

## Kako izraditi web sjedište istraživačke grupe?

* očekivanja tipične istraživačke grupe su:
    * jednostavan jezik za stvaranje sadržaja: web jezici (HTML/CSS/JS/...) su nepotrebno rječiti
    * prenosiv jezik za stvaranje sadržaja: želimo moći lako seliti već stvoreni sadržaj među platformama/alatima/rješenjima
    * ukupna cijena rješenja bez našeg rada mora biti 0 EUR ili blizu tome

![Open source Swiss knife bg right:40%](https://upload.wikimedia.org/wikipedia/commons/c/c7/121212_2_OpenSwissKnife.png)

Izvor slike: [Wikimedia Commons File:121212 2 OpenSwissKnife.png](https://commons.wikimedia.org/wiki/File:121212_2_OpenSwissKnife.png)

---

## Znate li koristiti LaTeX?

* znam LaTeX i koristim ga
* znam da LaTeX postoji, ali ga ne koristim
* prvi put čujem za LaTeX

![Primjer LaTeX-a bg right](https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/LaTeX_example.svg/744px-LaTeX_example.svg.png)

Izvor slike: [Wikimedia Commons File:LaTeX example.svg](https://commons.wikimedia.org/wiki/File:LaTeX_example.svg)

---

## Jezik Markdown

* obični tekst s vrlo malo dodataka
* "LaTeX weba": LaTeX -> PDF je kao Markdown -> HTML
* mnogo rješenja temeljenih na Markdownu: Hugo, Docusaurus, Jekyll, Hexo, MkDocs, mdBook, Pelican [itd.](https://jamstack.com/generators/)
    * mi koristimo [MkDocs](https://www.mkdocs.org/) i [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)

![Markdown logo bg right 90%](https://upload.wikimedia.org/wikipedia/commons/4/48/Markdown-mark.svg)

Izvor slike: [Wikimedia Commons File:Markdown-mark.svg](https://commons.wikimedia.org/wiki/File:Markdown-mark.svg)

---

## Tehnički postupak izrade web sjedišta

1. stvaranje novog web sjedišta

    ``` shell
    $ mkdocs new .
    ```

1. punjenje sjedišta sadržajem: dani, mjeseci, godine, desetljeća...
1. lokalni pregled izrađenog web sjedišta

    ``` shell
    $ mkdocs serve
    ```

1. postavljanje sjedišta na web, odnosno osvježavanje verzije koja već postoji na webu novim sadržajem

    ``` shell
    $ mkdocs gh-deploy
    ```

---

## Organizacija sadržaja

* informacije o grupi, opis projekata, popis radova...
* hrvatska i engleska verzija?
    * moraju li sve stranice postojati u obje verzije?
* blog? kategorije objava na blogu?
* nastavni materijali?
* pretraživanje?
* oznake na stranicama?

![Googleov sitemap bg right:40% 80%](https://upload.wikimedia.org/wikipedia/commons/2/20/Sitemap_google.jpg)

Izvor slike: [Wikimedia Commons File:Sitemap google.jpg](https://commons.wikimedia.org/wiki/File:Sitemap_google.jpg)

---

## Radionica tijekom prve polovice iduće godine?

* tehnički dio: Markdown + Material for MkDocs
* organizacijski dio: obavezni elementi, dodatni elementi, umetanje multimedijskog sadržaja

![Material for MkDocs Illustration bg left:60%](https://raw.githubusercontent.com/squidfunk/mkdocs-material/master/docs/assets/images/illustration.png)

Izvor slike: [Material for MkDocs Documentation](https://squidfunk.github.io/mkdocs-material/)

---

## Hvala na pažnji!

![GASERI Logo](../../images/gaseri-logo-animated.webp)

GASERI website: [group.miletic.net](../../index.md) ([SimilarWeb za miletic.net](https://www.similarweb.com/website/miletic.net/))

![Vedran bg left](https://vedran.miletic.net/images/vm.jpg)

![Matea bg right](https://mateaturalija.github.io/images/profile.jpg)
