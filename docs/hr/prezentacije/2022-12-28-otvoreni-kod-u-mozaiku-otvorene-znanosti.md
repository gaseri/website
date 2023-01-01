---
marp: true
author: Vedran Miletić
title: Otvoreni kod u mozaiku otvorene znanosti
description: Predavanje na Mutimiru 2022 u organizaciji Udruge Penkala
keywords: free software, open source, open science
theme: default
class: _invert
paginate: true
abstract: |
  U posljednjih 30-ak godina slobodni softver otvorenog kod prešao je velik put od prakse hobista i entuzijasta preko prvih industrijskih projekata koji su vremenom u suradnji IT tvrtki i zajednice razvili alate poput web preglednika Firefox i uredskog paketa LibreOffice pa do stanja integracije u IT rješenja u kojem je nezamislio obaviti išta na internetu bez korištenja nekog programa ili biblioteke otvorenog koda. Osim u industriju, prakse slobodnog sotvera otvorenog koda polako su se probile i u znanstvenu zajednicu pa se se kroz posljednjih 20-ak godina sve veći broj projekata odlučuje za otvoreno licenciranje (u nekom obliku) nauštrb tipičnog akademskog (besplatno korištenje u akademske svrhe i posebno licenciranje za komercijalnu upotrebu). Predavanje će govoriti o prednostima slobodnog softvera otvorenog koda iz akademske perspektive, dosadašnjim uspjesima i primjerima dobre prakse te ulozi otvorenog koda u pokretu otvorene znanosti.
---

# Otvoreni kod u mozaiku otvorene znanosti

## Vedran Miletić, Fakultet informatike i digitalnih tehnologija

![FIDIT logo h:400px](https://upload.wikimedia.org/wikipedia/commons/1/14/FIDIT-logo.svg)

### Mutimir 2022 (Udruga Penkala), Hotel Matija Gubec, 28. prosinca 2022.

---

## Predstavljanje predavača

* Docent, [Fakultet informatike i digitalnih tehnologija](https://www.inf.uniri.hr/), Sveučilište u Rijeci
    * Voditelj Grupe za aplikacije i usluge na ekaskalarnoj istraživačkoj infrastrukturi (engl. **G**roup for **A**pplications and **S**ervices on **E**xascale **R**esearch **I**nfrastructure)
* Računarstvo (formalno), računalna biokemija (realno)
    * Razvoj istraživačkog softvera za superračunala i računalne oblake
    * Fokus: simulacija molekulske dinamike biomolekula

![GASERI logo bg right:30% 70%](../../images/gaseri-logo-koleda.png)

---

## Otvorena znanost

![File:Osc2021-unesco-open-science-no-gray.png](https://upload.wikimedia.org/wikipedia/commons/2/28/Osc2021-unesco-open-science-no-gray.png)

---

## Mentimeter: Što je za vas otvoreni kod / slobodni softver?

### Posjetite `www.menti.com` i upišite kod `1866 0700`

---

## Crtice iz povijesti

* Richard Stallman, FSF i GNU, 1983.
    * slobodni softver (engl. *free software*)
* Netscape/Mozilla 1998., preteča Firefoxa
    * kontekst: [dot-com boom](https://en.wikipedia.org/wiki/Dot-com_bubble), počeci bogatih web aplikacija (web 2.0)
    * otvoreni kod (engl. *open source*)
    * posljedica: Google Chrome, [Microsoft is dead](http://www.paulgraham.com/microsoft.html)

![Saint Ignucius bg right:40% 80%](https://stallman.org/saintignucius.jpg)

---

## Tradicionalni tzv. akademski pristup licenciranju softvera

* akademska primjena bez naknade
* komercijalna primjena zahtijeva posebnu licencu
* pristup izvornom kod dozvoljen suradnicima na projektima

---

## Stanje znanstvenog softvera (1/2)

* [List of protein-ligand docking software](https://en.wikipedia.org/wiki/List_of_protein-ligand_docking_software)
* [List of quantum chemistry and solid-state physics software](https://en.wikipedia.org/wiki/List_of_quantum_chemistry_and_solid-state_physics_software)
* [List of systems biology modeling software](https://en.wikipedia.org/wiki/List_of_systems_biology_modeling_software)
* [List of bioinformatics software](https://en.wikipedia.org/wiki/List_of_bioinformatics_software)
* [List of free geology software](https://en.wikipedia.org/wiki/List_of_free_geology_software)

---

## Stanje znanstvenog softvera (2/2)

* Python/Jupyter, R, Julia
* program-prevoditelji i biblioteke za [Fortran](https://fortran-lang.org/), C, [C++](https://en.cppreference.com/w/cpp/links/libs) i [Rust](https://www.rust-lang.org/)
* [Comparison of deep learning software](https://en.wikipedia.org/wiki/Comparison_of_deep_learning_software)
* Još primjera?

---

## *Everything as a Service*

* aplikacije i usluge dostupne u oblaku (GMail, [Adobe CC](https://www.adobe.com/creativecloud.html), [Overleaf](https://www.overleaf.com/), Office 365, [Figma](https://www.figma.com/), [BioRender](https://biorender.com/) itd.)
* tzv. *web serveri* u znanosti: [Charmm-gui](https://charmm-gui.org/), [SwissParam](https://www.swissparam.ch/), [HADDOCK](https://wenmr.science.uu.nl/haddock2.4/), [webSDA](https://websda.h-its.org/webSDA) itd.
* sadašnjost i budućnost znanosti: [ELIXIR Services](https://elixir-europe.org/services)
* programski kod usluge i pripadna dokumentacija mogu biti otvoreni, svatko može pokrenuti svoju instancu aplikacije ili usluge
    * mogućnost federacije, npr. kao [Mastodon](https://joinmastodon.org/)
    * mogućnost dobrovoljnog računanja, npr. kao [Folding@home](https://foldingathome.org/)

---

## Načela slobodnog softvera otvorenog koda i prednost u otvorenoj znanosti

* > The freedom to run the program, for any purpose.
    * reproducibilnost znanstvenog procesa
* > The freedom to study how the program works, and change it so it does your computing as you wish.
    * istraživački rad
* > The freedom to redistribute copies so you can help your neighbor.
    * suradnja
* > The freedom to distribute copies of your modified versions, giving the community a chance to benefit from your changes.
    * objava rezultata

---

## Zaključak

* Doba licenciranja softvera po *defaultu* kao slobodnog softvera otvorenog koda nije privremena povijesna epizoda; ono je epoha
    * [Tracking the explosive growth of open-source software](https://techcrunch.com/2017/04/07/tracking-the-explosive-growth-of-open-source-software/)
    * [Open Source is Taking Over Europe!](https://itsfoss.com/open-source-adoption-europe/)
    * [Open Source Survey](https://opensourcesurvey.org/2017/)
* Doba besplatnog tzv. akademskog softvera je završilo 2000-ih omasovljenjem korištenja interneta i dolaskom Weba 2.0
    * [What is the price of open-source fear, uncertainty, and doubt?](https://gaseri.org/en/blog/2015-09-14-what-is-the-price-of-open-source-fear-uncertainty-and-doubt/)
