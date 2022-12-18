---
marp: true
author: Vedran Miletić
title: Timski razvoj web aplikacija i upravljanje verzijama programskog koda
description: Razvoj web aplikacija i usluga
keywords: razvoj web aplikacija usluga
theme: default
class: _invert
paginate: true
---

# Timski razvoj web aplikacija i upravljanje verzijama programskog koda

## doc. dr. sc. Vedran Miletić, vmiletic@inf.uniri.hr, [vedran.miletic.net](https://vedran.miletic.net/)

### Fakultet informatike i digitalnih tehnologija, akademska 2021./2022. godina

---

## Motivacija

Voditelj ste tima za razvoj aplikacija. Vaši programski inženjeri (kolokvijalno programeri, *developeri*) vam na redovnoj bazi šalju promjene u izvornom kodu koje su izveli i vi ih integrirate te vršite postavljanje aplikacije.

Na koji način ćete osigurati:

- da rad različitih programskih inženjera nije u konfliktu,
- da verzija koju dobijete nakon integracije promjena radi barem jednako dobro kao trenutna, postojeća i
- da proces postavljanja aplikacije svaki put izvedete bez pogreške?

---

## Upravljanje verzijama

- Sinonimi: *verzioniranje*, *upravljanje konfiguracijom*
- Uključuje pamćenje povijesti promjena na izvornom kodu softverskog projekta koji razvijate i njegovoj pripadnoj dokumentaciji ([Wikipedia](https://en.wikipedia.org/wiki/Version_control))

![bg 50% right:45%](https://upload.wikimedia.org/wikipedia/commons/a/af/Revision_controlled_project_visualization-2010-24-02.svg)

---

## Razlozi za korištenje upravljanja verzijama

- Omogućuje povratak na prethodno stanje u slučaju potrebe pa možete odgovoriti na pitanja kao što su:
    - Je li neka pogreška postojala u prethodnoj verziji?
    - Tko je napisao ovih 20-ak linija koda koje rade problem?
    - Koje su točno promjene od verzije 1.2.4 do verzije 1.2.7?
- Omogućuje istovremeni rad više developera na više verzija softvera
    - Primjerice, dva developera održavaju stabilnu verziju i popravljaju bugove u njoj dok pet developera razvija novu verziju
- Omogućuje kontinuiranu integraciju i kontinuiranu isporuku/postavljanje

---

## Povijesni pregled

- Source Code Control System (SCCS) na Unixu, 1970-te
- Revision Control System (RCS)
- Concurrent Versions System (CVS)
- [Subversion](https://subversion.apache.org/) (SVN, moto projekta: "CVS done right")
    - > I see Subversion as being the most pointless project ever started. (...) There is no way to do CVS right.

        -- Linus Torvalds, Google Tech talk on Git, 2007. ([snimka](https://youtu.be/4XpnKHJAok8))

---

## Distribuirani pristup verzioniranju

- Kod centraliziranog pristupa sva povijest promjena je na jednom mjestu (CVS/SVN poslužitelj) i klijenti je povlače
- Kod decentraliziranog pristupa sva povijest promjena je kod svakog sudionika u razvoju softvera i moguća je međusobna razmjena P2P (kao BitTorrent)
- [Git](https://git-scm.com/) je *de facto* standard, ostali (Mercurial, BitKeeper, GNU Bazaar, Fossil itd.) su danas vrlo malo korišteni

![Git bg 90% right:40%](https://upload.wikimedia.org/wikipedia/commons/e/e0/Git-logo.svg)

---

## Osnovni pojmovi

- Lokalni repozitorij (Git, [Visual Studio Code](https://code.visualstudio.com/docs/editor/versioncontrol))
    - Spremanje promjena: `commit`
- Udaljeni repozitorij ([GitLab](https://about.gitlab.com/), [GitHub](https://github.com/)); sinkronizacija s lokalnim:
    - Povlačenje promjena s udaljenog repozitorija: `pull`
    - Guranje promjene na udaljeni repozitorij: `push`
- Različiti programeri rade promjene neovisno na različitim granama pa se kasnije te grane spajaju u jednu granu koja sadrži sve promjene
    - Grananje: `branch`
    - Spajanje grana: `merge`

---

## Što sve verzionirati?

- Sve! (*ne mora biti sve u jednom repozitoriju*)
    - Primjerice, [Grupa za strukturu i funkciju biomolekula](https://svedruziclab.github.io/) verzionira statičko web sjedište ([repozitorij na GitHubu](https://github.com/svedruziclab/svedruziclab.github.io))
- Izvorni kod
- Testove
- Skripte za izgradnju i punjenje baze podataka
- Skripte i konfiguracijske datoteke za izgradnju i isporuku aplikacije
- Dokumentaciju
- Postavke DNS-a, postavke vatrozida

---

## Što ne verzionirati?

- Datoteke koje se preuzmu ili stvore kod izgradnje softvera, primjerice:
    - Node.js: direktorij `node_modules`, stvorit će ga npm kod izgradnje naredbom `npm install` na temelju podataka u `package.json`
    - PHP: direktorij `vendor`, stvorit će Composer kod izgradnje naredbom `composer update` na temelju podataka u datoteci `composer.json`
- [Git Large File Storage](https://git-lfs.github.com/) brine o velikim binarnim datotekama (slike, audiovizualne datoteke i sl.)

---

## Kako verzionirati

- Redovito
- Svako spremanje promjena čini cjelinu (npr. ako ne prolazi izgradnja softvera, popravite greške prije commita)
- Nakon izvođenja testova
- Inkrementalne promjene
- Jasne poruke o promjenama u commitovima; usporedite:
    - "Popravljena neka pogreška kod izvođenja"
    - "Promijenjen `QuuxFactory` da odbije stvoriti prazne objekte kad je varijabla okoline `HARDENED_DEFAULTS` postavljena na vrijednost `enforce`"

---

## Verzioniranje tuđeg koda

- Pod *tuđim kodom* podrazumijevamo biblioteke o kojima vaš softver ovisi
- Nije vaš posao (tm)
- Točne verzije navedene u datoteci s popisom paketa o kojima vaša aplikacija ovisi povući će [npm](https://www.npmjs.com/) za Node.js, [Composer](https://getcomposer.org/) za PHP, [pip](https://pip.pypa.io/) za Python, [RubyGems](https://rubygems.org/) za Ruby, [Maven](https://maven.apache.org/) za Javu, [Nuget](https://www.nuget.org/) za C#, [Cargo](https://doc.rust-lang.org/cargo/) za Rust itd.
    - Datoteku s popisom zavisnosti ćete verzionirati

---

![Relax it's NPM bg 95% left:70%](https://www.monkeyuser.com/assets/images/2017/52-npm-delivery.png)

## NPM isporuka

Izvor: [NPM Delivery](https://www.monkeyuser.com/2017/npm-delivery/) (MonkeyUser, 4th July 2017)

---

![Dependency hell is a colloquial term for the frustration of some software users who have installed software packages which have dependencies on specific versions of other software packages. --Wikipedia bg 77% left:70%](https://www.monkeyuser.com/assets/images/2021/226-update.png)

## Nadogradnja

Izvor: [Update](https://www.monkeyuser.com/2021/update/) (MonkeyUser, 5th October 2021)

---

## Verzioniranje konfiguracijskih datoteka

- Dvanestofaktorska aplikacija nema konfiguracijske datoteke, već koristi varijable okoline za konfiguraciju
- Ako ih ima, verzioniraju se odvojeno od osnovne aplikacije
    - npr. datoteka u formatu JSON sadrži `"domain": "www.example.com"`, `"database-driver": "mongo"`, `"database-server-port": 5502`, `"file-storage-path": "/var/lib/fizz-buzz-enterprise-edition"`
- Nemojte verzionirati zaporke, parove (SSH) ključeva i certifikate

---

## Načini korištenja Gita u razvoju softvera

- [GitLabov pregled nekoliko mogućih načina korištenja Gita](https://about.gitlab.com/topics/version-control/what-is-git-workflow/)
- [Atlassianov opis mogućih načina korištenja Gita](https://www.atlassian.com/git/tutorials/comparing-workflows):
    - [Feature branch](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow)
    - [Gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)
    - [Forking](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow)
