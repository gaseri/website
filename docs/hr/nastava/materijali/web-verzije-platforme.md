---
marp: true
author: Vedran Miletić
title: Podrška web aplikacija za izvođenje na različitim verzijama platforme
description: Razvoj web aplikacija i usluga
keywords: razvoj web aplikacija usluga
theme: default
class: _invert
paginate: true
---

# Podrška web aplikacija za izvođenje na različitim verzijama platforme

## doc. dr. sc. Vedran Miletić, vmiletic@inf.uniri.hr, [vedran.miletic.net](https://vedran.miletic.net/)

### Fakultet informatike i digitalnih tehnologija Sveučilišta u Rijeci, akademska 2021./2022. godina

---

## Verzije platforme

Vaša aplikacija eksplicitno podržava samo:

- one verzije interpretera programskog jezika koje ste eksplicitno testirali,
- one verzije okvira i biblioteka koje ste eksplicitno testirali,
- one verzije sustava za upravljanje bazom podataka koje ste eksplicitno testirali,
- one verzije sustava za predmemoriju koje ste eksplicitno testirali,
- itd.

Kod Pythona se potrebne verzije modula navode u datoteci `requirements.txt` ([primjer korištenja](https://www.jetbrains.com/help/pycharm/managing-dependencies.html)).

---

## Podrška za više verzija interpretera programskog jezika (1/2)

- Korisno ako isporučujete aplikaciju korisnicima u obliku izvornog koda koji onda oni sami postavljaju na svojoj infrastrukturi; primjerice:
    - [WordPress](https://make.wordpress.org/plugins/2019/04/01/wordpress-to-move-to-php-5-6/) podržava [PHP 5.6](https://www.php.net/releases/5_6_0.php) ([službeno nepodržan od 2018.](https://www.php.net/eol.php)) i noviji
    - [Phabricator](https://www.phacility.com/phabricator/) podržava [PHP 5.5](https://www.php.net/releases/5_5_0.php) ([službeno nepodržan od 2016.](https://www.php.net/eol.php)) i sve novije verzije uz iznimku [PHP-a 7.0](https://www.php.net/releases/7_0_0.php) ([nedostaje asinkrono baratanje signalima](https://secure.phabricator.com/T12101))
    - [django CMS](https://www.django-cms.org/) zahtijeva [Django 1.11 ili noviji i Python 3.3 ili noviji](https://docs.django-cms.org/en/latest/index.html#software-version-requirements-and-release-notes)

---

## Podrška za više verzija interpretera programskog jezika (2/2)

- Potrebno je koristiti isključivo *presjek značajki* svih podržanih verzija interpretera; provjerite popis promjena u verzijama interpretera, upute za migraciju itd.
    - Ne možete koristiti funkcionalnost koja postoji u starijoj verziji, ali je zastarjela pa je maknuta u novijoj verziji
    - Ne možete koristiti funkcionalnost dodanu u novijoj verziji jer je nema u starijoj
- Dijelove koda moguće je isprobati u različitim verzijama interpretera na servisima kao što su [3v4l.org](https://3v4l.org/) (PHP), [Coliru](https://coliru.stacked-crooked.com/) (C++) i [Godbolt](https://godbolt.org/) (razni jezici)
    - Ciljano napravljeni da budu neupotrebljivi za velike aplikacije

---

## Podrška za više verzija okvira, biblioteka, sustava za upravljanje bazom podataka, sustava za predmemoriju i ostalih

1. Postavljanje lokalne okoline za inicijalno isprobavanje koja sadrži željene verzije okvira, biblioteka, sustava za upravljanje bazom podataka, sustava za predmemoriju itd.
2. Postavljanje iste okoline korištenjem Docker kontejnera ili sl. kako bi bila lako ponovljiva na različitim računalima.
3. Uključivanje okoline u CI/CD za redovito testiranje i pokretanje programa u navedenoj okolini.

Primjeri CI/CD s više različitih okolina: [Buildbot](https://github.com/buildbot/buildbot), [Grav](https://github.com/getgrav/grav), [Trac](https://github.com/edgewall/trac), [Express](https://github.com/expressjs/express) i [Sphinx](https://github.com/sphinx-doc/sphinx).

Za usporedbu, primjeri CI/CD sa samo jednom okolinom: [Discourse](https://github.com/discourse/discourse) i [Invidious](https://github.com/iv-org/invidious).
