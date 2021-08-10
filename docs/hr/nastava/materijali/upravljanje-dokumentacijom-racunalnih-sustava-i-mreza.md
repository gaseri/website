---
author: Vedran Miletić
---

# Upravljanje dokumentacijom računalnih sustava i mreža

Kako sistemaš radi s mnogo različitih skupova usluga na stvarnim i virtualnim strojevima, često u više različitih konfiguracija, potrebno je održavati dokumentaciju koja opisuje specifičnosti instalacije i konfiguracije postojećih sustava.

## Tehnološka rješenja

Za održavanje dokumentacije mogu se koristiti programi za obradu teksta kao što su [LibreOffice Writer](https://www.libreoffice.org/discover/writer/) i [Microsoft Word](https://products.office.com/word). Kako takvi alati imaju vrlo složene i upitno prenosive formate za pohranu podataka, bolja je praksa koristiti [neki od wikija](https://en.wikipedia.org/wiki/List_of_wiki_software) ili [generatora statičkih web sjedišta](https://www.staticgen.com/) zasnovanih na jednostavnim i svima čitljivim formatima označavanja običnog teksta. Bez obzira na wiki ili generator, tri se formata vrlo često koriste:

- [Markdown](https://daringfireball.net/projects/markdown/), najčešće [varijanta koje je razvio GitHub](https://github.github.com/gfm/) za [pisanje prijava grešaka, komentara i drugog teksta](https://help.github.com/categories/writing-on-github/), iako [postoje i druge](https://github.com/commonmark/CommonMark/wiki/Markdown-Flavors),
- [reStructuredText](http://docutils.sourceforge.net/rst.html), najčešće [proširenje koje nudi Sphinx](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html), popularni generator dokumentacije inicijalno namijenjen za [dokumentaciju programskog jezika Python](https://docs.python.org/), a kasnije uopćen,
- [AsciiDoc](https://asciidoc.org/), danas najčešće korišten u putem generatora dokumentacije [Asciidoctor](https://asciidoctor.org/) i [Antora](https://antora.org/).

Za rad s dokumentacijom koristit ćemo [Markdown](https://commonmark.org/help/), online uređivač [HackMD](https://hackmd.io/), desktop uređivač [Visual Studio Code](https://code.visualstudio.com/) i sustav za izradu dokumentacije [MkDocs](https://www.mkdocs.org/).

## Stilska pravila

Kod pisanja interne dokumentacije vrijedi slijediti nekoliko pravila.

- Nije potrebno pisati općenite informacije koje pišu u službenoj dokumentaciji softvera koji se koristi, npr. sve parametre pojedine naredbe ili sve dozvoljene vrijednosti neke varijable u konfiguracijskoj datoteci. Specifične informacije, npr. značenje konfiguracijske naredbe koja se koristi ili razlog za korištenje određene vrijednosti određenog parametra, svakako treba zapisati. Službena dokumentacija će vrlo vjerojatno biti dostupna tijekom čitavog životnog ciklusa softvera.
- Potrebno je zapisati upute za instalaciju i konfiguraciju preuzete s nekog foruma, bloga, društvene mreže, web sjedišta o tehnologiji ili sl. Moguće je da te stranice prestanu biti dostupne tijekom životnog ciklusa softvera.

    - Specifično, u slučaju da su upute dane u video formatu, dobro ih je prepisati u tekstualni za lakše kasnije pregledavanje.

- Dobra struktura dokumentacije je opis -- `naredbe` -- opis -- `naredbe` -- opis -- `sadržaj (dijela) konfiguracijske datoteke` -- ..., gdje se prvo navede što naredbe ili konfiguracija rade, a zatim ih se napiše kao `blok koda` da ih se kasnije može lakše pregledavati i kopirati.
- Zaporke se ne čuvaju u internoj dokumentaciji, već u specifičnim alatima kao što su [KeePassXC](https://keepassxc.org/), [KeePass](https://keepass.info/) i [Bitwarden](https://bitwarden.com/).
- Imena domaćina, IP adrese i slični podaci mogu se čuvati u internoj dokumentaciji, ali treba paziti da interna dokumentacija ostane interna. U slučaju da se dijelovi dokumentacije prerađuju za javnu objavu, imena i adrese treba zamijeniti generičkima.
- Promjene u tekstu treba čuvati u nekom sustavu za upravljanje verzijama.

## Dokumentiranje u vježbanju sistemašenja

(Napisano prema [4. epizodi podcasta 2.5 Admins nazvanoj Zooming from pets to cattle](https://2.5admins.com/2-5-admins-04/), 22. i 23. minuti gdje [Allan Jude](https://twitter.com/allanjude) govori kako postati sistemaš.)

Dokumentiranje je neizostavni dio učenja sistemašenja. Kad učite biti sistemaš, učite kroz rad (instalaciju operacijskih sustava, postavljanje određenih datotečnih sustava na određene medije za pohranu podataka, konfiguriranje određenih mrežnih usluga i sl.) i istovremeno dokumentirate što radite.

Kad ste gotovi, izbrišete sve što ste upravo postavili i po svojoj dokumentaciji ponovite postupak. Ako je postupak savršeno ponovljiv po dokumentaciji koju ste napisali bez da morate ručno razmišljati o dodatnim koracima koji ne pišu i gledati ostale izvore informacija, izvrsno. Ako nije, dopunite dokumentaciju i ponovite postupak.
