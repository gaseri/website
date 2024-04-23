---
marp: true
author: Vedran Miletić, Matea Turalija
title: Izradite svoj web u 4 sata! (radionica)
description: Radionica u organizaciji Udruge Penkala
keywords: web site, markdown, mkdocs, material for mkdocs
theme: default
class: _invert
paginate: true
abstract: |
  Najcitiraniji znanstvenici imaju svoje web stranice. U Americi preko 50% znanstvenika ima vlastite web stranice, neovisne o institutu ili sveučilištu na kojem rade.  Postanite i vi jedan od njih kroz radionicu gdje ćete u 4h korišteći Markdown izraditi osobne web stranice, web stranice vaše istraživačke i/ili pokrenuti svoj blog!
---

# Izradite svoj web u 4 sata! (radionica)

Nastavak i praktična strana predavanja [Zašto i kako izraditi web sjedište istraživačke grupe](2023-12-28-zasto-i-kako-izraditi-web-sjediste-istrazivacke-grupe.md) s [Mutimira](https://udruga-penkala.hr/mutimir/) [2023](https://udruga-penkala.hr/mutimir2023/).

---

## Najava radionice

[Penkalin web](https://udruga-penkala.hr/radionica-izradite-svoj-web-u-4-sata/2024/) kaže:

> **Za koga:**
>
> - Doktorande, poslijedoktorande, voditelje grupa ili bilo koga!
> - Za one kojima nije baš do istraživanja sami kako to napraviti, i žele brzo, jednostavno i besplatno rješenje bez potrebe za kasnijim održavanjem
> - Za one koji ne vole kodirati, niti ih zanimaju postavljanje domena i servera
>
> **Od koga:** doc. dr. sc. [Vedran](https://vedran.miletic.net/) [Miletić](https://www.miletic.net/) i [Matea Turalija](https://mateaturalija.github.io/), [GASERI](../index.md), [FIDIT](https://www.inf.uniri.hr/)/[MedRi](https://medri.uniri.hr/), [Sveučilište u Rijeci](https://uniri.hr/)
>
> **Kada i gdje:** Online, u 2 termina po 2h, 24.4. i 8.5.2024.

---

## Najava radionice (nast.)

[Penkalin web](https://udruga-penkala.hr/radionica-izradite-svoj-web-u-4-sata/2024/) dalje kaže:

> **Zašto:**
>
> - Svi projekti vam ne stanu na CV, a web stranice su pravo rješenje za portfolio vaših istraživanja, projekata, publikacija, kodova itd.
> - Vlastite web stranice povećavanju vašu vidljivost kao znanstvenika
> - Znanstvena komunikacija vam je u planu, ali nikad niste sjeli i smislili kako
> - Osobni web ili web grupe povećava šanse za suradnjama
> - Izrada weba može biti jednostavna i brza ako znate prave alate

---

## Ishod učenja

Imati na javnom webu postavljeno vlastito sjedište slično onome kakvo ima [Matea Turalija](https://mateaturalija.github.io/).

## Neishodi učenja

- Kupnja i konfiguracija vlastite domene.
- Optimizacija sadržaja za tražilice.
- Dizajn web sjedišta.
- Suradnički razvoj i verzioniranje.

---

## Sadržaj radionice

- 24\. 4. 2024.
    - Stvaranje projekta web sjedišta (Python, MkDocs)
    - Pisanje web stranica u Markdownu (Visual Studio Code i proširenja markdownlint, Markdown Theme Kit i Word Count)
- 8\. 5. 2024.
    - Mogućnosti konfiguracija projekta web sjedišta (Material for MkDocs)
    - Postavljanje sjedišta na web (Git, GitHub, GitHub Pages)

---

## Priprema za radionicu

Potrebno je registrirati korisnički račun na [GitHubu](https://github.com/).

### Potrebni softveri

- [Windows Terminal](https://learn.microsoft.com/en-us/windows/terminal/)
- [Microsoft PowerToys](https://learn.microsoft.com/en-us/windows/powertoys/)
- [Visual Studio Code](https://code.visualstudio.com/)
    - [markdownlint](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint), [Markdown Theme Kit](https://marketplace.visualstudio.com/items?itemName=ms-vscode.Theme-MarkdownKit) i [Word Count](https://marketplace.visualstudio.com/items?itemName=ms-vscode.wordcount)
- [Python](https://www.python.org/)
    - [MkDocs](https://www.mkdocs.org/) i [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Git](https://git-scm.com/)

---

### Upute za instalaciju potrebnih softvera na operacijskom sustavu Windows, verzija 10 ili 11

Preduvjeti za instalaciju potrebnih softvera:

1. Instalirajte sve dostupne nadogradnje putem sustava [Windows Update](https://support.microsoft.com/en-us/windows/update-windows-3c5ae7fc-9fb6-9af1-1984-b5e0412c556a). To može potrajati i do nekoliko sati te može biti potrebno jedno ili više ponovnih pokretanja računala. Ovaj korak je nužan zato što softveri koje koristimo zahtijevaju relativno svježe podverzije Windowsa 10 ili 11.
1. Putem [Microsoft Storea](https://apps.microsoft.com/home) instalirajte [Windows Terminal](https://apps.microsoft.com/detail/9n0dx20hk701) (ako već nije instaliran).

---

Iskoristit ćemo [Windows Package Manager](https://learn.microsoft.com/en-us/windows/package-manager/), koji pokrećemo naredbom `winget` u Terminalu, za jednostavnu instalaciju svih potrebnih softvera. Pokrenite Windows Terminal (putem [izbornika Start](https://support.microsoft.com/en-us/windows/open-the-start-menu-4ed57ad7-ed1f-3cc9-c9e4-f329822f5aeb)) i upišite redom naredbe:

``` shell
winget install --id Microsoft.PowerToys -e
winget install --id Microsoft.VisualStudioCode -e
code --install-extension DavidAnson.vscode-markdownlint
code --install-extension ms-vscode.Theme-MarkdownKit
code --install-extension ms-vscode.wordcount
winget install --id Python.Python.3.12 -e
pip install mkdocs-material
winget install --id Git.Git -e
```

U slučaju da imate pitanja, obratite se [Vedranu Miletiću putem e-maila](https://vedran.miletic.net/#contact).
