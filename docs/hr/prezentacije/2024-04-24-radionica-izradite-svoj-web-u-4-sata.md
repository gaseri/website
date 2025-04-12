---
marp: true
theme: default
class: default
author: Vedran Miletić, Matea Turalija
title: Izradite svoj web u 4 sata! (radionica)
description: Radionica u organizaciji Udruge Penkala
keywords: web site, markdown, mkdocs, material for mkdocs
abstract: |
  Najcitiraniji znanstvenici imaju svoje web stranice. U Americi preko 50% znanstvenika ima vlastite web stranice, neovisne o institutu ili sveučilištu na kojem rade.  Postanite i vi jedan od njih kroz radionicu gdje ćete u 4h korišteći Markdown izraditi osobne web stranice, web stranice vaše istraživačke i/ili pokrenuti svoj blog!
---

# Izradite svoj web u 4 sata! (radionica)

Nastavak i praktična strana predavanja [Zašto i kako izraditi web sjedište istraživačke grupe](2023-12-28-zasto-i-kako-izraditi-web-sjediste-istrazivacke-grupe.md) s [Mutimira](https://udruga-penkala.hr/mutimir/) [2023](https://udruga-penkala.hr/mutimir2023/).

---

<!--paginate: true -->

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

## CMS ili Jamstack?

- sustav za upravljanje sadržajem (*content management systems*, CMS)
    - [WordPress](https://wordpress.org/40-percent-of-web/), Shopify, Wix, Squarespace, [Joomla](https://www.joomla.org/), [Drupal](https://www.drupal.org/) i brojni drugi
- generator statičkog web sjedišta, kolokvijalno Jamstack
    - [Hugo](https://gohugo.io/), [Jekyll](https://jekyllrb.com/), [Docusaurus](https://docusaurus.io/), [MkDocs](https://www.mkdocs.org/) i brojni drugi

---

## Sadržaj radionice

- 24\. 4. 2024.
    - Stvaranje projekta web sjedišta (Python, MkDocs)
    - Pisanje web stranica u Markdownu (Visual Studio Code i proširenja markdownlint, Markdown Theme Kit i Word Count)
- 8\. 5. 2024.
    - Postavljanje sjedišta na web (Git, GitHub, GitHub Pages)
    - Mogućnosti konfiguracija projekta web sjedišta (Material for MkDocs)

---

## Priprema za radionicu

Potrebno je registrirati korisnički račun na [GitHubu](https://github.com/).

### Potrebni softveri

- [Windows Terminal](https://learn.microsoft.com/en-us/windows/terminal/) (samo na Windowsima)
- [Microsoft PowerToys](https://learn.microsoft.com/en-us/windows/powertoys/) (samo na Windowsima)
- [Visual Studio Code](https://code.visualstudio.com/)
    - [markdownlint](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint), [Markdown Theme Kit](https://marketplace.visualstudio.com/items?itemName=ms-vscode.Theme-MarkdownKit) i [Word Count](https://marketplace.visualstudio.com/items?itemName=ms-vscode.wordcount)
- [Python](https://www.python.org/)
    - [MkDocs](https://www.mkdocs.org/) i [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Git](https://git-scm.com/)

---

### Upute za instalaciju potrebnih softvera

#### Preduvjeti za instalaciju potrebnih softvera

=== "🪟 Windows"

    1. Instalirajte sve dostupne nadogradnje putem sustava [Windows Update](https://support.microsoft.com/en-us/windows/update-windows-3c5ae7fc-9fb6-9af1-1984-b5e0412c556a). To može potrajati i do nekoliko sati te može biti potrebno jedno ili više ponovnih pokretanja računala. Ovaj korak je nužan zato što softveri koje koristimo zahtijevaju relativno svježe podverzije Windowsa 10 ili 11.
    1. Putem [Microsoft Storea](https://apps.microsoft.com/home) instalirajte [Windows Terminal](https://apps.microsoft.com/detail/9n0dx20hk701) (ako već nije instaliran).

=== "🐧 Linux"

    !!! note
        Upute u nastavku odnose se na [Linux Mint](https://linuxmint.com/), [Ubuntu](https://ubuntu.com/) i [Debian](https://www.debian.org/) [GNU/Linux](https://wiki.debian.org/DebianGnuLinux).

    U terminalu upišite naredbu:

    ``` shell
    sudo apt update && sudo apt upgrade
    ```

    Po potrebi, ponovno pokrenite računalo.

---

#### Naredbe za instalaciju potrebnih softvera

=== "🪟 Windows"

    Iskoristit ćemo [Windows Package Manager](https://learn.microsoft.com/en-us/windows/package-manager/), koji pokrećemo naredbom `winget` u Terminalu, za jednostavnu instalaciju svih potrebnih softvera. Pokrenite Windows Terminal (putem [izbornika Start](https://support.microsoft.com/en-us/windows/open-the-start-menu-4ed57ad7-ed1f-3cc9-c9e4-f329822f5aeb)) i upišite redom naredbe u nastavku.

    Skup pomoćnih alata [Microsoft PowerToys](https://learn.microsoft.com/en-us/windows/powertoys/):

    ``` shell
    winget install --id Microsoft.PowerToys -e
    ```

=== "🐧 Linux"

    Iskoristit ćemo [APT](https://en.wikipedia.org/wiki/APT_(software)), koji pokrećemo naredbom `apt` u terminalu, za jednostavnu instalaciju svih potrebnih softvera. U terminalu upišite redom naredbe:

---

Radno okruženje [Visual Studio Code](https://code.visualstudio.com/):

=== "🪟 Windows"

    ``` shell
    winget install --id Microsoft.VisualStudioCode -e
    ```

    Nakon ovog koraka zatvorite Terminal pa ga ponovno pokrenite prije nastavka.

=== "🐧 Linux"

    ``` shell
    curl -L -o code.deb 'https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64'
    sudo apt install ./code.deb
    ```

Proširenja [markdownlint](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint), [Markdown Theme Kit](https://marketplace.visualstudio.com/items?itemName=ms-vscode.Theme-MarkdownKit) i [Word Count](https://marketplace.visualstudio.com/items?itemName=ms-vscode.wordcount):

``` shell
code --install-extension DavidAnson.vscode-markdownlint
code --install-extension ms-vscode.Theme-MarkdownKit
code --install-extension ms-vscode.wordcount
```

---

[Python](https://www.python.org/):

=== "🪟 Windows"

    ``` shell
    winget install --id Python.Python.3.12 -e
    ```

    Nakon ovog koraka zatvorite Terminal pa ga ponovno pokrenite prije nastavka.

=== "🐧 Linux"

    ``` shell
    sudo apt install python-is-python3 python3-pip
    ```

[MkDocs](https://www.mkdocs.org/) i [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/):

``` shell
pip install mkdocs-material
```

---

[Git](https://git-scm.com/):

=== "🪟 Windows"

    ``` shell
    winget install --id Git.Git -e
    ```

=== "🐧 Linux"

    ``` shell
    sudo apt install git
    ```

U slučaju da imate pitanja, obratite se [Vedranu Miletiću putem e-maila](https://vedran.miletic.net/#contact).

---

## Osnovna literatura

Visual Studio Code i proširenja:

- [Basic Editing in Visual Studio Code](https://code.visualstudio.com/docs/editor/codebasics)
- [Terminal Basics in Visual Studio Code](https://code.visualstudio.com/docs/editor/codebasics)
- [Markdown and Visual Studio Code](https://code.visualstudio.com/docs/languages/markdown)
- [Collaborate on GitHub](https://code.visualstudio.com/docs/sourcecontrol/github)

MkDocs i Material for MkDocs:

- [Command Line Interface -- MkDocs](https://www.mkdocs.org/user-guide/cli/)
- [Writing your docs -- MkDocs](https://www.mkdocs.org/user-guide/writing-your-docs/)
- [Deploying your docs -- MkDocs](https://www.mkdocs.org/user-guide/deploying-your-docs/)
- [Configuration -- MkDocs](https://www.mkdocs.org/user-guide/configuration/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)

---

## Sadržaj prvog dijela radionice

- Stvaranje projekta web sjedišta (Python, MkDocs)
- Pisanje web stranica u Markdownu (Visual Studio Code i proširenja markdownlint, Markdown Theme Kit i Word Count)

---

## Započinjanje projekta

U (Windows) Terminalu:

``` shell
mkdocs new mojnoviweb
```

U Visual Studio (VS) Code: izbornik `File` pa `Open Folder...` pa pronađite mapu `mojnoviweb` i otvorite ju.

U VS Code ugrađenom terminalu:

``` shell
mkdocs serve
```

pa ++ctrl++ i klik na `http://127.0.0.1:8000/` u izlazu naredbe za otvaranje pregleda weba.

---

## Pisanje Markdowna

- [Writing with Markdown (Writing your docs -- MkDocs)](https://www.mkdocs.org/user-guide/writing-your-docs/#writing-with-markdown)
- [Suradnički uređivač teksta HackMD i jezik Markdown](../nastava/materijali/hackmd-markdown.md#)
- ([Zettlr workshop](../../en/tutorials/zettlr.md))

---

## Sadržaj drugog dijela radionice

- Postavljanje sjedišta na web (Git, GitHub, GitHub Pages)
- Mogućnosti konfiguracija projekta web sjedišta (Material for MkDocs)

---

## Postavljanje web sjedišta na GitHub

Preduvjeti:

- postavljanje imena i e-maila u konfiguraciji Gita
- stvaranje SSH ključa
- dodavanje ključa na GitHub
- provjera mogućnosti povezivanja
- stvaranje repozitorija
- povezivanje repozitorija
- uključivanje GitHub Pages

---

## Postavljanje imena i e-maila u konfiguraciji Gita

``` shell
git config --global user.name "Cool Gaser"
git config --global user.email "cool@gaser.zg"
```

## Stvaranje SSH ključa

U Terminalu pokrenite naredbu (pripazite da ne prepišete ključ koji već postoji):

``` shell
ssh-keygen
```

---

## Dodavanje SSH ključa na GitHub

Na GitHubovom web sjedištu u web pregledniku:

1. Klik na svoj avatar u gornjem desnom uglu pa u izborniku s desne strane `Settings` pa u izborniku s lijeve strane `SSH and GPG keys` ili otvorite izravno <https://github.com/settings/keys>.
1. GPG ključevi nam zasad nisu bitni. Klik na `New SSH key` ili otvorite izravno <https://github.com/settings/ssh/new>, postavite naslov u polju `Title` po želji, `Key type` neka bude `Authentication key` i u `Key` zalijepite izlaz naredbe:

    ``` shell
    cat .ssh/id_rsa.pub
    ```

---

## Provjera mogućnosti povezivanja na GitHub

U Terminalu:

``` shell
ssh -T git@github.com
```

U slučaju da je sve ispravno konfigurirano:

``` shell-session
Hi coolgaser! You've successfully authenticated,
but GitHub does not provide shell access.
```

---

## Stvaranje repozitorija web sjedišta na GitHubu

Klik na `+` gore desno pa `New repository` ili otvorite izravno <https://github.com/new>. Ime u polju `Repository name` postavite da bude oblika `vasekorisnickoime.github.io`, pri čemu na mjesto `vasekorisnickoime` umetnite svoje korisničko ime (npr. `coolgaser`).

Repozitorij neka bude javan (`Public`), a pod `Initialize this repository:` ne uključujte ništa (ni `README`, ni `.gitignore`, ni licencu).

---

## Stvaranje lokalnog repozitorija i povezivanje s repozitorijem na GitHubu

U VS Code terminalu *nakon otvaranja projekta* `mojnoviweb`:

1. stvaranje lokalnog repozitorija:

    ``` shell
    git init
    git add .
    git commit -m "first commit"
    git branch -M main
    ```

1. povezivanje s repozitorijem na GitHubu:

    ``` shell
    git remote add origin git@github.com:vasekorisnickoime/vasekorisnickoime.github.io.git
    git push -u origin main 
    ```

1. priprema web sjedišta za prvo postavljanje

    ``` shell
    mkdocs gh-deploy
    ```

---

## Uključivanje GitHub Pages

1. Na stranici repozitorija odaberite karticu `Settings` pa u izborniku s lijeve strane `Pages`. Pod `Build and deployment`:

    1. `Source` postavite na `Deploy from a branch`.
    1. `Branch` postavite na `gh-pages` i zatim prisnite gumb `Save`.

1. (Uočite da postoji i odjeljak `Custom domain`.)
1. Otvorite u web pregledniku adresu:

    ``` text
    https://vasekorisnickoime.github.io/
    ```

---

## Postavke značajki web sjedišta

- [Boje](https://squidfunk.github.io/mkdocs-material/setup/changing-the-colors/)
- [Oblici pisma](https://squidfunk.github.io/mkdocs-material/setup/changing-the-fonts/)
- [Jezik](https://squidfunk.github.io/mkdocs-material/setup/changing-the-language/)
- [Pretraga](https://squidfunk.github.io/mkdocs-material/setup/setting-up-site-search/)
- [Zaglavlje](https://squidfunk.github.io/mkdocs-material/setup/setting-up-the-header/)
- [Podnožje](https://squidfunk.github.io/mkdocs-material/setup/setting-up-the-footer/)
- [Oznake](https://squidfunk.github.io/mkdocs-material/setup/setting-up-tags/)

---

## Postavke značajki web sjedišta (nast.)

- [Blog](https://squidfunk.github.io/mkdocs-material/setup/setting-up-a-blog/)
- [Slike](https://squidfunk.github.io/mkdocs-material/reference/images/)
- [Tablice](https://squidfunk.github.io/mkdocs-material/reference/data-tables/)
- [Fusnote](https://squidfunk.github.io/mkdocs-material/reference/footnotes/)
- [Matematika](https://squidfunk.github.io/mkdocs-material/reference/math/)
- [Blokovi izvornog koda](https://squidfunk.github.io/mkdocs-material/reference/code-blocks/)
- [Dijagrami](https://squidfunk.github.io/mkdocs-material/reference/diagrams/)

---

## Za one koji žele znati više

- [Ikone](https://squidfunk.github.io/mkdocs-material/setup/changing-the-logo-and-icons/)
- [Navigacija](https://squidfunk.github.io/mkdocs-material/setup/setting-up-navigation/)
- [Analitika](https://squidfunk.github.io/mkdocs-material/setup/setting-up-site-analytics/)
- [Kartice za društvene mreže](https://squidfunk.github.io/mkdocs-material/setup/setting-up-social-cards/)
- [Povezivanje repozitorija](https://squidfunk.github.io/mkdocs-material/setup/adding-a-git-repository/)
- [Komentari](https://squidfunk.github.io/mkdocs-material/setup/adding-a-comment-system/)
- [Proširenja](https://squidfunk.github.io/mkdocs-material/setup/extensions/python-markdown-extensions/)
- [Rešetke](https://squidfunk.github.io/mkdocs-material/reference/grids/)
- [Opisi](https://squidfunk.github.io/mkdocs-material/reference/tooltips/)

---

## Zahvale

- [Una Pale](https://www.una-pale.from.hr/) i [ostali aktivni članovi Udruge Penkala](https://udruga-penkala.hr/iza-penkale/)
- [Martin Donath](https://github.com/squidfunk) (autor teme [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)) i čitava zajednica oko [MkDocsa](https://www.mkdocs.org/)
