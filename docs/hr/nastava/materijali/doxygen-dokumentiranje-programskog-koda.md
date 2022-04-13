---
author: Vedran Miletić
---

# Dokumentiranje programskog koda alatom Doxygen

[Doxygen](https://www.doxygen.nl/) je vjerojatno najkorišteniji alat za generiranje dokumentacije iz bilješki u C++ kodu, ali podržava i druge jezike kao što su C, Objective-C, C#, PHP (u praksi se češće koriste [phpDocumentor](https://phpdoc.org/) i [Doctum](https://doctum.long-term.support/)), Java (u praksi se češće koristi [Javadoc](https://www.oracle.com/java/technologies/javase/javadoc.html)), Python (u praksi se češće koristi [Docstrings](https://www.python.org/dev/peps/pep-0257/)), Fortran i drugi.

Doygen iz komentara izvornog koda može generirati HTML i PDF (korištenjem [LaTeX-a](https://www.latex-project.org/)). Osim komentara s dokumentacijom, može u dokumentaciju uključiti i strukturu izvornog koda nedokumentiranih datoteka i proizvoljnu dodatnu dokumentaciju.

## Primjeri gotove dokumentacije

Neki od projekata koji koriste Doxygen za generiranje dokumentacije aplikacijskog programskog sučelja su:

- [D-Bus](https://dbus.freedesktop.org/doc/api/html/)
- [Eigen](https://eigen.tuxfamily.org/dox/)
- [KDE](https://api.kde.org/)
- [ns-3](https://www.nsnam.org/docs/doxygen/)
- ROCm-ove biblioteke [rocPRIM](https://codedocs.xyz/ROCmSoftwarePlatform/rocRAND/) i [rocRAND](https://codedocs.xyz/ROCmSoftwarePlatform/rocRAND/)
- [RxDock](https://rxdock.gitlab.io/api-documentation/devel/html/)
- [VTK](https://vtk.org/doc/nightly/html/)

## Osnovna konfiguracija

[Službena dokumentacija](https://www.doxygen.nl/manual/) ima mnogo detalja o mogućnostima koji Doxygen ima, a mi ćemo se u nastavku ograničiti na osnovno korištenje. Mi ćemo koristiti naredbu `doxygen`, a dodatno se može koristiti [Doxygenovo grafičko korisničko sučelje](https://www.doxygen.nl/manual/doxywizard_usage.html) (naredba `doxywizard`) koje olakšava pojedine korake opisane u nastavku.

Za primjer korištenja Doxygena, naredbom

``` shell
$ doxygen -g
```

stvorit ćemo datoteku `Doxyfile` sa zadanom konfiguracijom. U njoj ne moramo mijenjati ništa, samo se uvjerimo se da značka `FILE_PATTERNS` uključuje `*.h`

```
FILE_PATTERNS          = *.c \
                         *.cc \
                         *.cxx \
                         *.cpp \
(...)
                         *.h \
(...)
```

## Dokumentiranje koda

U istom direktoriju stvorimo datoteku `hello.h` s dokumentacijom unutar komentara sadržaja:

``` c++
/**
 * \file hello.h
 * \brief Dokumentirana datoteka.
 *
 * Datoteka u kojoj su dokumentirane funkcije.
 */

/**
 * \brief Funkcija vraća 1.
 *
 * Funkcija ne prima nikakve ulazne parametre i uvijek vraća vrijednost 1.
 *
 * \return cjelobrojna vrijednost 1
 */
int funkcija1() { return 1; }
```

Uočimo kako Doxygenov blok kreće znakom višeretčanog komentara `/*` i još jednim dodatnim znakom zvjezdice `*`. Prvi od blokova je dokumentacija same datoteke koja kreće naredbom `\file` i imenom datoteke. Zatim imamo krati opis naveden korištenjem naredbe `\brief` i dugi opis odmaknut od njega za jedan prazni redak.

Dokumentacija pojedine funkcije također ima kratak (`\brief`) i dugi opis te navodi što funkcija vraća naredbom `\return`.

Dodajmo u istu datoteku još jednu dokumentiranu funkciju `zbroji()` oblika:

``` c++
/**
 * \brief Funkcija vrši zbrajanje brojeva.
 *
 * Funkcija računa zbroj dva cijela broja koje prima kao ulazne parametre.
 *
 * \param a prvi cijeli broj
 * \param b drugi cijeli broj
 * \return zbroj
 */
int zbroji(int a, int b) { return a + b; }
```

Ovdje vidimo i korištenje naredbi `\param` kojima navodimo pojedine parametre s njihovim imenima i ulogom. Ukupno je sada kod oblika:

``` c++
/**
 * \file hello.h
 * \brief Dokumentirana datoteka.
 *
 * Datoteka u kojoj su dokumentirane funkcije.
 */

/**
 * \brief Funkcija vraća 1.
 *
 * Funkcija ne prima nikakve ulazne parametre i uvijek vraća vrijednost 1.
 *
 * \return cjelobrojna vrijednost 1
 */
int funkcija1() { return 1; }

/**
 * \brief Funkcija vrši zbrajanje brojeva.
 *
 * Funkcija računa zbroj dva cijela broja koje prima kao ulazne parametre.
 *
 * \param a prvi cijeli broj
 * \param b drugi cijeli broj
 * \return zbroj
 */
int zbroji(int a, int b) { return a + b; }
```

## Generiranje HTML-a i PDF-a

Pokrenimo generiranje dokumentacije Doxygenom:

``` shell
$ doxygen
Doxygen version used: 1.8.20
Searching for include files...
Searching for example files...
Searching for images...
Searching for dot files...
Searching for msc files...
Searching for dia files...
Searching for files to exclude
Searching INPUT for files to process...
(...)
```

U direktoriju `html` dobit ćemo izgrađenu dokumentaciju koju možemo krenuti pregledavati otvaranjem datoteke `index.html`, a u direktoriju `latex` dobit ćemo dokumentaciju pisanu u LaTeX-u koju, ako imamo instaliran [pdfLaTeX](https://www.tug.org/applications/pdftex/) i sve potrebne pakete, možemo izgraditi naredbom `make` na način:

``` shell
$ cd latex
$ make
rm -f *.ps *.dvi *.aux *.toc *.idx *.ind *.ilg *.log *.out *.brf *.blg *.bbl refman.pdf
pdflatex refman
This is pdfTeX, Version 3.14159265-2.6-1.40.21 (TeX Live 2020/Debian) (preloaded format=pdflatex)
restricted \write18 enabled.
entering extended mode
(./refman.tex
LaTeX2e <2020-02-02> patch level 5
L3 programming layer <2020-09-24>
(/usr/share/texlive/texmf-dist/tex/latex/base/book.cls
Document Class: book 2019/12/20 v1.4l Standard LaTeX document class
(...)
```

čime ćemo dobiti nekoliko privremenih datoteka (`refman.ind`, `refman.ilg`, `refman.toc`, `refman.out`, `refman.idx`, `refman.aux`), rezultirajući PDF (`refman.pdf`) i log datoteku s detaljima prevođenja iz LaTeX-a u PDF (`refman.log`).

!!! admonition "Zadatak"
    Deklarirajte klasu `PodcastPlayer` te metode `playEpisode(int episodeId, std::chrono::duration intervalToSkip)`, koja započinje emitiranje epizode podcasta s danim rednim brojem i to tako da preskoči dano vrijeme od početka, i `stop()`, koja zaustavlja emitiranje epizode koja se trenutno emitira. (Metode ne morate implementirati.) Dokumentirajte metode korištenjem Doxygena, a zatim izvedite generiranje dokumentacije. (*Uputa:* dokumentiranje klasa vrši se na isti način kao dokumentiranje funkcija, a primjer možete pronaći u dijelu Doxygenove dokumentacije [Documenting the code](https://www.doxygen.nl/manual/docblocks.html).)
