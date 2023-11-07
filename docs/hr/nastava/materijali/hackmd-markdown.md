---
author: Matea Turalija, Dejan Ljubobratović
---

# Suradnički uređivač teksta HackMD i jezik Markdown

[Markdown](https://daringfireball.net/projects/markdown/) je jednostavan jezik za pisanje oblikovanog teksta korištenjem čistog teksta. Njegova osnova karakteristika je jednostavnost i lakoća upotrebe, čineći ga time popularnim alatom za oblikovanje i uređivanje teksta bez potrebe za naprednim tehničkim vještinama. Najčešće se koristi za [pisanje znanstvenih članaka](https://jaantollander.com/post/scientific-writing-with-markdown/), README datoteka i objava na forumima. Više detalja o Markdownu moguće je pronaći na [Wikipedijinoj stranici Markdown](https://en.wikipedia.org/wiki/Markdown).

Datoteke napisane u Markdownu obično koriste ekstenzije kao što su `.md` ili `.markdown`. Na primjer, `primjer.md` je naziv tipične datoteke sa sadržajem u obliku Markdown.

## Uređivači Markdowna

Offline:

- [Visual Studio Code](https://code.visualstudio.com/docs/languages/markdown)
- [Zettlr](https://www.zettlr.com/)
- [ghostwriter](https://ghostwriter.kde.org/)
- [MarkText](https://www.marktext.cc/)
- [RStudio](https://posit.co/download/rstudio-desktop/) ([R Markdown](https://rmarkdown.rstudio.com/) kombinira jezike [R](https://www.r-project.org/) i Markdown)

Online:

- [HackMD](https://hackmd.io/)
- [Visual Studio Code](https://vscode.dev/)
- [Google Colab](https://colab.google/)
- [Jupyter notebooks](https://jupyter.org/)
- [StackEdit](https://stackedit.io/)
- [Dilinger](https://dillinger.io/)
- [Typora](https://typora.io/)
- [Editor.md](https://pandao.github.io/editor.md/en.html)

## Mrežni uređivač Markdowna HackMD

### Kreiranje računa i prijava

Posjetite web stranicu [hackmd.io](https://hackmd.io/) i registrirate se putem ekrana za registraciju. Unesite svoju željenu e-mail adresu i slijedite upute za kreiranje novog korisničkog računa.

Možete koristiti i druge servise za prijavu poput Google ili GitHub računa.

### Moj radni prostor

Kada se prijavite nalazit ćete se u vlastitom radnom prostoru (engl. *My workspace*). Po potrebi možete stvarati odvojene timske radne prostore odabirom *Create new team*.

Novu bilješku možete kreirati klikom na zelenu ikonicu *New note*.

[Sučelje HackMD](https://hackmd.io/c/tutorials/%2Fs%2Ftutorials)-a sastoji se od više ekrana, koji služe za:

- uređivanje,
- pregled,
- podjelu ekrana,
- kreiranje osobne bilješke,
- pomoć,
- pretragu bilješki,
- dijeljenje...

### Uređivanje teksta

#### Naslovi

Naslove označavamo na način da ispred teksta stavimo oznaku `#`. Jedan znak `#` ispred naslova je glavni naslov, dva znaka `##` je naslov druge razine i tako dalje.

!!! admonition "Zadatak"
    - Napišite naslov `Suradnički uređivač teksta HackMD i jezik Markdown` s podnaslovom `Moje bilješke s vježbi`.
    - Isprobavanjem otkrijte koliko razina poglavlja podržava jezik Markdown.

#### Podebljano, kurziv i precrtano

Jednostavno oblikovanje teksta poput korištenja kurziva, podebljanih ili precrtanih riječi pomaže istaknuti važne koncepte unutar dokumenta i učiniti ga čitljivijim.

Podebljana slova (engl. *bold*) u tekstu označavaju se dvjema zvjezdicama `**` prije i nakon riječi. Neki uređivači Markdowna uz zvjezdicu podržavaju i korištenje donjih crta `_` i `__` (engl. *underscore*).

Ukošena slova (engl. *italic*) u tekstu označavaju se jednom zvjezdicom `*` prije i nakon riječi.

Istovremeno podebljana i ukošena slova možemo dobiti korištenjem triju zvjezdica `***` ili kombinacijom s donjim crtama `**_`, `__*`.

Precrtana slova (eng. *strikethrough*) u tekstu označavaju se s dvije tilde `~~` (++alt-graph+1++) prije i nakon teksta.

!!! admonition "Zadatak"
    Napišite sljedeću rečenicu:

    > Ovo je primjer gdje ću koristiti **podebljana**, *ukošena*, ***istovremeno podebljana i ukošena*** te ~~precrtana~~ slova.

#### Komentar

Ukoliko želimo u dokumentu imati komentar koji se neće prikazivati u oblikovanom tekstu, koristimo sljedeću sintaksu:

`<!-- komentar -->`

``` markdown
<!--
Ovaj tekst se
neće vidjetu u izlazu,
to je moj komentar.
-->
```

#### Boja slova

Za promjenu boje slova koristimo [HTML](https://en.wikipedia.org/wiki/HTML) tag `font color` na primjer: `<font color="yellow"> Žuti tekst </font>`.

#### Citiranje

Citat se piše tako da se na početak retka stavi znak `>`.

!!! admonition "Zadatak"
    Napišite sljedeći citat:

    > Ja ne mogu nikoga ništa naučiti, ja ih samo mogu natjerati da misle.

    -- Sokrat

#### Numerirane i nenumerirane liste

Numeriranu listu dobijemo stavljajući redne brojeve prije svakog retka:

``` markdown
Za napraviti:

1. zaliti cvijeće,
2. nahraniti psa,
3. prošetati se.
```

Nenumeriranu listu dobijemo stavljajući znakove `-` ili `*` prije svakog retka:

``` markdown
Moram kupiti:

- kruh,
- mlijeko,
- novine.
```

Potvrdni okvir (engl. *checkbox*) dobijemo stavljajući znak `- [ ]`, odnosno `- [x]` prije svakog retka:

``` markdown
- [ ] za neoznačeni potvrdni okvir
- [x] za označeni potvrdni okvir
```

!!! admonition "Zadatak"
    Napravite sljedeću listu:

    * jezik Markdown
        * jednostavan jezik za pisanje teksta
    * najpoznatiji uređivači jezika Markdown
        1.  Visual Studio Code
            * popularno razvojno okruženje
        2.  HackMD
            * online uređivač
        3. Zettlr
            * offline uređivač

#### Linkovi

Za umetanje linka koristi se sljedeća sintaksa:`[Naziv linka](http://gaseri.org)`.

Ukoliko ne želimo koristiti posebni tekst kao link onda je dovoljno unijeti adresu web stranice unutar znakova `<` i `>`, a Markdown će automatski kreirati link: `<http://gaseri.org>`.

#### Umetanje slika

Za umetanje slike s računala dovoljno kliknuti na *Insert image* ikonicu na alatnoj traci. Takvo ubacivanje slike u nekim markdown uređivačima napravit će gomilu ružnog koda. Zato se preporučuje korištenje online slika.

Za umetanje online slika, dovoljno je u zagradu staviti link na sliku: `![Naziv slike](https://group.miletic.net/images/gaseri-logo-animated.webp)`.

#### Escape znak

Da bi mogli prikazati neku naredbu ili specijalni znak, koristi se escape znak `\` ispred rezervirane riječi ili znaka.

!!! admonition "Zadatak"
    Napišite rečenicu koja će u izlazu Markdowna izgledati ovako:

    > Riječi možemo napisati \*ukošeno*, \*\*podebljano** ili \~\~precrtano~~.

#### Navođenje dijelova koda

Da bi mogli napisati dijelove koda koristimo oznaku `` ` `` (++alt-graph+7++) na početku i na kraju citiranog koda. Ako želimo da se prikaže više linija koda, koristiti ćemo tri oznake `` ``` `` ili tri tilde `~~~` na početku i na kraju citiranog koda.

Ukoliko želimo da se kod formatira za određeni programski jezik, pored tri oznake na početku koda napisat ćemo skraćenicu programskog jezika, primjerice `` ``` python``. Primjer:

```` markdown
``` python
for i in [1, 2, 3, 4]:
    print("broj", i)
```
````

#### Tablice

Tablice se kreiraju korištenjem okomitih crta (++alt-graph+w++). Ispod zaglavlja potrebno je staviti crtice `-`:

``` markdown
| broj | ime | oznaka |
| ---- | --- | ------ |
| 1 | Ivo | $4A |
| 5 | Eva | $M5 |
| 2 | Ira | $8D |
```

Za tablice bez zaglavlja ostavljamo prvi redak praznim. Širine ćelija će se automatski prilagođavati širini teksta:

``` markdown
|   |   |   |
| - | - | - |
| 1 | Ivo | $4A |
| 5 | Eva | $M5 |
| 2 | Ira | $8D |
```

##### Poravnavanje tablica

Različita poravnanja teksta u tablicama dobivamo dodavanjem znaka ':' u dio tablice sa crticama:

- `:---` lijevo poravnanje (često zadana postavka, ali ne mora biti),
- `:---:` centrirano poravnanje i
- `---:` desno poravnanje.

!!! admonition "Zadatak"
    Promijenite tablicu iz primjera tako da:

    - dodate još jedan redak s podacima po vašoj želji,
    - dodate još stupac prezime s pomno odabranim prezimenima s [Acta Croatice](https://actacroatica.com/),
    - prvi stupac poravnate desno,
    - stupce s imenom i prezimenom poravnate centrirano te
    - stupac s oznakom poravnate lijevo.

#### Matematičke formule

Markdown koristi jezik [LaTeX](https://www.latex-project.org/) za pisanje matematičkih formula. Da bi napisali matematičku formulu, koristimo znak `$` ukoliko je formula u tekstu ili `$$` za samostalnu formulu (u svom retku centrirano). Detaljnije o pisanju matematičkih formula imate na [Wikibooks stranici LaTeX/Mathematics](https://en.wikibooks.org/wiki/LaTeX/Mathematics).

!!! admonition "Zadatak"
    Napišite sljedeće dvije rečenice:

    > Pitagorin poučak glasi: $c^2 = a^2 + b^2$.

    i

    > Jednadžba pravca zadana je izrazom:
    >
    > $$y = ax + b,$$
    >
    > pri čemu $a$ predstavlja koeficijent smjera.

Za pisanje kemijskih formula također koristimo jezik LaTeX i pritom koristimo simbole atoma kao `\text{}`, npr `\text{H}`.

!!! admonition "Zadatak"
    Napišite sljedeću kemijsku reakciju:

    > $\text{H}_2\text{S}\text{O}_4 + 2\text{Na}\text{O}\text{H} \to 2\text{H}_2\text{O} + \text{Na}^+ + \text{S}\text{O}_4^{2-}$.

    Za strelicu koristite oznaku `\to`.

#### Emoji

Markdown jezik podržava korištenje emoji simbola, na način da se emoji kod ubaci između znaka `:`. Popis emoji kodova možete pronaći na [GitHubu ikatyang/emoji-cheat-sheet](https://github.com/ikatyang/emoji-cheat-sheet).

!!! admonition "Zadatak"
    Napišite sljedeće:

    > :information_source: [Jedna](../../vrlo-vazne-informacije/identitet.md) [vrlo](../../vrlo-vazne-informacije/hijerarhija-gasera.md) [važna](../../vrlo-vazne-informacije/index.md) [informacija](../../vrlo-vazne-informacije/cesto-postavljana-pitanja.md)!  
    > Ja jako :heart: učiti Markdown korištenjem materijala s :sunglasses: web stranica.  
    > I zato sam :star_struck:.

### Izrada prezentacije

Slajdovi u prezentaciji odvajaju se s tri crtice `---`, ali obavezno mora biti jedan prazan redak prije i nakon njih. Pod `Customize slides options` unosimo osnovne postavke naše prezentacije u obliku [YAML](https://yaml.org/) preslikavanja:

``` yaml
theme: solarized
transition: fade
```

!!! admonition "Zadatak"
    Kreirajte novi dokument s naslovom "Moja prva prezentacija" koristeći `.yaml` zaglavlje iz prethodnog primjera.

#### Teme prezentacije

Možemo birati između desetak osnovnih tema (engl. *theme*):

|   |   |
| - | - |
| `black` | Black background, white text, blue links (default) |
| `white` | White background, black text, blue links |
| `league` | Gray background, white text, blue links |
| `beige` | Beige background, dark text, brown links |
| `sky` | Blue background, thin dark text, blue links |
| `night` | Black background, thick white text, orange links |
| `serif` | Cappuccino background, gray text, brown links |
| `simple` | White background, black text, blue links |
| `solarized` | Cream-colored background, dark green text, blue links |
| `blood` | Dark background, thick white text, red links |
| `moon` | Dark blue background, thick grey text, blue links |

#### Prijelazi između slajdova

Možemo birati između šest osnovnih prijelaza (engl. *transition*):

|   |   |
| - | - |
| `none` | Switch backgrounds instantly |
| `fade` | Cross fade — default for background transitions |
| `slide` | Slide between backgrounds — default for slide transitions |
| `convex` | Slide at a convex angle |
| `concave` | Slide at a concave angle |
| `zoom` | Scale the incoming slide up so it grows in from the center of the screen |

Kroz prezentaciju se krećemo uobičajeno: strelicama lijevo desno ili razmaknicom, a možemo i mišem na način da kliknemo na strelicu pri dnu slajda.

#### Podslajd

Za razliku od klasičnih prezentacija, u Markdownu možete imati i podslajdove ili sekcije. Njih u prezentaciji pregledavamo klikom na strelicu dole. Vrlo je korisno jer dodaje novu dimenziju prezentaciji. Podslajd kreiramo tako da slajd odvojimo s četiri `----` crtice umjesto tri `---` kao do sada.

!!! admonition "Zadatak"
    Napravite prezentaciju s barem 5 slajdova koja sadrži sljedeće elemente: formulu, tablicu, listu, emoji, link do neke web stranice, sliku i podslajd.

Kada smo gotovi s izradom prezentacije možemo ju prikazati klikom na `Share` pa pod `Slide mode` odaberemo `Preview`.
