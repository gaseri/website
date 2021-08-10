---
author: Vedran Miletić
---

# Dokumentiranje programa alatom MkDocs

[MkDocs](https://www.mkdocs.org/) je brz i jednostavan alat za generiranje projektne dokumentacije. Izvorni kod dokumentacije piše se u [formatu Markdown](https://commonmark.org/help/), a MkDocs generira izlazni HTML u skladu s postavkama navedenim u konfiguracijskoj datoteci u [formatu YAML](https://yaml.org/).

## Stvaranje početnih datoteka

Naredbom `mkdocs` s argumentom `new` možemo stvoriti početnu konfiguraciju (datoteka `mkdocs.yml`) i primjer dokumentacije (datoteka `docs/index.md`):

``` shell
$ mkdocs new moja-dokumentacija
$ cd moja-dokumentacija
```

Prevođenje izvornog koda u HTML za pregled vršimo naredbom `mkdocs` s argumentom `serve`:

``` shell
$ mkdocs serve
INFO    -  Building documentation...
INFO    -  Cleaning site directory
INFO    -  Documentation built in 0.22 seconds
[I 201101 19:13:50 server:296] Serving on http://127.0.0.1:8000
[I 201101 19:13:50 handlers:62] Start watching changes
```

Izlaz možemo pregledati otvaranjem web preglednika na adresi `127.0.0.1` i vratima `8000`.

## Uređivanje konfiguracije i sadržaja

Markdown i YAML datoteke možemo uređivati korištenjem bilo kojeg uređivača teksta. Primjerice, možemo koristiti Visual Studio Code koji ima [ugrađenu podršku za Markdown](https://code.visualstudio.com/Docs/languages/markdown) i [Red Hatovo proširenje za YAML](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml) tako da u njemu otvorimo direktorij `moja-dokumentacija`.

U datoteci `mkdocs.yml` promijenimo ime stranice promjenom postavke `site_name`:

``` yaml
site_name: Moja dokumentacija
```

Iskoristimo [Lorem Markdownum](https://jaspervdj.be/lorem-markdownum/) za generiranje sadržaja i spremimo rezultat u datoteku `docs/lorem.md`. Definirajmo navigaciju da uključuje obje datoteke tako da u `mkdocs.yml` dodamo naredbu `nav` i navedemo obje datoteke s odgovarajućim naslovima:

``` yaml
site_name: Moja dokumentacija
nav:
    - Početna: index.md
    - 'Lorem Markdownum': lorem.md
```

Pritom smo naslov `'Lorem Markdownum'` stavili pod navodnike zbog razmaka u riječi. Nakon spremanja MkDocs će sam uočiti da su datoteke izmijenjene i ponovno izgraditi HTML datoteke. Kad smo zadovoljni rezultatom, izgradnju web sjedišta pokrenut ćemo naredbom:

``` shell
$ mkdocs build
```

što će stvoriti direktorij `site` i u njemu HTML datoteke `index.html` i `lorem/index.html`, zbog čega ćemo moći imati lijepe URL-e `/` i `/lorem/` (respektivno). Ova naredba ima nekoliko parametara čiji je opis moguće dobiti parametrom `--help`, a istaknut ćemo samo parametar `--clean` koji će isprazniti direktorij `site` prije izgradnje sjedišta, što je korisno za brisanje datoteka pod starim imenima nakon preimenovanja.

!!! admonition "Zadatak"
    Proučite upute za generiranje hijerarhije u navigaciji u dijelu [Writing your docs](https://www.mkdocs.org/user-guide/writing-your-docs/). Zatim generirajte dva nova dokumenta korištenjem Lorem Markdownuma i spremite ih u datoteke `lorem-est.md` i `lorem-qui.md` te sva tri dokumenta postavite u navigaciji pod 'Lorem Markdownum' s naslovima pojedinih dokumenata u skladu s naslovima koje je generator dao.

!!! admonition "Zadatak"
    Proučite upute za konfiguraciju [Configuration](https://www.mkdocs.org/user-guide/configuration/) pa postavite ime autora, opis sjedišta i informacije o autorskom pravu, a zatim pogledajte izvorni kod stvorenog HTML-a te u njemu pronađite postavljene informacije. Uključite ekstenziju [SmartyPants](https://python-markdown.github.io/extensions/smarty/) te u nekom od dokumenata isprobajte navodnike te dvostruke i trostruke crtice da se uvjerite da radi.
