---
author: Vedran Miletić
---

# Rad s Python okvirom waf

!!! note
    Materijali su složeni prema [Waf Tutorialu](http://docs.waf.googlecode.com/git/apidocs_17/tutorial.html). Za više detalja proučite [ostalu Waf dokumentaciju](http://docs.waf.googlecode.com/git/apidocs_17/index.html) i [Waf Book](http://docs.waf.googlecode.com/git/book_17/single.html).

[Sustavi za automatizaciju izgradnje softvera](https://en.wikipedia.org/wiki/Build_automation) (engl. *build automation system*) su aplikacije koje automatiziraju ili skriptiraju aktivnosti u procesu izgradnje softvera.

Primjeri takvih alata su [Autotools](https://en.wikipedia.org/wiki/GNU_build_system), [CMake](https://en.wikipedia.org/wiki/CMake) [qmake](https://en.wikipedia.org/wiki/Qmake) (dio [Qt](https://en.wikipedia.org/wiki/Qt_(software))-a, [SCons](https://en.wikipedia.org/wiki/SCons)>__, [Waf](https://en.wikipedia.org/wiki/Waf) i drugi.

Autotools i CMake rade tako da generiraju datoteku `./configure`, a ponekad i datoteku `Makefile`. Korisnik ih najčešće pokreće putem prethodno napravljenih skripti za kompajiranje programa, a za samu izgradnju koristi `make`.

[Waf](http://code.google.com/p/waf/) je okvir za konfiguraciju, prevođenje i instalaciju aplikacija. Za razliku od Autotools-a i CMake-a, Waf ne koristi `make`.

Da bi iskoristili Waf na svojem projektu, postupak je vrlo jednostavan:

- `$ cd myproject`
- `$ curl -o waf http://ftp.waf.io/pub/release/waf-1.7.16`
- `$ chmod +x waf`
- u istom direktoriju treba napravit `wscript` sa sadržajem

    ``` python
    def configure(conf):
        print("configure!")

    def build(bld):
        print("build!")
    ```

    čime smo omogućili pokretaje naredbi:

    - `$ ./waf configure`
    - `$ ./waf build`

!!! todo
    Ovdje nedostaje zadatak.

Faze izgradnje softvera su:

- `configure`
- `build`
- `install`
- `uninstall`
- `dist` -- stvara datoteku s izvornim kodom
- `clean` -- čisti datoteke stvorene u `build` fazi
