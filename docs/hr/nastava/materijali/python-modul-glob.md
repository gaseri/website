---
author: Vedran Miletić
---

# Python: ekspanzija uzoraka imena putanje u stilu operacijskih sustava sličnih Unixu

- `module glob` ([službena dokumentacija](https://docs.python.org/3/library/glob.html)) omogućuje ekspanziju imena putanje koja sadrže posebne znakove (tzv. [glob](https://en.wikipedia.org/wiki/Glob_(programming))), slično kao što radi ljuska Bash
- `glob.glob(pathname)` vraća popis datoteka koje se podudaraju s glob uzorkom `pathname` kao listu

    ``` python
    slike = glob.glob('./*.jpg') # vraća sve JPEG slike iz trenutnog direktorija
    confs = glob.glob('/etc/????.conf') # vraća sve datoteke čija imena imaju četiri znaka i nastavak .conf
    ```

- `glob.iglob(pathname)` vraća popis datoteka koje se podudaraju s glob uzorkom `pathname` kao iterator koji je preferiran pred listom za korištenje u for petlji, npr. `for path in glob.iglob("/home/korisnik/Glazba/*.m4a"): (..)`

!!! admonition "Zadatak"
    - Dohvatite sve datoteke u `/var/log` koje imaju nastavak `.log`.
    - Dohvatite sve datoteke u `/var/log` čije ime počinje malim slovom, a nastavak završava brojem (npr. `btmp.1`, `wtmp.1`, `dmesg.0`, `auth.log.1`).
