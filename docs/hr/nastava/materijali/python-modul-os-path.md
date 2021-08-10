---
author: Vedran Miletić
---

# Python: općenite usluge operacijskog sustava: baratanje imenom putanje

- `module os.path` ([službena dokumentacija](https://docs.python.org/3/library/os.path.html)) pruža prenosiv način za baratanje imenom putanje

    - kao i kod modula `os`, prenosiv u terminima da radi na više različitih operacijskih sustava (operacijski sustavi slični Unixu i Windows) pa u dokumentaciji svake pojedine funkcije piše dostupnost

- `os.path.abspath(path)` vraća apsolutnu putanju za danu relativnu ili apsolutnu putanju `path`
- `os.path.basename(path)` vraća ime datoteke ili direktorija za danu relativnu ili apsolutnu putanju `path`
- `os.path.isabs(path)` vraća `True` ako je putanja apsolutna
- `os.path.exists(path)` vraća `True` ako putanja `path` postoji
- `os.path.realpath(path)` vraća putanju do datoteke u kanonskom obliku: ako je `path` simbolička poveznica, slijedi tu simboličku poveznicu; ako je rezultat opet simbolička poveznica, slijedi je i tako redom dok ne dođe do odredišta koje nije simbolička poveznica

!!! admonition "Zadatak"
    Stvorite u direktoriju u kojem ćete pokrenuti Python skriptu dvije simboličke poveznice: prva je imena `lnkhosts` na datoteku `/etc/hosts`, a druga je `nema-dat` na `/tmp/nepostojeca-datoteka`. U Python skripti prvo saznajte apsolutne putanje do stvorenih poveznica, a zatim provjerite postoje li vaše poveznice i postoje li datoteke na koje pokazuju.

- `os.path.getatime(path)` dohvaća zadnje vrijeme pristupa datoteci kao varijablu tipa `float` u kojoj je zapisan broj sekundi od epohe
- `os.path.getmtime(path)` dohvaća zadnje vrijeme promjene datoteke kao varijablu tipa `float` u kojoj je zapisan broj sekundi od epohe
- `os.path.getctime(path)` dohvaća zadnje vrijeme promjene metapodataka datoteke kao varijablu tipa `float` u kojoj je zapisan broj sekundi od epohe
- `os.path.getsize(path)` dohvaća veličinu u bajtovima

- `os.path.isfile(path)` vraća `True` ako je `path` putanja do obične datoteke ili simbolička poveznica na datoteku
- `os.path.isdir(path)` vraća `True` ako je `path` putanja do direktorija ili simbolička poveznica na direktorij
- `os.path.islink(path)` vraća `True` ako je `path` putanja simboličke poveznice

!!! admonition "Zadatak"
    Saznajte posljednje vrijeme pristupa, vrijeme promjene metapodataka i veličinu datoteke `/etc/hosts`, a zatim i simboličke poveznice `lnkhosts`. Uvjerite se da se zaista radi o datoteci i simboličkoj poveznici.

- `os.path.expanduser(path)` zamjenjuje `~` ili `~guido` putanjom do korisničkog direktorija trenutnog korisnika, odnosno korisnika `guido` (ako taj korisnik postoji); npr. `os.path.expanduser("~")` vratit će `"/home/korisnik"`
- `os.path.expandvars(path)` zamjenjuje varijablu ljuske njezinom vrijednošću; npr. `os.path.expandvars("$HOME/Radna površina")` vratit će `"/home/korisnik/Radna površina"`
- `os.path.join(path, *paths)` spaja dijelove putanje razdjelnicima putanje odgovarajućim za operacijski sustav na kojem radi (`/` na operacijskim sustavima sličnim Unixu, `\` na Windowsima); npr. `os.path.join("/home", "korisnik", "Radna površina")` vratit će `"/home/korisnik/Radna površina"`
- `os.path.split(path)` cijepa putanju u dva dijela, prvi je putanja do datoteke ili direktorija, a drugi je ime datoteke ili direktorija; npr. `os.path.split("/home/korisnik/Radna površina")` vraća uređeni par `("/home/korisnik", "Radna površina")`
