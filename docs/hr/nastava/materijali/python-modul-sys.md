---
author: Vedran Miletić
---

# Python: parametri i funkcije ovisni o sustavu

- `module sys` ([službena dokumentacija](https://docs.python.org/3/library/sys.html)) nudi pristup objektima koje interpreter koristi i s kojima komunicira

!!! admonition "Zadatak"
    Pokrenite interpreter Pythona 3 u terminalu ili u razvojnom okruženju i učitajte modul `sys`.

    - Otkrijte kojeg su tipa objekti `sys.ps1` i `sys.ps2`.
    - Promijenite im vrijednost na drugu vrijednost po vašem izboru. Što uočavate? Fora, ha? ;-)

- `sys.version` je verzija interpretera Pythona i program prevoditelja kojim je preveden, npr. `'3.8.1 (default, Jan 22 2020, 06:38:00) \n[GCC 9.2.0]'` ili `'3.7.6 (default, Jan  2 2020, 01:19:56) \n[Clang 6.0.1 (tags/RELEASE_601/final 335540)]'`
- `sys.platform` je platforma (operacijski sustav), npr. `'linux'` ili `'freebsd12'`
- `sys.modules` je rječnik učitanih modula
- `sys.getdefaultencoding()` vraća najčešće `'utf-8'` (slučajevima kad nije tako se ovdje nećemo baviti)
- `sys.getfilesystemencoding()` vraća najčešće `'utf-8'` (slučajevima kad nije tako se ovdje nećemo baviti)
- `sys.getsizeof(object)` vraća veličinu objekta `object` u bajtovima
- `sys.int_info` daje informacije o načinu spremanja cijelih brojeva
- `sys.maxsize` daje informacije o maksimalnoj duljini liste (`list`), uređene n-torke (`tuple`), skupa (`set`) ili rječnika (`dict`)
- `sys.exit([status])` izlazi iz interpretera s povratnim kodom `status`

    - raspon povratnih kodova (izlaznih statusa) je 0--127 i 129--192

        - postoje pokušaji standardizacije značenja svakog pojedinog broja, primjerice `sysexits.h`
        - primjerice, naredba `man` ima izlazni status 0 nakon izlaska iz pregledavanja neke stranice tipkom `q`, izlazni status 1 i poruku `What manual page do you want?` u slučaju da nije navedeno ime stranice te izlazni status 16 u slučaju da stranica ne postoji ili ne postoji u navedenom odjeljku

    - u ljusci `bash` varijabla ljuske `$?` sadrži izlazni status i može se ispisati naredbom `echo $?`

- `sys.argv` je lista argumenata proslijeđenih Python skripti kod pokretanja

    - `argv[0]` je ime Python skripte
    - `argv[1]`, `argv[2]`, ... su redom prvi, drugi, ... argument naveden nakon imena skripte
    - primjerice, pokrenemo li Python skriptu na način `./program.py -x -z foo bar baz`, tada će `argv` biti sadržaja `['./program.py', '-x', '-z', 'foo', 'bar', 'baz']` pa će `argv[0]` biti `'./program.py'`, a `argv[1]`, `argv[2]`, ..., `argv[5]` redom `'-x'`, `'-z'`, `'foo'`, `'bar'`, `'baz'`

!!! admonition "Zadatak"
    Napišite Python skriptu imena `zadatak-argv.py` koja će nakon što je iz naredbenog retka pozovemo s argumentom `corona` (`./zadatak-argv.py corona` ili `python3 zadatak-argv.py corona`) ispisati:

    - `Pozvali ste skriptu imena zadatak-argv.py s argumentom corona`, pri čemu su `zadatak-argv.py` i `corona` vrijednosti varijabli.
    - Ako je prvi argument jednak `pythonversion`, onda će taj program pored toga ispisati verziju Python interpretera kojeg koristimo, u našem slučaju `'3.8.1'`.
    - Ako je prvi argument jednak `compilerversion`, onda će taj program ispisati verzi GCC-a kojom je korišteni Python interpreter preveden, u našem slučaju `'GCC 9.2.0'`.

    (**Uputa:** razmislite kako ćete iz `sys.version` izvući vrijednosti koje vam trebaju.)

!!! admonition "Zadatak"
    Modificirajte program iz prethodnog zadatka tako da:

    - U slučaju da korisnik unese nedozvoljnu vrijednost prvog argumenta skripta na standardni izlaz za greške ispisuje poruku o tome, a zatim završava izvođenje (izlazi iz interpretera) s izlaznim statusom 1.
    - U slučaju da korisnik unese manje ili više argumenata skripta na standardni izlaz za greške ispisuje poruku o tome, a zatim završava izvođenje (izlazi iz interpretera) s izlaznim statusom 2.

    (**Uputa:** Pogledajte u pomoći za funkciju `print()` kako ispisati sadržaj na standardni izlaz za greške).

!!! admonition "Zadatak"
    Napišite vlastiti `cat`, odnosno Python skriptu imena `zadatak-cat.py` koja će nakon što je iz naredbenog retka pozovemo s argumentom imena datoteke ispisati sadržaj datoteke (primjerice, ako želimo ispisati sadržaj datoteke `/etc/fstab`, to ćemo napraviti pozivom `./zadatak-cat.py /etc/fstab` ili `python3 zadatak-cat.py /etc/fstab`). (**Uputa:** podsjetite se kako se čita sadržaj datoteke u Pythonu.)

    Omogućite da vaša naredba prima argument `-n` (npr. `./zadatak-cat.py -n /etc/fstab` ili `python3 zadatak-cat.py -n /etc/fstab`) i tada ispisuje broj linije uz svaku liniju koju ispiše, slično kako to radi naredba `cat` s parametrom `-n`. U slučaju da korisnik ne unese `-n`, naredba ispisuje samo sadržaj datoteke bez brojeva linija. (**Uputa:** podsjetite se kako se cijepa tekst po linijama da možete na svaku dodati broj.)

- opisnik datoteke (engl. *file descriptor*) je cjelobrojni indeks u tablici otvorenih datoteka vezanih uz proces
- svaki proces ima barem 3 opisnika: standardni ulaz, standardni izlaz i standardni izlaz za greške

    - objekt `sys.stdin` reprezentira standardni ulaz (opisnik 0)
    - objekt `sys.stdout` reprezentira standardni izlaz (opisnik 1)
    - objekt `sys.stderr` reprezentira standardni izlaz za greške (opisnik 2)

- naredba `lsof` ispisuje popis otvorenih datoteka na sustavu

    - `lsof -p pid` ispisuje popis datoteka koje je otvorio proces s PID-om `pid`

!!! admonition "Zadatak"
    - Isprobajte funkciju `sys.stdout.write()` i objasnite na koji način radi različito od `print()`. Prima li više od jednog argumenta? Dodaje li znak za novi redak na kraju ispisa? Možete li ispisati listu, cijeli ili realan broj?
    - Otvorite u Python interpreteru proizvoljnu datoteku iz vašeg direktorija. Pronađite u popisu otvorenih datoteka na sustavu vaš proces, 3 osnovna opisnika i opisnik te datoteke.
