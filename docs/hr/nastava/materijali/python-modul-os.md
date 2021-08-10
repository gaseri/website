---
author: Vedran Miletić
---

# Python: općenite usluge operacijskog sustava: osnovna sučelja

- `module os` ([službena dokumentacija](https://docs.python.org/3/library/os.html)) pruža prenosiv način za pristup sučeljima operacijskog sustava

    - prenosiv u terminima da radi na više različitih operacijskih sustava (operacijski sustavi slični Unixu i Windows)
    - u dokumentaciji svake pojedine funkcije piše dostupnost; velik broj funkcija su specifične za operacijske sustave slične Unixu ili dostupne na operacijskim sustavima sličnim Unixu i na Windowsima, a manji broj funkcija su isključivo dostupne na Windowsima

## Dohvaćanje informacija i izvođenje osnovnih operacija

- `os.uname()` vraća informacije o operacijskom sustavu u objektu koji ima atribute `sysname`, `nodename`, `release`, `version` i `machine`, slično kao naredba `uname`; primjerice, kod

    ``` python
    import os

    uname = os.uname()
    print(uname.sysname)
    ```

    ispisat će `Linux`, `FreeBSD`, ... (radi samo na operacijskim sustavima sličnim Unixu)

- `os.getlogin()` vraća korisničko ime korisnika kojem pripada proces (radi na operacijskim sustavima sličnim Unixu i na Windowsima)
- `os.kill(pid, sig)` šalje signal `sig` procesu s PID-om `pid`, slično kao naredba `kill`
- `os.nice(increment)` povećava niceness procesa, slično kao naredba `nice`
- `os.getpid()` vraća PID procesa interpretera
- `os.getppid()` vraća PID roditelja (PPID) procesa interpretera

!!! admonition "Zadatak"
    Provjerite u službenoj dokumentaciji za preostale funkcije istaknute iznad rade li na (nekim) operacijskim sustavima sličnim Unixu i/ili na Windowsima.

!!! admonition "Zadatak"
    Napravite program koji radi jednostavan login. Korisnika se pita da upiše korisničko ime, a onda se provjerava je li to korisničko ime isto kao ime korisnika kojem pripada proces. Ako je, ispisuje se PID procesa i PID roditelja procesa, a korisniku se nudi mogućnost unosa dva broja: prvi broj je PID procesa, a drugi je signal. Program zatim šalje uneseni signal procesu s PID-em koji je korisnik unio.

- razlikujemo dva tipa korisničkih (UID) i grupnih ID-eva (GID)

    - stvarni UID i GID utječu na dozvole za slanje signala procesima
    - efektivni UID i GID utječu na dozvole za stvaranje i pristup datotekama

- uglavnom će biti isti dok ih ne promijenimo funkcijama za postavljanje ID-eva

- `os.getuid()` dohvaća stvarni UID
- `os.setuid(uid)` postavljan stvarni UID na brojačnu vrijednost `uid`
- `os.getgid()` dohvaća stvarni GID
- `os.setgid(gid)` postavljan stvarni GID na brojčanu vrijednost `gid`

- `os.geteuid()` dohvaća efektivni UID
- `os.seteuid(euid)` postavljan efektivni UID na brojčanu vrijednost `euid`
- `os.getegid()` dohvaća efektivni GID
- `os.setegid(egid)` postavljan efektivni GID na brojčanu vrijednost `egid`

- `os.getgroups()` dohvaća popis grupa korisnika
- `os.setgroups(groups)` postavlja popis grupa korisnika na listu brojčanih vrijednosti `groups`

!!! admonition "Zadatak"
    Provjerite koji su stvarni UID i GID te efektivni UID i GID procesa, a zatim ih pokušajte promijeniti na vrijednost 0 (UID korisnika `root` i GID grupe `root` ili `wheel`). Objasnite zašto to možete ili ne možete napraviti.

!!! admonition "Zadatak"
    Na temelju:

    - stvarnog UID-a i sadržaja datoteke `/etc/passwd`, te
    - stvarnog GID-a i sadržaja datoteke `/etc/group`,

    saznajte ime korisnika i grupe.

- `os.umask(mask)` postavlja korisničku masku na vrijednost `mask`
- `os.getcwd()` vraća trenutni radni direktorij procesa
- `os.chdir(path)` mijenja trenutni radni direktorij procesa u `path`
- `os.listdir(path)` izlistava sadržaj direktorija, kao rezultat vraća listu znakovnih nizova koji su imena datoteka i poddirektorija

!!! admonition "Zadatak"
    Implementirajte naredbu `ls`, odnosno napravite Python program koji:

    - ako je pozvan bez argumenata, izlistava trenutni radni direktorij,
    - ako je pozvan s jednim argumentom, izlistava sadržaj direktorija čija je putanja navedenu u argumentu (nije potrebno provjeriti je li ispravno unesena od strane korisnika).

- `os.ttyname(fd)` vraća ime terminala koji je povezan na opisnik datoteke `fd`

    - opisnici datoteka koje ima svaki proces su `0` (standardni ulaz), `1` (standardni izlaz) i `2` (standardni izlaz za greške); npr. `os.ttyname(0)` vratit će ime terminala povezanog na standardni ulaz
    - otvaranjem datoteka funkcijom `open()` proces dobiva opisnike datoteka `3`, `4`, itd.

!!! admonition "Zadatak"
    Provjerite imena terminala povezanih na opisnike datoteka `0`, `1`, `2` u situacijama:

    - kada je Python program pokrenut bez preusmjeravanja ulaza i izlaza,
    - kada je Python program pokrenut s preusmjeravanjem standardnog izlaza, te
    - kada je Python program pokrenut s preusmjeravanjem standardnog izlaza i standardnog izlaza za greške.

- `os.environ` je rječnik varijabli okoline oblika `{'VAR1': 'vrijednost1', 'VAR2': 'vrijednost2', ... }`

    - dohvaćanje vrijednosti pojedine varijable: `os.environ['IME_VARIJABLE']` ili `os.getenv('IME_VARIJABLE')`
    - dodavanje nove varijable i postavljanje njene vrijednosti: `os.environ['IME_VARIJABLE'] = 'vrijednost varijable'`

!!! admonition "Zadatak"
    - Ispište prvo u ljusci, a zatim u Python programu vrijednosti varijabli `USER`, `SHELL`, `LANG`, `PPID`, `PWD`, `RANDOM` i `_`. Postoje li sve u oba slučaja i imaju li iste vrijednosti?
    - Postavite u programu varijablu okoline `KOLEGIJ` na vrijednost `Operacijski sustavi 2` i uvjerite se prvo da je zaista uspješno postavljena, a zatim provjerite je li postavljena i u ljusci nakon završetka izvođenja programa.

## Pokretanje procesa i podprocesa

- `os.exec{l,v{,p,e,pe}}()` pokreće proces koji zamjenjuje trenutni proces

    - novi proces zadržava PID i okolinu
    - stog, hrpa i podaci procesa zamjenjuju se novim
    - **l** -- argumenti naredbe navedeni su kao argumenti funkcije `execl()`
    - **v** -- argumenti naredbe navedeni su kao lista koja je jedan arugment funkcije `execv()`
    - **p** -- iskoristi varijablu okoline `$PATH` za pronalaženje datoteke za pokretanje
    - **e** -- kod pokretanja naredbe prosljeđuje se i rječnik varijabli okoline koje se dodaju `os.environ`

- `os.execl()` ima četiri varijante; `arg0` je ime naredbe, `arg1` i ostali su argumenti koje naredba prima:

    - `os.execl(path, arg0, arg1, ...)`
    - `os.execle(path, arg0, arg1, ..., env)`
    - `os.execlp(file, arg0, arg1, ...)`
    - `os.execlpe(file, arg0, arg1, ..., env)`

- `os.execv()` također ima četiri varijante; `args[0]` je ime naredbe, `args[1]` i ostali su argumenti koje naredba prima:

    - `os.execv(path, args)`
    - `os.execve(path, args, env)`
    - `os.execvp(file, args)`
    - `os.execvpe(file, args, env)`

- primjerice, kako bismo pokrenuli `cal` koja se nalazi u `/usr/bin/cal` s parametrom `-3` i argumentima `5` i `2020` (prikaz kalendara za travanj, svibanj i lipanj 2020. godine, naredba u ljusci `cal -3 5 2020`), pozvat ćemo bilo koju od sljedećih funkcija:

    - `os.execl('/usr/bin/cal', 'cal', '-3', '5', '2020')`
    - `os.execv('/usr/bin/cal', ['cal', '-3', '5', '2020'])`
    - `os.execlp('cal', 'cal', '-3', '5', '2020')`
    - `os.execvp('cal', ['cal', '-3', '5', '2020'])`

- za ilustraciju prosljeđivanja varijabli okoline, razmotrimo [DEC VT52](https://en.wikipedia.org/wiki/VT52), hardverski terminal iz 1975. godine koji se može emulirati postavljanjem varijable ljuske `$TERM` na vrijednost `vt52` i na kojemu ne postoji mogućnost inverzije boje pozadine i boje teksta pa naredba `cal` kod ispisa ne označava trenutni datum; naredba u ljusci `TERM=vt52 cal -3 5 2020` imat će isti rezultat kao pozivi funkcija:

    - `os.execl('/usr/bin/cal', 'cal', '-3', '5', '2020', {'TERM': 'vt52'})`
    - `os.execv('/usr/bin/cal', ['cal', '-3', '5', '2020'], {'TERM': 'vt52'})`
    - `os.execlp('cal', 'cal', '-3', '5', '2020', {'TERM': 'vt52'})`
    - `os.execvp('cal', ['cal', '-3', '5', '2020'], {'TERM': 'vt52'})`

!!! admonition "Zadatak"
    - Napišite Python skriptu koja pokreće `ls` na direktorij `/etc` tako da iskoristite varijantu funkcije `exec()` koja za traženje naredbe koristi varijablu okoline `$PATH`.
    - Napišite Python skriput koja pokreće `cal` da ispiše kalendar srpanj 2020. godine i to tako da se kod pokretanja okolina modificira tako da `$LANG` poprimi vrijednost `en_US.UTF-8`.

- zombie proces, ponekad nazvan mrtav proces (engl. *defunct process*), proces koji je završio s izvođenjem ali i dalje ima unos u tablici procesa zbog potrebe da njegov roditelj pročita izlazni status

- `os.fork()` stvara kopiju procesa; procesu roditelju vraća PID procesa djeteta, procesu djetetu vraća 0 (na temelju toga ih razlikujemo)

    ``` python
    import os

    child_pid = os.fork()
    print("Ovo ispisuju oba procesa")
    if child_pid == 0:
        print("Ovo ipisuje samo proces dijete")
    else:
        print("Ovo ispisuje samo proces roditelj")
    ```

- `os.wait()` čeka na završetak izvođenja procesa djeteta; vraća uređeni par `(pid, status)`

    ``` python
    import os

    child_pid = os.fork()
    print("Ovo ispisuju oba procesa")
    if child_pid == 0:
        print("Ovo ipisuje samo proces dijete")
    else:
        print("Ovo ispisuje samo proces roditelj")
        pid, status = os.wait()
        print("Proces dijete s PID-om", pid, "završio je izvođenje s izlaznim statusom", status)
    ```

!!! admonition "Zadatak"
    Napišite Python skriptu koja radi `fork()` i u procesu djetetu pokreće proces koji izlistava trenutni direktorij, a zatim čeka na završetak procesa djeteta i ispisuje na ekran njegov izlazni status. Razmislite na koji ćete način osigurati da se naredba pokreće samo u procesu djetetu.

    (Vremenom ćemo naučiti koristiti elegantnije sučelje za baratanje potprocesima, u sklopu modula `subprocess`.)

- grupa procesa je skup koji se sastoji od jednog ili više procesa

    - identifikator grupe procesa (engl. *Process Group Identifier*, PGID) jednak je PID-u procesa koji je voditelj grupe
    - specijalno, kontrolne grupe (engl. *control groups*, kraće [cgroups](https://en.wikipedia.org/wiki/Cgroups)) omogućuju ograničavanje i praćenje korištenja resursa od strane grupa procesa

- sesija je skup koji se sastoji od jedne ili više procesnih grupa

    - postoji definiran proces koji je voditelj sesije
    - procesi jedne sesije mogu stvarati procesne grupe samo unutar te sesije

- `os.getsid(pid)` vraća id sesije procesa s PID-om `pid`
- `os.setsid()` stvara novu sesiju u kojoj trenutni proces postaje voditelj i sesije i vlastite procesne grupe

- upravljački terminal (engl. *controlling terminal*, CTTY) je terminal koji može biti pridružen sesiji

    - odlučuje o stvarima kao što su koji proces prima ulaz s tipkovnice i kako procesi primaju informacije o promjeni veličine prozora terminala
    - [upravljački terminal se može mijenjati](https://blog.nelhage.com/2011/02/changing-ctty/)
    - sesija koja nema upravljački terminal dobiva ga [kad voditelj otvori prvu datoteku](https://blog.nelhage.com/2011/01/reptyr-attach-a-running-process-to-a-new-terminal/)

- [daemon proces](https://en.wikipedia.org/wiki/Daemon_(computer_software)) je proces pokrenut u pozadini, bez terminala koji njime upravlja, često bez roditelja zbog čega ga posvaja proces s PID-om 1 (`init`)

    - primjeri takvih procesa su `sshd`, `httpd`, `syslogd`, `ospfd`, `ntpd`, `acpid`, ...
    - nastaje dvostrukim forkanjem (tzv. *double fork magic*, poznat i pod nazivom [daemon fork](https://stackoverflow.com/q/4192472)) uz neke dodatne operacije:

        - prvi `fork()` služi za stvaranje procesa djeteta koji zatim mijenja radni direktorij u `/`, postavlja korisničku masku na 0 i odvaja se od upravljačkog terminala i procesa roditelja tako što postaje voditelj vlastite sesije i procesne grupe pozivom funkcije `setsid()`;
        - drugi `fork()` osigurava da proces prestaje biti voditelj sesije, kako ne bi bio spojen na upravljački terminal.
        - nakon svakog `fork()`-a prekida se izvođenja procesa roditelja (npr. pozivom funkcije `sys.exit()`), što se popularno naziva *fork off and die*

!!! admonition "Zadatak"
    Napišite program koji izvodi dvostruki fork i postaje daemon, a zatim ga pokrenite i uvjerite se pregledom izlaza naredbe `pstree` da je `init` posvojio vaš pokrenuti proces.

!!! admonition "Zadatak"
    Modificirajte kod koji radi dvostruki fork tako da proces koji je daemon čini sljedeće:

    - u kućnom direktoriju korisnika koji ga je pokrenuo ili u direktoriju `/tmp` (prema vašem izboru) otvara datoteku `mojdaemon.log` za zapisivanje,
    - saznaje UID i korisničko ime korisnika koji ga je pokrenuo i ispisuje ga na ekran, a zatim u datoteku,
    - saznaje ime operacijskog sustava i ispisuje ga na ekran, a zatim u datoteku,
    - spava 5 sekundi,
    - ispisuje na ekran trenutni datum i vrijeme, a zatim u datoteku,
    - ispisuje na ekran vrijednost varijable okoline `$PATH`,
    - uspisuje u datoteku niz znakova `=== kraj izvođenja procesa s PID-em {} ===` (na mjestu `{}` je PID procesa), a zatim zatvara datoteku.
