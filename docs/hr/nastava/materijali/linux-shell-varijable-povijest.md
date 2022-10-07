---
author: Vedran Miletić, Vanja Slavuj, Sanja Pavkov, Anja Vrbanjac
---

# Varijable ljuske i okoline, povijest ljuske

- [Bash](https://en.wikipedia.org/wiki/Bash_(Unix_shell)) (naredba `bash`) je ljuska koju koristimo dok radimo u komandnoj liniji

    - nudi nam prompt oblika `[studXYZ@akari ~]$` gdje unosimo naredbe
    - hvata `<Tab>` kod tab kompletiranja, interpretira glob, interpretira naredbe koje unosimo, a zatim poziva odgovarajuće programe ili vraća određene poruke o grešci (primjerice, kada naredba ne postoji)
    - podržava varijable, funkcije i standardne programske naredbe `if`, `for`, `do/while`, ...
    - dizajnirana po uzoru na stariju Bourne ljusku; Bash je kratica za Bourne Again SHell, što zvuči slično kao *born again* shell

## Varijabla ljuske

- varijabla ljuske

    - ime počinje znakom `$`, vrijednost se može ispisati naredbom `echo`
    - `$HOME`, `$PATH`, `$BASH`, `$BASH_VERSION`, `$COLUMNS`, `$USER`, `$GROUPS`, ...

!!! admonition "Zadatak"
    - Što je točno sadržano u varijablama koje smo upravo vidjeli (`$HOME`, `$PATH`, `$BASH_VERSION`, `$COLUMNS`, `$USER`, `$GROUPS`)? Ispišite njihovu vrijednost na ekran i zaključite.
    - Što je sadržano u sljedećim varijablama:

        - `$HOSTNAME`
        - `$HOSTTYPE`
        - `$MACHTYPE`
        - `$RANDOM`
        - `$SECONDS`

- razlikujemo dvije vrste [dostupnih varijabli ljuske](https://www.gnu.org/software/bash/manual/html_node/Shell-Variables.html)

    - [varijable ljuske koje postoje u Bourne ljusci](https://www.gnu.org/software/bash/manual/html_node/Bourne-Shell-Variables.html)
    - [varijable ljuske specifične za Bash](https://www.gnu.org/software/bash/manual/html_node/Bash-Variables.html)

- pridruživanje vrijednosti varijabli je oblika `MOJA_VARIJABLA=vrijednost` (bez razmaka)

    - primijetite da ne postoji znak dolara ispred naziva varijable
    - vrijednost varijable može biti bilo što: integer, array i sl.

- znak `$` služi za dohvaćanje vrijednosti varijable; **primjer:**

    ``` shell
    $ MOJA_VARIJABLA = 5 # pridruživanje vrijednosti varijabli MOJA_VARIJABLA
    $ echo $MOJA_VARIJABLA # dohvaćanje vrijednosti varijable MOJA_VARIJABLA
    ```

!!! admonition "Zadatak"
    - Pridružite varijabli `IME` vrijednost svoga imena. Ispišite vrijednost te varijable na ekran.
    - Pridružite varijabli `STUDENT` vaše ime i prezime. Što se događa? Kako to izbjeći?
    - Radi li pridruživanje vrijednosti s razmacima oko znaka jednakosti? Probajte objasniti zašto.

## Varijabla okoline

- operacijski sustavi slični Unixu imaju mehanizam prijenosa okoline svim procesima djeci koje je stvorio neki proces roditelj
- vrijednost varijable definirane u ljusci neće se naslijediti u nekoj skripti ljuske; **primjer:**

    ``` shell
    # u promptu pišemo:
    $ VAR1=1
    $ emacs pr1.sh

    # unutar skripte pr1.sh pišemo

    #! /bin/bash
    echo "Vrijednost varijable je $VAR1"
    VAR1=2
    echo "Vrijednost varijable je $VAR1"

    $ chmod +x pr1.sh # označimo skriptu za pokretanje
    $ ./pr1.sh # pokrećemo skriptu
    ```

- da varijabla ljuske postane [varijabla okoline](https://en.wikipedia.org/wiki/Environment_variable) (engl. *environment variable*) potrebno je napraviti izvoz (engl. *export*)

    - **sintaksa:** `export MOJA_VARIJABLA`; **primjer:**

        ``` shell
        $ MOJA_VARIJABLA = 5 # varijabla ljuske poprima vrijednost
        $ export MOJA_VARIJABLA # čini da varijabla postane dijelom okoline
        ```

    - svaka varijabla okoline je varijabla ljuske, ali obrat ne vrijedi
    - izvoz varijable vrijedi samo za trenutnu sesiju i to samo za trenutnog korisnika

- kada više nije potrebna može se napraviti brisanje varijable s `unset MOJA_VARIJABLA`; **primjer:**

    ``` shell
    $ unset MOJA_VARIJABLA # brisanje varijable
    $ echo $MOJA_VARIJABLA # ispisuje prazan redak jer varijabla nema vrijednost
    ```

!!! admonition "Zadatak"
    - Objasnite zašto kod naredbe `unset` nismo koristili `$` pored naziva varijable.
    - Varijabli `TEXT_EDITOR` pridružite vrijednost kojom ćete pokrenuti Emacs alat. Pokrenite navedeni alat pomoću varijable kojoj ste upravo dodijelili vrijednost.
    - Varijabli `CRVENI_EDITOR` pridružite vrijednost `emacs --fg red` kojom se pokreće Emacs u crvenoj boji slova. Pokrenite Emacs tom varijablom.
    - Izbrišite obje varijable.

## Povijest unesenih naredbi

- povijest unesenih naredbi ljuska sprema u `$HISTFILE`
- naredba `history` ispisuje do `$HISTSIZE` prethodno unesenih naredbi
- moguće je po povijesti kretati se strelicama gore i dolje
- `!n` pokreće naredbu sa `n`-tog mjesta u povijesti
- `!niz_znakova` ako postoji, pokreće posljednju naredbu koja na početku ima dani niz znakova
- `^R` pretraživanje povijesti unatrag za specifičnom naredbom

!!! admonition "Zadatak"
    - Pokrenite naredbu na petom mjestu u povijesti. Što se događa? Zašto?
    - Pokrenite posljednju naredbu u povijesti koja počinje s `emacs`.
    - Objasnite što radi `!!`, a što `!-2`.
    - Iskoristite pretraživanje povijesti da pokrenete naredbu posljednju naredbu koja sadrži određeni niz znakova `red`, ali ne nužno na početku.

## Vrste naredbi i funkcije ljuske

- `type` je naredba ljuske koja pretražuje naredbe ljuske; tipovi naredbi mogu biti:

    - naredbe koje su ugrađene u ljusku (shell) odnosno hard coded naredbe (npr. `cd`)
    - vanjske naredbe koje poziva ljuska (npr. `man`) (ako samo kroz ljusku pozvali neku od vanjskih naredbi, ona nakon nalaženja u nekom od direktorija navedenih u varijabli okoline `PATH` ostaje spremljena u hash tablici ljuske te je kasnije nalaženje iste naredbe puno kraće)
    - aliase naredbi (npr. `ls` može biti alias na `ls --color=auto`)
    - funkcije ljuske (npr. `_expand`)

!!! admonition "Zadatak"
    Pronađite još jedan primjer ugrađene naredbe, vanjske naredbe, aliasa i funkcije ljuske.

!!! admonition "Dodatni zadatak"
    Proučite sintaksu i funkcionalnosti naredbe `alias`. Pronađite praktičnu primjenu ove naredbe.

!!! todo
    Ovdje bi trebalo dodati dodati i `&&`, `||`, `;`, `$()`, `true`, `false`.

!!! admonition "Ponovimo!"
    - Što je `bash`, a što je shell?
    - Što su varijable ljuske?
    - Koje vrijednosti mogu imati varijable ljuske? Dokažite.
    - Zbog čega se vrši izvoz varijable ljuske? Objasnite.
    - Gdje se sprema povijest rada s komandnom linijom?
    - Koja je razlika između `HISTSIZE` i `HISTFILE`? Što ti nazivi zapravo predstavljaju?
    - Kako se vrši spremanje novih zapisa u povijest? Što se pritom događa sa starima?
    - Što radi naredba !-6?
