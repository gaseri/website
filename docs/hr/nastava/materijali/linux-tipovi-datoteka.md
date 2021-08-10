---
author: Vedran Miletić, Vanja Slavuj
---

# Tipovi datoteka

- doslovno sve je datoteka

    - pisač, skener, hard disk, USB port, Ethernet kartica, zvučna kartica, web kamera, DVD snimač, ...
    - baza podataka, web stranica, instalirane aplikacije i njihove postavke, ...
    - obične datoteke, poveznice, procesi, socketi, ...

- tipovi datoteka

    - obična datoteka, direktorij, simbolička poveznica, specijalna znakovna datoteka, specijalna blok datoteka, fifo datoteka (imenovana cijev), socket
    - informacija o tipu datoteke zapisana u inodeu, može se pročitati pomoću `stat`-a

- naredba `file` prepoznaje tip datoteke (kod običnih datoteka korištenjem tvz. magičnih testova *po sadržaju* prepoznaje i radi li se o tekstualnoj datoteci, rasterskoj grafici, video zapisu, aplikaciji, ...)

    - **Napomena:** Linux ne razlikuje datoteke po ekstenziji (kao Windowsi), već po sadržaju; za Linux, ekstenzija je samo dio imena, i uredno se može raditi sa PNG datotekom koja se zove `fotka1587.txt` (ako je zaista PNG datoteka).
    - Već samo zbog toga je Linux prilično imun na viruse i zlonamjerni softver; ukoliko vam netko pošalje izvršnu datoteku s ekstenzijom JPG, prije nego je nehotice pokrenete dobiti ćete informaciju da se ne radi o slici, već o izvršnoj datoteci.

- obična datoteka (`-`)

    - datoteka korisničke aplikacije; **primjer:**

        ``` shell
        $ ls -l /etc/hosts
        ```

- direktorij (`d`)

    - može sadržavati datoteke i poddirektorije; **primjer:**

        ``` shell
        $ ls -l /home/miran
        ```

- simbolička poveznica (`l`)

    - njihov sadržaj je putanja do datoteke na koju pokazuju (može biti apsolutna i relativna); **primjer:**

        ``` shell
        $ ls -l /usr/bin/python
        ```

- specijalna blok datoteka (`b`)

    - buffered: kod čitanja i pisanja podaci se prvo spremaju u međuspremnik
    - omogućuje asinkrono zapisivanje (engl. *asynchronous write*); **primjer:**

        ``` shell
        $ ls -l /dev/sda1
        ```

- specijalna znakovna datoteka (`c`)

    - unbuffered: podaci se čitaju direktno, nema međuspremnika; **primjer:**

        ``` shell
        $ ls -l /dev/input/mice
        ```

- `mknod` stvara specijalne blok i specijalne znakovne datoteke (spominjemo radi potpunosti)

!!! admonition "Zadatak"
    - Pronađite još tri primjera običnih datoteka i po još jedan primjer za svaku od ostalih vrsta datoteka. (**Uputa:** koristite rekurzivno izlistavanje.)
    - Ispitajte tip običnih datoteka koje ste pronašli.

## Imenovane cijevi i utičnice

- fifo datoteka ([imenovana cijev](https://en.wikipedia.org/wiki/Named_pipe)) (`p`)

    - radi slično kao cijevi koje smo već koristili, jedino što ima ime
    - radi isključivo lokalno na računalu; **primjer:**

        ``` shell
        $ ls -l /home/vedran/python-samples/fifo1
        ```

- `mkfifo` stvara fifo datoteku danog imena
- [socket](https://en.wikipedia.org/wiki/Unix_domain_socket) (`s`)

    - datoteka koja služi za međuprocesnu komunikaciju
    - radi lokalno i preko mreže
    - npr. lokalna komunikacija dva procesa, otvorena HTTP veza prema nekom poslužitelju za vrijeme preuzimanja datoteke ili otvorena SSH veza s nekog drugog poslužitelja; **primjer:**

        ``` shell
        $ ls -l /var/run/acpid.socket
        ```

    - time ćemo se više baviti kad budemo radili modul `socket` u Pythonu

!!! admonition "Zadatak"
    Stvorite fifo datoteku pod nazivom `cijev1`. Pokrenite dva terminala.

    - U jednom terminalu pokrenite `cat < cijev1`, a u drugom `cat > cijev1`. Uočite da se ono što unesete u drugom terminalu ispisuje u prvom.
    - Pod kojim uvjetom se oba procesa prekidaju? Mogu li obje strane izazvati prekid?
    - Postoji li međuspremnik?
