---
author: Vedran Miletić
---

# Praćenje korištenja programskih sučelja sustava korištenjem alata strace i ltrace

- poziv u sustav (engl. *system call*) je način na koji program traži uslugu od jezgre operacijskog sustava
- poziv biblioteci (engl. *library call*) je način na koji program traži uslugu od neke od biblioteka
- naredba `strace` prati pozive u sustav
- naredba `ltrace` prati pozive bibliotekama
- obje naredbe ispisuju na standardni izlaz za greške (`stderr`) koji možemo preusmjeriti slično kao standardni izlaz korištenjem `2>`, `2>>`

- često korišteni odjeljci `man` stranica

    - **1**: *Executable programs or shell commands*
    - **2**: *System calls (functions provided by the kernel)*
    - **3**: *Library calls (functions within program libraries)*

- naredba `man [ODJELJAK] STRANICA` daje stranicu `STRANICA` u odjeljku `ODJELJAK`

    - potrebno je navesti odjeljak slučaju kad istoimena stranica postoji u više odjeljaka

!!! admonition "Zadatak"
    - Usporedite `man` stranice imena `kill` u navedena tri odjeljka ako postoje. Po čemu se razlikuju? Što svaka od njih opisuje?
    - Pratite pozive u sustav koje radi naredba `kill` kada:

        - proces s danim PID-om ne postoji,
        - proces s danim PID-om postoji, ali nemate dozvolu slanja signala tom procesu,
        - proces s danim PID-om biva uspješno ubijen.

    - Objasnite što naredba `kill` radi prije poziva funkcije `kill()`. Poziva li se funkcija `kill()` u svakom od danih slučajeva? Razlikuju li se njene povratne vrijednosti?
