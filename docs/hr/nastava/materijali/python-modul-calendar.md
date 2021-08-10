---
author: Vedran Miletić
---

# Python: stvaranje i ispis kalendara

- `module calendar` ([službena dokumentacija](https://docs.python.org/3/library/calendar.html)) definira tipove podataka i nudi funkcije za ispis kalendara u obliku čistog teksta i u HTML-u
- `calendar.Calendar` zajednički dio klasa `TextCalendar` i `HTMLCalendar`
- `calendar.TextCalendar` omogućuje ispis kalendara u obliku čistog teksta

    - funkcija `formatmonth()` oblikuje kalendar mjeseca sličan naredbi `cal` s dva argumenta; primjerice ispis mjeseca travnja 2020. godine izvest ćemo naredbom `cal 4 2020` i kodom

        ``` python
        import calendar

        c = calendar.TextCalendar()
        travanj2020 = c.formatmonth(2020, 4)
        print(travanj2020)
        ```

    - funkcija `formatyear()` oblikuje kalendar godine sličan naredbi `cal` s jednim argumentom; primjerice ispis 2020. godine izvest ćemo naredbom `cal 2020` i kodom

        ``` python
        import calendar

        c = calendar.TextCalendar()
        godina2020 = c.formatyear(2020)
        print(godina2020)
        ```

- `calendar.HTMLCalendar` omogućuje ispis kalendara u HTML-u (koristi se analogno kao `TextCalendar`)

!!! admonition "Zadatak"
    Napišite Python program koji radi kao `cal -3`, odnosno pita korisnika da unese godinu i mjesec, a zatim ispisuje 3 mjeseca: mjesec prije, navedeni mjesec i mjesec poslije. (Vaši ispisani mjeseci će ići vertikalno umjesto horizontalno kako ih ispisuje `cal -3`, zanemarite tu razliku.)

!!! admonition "Zadatak"
    Napišite Python program koji korisnik poziva u naredbenom retku s argumentom godine (npr. `./program.py 2020`) i program ispisuje kalendar za tu godinu.

    - Poopćite program tako da ispisuje kalendare za više godina odjednom ako su argumenti navedeni odvojeni zarezima, npr. `./program.py 2019,2020,2021`.
    - Poopćite program tako da ispisuje kalendare za više godina odjednom ako su argumenti navedeni odvojeni razmacima, npr. `./program.py 2019 2020 2021`.
