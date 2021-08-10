---
author: Vedran Miletić
---

# Python: osnovni tipovi podataka datuma i vremena

- `module datetime` ([službena dokumentacija](https://docs.python.org/3/library/datetime.html)) definira osnovne tipove podataka datuma i vremena
- `datetime.date` sadrži datum

    - `datetime.date(1997, 4, 21)` vraća `date` koji sadrži vremenski trenutak dana 21. travnja 1997.
    - `datetime.date.today()` vraća trenutni datum tipa `date`
    - moguća usporedba dva datuma, npr. `datetime.date(1998, 4, 21) > datetime.date(1997, 4, 21)` daje rezultat `True`
    - razlika dva datuma je `timedelta`

        ``` python
        import datetime

        datum1 = datetime.date(1997, 4, 21)
        datum2 = datetime.date(1998, 4, 21)
        razlika = datum2 - datum1 # bit će točno 365 dana
        ```

- `datetime.datetime` sadrži datum i vrijeme

    - `datetime.datetime(1997, 4, 21, 12, 35, 41)` vraća `datetime` koji sadrži vremenski trenutak dana 21. travnja 1997. u 12:35:41 sati
    - `datetime.datetime.now()` vraća trenutni datum i vrijeme tipa `datetime`
    - moguća usporedba dva datuma, npr. `datetime.datetime(1998, 4, 21, 12, 35, 41) < datetime.datetime(1997, 4, 21, 12, 35, 41)` daje rezultat `False`
    - razlika dva datuma i vremena je `timedelta`

        ``` python
        import datetime

        datumvrijeme1 = datetime.datetime(1997, 4, 21, 12, 35, 41)
        datumvrijeme2 = datetime.datetime(1998, 4, 21, 12, 35, 41)
        razlika = datumvrijeme2 - datumvrijeme1 # bit će točno 365 dana
        ```

- `datetime.time` sadrži vrijeme
- `datetime.timedelta` je trajanje vremena, rezultat razlike dva vremenska trenutka (`date`, `time`, `datetime`)

    - interno izražen u danima, sekundama i mikrosekundama, ali može se zapisati u jedincama po želji
    - omogućuje brojne aritmetičke operacije: zbrajanje, oduzimanje, množenje i dijeljenje cijelim brojem i brojem s pomičnim zarezom, modulo dijeljenje i druge ([detaljnije](https://docs.python.org/3/library/datetime.html#timedelta-objects))

!!! admonition "Zadatak"
    Odredite koliko je dana prošlo od 1. siječnja 1970.

!!! admonition "Zadatak"
    Neki od operacijskih sustava sličnih Unixu vrijeme spremaju kao 32-bitni cijeli broj pa će 19. siječnja 2038. u 03:14:07 po koordiniranom svjetskom vremenu doživjeti prelijevanje vrijednosti vremena (tzv. [Year 2038 problem](https://en.wikipedia.org/wiki/Year_2038_problem)). Odredite koliko još vremena (u sekundama) imamo do tada.

- `datetime.tzinfo` i `datetime.timezone` služe klasama `datetime` i `time` za pohranu informacija o vremenskoj zoni kao što je odstupanje lokalnog vremena od [koordiniranog svjetskog vremena (Coordinated Universal Time, UTC)](https://en.wikipedia.org/wiki/Coordinated_Universal_Time)

    - `datetime.timezone.utc` je vremenska zona UTC-a
    - naša vremenska zona [srednjoeuropsko vrijeme (Central European Time, CET)](https://en.wikipedia.org/wiki/Central_European_Time) i [srednjoeuropsko ljetno vrijeme (Central European Summer Time, CEST)](https://en.wikipedia.org/wiki/Central_European_Summer_Time) zimi ima odmak +1 sat od UTC-a, a ljeti +2 sata; kako su odmaci ovdje `timedelta`, te vremenske zone konstruirat ćemo kao `datetime.timezone(datetime.timedelta(hours=1))` i `datetime.timezone(datetime.timedelta(hours=2))`

        ``` python
        import datetime

        datumvrijeme1 = datetime.datetime(1997, 4, 21, 12, 35, 41, tzinfo=datetime.timezone.utc)
        datumvrijeme2 = datetime.datetime(1997, 4, 21, 14, 35, 41, tzinfo=datetime.timezone(datetime.timedelta(hours=1)))
        razlika = datumvrijeme2 - datumvrijeme1 # bit će 3600 sekundi, odnosno 1 sat
        ```

    - bazu vremenskih zona [održava IANA](https://www.iana.org/time-zones)

!!! admonition "Zadatak"
    Napišite program koji korisnika traži unos odmaka dvaju vremenskih zona između kojih će se vršiti pretvorba vremena. Korisnik unosi datum i vrijeme tako da redom unosi godinu, mjesec, dan, sat, minute i sekunde, a zatim navodi u kojoj od dvije vremenske zone je to vrijeme uneseno. Program pretvara to vrijeme u vrijeme u drugoj vremenskoj zoni. (Uputa: pretvorbu izvršite u dva koraka: prvo oduzmite odmak vremenske zone da dobijete vrijem po UTC-u, a zatim dodajte drugi odmak.)

- formatiranje datuma i vremena ([popis kodova za formatiranje](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes))

    - `date.strftime()` omogućuje formatiranje datuma
    - `datetime.strftime()` omogućuje formatiranje datuma i vremena

        ``` python
        import datetime

        datvr = datetime.datetime(1997, 4, 21, 12, 35, 41)
        print(datvr.strftime("Dana %d.%m.%Y. u %H sati, %M minuta i %S sekundi"))
        ```

    - `time.strftime()` omogućuje formatiranje vremena

- parsiranje datuma i vremena ([popis kodova za parsiranje, isti kao za formatiranje](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes))

    - `date.strptime()` omogućuje formatiranje datuma
    - `datetime.strptime()` omogućuje formatiranje datuma i vremena

        ``` python
        import datetime

        datvr = datetime.datetime.strptime("15/11/19 15:30", "%d/%m/%y %H:%M")
        print(datvr) # 2019-11-15 15:30:00
        ```

    - `time.strptime()` omogućuje formatiranje vremena

!!! admonition "Zadatak"
    Promijenite rješenje prethodnog zadatka tako da korisnik unosi vremena u obliku `2020-03-26 08:37:54` umjesto pojedinačnih unosa godina, mjeseca, dana, sati, minuta i sekundi (pritom nije potrebno baratati pogrešnim unosima) te da se rezultat pretvorbe ispisuje u obliku `Date: 26. March 2020. Time: 08.37.54`.
