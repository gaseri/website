---
author: Vedran Miletić
---

# Python: općenite usluge operacijskog sustava: vrijeme

- `module time` ([službena dokumentacija](https://docs.python.org/3/library/time.html)) nudi funkcije vezane uz vrijeme
- `time.time()` vraća trenutno vrijeme sustava, zapis tipa `float` koji označava broj sekundi od [epohe](https://en.wikipedia.org/wiki/Epoch) (u ovom slučaju 1. siječnja 1970. godine kad počinje [brojanje vremena po Unixu](https://www.epochconverter.com/), a općenito može biti [prva Olimpijada](https://en.wikipedia.org/wiki/Olympiad), [osnutak Rima](https://en.wikipedia.org/wiki/Ab_urbe_condita) ili sl.)
- `time.ctime()` vraća trenutno vrijeme sustava, zapis tipa `str` oblika `'Sun Jun 20 20:35:27 1993'`
- `time.sleep(seconds)` čini da proces *spava* `seconds` sekundi

!!! admonition "Zadatak"
    Cilj je mjeriti vrijeme izvođenja. Napišite program koji na početku i na kraju zapisuje vrijeme u dvije različite varijable (recimo, `start_time` i `end_time`). Između ta dva pridruživanja vrijednosti učinite da program "spava" 5 sekundi, nakon toga ispiše na ekran `"spavao sam 5 sekundi"`.

    Nakon računanja ukupnog vremena izvođenja kao razlike krajnjeg i početnog vremena, program ispisuje na ekran `Ukupno vrijeme izvođenja:` i izračunato vrijeme. (**Napomena:** Razmislite zašto je razlika ta dva vremena jednaka proteklom vremenu te paše li vam za izračun više zapis tipa `float` ili zapis tipa `str`.)

!!! admonition "Zadatak"
    Proučite [primjere formatiranja znakovnih nizova u Pythonu](https://docs.python.org/3/library/string.html#formatexamples), a zatim modificirajte prethodni zadatak tako da ispisuje trajanje izvođenja u obliku:

    - `Ukupno vrijeme izvođenja:` i vrijeme s preciznošću na jedno decimalno mjesto.
    - `Ukupno vrijeme izvođenja: ... sekundi i ... stotinki`, gdje ćete na mjesta trotočke postaviti broj sekundi i broj stotinki u cjelobrojnom zapisu.

!!! admonition "Zadatak"
    Izmjerite brzinu izvođenja ovih operacija u Pythonu:

    - otvaranje datoteke `/etc/passwd`, čitanje njenog sadržaja i zatvaranje datoteke.
    - računanje produkta prvih 100 000 prirodnih brojeva,
    - računanje zbroja prvih 500 000 prirodnih brojeva.
