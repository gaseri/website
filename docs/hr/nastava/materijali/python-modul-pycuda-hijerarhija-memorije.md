---
author: Vedran Miletić, Kristijan Lenković
---

# Python modul PyCUDA: hijerarhija GPU memorije

- Globalna memorija (pripada rešetki, ključna riječ `__device__`)

    - Tu se kopiraju argumenti kod poziva funkcije u svim dosadašnjim primjerima

- Konstantna memorija (pripada aplikaciji, ključna riječ `__constant__`)

    - Vrši se međuspremanje za brži pristup varijablama od pristupa globalnoj memoriji
    - "Varijable" koje su `__constant__` moraju biti deklarirane izvan svih zrna

- Dijeljena memorija (pripada bloku, ključna riječ `__shared__`)

    - Može biti statički ili dinamički alocirana

- Lokalna memorija (pripada niti, nema ključne riječi)

    - Koristi se automatski kod polja deklariranih unutar niti

- Registri (pripadaju niti, nema ključne riječi)

    - Koristi se varijabli kod polja deklariranih unutar niti
    - Ponekad, ovisno o prevoditelju, može se koristiti i kod polja

!!! admonition "Zadatak"
    - Deklarirajte varijablu `pi` tipa float **u konstantnoj memoriji** i postavite ju na vrijednost `3.14159`.
    - Definirajte zrno `pi_zapisi(float *polje)`. Unutar modula, ali van zrna, definirajte polje `pi_polje` tipa float veličine 200 elemenata **u globalnoj memoriji**, u koje će svaka od niti zrna `pi_zapisi()` upisati vrijednost konstante `pi`. Pokrenite izvođenje zrna sa 20 niti po bloku i 10 blokova. (**Uputa:** za provjeru točnosti koda možete iskoristiti `printf()`.)
