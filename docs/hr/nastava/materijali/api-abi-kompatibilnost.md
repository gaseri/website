---
author: Vedran Miletić
---

# Binarna kompatibilnost i kompatibilnost na razini izvornog koda

Razlikujemo dvije vrste kompatibilnosti aplikacija s bibliotekama:

- kompatibilnost na razini izvornog koda (engl. *source code compatibility*)
- binarna kompatibilnost (engl. *binary compatibility*).

## Kompatibilnost na razini izvornog koda

Kompatibilnost na razini izvornog koda tiče se aplikacijskog programskog sučelja koje biblioteka pruža i koje aplikacija koristi. Ukoliko pretpostavimo biblioteku s postojećim sučeljem, neki od načina za lomljenje API kompatibilnosti su:

- uklanjanje određene funkcije,
- promjena potpisa određene funkcije promjenom broja ili tipa parametera,
- promjena poretka članova strukture ili klase.

!!! admonition "Zadatak"
    Razmotrite biblioteku koja ima API

    ``` c++
    void set_value1 (int *num);
    void set_value2 (int *num);
    ```

    te program koji ju koristi.

    - Promijenite prvu funkciju da ima argument tipa `double*`, a drugoj funkciji dodajte argument tipa `float*` te taj argument iskoristite unutar tijela na neki način. Prevedite biblioteku i njome zamijenite prvu varijantu, bez da ponovno prevodite program. Pokrenite program i uočite što se događa.
    - Pokušajte ponovno prevesti program i povezati ga s bibliotekom. Objasnite zašto to možete ili ne možete učiniti.

## Binarna kompatibilnost

!!! hint
    Za više detalja proučite [Binary Compatibility Issues With C++](https://techbase.kde.org/Policies/Binary_Compatibility_Issues_With_C++) i [Building Applications with the Linux Standard Base](https://wiki.linuxfoundation.org/en/Book).
