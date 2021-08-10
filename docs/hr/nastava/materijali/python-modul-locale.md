---
author: Vedran Miletić
---

# Python: internacionalizacija i lokalizacija

- `module locale` ([službena dokumentacija](https://docs.python.org/3/library/locale.html)) nudi pristup lokalnim i regionalnim postavkama operacijskog sustava

- `locale.setlocale(category, locale)` postavlja kategoriju `category` na lokalne postavke `locale`; `category` može biti `locale.LC_ALL`, `locale.LC_CTYPE`, `locale.LC_NUMERIC`, `locale.LC_TIME`, `locale.LC_COLLATE`, `locale.LC_MONETARY` i `locale.LC_MESSAGES` ([objašnjenja kategorija](https://www.gnu.org/software/libc/manual/html_node/Locale-Categories.html))
- `locale.getlocale(category)` vraća uređeni par koda lokalne postavke i kodiranja znakova; `category` može biti bilo koja od kategorija koje postoje u `locale.setlocale()` osim `locale.LC_ALL`

    ``` python
    import locale

    # nakon pokretanja Python postavlja samo LC_CTYPE u skladu s postavkama operacijskog sustava

    locale.getlocale(locale.LC_CTYPE) # vraća ('hr_HR', 'UTF-8')
    locale.getlocale(locale.LC_NUMERIC) # vraća (None, None)
    locale.getlocale(locale.LC_TIME) # vraća (None, None)
    locale.getlocale(locale.LC_COLLATE) # vraća (None, None)
    locale.getlocale(locale.LC_MONETARY) # vraća (None, None)
    locale.getlocale(locale.LC_MESSAGES) # vraća (None, None)

    locale.setlocale(locale.LC_ALL, 'hr_HR.UTF-8')
    # postavili smo i LC_NUMERIC, LC_TIME, LC_COLLATE, LC_MONETARY i LC_MESSAGES na hr_HR i UTF-8

    locale.getlocale(locale.LC_CTYPE) # vraća ('hr_HR', 'UTF-8')
    locale.getlocale(locale.LC_NUMERIC) # vraća ('hr_HR', 'UTF-8')
    locale.getlocale(locale.LC_TIME) # vraća ('hr_HR', 'UTF-8')
    locale.getlocale(locale.LC_COLLATE) # vraća ('hr_HR', 'UTF-8')
    locale.getlocale(locale.LC_MONETARY) # vraća ('hr_HR', 'UTF-8')
    locale.getlocale(locale.LC_MESSAGES) # vraća ('hr_HR', 'UTF-8')
    ```

- `locale.nl_langinfo(option)` (zahtijeva prethodni poziv `locale.setlocale()` jer ovisi o `locale.LC_*`) dohvaća informaciju o lokalnim postavkama u obliku niza znakova, npr. `option` postavljen na `locale.DAY_1` dohvaća lokalni naziv nedjelje (radi prema američkoj konvenciji o poretku dana u tjednu, a ne prema [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html)), a `locale.MON_1` dohvaća lokalni naziv siječnja; [čitav popis u službenoj dokumentaciji](https://docs.python.org/3/library/locale.html#locale.nl_langinfo)
- `locale.localeconv()` (zahtijeva prethodni poziv `locale.setlocale()` jer ovisi o `locale.LC_*`) dohvaća popis svih konvencija koje lokalne postavke sadrže, npr. znak koji se koristi za decimalnu točku, odvajanje tisućica, valutu; [čitav popis u službenoj dokumentaciji](https://docs.python.org/3/library/locale.html#locale.localeconv)

!!! admonition "Zadatak"
    Saznajte nazive dana i mjeseci te konvencije vezane uz ispis brojeva prvo u hrvatskim, a onda američkim engleskim lokalnim postavkama.

- `locale.strcoll(string1, string2)` vraća negativnu, pozitivnu ili vrijednost 0 ovisno o tome je li `string1` prije, poslije ili jednak `string2` (respektivno) ovisnu o lokalnoj postavci `LC_COLLATE`
- `locale.strxfrm(string)` pretvara znakovni niz koji sadrži znakove specifične za lokalne postavke u oblik koji se može koristiti za uobičajeno uspoređivanje znakovnih nizova operatorima `<`, `>` i `==`
- `locale.format_string(format, val)` formatira vrijednost `val` u obliku `format`, koji može biti `%d` za cijele brojeve te `%f` i `5.2f` za brojeve s pomičnim zarezom i brojeve s 5 mjesta prije decimalne točke i dva iza (respektivno)
- `locale.currency(val)` formatira vrijednost `val` kao novčanu vrijednost ovisnu o lokalnoj postavci `LC_MONETARY`
- `locale.atof(string)` pretvara znakovni niz u broj s pomičnim zarezom ovisno o lokalnoj postavci `LC_NUMERIC`
- `locale.atoi(string)` pretvara znakovni niz u cijeli broj ovisno o lokalnoj postavci `LC_NUMERIC`

!!! admonition "Zadatak"
    Prvo u hrvatskoj, a zatim u američkoj engleskoj lokalnoj postavci:

    - usporedite znakovne nizove Cunj, Čabar, Ćićarija i Dubrovnik i odredite koji po abecedi dolaze prije, a koji poslije,
    - formatirajte broj 1234567,89 prvo kao broj s pomičnim zarezom, a zatim kao novčanu vrijednost.
