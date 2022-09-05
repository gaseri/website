---
author: Vedran Miletić
---

# Osnovna sintaksa programskog jezika Python

- Greg Stein u [prezentaciji Python at Google](https://www.sauria.com/twl/conferences/pycon2005/20050325/Python%20at%20Google.html) kaže:

    - "Python has been an important part of Google since the beginning, and remains so as the system grows and evolved. Today dozens of Google engineers use Python, and we're looking for more people with skils in this language" ([Peter Norvig](https://norvig.com/), Director of Search Quality at Google)

    - Google's programming environment

        - Primary languages: C++, Java, Python

    - Python at [eShop](https://en.wikipedia.org/wiki/EShop):

        - 1995\. "What in the world is Python?"
        - 1996\. "This is great stuff."
        - (eShop gets assimilated by Microsoft)

    - Python at [Microsoft](https://en.wikipedia.org/wiki/Microsoft):

        - 1996\. "It's called what?"
        - 1997\. "You actually shipped Python code?" (MerchantServer 1.0)
        - 1998\. "Nice prototype. We'll rewrite it in the next version." And they did, in C++.

    - Python at [CollabNet](https://en.wikipedia.org/wiki/CollabNet) (poznati kao autori [Subversiona](https://en.wikipedia.org/wiki/Apache_Subversion)):

        - 2001\. "No, we don't really use Python here."
        - 2003\. "Definitely write that in Python."

    - Python at [Google](https://en.wikipedia.org/wiki/Google):

        - 2004\. "Of *course* we use Python. Why wouldn't we?" Python caught on here [at Google] like a virus, moving from developer to developer.

- [Programski jezik Python](https://en.wikipedia.org/wiki/Python_(programming_language)):

    - programski jezik visoke razine,
    - naglasak na čitljivost koda,
    - vrlo bogata standardna biblioteka,
    - sličan C/C++-u, Perlu, Javi, Rubyju, JavaScriptu,
    - referentna implementacija je [CPython](https://en.wikipedia.org/wiki/CPython):

        - napisana u C-u,
        - višeplatformska,
        - dostupna pod slobodnom licencom,
        - interpreter (nema kompajliranja).

    - danas vrlo korišten u raznim domenama, od numeričkog računanja u kvantnoj fizici i poravnavanja sekvenci gena do poslovne logike i web aplikacija

        - [Google pretraga za "who uses python"](https://lmgtfy.com/?q="who+uses+python") daje nekoliko popisa).
        - postoji [popis domena u kojima se Python koristi](https://www.python.org/about/success/)
        - postoji [popis korisnih modula](https://wiki.python.org/moin/UsefulModules) koji omogućuju primjenu u raznim domenama

- [Povijesni razvoj Pythona](https://en.wikipedia.org/wiki/History_of_Python):

    - 1991. verzija 0.9.0, Guido von Rossum na alt.sources,
    - 1994. verzija 1.0, 1995. 1.2, 1996. 1.4, ...,
    - 2000. verzije 1.6 i 2.0, Python razvojni tim napušta [CNRI](https://en.wikipedia.org/wiki/Corporation_for_National_Research_Initiatives), interes za Python u zajednici otvorenog koda raste,
    - 2008. verzija 3.0 poznata kao [Python 3000 ili kraće Py3k](https://en.wikipedia.org/wiki/History_of_Python#Version_3.0), nekompatibilna s verzijom 2.0, promjene u sintaksi, izbačeni zastarjeli dijelovi.

- Python 2 i Python 3 danas:

    - `python` == `python3` == `python3.8`
    - `python2` == `python2.7`

- Python 2 i Python 3 nekad:

    - `python` == `python2` == `python2.7`
    - `python3` == `python3.5`

- koristit ćemo **Python 3.8** jer je [Python 2.7 eksplicitno mrtav od 1. siječnja 2020. godine](https://pythonclock.org/)

## Rad s Python interpreterom

- interaktivni način rada

    - pokretanjem naredbe `python3` dobivamo ljusku u koju možemo unositi naredbe
    - `>>>` unos naredbe
    - `...` nastavak prethodnog retka

- pokretanje programa pomoću interpretera

    - spremimo niz naredbi u datoteku s ekstenzijom `.py` i pokrenemo je interpreterom `python3`
    - zaglavlje je oblika

        ``` python
        #!/usr/bin/env python3
        ```

    - `env` traži `python3` na sustavu; on bi mogao biti u `/usr/bin`, `/usr/local/bin` ili čak u `/opt/python3/bin`; autor skripte o tome ne mora brinuti

- Unicode

    - Python 3 ga koristi u zadanim postavkama
    - kod Pythona 2 potrebno eksplicitno navesti u drugom retku zaglavlja

        ``` python
        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        ```

- [Python dokumentacija](https://docs.python.org/3/)

    - vrlo detaljno i kvalitentno napisana, i mnogo ćemo je koristiti; možete je koristiti na kolokviju
    - naročito važni **Tutorial** i **Library Reference**

- dinamički tipovi (engl. *dynamic typing*) -- nema deklaracije tipa varijable, određuje se prema pridruženoj vrijednosti

    ``` python
    a = 3
    b = "rijec"
    ```

    ``` c++
    // ekvivalentan C++ kod
    int a = 3;
    char* b = "rijec";
    ```

- tipovi (klase) varijabli

    - objektno orijentiran jezik: svaka varijabla je *objekt*

    ``` python
    type() # vraća tip (klasu) objekta
    ```

- osnovni tipovi varijabli

    - `bool`, `int`, `long`, `float`, `complex`

        - `float` je pandan C++-ovom `float` i `double`

    - nepromjenjivi: `str`, `bytes`, `tuple`, `frozenset`

        - `str` i `bytes` su pandani C++-ovom `std::string` i `char*`
        - `tuple` je pandan C++-ovim poljima

    - promjenjivi: `list`, `set`, `dict`, `bytearray`

        - `list` je pandan C++-ovom `std::list` i `std::vector`
        - `set` je pandan C++-ovom `std::set`
        - `dict` je pandan C++-ovom `std::map`

    - za svaki od tipova postoji istoimena funkcija (tzv. *konstruktor*) koja služi za stvaranje instanci klase i pretvaranje među različitim tipovima

- osnovni operatori

    - `=`, `+`, `-`, `*`, `**`, `/`, `//`, `%`
    - pridruživanje, zbrajanje, oduzimanje, množenje, potenciranje, dijeljenje, cjelobrojno dijeljenje, modulo dijeljenje

- "hello world": funkcija `print()` vrši ispis na ekran

    ``` python
    print("Pozdrav Rijeci")
    ```

    ``` c++
    // ekvivalentan C++ kod
    #include <iostream>
    using namespace std;

    int main ()
    {
      cout << "Pozdrav Rijeci" << endl;
      return 0;
    }
    ```

- sustav pomoći `help()` --  interpreter ima u sebi ugrađenu dokumentaciju

    ``` python
    help(xyz) # pomoć za funkciju xyz()
    help("xyz") # pomoć na temu xyz
    ```

## Rad s ljuskom IPython

Ljuska [IPython](https://ipython.org/) proširuje funkcionalnost osnovnog intepretera.

- `<Tab>` kompletira uneseni niz znakova do imena naredbe/varijable/funkcije/tipa/modula, dvostruki `<Tab>` nudi sve mogućnosti (isto kao u ljusci `bash`).
- `help(modul/funkcija/objektu)` ili `modul/funkcija/objekt?` daje detaljne informacije o modulu/funkciji/tipu, koje su pored toga *lijepo formatirane*.

!!! note
    Intrepreter `ipython` je moguće testirati i [izravno u web pregledniku](https://www.pythonanywhere.com/try-ipython/).

## Podrška za Python u uređivaču teksta Emacs

Emacs u standardnoj distribuciji ima `python.el` koji mu omogućuje [napredno baratanje Python kodom](https://www.saltycrane.com/blog/2010/05/my-emacs-python-environment/):

- `python-shift-left`, `C-c C-<`: Decrease indentation of the region
- `python-shift-right`, `C-c C-<`: Increase indentation of the region
- `python-switch-to-python`, `C-c C-z`: Start (or switch) to a Python shell
- `python-send-buffer`, `C-c C-c`: Run the current buffer in the Python interpreter
- `python-send-region`, `C-c C-r`: Run the selected code in the Python interpreter
- `python-describe-symbol`, `C-c C-f`: Get help on a Python symbol (Better than visiting the slow Python website, right?)

!!! admonition "Zadatak"
    Napišite "hello world" program u Pythonu s pozdravom po želji, a zatim ga modificirajte kako je opisano.

    - Pokušajte spomenuti način ispisati i varijable tipa `int` i `float`.
    - Isprobajte na koji način radi ispis znakovnih nizova s našim znakovima.
    - Ostavite proizvoljan znakovni niz sam u svom retku i pokušajte pokrenuti program. Što se ispisuje na ekran kod pokretanja programa? Što se mijena kod interaktivnog načina rada?

## Objektni pristup programiranju

- atribut objekta je konstanta vezana uz objekt
- metoda objekta je funkcija vezana uz objekt
- funkcija `dir()` vraća popis atributa i metoda

``` python
a = 5 # varijabla a će dinamički postati tipa int
dir() # vraća popis atributa i metoda tipa int
```

!!! admonition "Zadatak"
    U Pythonu se varijabli može pridružiti vrijednost kompleksnog broja na nekoliko načina, a najjednostavniji je `a = 2+3j` (naravno, vrijednosti 2 i 3 mogu se zamijeniti bilo kojim `int` ili `float` brojevima).

    - Pronađite način da ispište realni i imaginarni dio kompleksnog broja.
    - Pronađite način da ispište kompleksni konjugat tog broja. Objasnite koju razliku pravi par zagrada `()` nakon metode.

## Naredbe `if`, `else`, `for`, `while`

- Python *zahtjeva* točnu uvlaku koda; u protivnom, interpreter javlja grešku

    - nedostatak uvlake: `IndentationError: expected an indented block`
    - višak uvlake: `IndentationError: unexpected indent`
    - Emacs za Python uvlaku koristi 4 razmaka (preporučeno, [PEP 8](https://www.python.org/dev/peps/pep-0008/)) i *automatski pravilno uvlači* pritiskom na tipku `Tab`

- `None` -- tip podataka *ništa*; (`a` nedefiniran) != (`a = None`)
- `True` i `False` -- boolean tip podataka; `False` su `int(0)`, `float(0.0)`, `str("")`, `list([])`
- operatori `and` i `or`
- naredbe `if` i `else`

    ``` python
    if x < 0:
      print("Negative")
    elif x == 0:
      print("Zero")
    else:
      print("Positive")
    ```

    ``` c++
    // ekvivalentan C++ kod
    #include <iostream>
    using namespace std;

    int main ()
    {
      if (x < 0)
        {
          cout << "Negative" << endl;
        }
      else if (x == 0)
        {
          cout << "Zero" << endl;
        }
      else
        {
          cout << "Positive" << endl;
        }
      return 0;
    }
    ```

- tip podataka `range`

    - funkcija `range([start,] stop[, step])` vraća niz brojeva od `start` do `step` u koracima `step`
    - "početak uključen, kraj isključen" (prema idejama knjige [C Traps and Pitfalls](https://en.wikipedia.org/wiki/C_Traps_and_Pitfalls) [Andrewa Koeniga](https://en.wikipedia.org/wiki/Andrew_Koenig_(programmer)))

- `in` označava iteraciju po "elementima" nekog objekta (liste, uređene n-torke, rječnika, ...)
- naredba `for`

    ``` python
    for x in range(2, 7):
      print(x, x**2)
    ```

    ``` c++
    // ekvivalentan C++ kod
    #include <iostream>
    using namespace std;

    int main ()
    {
      for (int x = 2; x < 7; x++)
        {
           cout << x << " " << x * x << endl;
        }
    }
    ```

- naredba `while`

    ``` python
    x = 0
    while x < 5:
      x += 1
      print(x, x**2)
    ```

    ``` c++
    // ekvivalentan C++ kod
    #include <iostream>
    using namespace std;

    int main ()
    {
      int x = 0;
      while (x < 5)
        {
           x++;
           cout << x << " " << x * x << endl;
        }
    }
    ```

!!! admonition "Zadatak"
    Napišite program koji ispisuje:

    - sve neparne brojeve u rasponu od 1 do 101 (uključujući 101),
    - sve brojeve djeljive s 3 u rasponu od 5 do 45 (uključujući 45).

## Rad sa znakovnim nizovima

- Python koristi i jednostruke i dvostruke navodnike za navođenje znakovnih nizova

    - `"...'..."`, `'..."...'` ili `"...\"..."`
    - mi ćemo koristiti dvostruke navodnike, s prekidnim znakom po potrebi

- podnizovi znakovnih nizova rade po pravilu "početak uključen, kraj isključen"

    ``` python
    a[i] # i-ti znak u nizu; ista sintaksa kao C++
    a[i:j] # znakovi od i-tog do j-tog
    a[i:j:k] # svaki k-ti znak od i-tog do j-tog
    ```

- `str.strip()` miče razmake s početka i s kraja
- `str.split()` cijepa znakovni niz u listu po razmacima
- `str.lower()`
- `str.upper()`

!!! admonition "Zadatak"
    Neka je `a = "AndrewKoenig"`.

    - Isprobajte čemu je jednako, pa objasnite zašto za iduće podnizove:

        - `a[1:5]`,
        - `a[:3]`,
        - `a[2:]`,
        - `a[-3]`,
        - `a[:-1]`,
        - `a[1::2]`,
        - `a[::-1]`.

    - Pretvorite `a` u listu (korištenjem funkcije `list()`) i spremite je u varijablu `b`. Dobivate li iste rezultate kada tražite podliste?

!!! admonition "Zadatak"
    Neka je `a = "Miami 2 Ibiza"`.

    - Pretvorite `a` u listu (korištenjem funkcije `list()`) i spremite je u varijablu `b`.
    - Pretvorite `a` u listu korištenjem metode split i spremite je u varijablu `c`.
    - Je li `b` jednako `c`? Objasnite zašto je ili zašto nije.

## Rad s listama

- ista sintaksa pristupanja pojedinom elementu kao kod znakovnih nizova
- `list.append()` dodaje element na kraj liste
- `list.pop()` miče element s kraja liste
- `list.insert()` ubacuje element u listu na određeno mjesto

!!! admonition "Zadatak"
    Neka je `a = [1, "Bok", [24, 7, 365], ["Da", "Ne"]]`.

    - Isprobajte čemu je jednako, pa objasnite zašto za iduće podliste:

        - `a[1]`,
        - `a[2]`,
        - `a[2][0]`,
        - `a[2][4]`,
        - `a[3][1]`.

    - Promijenite vrijednost elementa `a[0]` na 5.
    - Umetnite element "Swedish House Mafia" na mjesto 1.
    - Maknite element s kraja liste, a zatim na kraj liste umetnite listu koja sadrži brojeve 5, 4, 2.
