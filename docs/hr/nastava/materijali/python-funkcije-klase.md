---
author: Vedran Miletić
---

# Python: funkcije i klase

## Funkcije

- funkcija ima `def` i opcionalno `return`, moguće je zadati podrazumijevane vrijednosti argumenata

    ``` python
    def funkcija1(arg1, arg2=0):
      return 2 * arg1 + arg2
    def funkcija2(arg1=1, arg2=0):
      print("Rezultat: ", 2 * arg1 + arg2)
    ```

    ``` c++
    // ekvivalentan C++ kod
    #include <iostream>
    using namespace std;

    int funkcija1 (int arg1, int arg2 = 0)
    {
      return 2 * arg1 + arg2;
    }

    void funkcija2 (int arg1 = 1, int arg2 = 0)
    {
      cout << "Rezultat: " << 2 * arg1 + arg2 << endl;
    }
    ```

!!! admonition "Zadatak"
    - Napišite funkciju `prosjek(lista)` koja ispisuje na ekran prosječnu vrijednost elemenata liste `(suma elemenata / broj elemenata)`. Pretpostavite da lista ima samo podatke tipova `int` i `float`.
    - Napišite funkciju `najduzi_niz_znakova(lista_nizova_znakova)` koja traži najduži niz znakova u listi nizova znakova i vraća ga. Pretpostavite da lista ima samo podatke tipa `str`.

!!! admonition "Zadatak"
    Uzmite da sustav za e-učenje predanu domaću zadaću studenta Ivo Ivić sprema kao `ivo_ivic.tar.gz`. Napišite funkciju koja će na temelju liste predanih zadaća oblika

    ``` python
    [ "ivo_ivic.tar.gz", "ana_anic.tar.gz", "marko_horvat.tar.gz"]
    ```

    ispisati prezimena i imena studenata koji su predali zadaću. Pritom zanemarite hrvatske znakove, međutim pripazite da prezime i ime počinju velikim početnim slovom.

!!! admonition "Zadatak"
    - Definirajte funkciju `select_from(table, id)` kojoj se šalje "tablica" oblika

        ``` python
        tablica = [[3, "Marko", "Ivić"], [6, "Ana", "Anić"], [4, "Hrvoje", "Horvat"]]
        ```

        i koja kao rezultat vraća traženi "redak" ili `None` ako redak ne postoji.
    - Definirajte funkciju `delete_from(table, id)` koja mijenja "tablicu" `table` tako da iz nje briše redak koji ima dani `id`. Funkcija ispisuje novu tablicu, ali ne vraća vrijednost.

## Generatori

!!! todo
    Ovaj dio treba napisati u cijelosti.

## Klase

- klasa je slična strukturi (i klasi) u C++-u i omogućuje nasljeđivanje atributa i metoda (više o tome čuti ćete na Objektno orijentiranom programiranju)
- `class` se koristi za definiranje klase

    ``` python
    class Iznos:
      broj = 0
      slovima = "nula"
      def ispis(self):
        return str(broj) + " (" + slovima + ")"
    ```

    ``` c++
    // ekvivalentan C++ kod
    #include <iostream>
    using namespace std;

    class Iznos
    {
      int broj = 0;
      string slovima = "nula";
      void ispis()
      {
        cout << broj << " (" << slovima << ")" << endl;
      }
    };
    ```

!!! todo
    Ovdje treba dati opisati nasljeđivanje.

!!! admonition "Zadatak"
    - Definirajte klasu `Covjek` koja sadrži atribute `visina`, `tezina`, `dob` i `budnost` te funkciju `say_hello()`.
    - Definirajte klasu `Student` koja nasljeđuje klasu `Covjek` i dodaje atribute `fakultet` i `godina_studija`.

!!! todo
    Učini ovaj zadatak smislenim.

- objekt je instanca klase
- `self` je referenca na objekt
- konstruktor je funkcija istog imena kao i pripadna klasa, stvara objekt koji je instanca te klase

    - promjenjiv uz redefiniranje metode `__init__()`

!!! todo
    Ovdje treba osmisliti primjere i zadatak.
