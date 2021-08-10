---
author: Vedran Miletić
---

# Python: ulaz i izlaz

## Rad s korisničkim unosom

- ulaz s tipkovnice prima se kao znakovni niz, potrebno je napraviti pretvorbu u odgovarajući tip podataka

    ``` python
    ulaz = input("Unesite cijeli broj: ") # ulaz je tipa str
    x = int(ulaz) # x je tipa int
    print("Unijeli ste " + str(x) + ".")
    ```

    ``` c++
    // ekvivalentan C++ kod
    #include <iostream>
    using namespace std;

    int main ()
    {
      int x = 0;
      cout << "Unesite cijeli broj: ";
      cin >> x;
      cout << "Unijeli ste " << x << "." << endl;
    }
    ```

!!! admonition "Zadatak"
    Napišite program u kojem korisnik unosi svoje ime i godinu rođenja, a zatim ga program pozdravlja sa `"bok <ime>"`. U novoj liniji se ispisuje `"<ime> je rođen <godina_rođenja> godine."`, a zatim se za taj broj određuje:

    - broj djeljitelja tog broja,
    - sumu djeljitelja tog broja,
    - produkt djeljitelja tog broja,
    - sumu kvadrata djeljitelja tog broja.

!!! admonition "Zadatak"
    Napravite "kalkulator" koji korištenjem *jedne* funkcije `input()` očekuje od korisnika unos *izraza* oblika:

    - `'24 + 8'` (zbroj),
    - `'72 - 15'` (razlika),
    - `'39 * 2'` (umnožak),
    - `'22 / 7'` (količnik).

    Dakle, korisnik unosi izraz koji sadrži i brojeve i operaciju na jedan od četiri opisana načina, a zatim mu se se vraća rezultat odgovarajuće operacije. Nije potrebno implementirati baratanje krivo unesenim izrazima.

## Rad s tekstualnim datotekama

- za otvaranje tekstualne datoteke koristi se `open()`, čiji je prvi argument apsolutna ili relativna putanja do datoteke, a drugi argument način (`"r"`, `"r+"`, `"w"`, `"a"`)

    ``` python
    datoteka = open("dragon.txt", "r")
    ```

    ``` c++
    // ekvivalentan C++ kod
    #include <iostream>
    #include <fstream>
    using namespace std;

    int main ()
    {
      ifstream datoteka;
      datoteka.open("dragon.txt");
      // string linija;
      // datoteka >> linija;
      // cout << linija << endl;
    }
    ```

- za čitanje tekstualne datoteke koristi se `read()`, `readline()` ili `readlines()`

    ``` python
    datoteka.read() # čitav sadržaj datoteke
    datoteka.readline() # jedna linija datoteke
    datoteka.readlines() # sve linije datoteke u listi
    ```

- za zatvaranje tekstualne datoteke koristi se `close()`

    ``` python
    datoteka.close()
    # ovo se prečesto zaboravlja
    ```

!!! admonition "Zadatak"
    S adrese `http://www.textfiles.com/art/dragon.txt` preuzmite u vaš kućni direktorij zmaja nacrtanog u ASCII artu. Pod Linuxom za to možete iskoristiti `wget` ili `curl`, a pod FreeBSD-om `fetch`.

    - Otvorite datoteku za čitanje.
    - Izvedite čitanje čitavog sadržaja datoteke, spremite ga u varijablu `sadrzaj` i ispišite ga na ekran.
    - Razdijelite sadržaj datoteke po znaku `'\n'` i spremite ga u varijablu `sadrzaj_split`.
    - Ispišite svaku treću liniju datoteke. (**Napomena:** Postoji više od jednog načina da to napravite.)

- za zapisivanje u tekstualnu datoteku koristi se `write()`

    ``` python
    # potrebno je prethodno stvoriti datoteku mojtekst.txt
    datoteka = open("mojtekst.txt", "w")
    datoteka.write("Nova linija\n") # zapisuje sadržaj na trenutnoj poziciji objekta pridruženog datoteci
    # datoteka.close() čini da se linije zaista zapišu u datoteku
    ```

    ``` c++
    // ekvivalentan C++ kod
    #include <iostream>
    #include <fstream>
    using namespace std;

    int main ()
    {
      ofstream datoteka;
      // potrebno je prethodno stvoriti datoteku mojtekst.txt
      datoteka.open("mojtekst.txt");
      datoteka << "Nova linija\n";
      // datoteka.close() po završetku rada
    }
    ```

- trenutnu poziciju u tekstualnoj datoteci daje `tell()`

    ``` python
    datoteka.tell() # trenutna poziciju objekta pridruženog datoteci
    ```

- za kretanje kroz tekstualnu datoteku koristi se `seek()`, a pozicija se može navesti u odnosu na: početak datoteke (0), trenutnu poziciju objekta (1), kraj datoteke (2)

    ``` python
    datoteka.seek(0, 6) # postavlja poziciju objekta pridruženog datoteci na sedmi znak
    datoteka.seek(-1, 2) # postavlja poziciju objekta pridruženog datoteci na predzadnji znak
    ```

!!! admonition "Zadatak"
    Napišite program u kojem se korisniku dozvoljava unos proizvoljnih znakovnih nizova koji se zatim spremaju u datoteku `dat-unosi.txt`.

    - Ukoliko je uneseni znakovni niz oblika `"b.*"` u regex notaciji (počinje nulom, a zatim ima proizvoljne znakove), sprema se na početak datoteke i eventualno briše postojeći sadržaj.
    - U protivnom, unesenom nizu se na kraj dodaje znak za novi redak i niz se sprema na kraj datoteke.

    Za isprobavanje koristite znakovne nizove različitih duljina. Što se događa kod unosa niza znakova veće duljine od sadržaja prvog retka datoteke ako niz počinje slovom `b`? Objasnite zašto.

!!! admonition "Zadatak"
    Napišite program koji učitava tekstualnu datoteku i tekst iz nje obrađuje na način da briše sve vokale i sve razmake duplira, te rezultat zapisuje u novu datoteku. Datoteku obrađujte liniju po liniju.
