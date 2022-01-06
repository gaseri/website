---
author: Vedran Miletić, Vanja Slavuj, Sanja Pavkov
---

# Informacijski čvorovi, vremena i poveznice datoteka

## Informacijski čvorovi i vremena datoteka

- inode sadrži metapodatke o datoteci: type, mode, link count, user ID, group ID, size, mtime, ctime, atime, device ID, pokazivače na blokove diska u kojima je spremljen sadržaj datoteke

    - jednoznačno ga identificira inode broj koji jednoznačno određuje jednu i samo jednu datoteku

- mtime, ctime, atime -- vrijeme promjene datoteke, vrijeme promjene inode-a i vrijeme zadnjeg pristupa
- `stat` prikazuje metapodatke koje sadrži inode datoteke

!!! admonition "Zadatak"
    - Stvorite neku datoteku naziva `vjezba1.txt`. Saznajte njezin inode broj, ID uređaja koji je sadrži te vremena.
    - Nadopišite u nju svoje ime i prezime. Usporedite inode broj i metapodatke sa onima iz prethodnog slučaja.
    - Ispišite njezin sadržaj na ekran i usporedite vremena prije i poslije ispisivanja.
    - Kopirajte datoteku u datoteku `vjezba2.txt`. Usporedite inode broj i metapodatke obje datoteke.
    - Primijenite naredbu `touch` na datoteku `vjezba2.txt`. Na koja vremena datoteke ta naredba djeluje?

    (**Napomena:** zadatak će nešto drugačije od očekivanog raditi na terminalima u računalnoj učionici.)

## Čvrste poveznice

- `ls -l` izlistava datoteke u tzv. dugom ispisu

    - vrsta datoteke, dozvole, broj čvrstih poveznica na datoteku, korisnik kojem datoteka pripada, grupa kojoj datoteka pripada, veličina, vrijeme zadnje promjene, ime
    - broj čvrstih poveznica kod direktorija uključuje i poveznice `.` i `..` pa ovisi o broju poddirektorija

- `ln datoteka poveznica` stvara čvrstu poveznicu na datoteku, koja ovisi o inode broju; **primjer:**

    ``` shell
    $ ln test.txt test_poveznica.txt
    ```

!!! admonition "Zadatak"
    - U svom kućnom direktoriju stvorite direktorij pod nazivom `dir1` i u njemu:

        - još jedan direktorij pod nazivom `vjezba`, te
        - neku datoteku `emacs1.txt` u koju ćete zapisati današnji datum.

    - Napravite u direktoriju `vjezba` čvrstu poveznicu `link` na `emacs1.txt` koristeći relativno referenciranje.
    - Ispišite sadržaj čvrste poveznice `link`.
    - Koliko je čvrstih poveznica na dirketorij `dir1`? Koje su to?
    - Napravite u direktoriju `dir1` još 2 direktorija po želji pa opet odgovorite na gornje pitanje.
    - U direktoriju `vjezba` napravite čvrstu poveznicu `link2` na direktorij `dir1` koristeći apsolutno referenciranje.

## Simboličke poveznice

- `ln -s datoteka poveznica` stvara simboličku poveznicu na datoteku, koja ovisi o imenu datoteke, slično shortcutima; **primjer:**

    ``` shell
    $ ln -s test.txt test_simbolicka_poveznica.txt
    ```

- jedan od zanimljivih problema koji mogu nastati ako se simboličke poveznice pogrešno koriste je [petlja simboličkih poveznica](https://tuxdna.wordpress.com/2011/12/10/symlink-loop-is-still-an-unsolved-problem/)

!!! admonition "Zadatak"
    - Stvorite u svom direktoriju tekstualnu datoteku `pepeljuga`, i upišite u nju sadržaj po želji.

        - Stvorite simboličku poveznicu na nju koja pristupa relativnim referenciranjem, nazovite je `princ`.
        - Stvorite simboličku poveznicu na nju koja pristupa apsolutnim referenciranjem, nazovite je `princeza`.
        - Stvorite na nju čvrstu poveznicu, nazovite je `dvorac`.
        - Objasnite izlaz naredbi `ls -s` i `ls -i`.
        - Izbrišite datoteku `pepeljuga`. Koje od poveznica pucaju?

    - Što rade sljedeće naredbe?

        - `ls -s`
        - `ls -i`

    - Pokušajte simboličkim poveznicama stvoriti ciklus. Što se dogodi? Objasnite.
    - Pokušajte napraviti čvrstu i simboličku poveznicu na datoteku koja ne postoji. Objasnite što se događa i zašto je to u skladu s načinom rada poveznica.
