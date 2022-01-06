---
author: Vedran Miletić, Vanja Slavuj, Sanja Pavkov
---

# Stablo direktorija

- **višekorijenski** operacijski sustav za više više particija ima više korijenskih direktorija

    - primjerice, operacijski sustav Windows particije označava kao `C:\`, `D:\`, `E:\`, ...

- **jednokorijenski** operacijski sustav za više particija imaju jedan korijenski direktorij

    - operacijski sustavi slični Unixu mogu imati particije u direktorijima `/`, `/home`, `/media/usbdisk`, ...

## Uloga pojedinih direktorija u sustavu

- [Filesystem Hierarchy Standard](https://en.wikipedia.org/wiki/Filesystem_Hierarchy_Standard) definira naziv u ulogu direktorija u datotečnom sustavu

    - dugo vremena je bila aktualna verzija 2.3, siječanj 2004., trenutna aktualna je verzija verzija 3.0, ljeto 2012.
    - više informacija možete pronaći na [stranici s dokumentima o FHS](https://www.pathname.com/fhs/)

- `/bin`, `/sbin`, `/lib` -- izvršne datoteke i bibliotečne datoteke

    - izvršne datoteke napravljene iz izvornog koda koji ima funkciju `main()`
    - bibliotečne datoteke napravljene iz izvornog koda koji nema funkciju `main()`

- `/dev` sadrži u sebi uređaje
- `/etc` sadrži konfiguracijske datoteke -- one određuju kako se programi ponašaju
- `/root`, `/home/<ime_korisnika>` su redom root direktorij administratora, kućni direktoriji ostalih korisnika
- `/proc`, `/sys` sadrže u sebi inforamcije o sustavu (hardveru)
- `/lost+found` sadrži dijelove datoteka koje su pronađene u datotečnom sustavu prilikom oporavka od pada sustava
- `/media`, `/mnt` sadrži direktorije putem kojih je dostupna vanjska memorija (USB, fotoaparat, digitalna kamera..)
- `/opt` unutar njega su poddirektoriji, najčešće sa komercijalnim programima koji se instaliraju drugačije od ostalog softvera
- `/tmp` sadrži privremene datoteke
- `/var` sadrži datoteke koje program koristi kod pokretanja (datoteke koje se mijenjaju dok program radi)
- `/usr`, `/usr/local` sadrži programe razdijeljene po poddirektoirjima `/usr/bin`, `/usr/sbin`, `/usr/lib`, `/usr/share`

    - konvencija je da se softver koji se instalira putem upravitelja paketima (sustav sličan Apple Store-u i Google Play-u) postavlja u `/usr`, a softver koji administrator instalira mimo upravitelja paketima u `/usr/local`

!!! admonition "Zadatak"
    - Izlistajte sve obične i sakrivene direktorije i datoteke u vašem kućnom direktoriju.
    - Izlistajte sve direktorije u `/home` direktoriju. Što vidite?
    - Objasnite što se nalazi u sljedećim direktorijima:

        - `/home/student12`
        - `/usr/include/c++/4.6`
        - `/dev/input`
        - `/etc/acpi`

    (**Uputa:** podsjetite se koje je značenje kratice ACPI.)

## Rad s direktorijima, naredbe `pwd` i `cd`

- `pwd` ispisuje putanju do radnog direktorija
- `cd` mijenja radni direktorij u dani

    - bez argumenata vraća vas u kućni direktorij

- `<Tab>` -- kompletiranje imena datoteka i direktorija

    - postoji razlika između `<Tab>` i `<Tab><Tab>`
    - na sličan način kompletiraju se imena naredbi
    - na sličan način kompletiraju se nazivi parametara *nekih* naredbi (u novijm verzijama `bash` ljuske, datoteka `/etc/bash_completion` i direktorij `/etc/bash_completion.d`)

!!! admonition "Zadatak"
    Isprobajte sljedeće naredbe i objasnite što rade:

    - `cd`
    - `cd .`
    - `cd ..`
    - `cd ~`
    - `cd -`
    - `cd ../..`
    - `cd ./././..`
    - `cd ../.././.`
    - `cd ../../../../../..`

- `.` referira na trenutni direktorij
- `..` referira na direktorij iznad trenutnog
- `../..` referira na direktorij iznad direktorija iznad trenutnog (tako možemo i dalje)

## Naredbe `mkdir` i `rmdir`

- `mkdir` stvara direktorij s danim imenom
- `rmdir` briše (prazan) direktorij danog imena

!!! admonition "Zadatak"
    - U Vašem kućnom direktoriju napravite predloženu strukturu direktorija:

        ```
        studentXY ---------- Ispiti ------------- Ispit1 -------- 15102012
                  |                        |
                  |                        |----- Ispit2
                  |
                  |--------- Kolokviji ---------- Kolokvij1
                  |                        |
                  |                        |----- Kolokvij2 ----- Rjesenja
                  |
                  |--------- Seminari
                  |
                  |--------- DZ ----------------- Grafovi
        ```

    - Uđite u direktorij `Ispiti` i pokušajte izbrisati direktorij `Ispit1`. Što se događa?
    - Pozicionirajte se u direktorij `Kolokviji` i u jednoj naredbi pokušajte izbrisati sve poddirektorije koji se ondje nalaze.
    - Vratite se do direktorija `Seminari` i uđite u njega. Pokušajte ga izbrisati. Što se događa? Zašto?
    - Otiđite do svog kućnog direktorija i od tamo pokušajte izbrisati direktorij `15102012`. Zašto ga ne možete izbrisati?
    - Uđite u direktorij `Kolokvij` i iz njega pokušajte izbrisati direktorij `DZ`. Možete li to učiniti? Zašto?

!!! admonition "Dodatni zadatak"
    - U svom kućnom direktoriju stvorite poddirektorij `trnoruzica`. Uđite u taj direktorij.
    - Stvorite direktorije `mikimaus` i `minimaus`.
    - Uđite u direktorij `mikimaus`.
    - Probajte sada izbrisati direktorij `minimaus`. Objasnite zašto to ne možete.
    - Vratite se u direktorij `trnoruzica` i izbrisite direktorij `minimaus`.
    - Isprobajte naredbe `cd -`, `cd ../.` i `cd -/..` te objasnite što rade.

## Apsolutno i relativno referenciranje

- `/home/vedran/radnidir`

    - apsolutno referira na `radnidir`, radi od svugdje
    - apsolutno referenciranje kreće od korijenskog direktorija i ide do traženog direktorija, uvijek započinje sa `/`

- `radnidir`

    - relativno referira na `radnidir`, radi samo kad se nalazimo u direktoriju `/home/vedran`
    - relativno referenciranje kreće od trenutnog direktorija i ide do traženog direktorija, nikad ne započinje znakom `/`

!!! admonition "Zadatak"
    - Izlistajte kućni direktorij korisnika `prof`:

        - apsolutnim referenciranjem iz svojeg kućnog direktorija,
        - apsolutnim referenciranjem iz korijenskog direktorija,
        - relativnim referenciranjem iz svojeg kućnog direktorija,
        - relativnim referenciranjem iz korijenskog direktorija.

    - Objasnite zašto prva dva dijela imaju isto rješenje.
    - Koji je vaš kućni direktorij na računalu na kojem trenutno radite?
    - U kojem se direktoriju nalazi moj kućni direktorij?
    - U kojem se direktoriju nalazi vaš kućni direktorij, a u kojem vaš `home` direktorij?
    - Imate li pristup kućnom direktoriju korisnika `student08`? Izlistajte sadržaj neke njegove datoteke.

## Rekurzivnost u radu s direktorijima i datotekama

- `ls -R` radi rekurzivno izlistavanje

    - izlista direktorij i njegove poddirektorije
    - na sličan način radi brisanje direktorija u kojem postoje poddirektoriji i datoteke

!!! admonition "Zadatak"
    - Koristeći naredbu `ls` izlistajte rekurzivno svoj kućni direktorij.
    - Izlistajte rekurzivno sljedeće direktorije:

        - `/usr`,
        - `/usr/local` (ovaj direktorij izlistajte dva puta: prvi put koristeći relativno, a drugi put apsolutno referenciranje).

    - Što se događa kada umjesto parametra `-R` koristite parametar `-r`? Isprobajte.

!!! admonition "Ponovimo!"
    - Navedite po jedan primjer za jednokorijenski i višekorijenski operacijski sustav.
    - Što je FHS?
    - Prisjetite se koja je uloga pojedinih direktorija u datotečnom sustavu.
    - Postoji li razlika između kućnog i `home` direktorija korisnika?
    - Na koji način možemo kompletirati nazive direktorija i datoteka?
    - Kako izglda referenca za trenutni (radni) direktorij? Kako za onaj jednu hijerarhijsku razinu više?
    - Ponovite razliku između apsolutnog i relativnog referenciranja.
    - Ako se nalazite u vlastitom kućnom direktoriju, napišite relativnu i apsolutnu adresu koja će vas odvesti do korijenskog direktorija.
    - Čemu služi rekurzivno izlistavanje direktorija?
    - Kakve informacije daje naredba `pwd`?
