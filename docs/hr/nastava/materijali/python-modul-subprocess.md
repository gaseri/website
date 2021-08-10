---
author: Vedran Miletić
---

# Python: međuprocesna komunikacija: podprocesi

- `module subprocess` ([službena dokumentacija](https://docs.python.org/3/library/subprocess.html)) omogućuje pokretanje novih (pod)procesa, povezivanje na njihove cijevi za ulaz, izlaz i greške, i dohvaćanje njihovih izlaznih kodova
- `subprocess.call([arg0, arg1, ...])` poziva naredbu s navedenim argumentima i vraća njezin izlazni status
- `subprocess.getoutput("arg0 arg1 ...")` poziva naredbu s navedenim argumentima i vraća njezin izlaz
- `subprocess.getstatusoutput("arg0 arg1 ...")` poziva naredbu s navedenim argumentima i vraća uređeni par `(status, output)`

``` python
subprocess.call(["ls", "-l"]) # vraća izlazni status 0 ako je naredba uspješna
subprocess.getoutput("ls -l") # ekvivalentno subprocess.call() iznad
subprocess.getstatusoutput("ls -l") # vraća (0, "znakovni niz koji sadrži izlaz naredbe ls -l") ako je naredba uspješna
```

- uočimo razliku u navođenju naredbe i parametara u obliku liste kod `subprocess.call()` nasuprot navođenju naredbe i parametara u obliku znakovnog niza kod `subprocess.getoutput()` i `subprocess.getstatusoutput()`

!!! admonition "Zadatak"
    Usporedite poziv brisanja datoteke koja postoji naredbom `rm` s pozivom brisanja datoteke koja ne postoji u terminima izlaznog statusa.

!!! admonition "Zadatak"
    - Usporedite izlaz funkcije `os.listdir()` s izlazom naredbe `ls` dohvaćenim korištenjem sučelja koja nudi modul `subprocess`. Ima li razlike u datotekama koje su izlistane?
    - Izvedite čitanje sadržaja datoteke `/etc/group` korištenjem standardnih Python sučelja za rad sa datotekama, a zatim dohvaćanjem izlaza naredbe `cat` na tu datoteku. Ima li razlike u pročitanom sadržaju?

- `subprocess.Popen()` vraća objekt pridružen potprocesu koji pokreće, a zatim se može koristiti za upravljanje tim procesom

    - `subprocess.Popen.communicate(input)` omogućuje interakciju s procesom koji se izvodi

        - prosljeđuje vrijednost opcionalnog parametra `input` na `stdin`; parametar mora biti tipa `bytes` i konverziju vršimo iz `str` u `bytes` vršimo metodom `str.encode()`
        - vraća uređeni par `(stdout, stderr)`, oba elementa su tipa `bytes`

    - `subprocess.Popen.wait()` čeka na završetak izvođenja procesa i vraća njegov izlazni status
    - `subprocess.Popen.terminate()` zaustavlja proces signalom `SIGTERM`
    - `subprocess.Popen.kill()` ubija proces signalom `SIGKILL`

- `subprocess.PIPE` je objekt koji se koristi da se standardni ulaz, standardni izlaz ili standardni izlaz za greške dohvate u Python interpreteru korištenjem metode `communicate()` objekta pridruženog procesu

``` python
p1 = subprocess.Popen(["ls", "-l"], stdout=subprocess.PIPE)
popis = p1.communicate()

# rm u interaktivnom načinu rada
p2 = subprocess.Popen(["rm", "-i", "datotekaKojaPostoji"], stdin=subprocess.PIPE)
p2.communicate("y\n".encode())
```

!!! admonition "Zadatak"
    Korištenjem objekta subprocess.Popen:

    - naredbom `ps` dohvatite popis procesa svih korisnika, sa i bez terminala,
    - dohvatite poruku koju naredba `man` ispisuje kada je pokrećete bez argumenata (razmislite na koji izlaz man ispisuje tu poruku),
    - naredbom `rm` pokušajte interaktivno (parametar `-i`) izbrisati neku datoteku za koju nemate dozvolu, npr. `/etc/hosts` i dohvatite izlazni status koji dobivate.

!!! admonition "Zadatak"
    Napravite Python skriptu koja traži od korisnika imena triju datoteka. Datoteke se brišu interaktivnim načinom rada, a od korisnika se traži unos na standardni ulaz "kao da izravno radi s naredbom" (odnosno unosi `y` ako želi obrisati i `n` ako ne želi, a to se prosljeđuje naredbi putem metode `communicate()` objekta pridruženog procesu).

!!! admonition "Zadatak"
    Za stvaranje korisnika na Linuxu koristi se naredba `useradd`. Proučite pripadnu man stranicu, a zatim napišite Python skriptu koja, uz pretpostavku da imate odgovarajuće ovlasti:

    - stvara korisnike `student01`, `student02`, ..., `student99` s pripadnim kućnim direktorijima,
    - u kućnom direktoriju svakog od korisnika stvara direktorij `python-samples` kojem postavlja dozvole na `rwxrwxrwx`,
    - na kraj datoteke `.bashrc` dodaje naredbu koja podešava korisničku masku na 077.
