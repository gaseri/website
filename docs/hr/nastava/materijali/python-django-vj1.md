---
author: Milan Petrović
---

# Django vježba 1: Postavljanje razvojnog okruženja web aplikacija na operacijskim sustavima sličnim Unixu i Windowsima. Stvaranje projekta i organizacija koda. Korištenje dokumentacije

## Instalacija

IDE u kojem se radi na vježbama je Visual Studio Code koji možete preuzeti na [linku](https://code.visualstudio.com/).

Nakon što preuzimete i instalirate VS Code. Pokrenite ga, zatim s kraticom `CTRL` + `SHIFT` + `X` otvorite dio za instalaciju ekstenzija, pronađite Python i instalirajte ga. Dodatna upustva za instalaciju Python ekstenzije unutar VS Codea možete pronaći na [linku](https://code.visualstudio.com/docs/python/python-tutorial).

Instalacija [pip](https://pypi.org/project/pip/)-a (kod sebe doma na WSL ili kakvom god Ubuntuu):

``` shell
$ sudo apt install python3 python3-pip python-is-python3 pylint
(...)
```

(Nepreporučeno jer je Django verzija 2.x) Instalacija pakirane verzije [Djanga](https://www.djangoproject.com/):

``` shell
$ sudo apt install python3-django
(...)
```

(Preporučeno jer je Django verzija 3.x) Instalacija Djanga korištenjem pip-a (kod sebe doma na WSL ili kakvom god Ubuntuu, uključujući i učionicu):

``` shell
$ pip3 install Django
(...)
```

ili

``` shell
$ python3 -m pip install Django
(...)
```

Primjer web sjedišta u Djangu:

- [Instagram](https://www.instagram.com/) (velikim dijelom)
- [RiTeh](http://www.riteh.uniri.hr/) (shoutout: [Zavod za računarstvo](http://www.riteh.uniri.hr/ustroj/zavodi/zr/))

## Stvaranje prvog projekta

Stvaranje direktorija gdje ćemo stvoriti naš projekt:

``` shell
$ mkdir moja_prva_stranica
(...)
```

Zatim idemo u taj direktorij naredbom:

``` shell
$ cd moja_prva_stranica
(...)
```

U njemu stvaramo Django projekt naredbom:

``` shell
$ django-admin startproject mysite
(...)
```

Ostale raspoložive naredbe možemo vidjeti s komandom `django-admin`, a za detaljnije o pojedinim naredbama koristite `django-admin help`, npr. za naredbu `startproject` na način:

``` shell
$ django-admin help startproject
(...)
```

Naredba `startproject` kreira novi direktorij naziva kojega smo proslijedili, u našem slučaju to je `mysite`.

Prebacimo se u direktorij `mysite` pomoću naredbe `cd mysite` i ispišimo sadržaj direktorija u obiliku stabla s naredbom `tree mysite`, možete koristiti i naredbu `ls mysite`.

``` shell
$ tree mysite
mysite
├── manage.py
└── mysite
    ├── asgi.py
    ├── __init__.py
    ├── settings.py
    ├── urls.py
    └── wsgi.py
```

Direktorij `mysite` je središnji direktorij naše stranice koju ćemo graditi. Ovo je "glavna" tj. središnja aplikacija. Njena uloga je usmjeravanje Djanga na druge aplikacije koje ćemo kreirati. Ono što ćemo tu razvijati je samo usmjeravanje na druge aplikacije koje razvijamo i podešavanje postavki.

### Projektni direktorij i njegov sadržaj

Sadržaj koji se nalazi u novo kreiranom projektu:

Izvan direktorija `mysite/`: korijenski je direktorij koji ima ulogu mjesta gdje se nalaze sve datoteke i direktoriji projekta.

Unutar direktorija `mysite/`: nalazi se Python package projekta, a njegov sadržaj je prikazan iznad.

Datoteka `mysite/__init__.py`: Govori interpreteru da sve unutar ovoga direktorija bude tretirano kao Python package.

Datoteka `mysite/settings.py`: Postavke/konfiguracija za Django projekt. Obratiti pažnju na `INSTALLED_APPS` koji nam je trenutno najbitniji dio u ovoj datoteci koji će nam trebati. Svaki puta kada se dodaje nova aplikacija na našu web stranicu, moramo ju ovdje ručno dodavati.

!!! danger
    Paziti na `SECRET_KEY` da nije vidljiv svima, npr. na GitHubu-u. Zlonamjerni pojedinac može to iskoristiti i upravljati vašom stranicom kao administrator.

Datoteka `mysite/urls.py`: Sadrži deklarirane URLove vezane za Django projekt, služi Django aplikaciji kao "kazalo sadržaja". Njegova uloga je 'kontroliranje' naše stranice/aplikacije. Pomoću njega 'pokazujemo/usmjeravamo' program na naše aplikacije.

!!! warning "Osvježite si znanje regularnih izraza"
    - [Regularni izrazi na kolegiju Operacijski sustavi](grep-sed-awk-tr.md#izdvajanje-linija-iz-tekstualnih-datoteka)
    - [Regularni izrazi i Python](https://docs.python.org/3/library/re.html#module-re)

Datoteke `mysite/wsgi.py i mysite/asgi.py`: ASGI je nasljednik WSGI, dugogodišnjeg Python standarda za kompatibilnost između web poslužitelja, frameworka i aplikacija. WSGI povećava mogućnosti u radu na webu pomoću Pythona, dok je cilj ASGI-a produžiti to za područje asinkronog Pythona.

Datoteka `manage.py`: Koristimo pri radu u terminalu za prosljeđivanje raznih naredbi koje prosljeđujemo programu da se izvrše nad projektom, više o tome u nastavku.

### Kreiranje prve aplikacije

Započnimo s kreiranjem prve aplikacije. Za početak dodat ćemo aplikacije na naš projekt `mysite`, to činimo naredbom: `./manage.py startapp main`. Primijetit ćete da u direktoriju `mysite` je stvoren novi poddirektorij naziva `main`.

!!! zadatak
    Provjerite sadržaj direktorija `mysite/mysite` i direktorija `mysite/main` s naredbama `tree` i `ls`.

#### Lokalni poslužitelj za razvoj

Pokrenimo sada naš lokalni poslužitelj (*eng.server*) na kojemu ćemo razvijati aplikaciju.

!!! warning "Napomena za pokretanje lokalnog poslužitelja"
    Za pokretanje poslužitelja koristite zasebni terminal. Poslužitelj aktivno radi za vrijeme razvoja i nije potrebno ponovno pokretati poslužitelj nakon svake promjene.

U zasebnom terminalu za pokretanje poslužitelja koristi se naredba:

``` shell
$ ./manage.py runserver
(...) 
```

!!! zadatak
    Provjerite rad poslužitelja posjetom adrese `http://127.0.0.1:8000/` u web pregledniku.

Ovo je poslužitelj na kojem ćemo razvijati Django aplikacije.

U terminalu možemo vidjeti HTTP zahtjeve na poslužitelj i statusni kod odgovora na njih.

!!! zadatak
    Provjerite ispis u terminalu prilikom osvježavanja stranice pritiskom na tipki`CTRL` + `R` ili `F5`.

### Dodatak: Čitanje dokumentacije i StackOverflow

> Tko radi taj i griješi, a iz grešaka se najbolje uči.

Djangovu službenu dokumentaciju možete pronaći [ovdje](https://docs.djangoproject.com/en/3.2/).

[Stack Overflow](https://stackoverflow.com/) je mjesto za pitanja i odgovore za profesionalne programere i entuzijaste. Sadrži pitanja i odgovore na širok raspon tema u računalnom programiranju. Više na [Wikipediji](https://en.wikipedia.org/wiki/Stack_Overflow).

Primjeri pitanja postavljenih na Stack Overflowu vezanih za probleme u Pythonu i Djangu:

- [Primjer 1](https://stackoverflow.com/questions/3500371/trouble-installing-django)
- [Primjer 2](https://stackoverflow.com/questions/28842901/how-to-start-a-new-django-project)
- [Primjer 3](https://stackoverflow.com/questions/755857/default-value-for-field-in-django-model)
