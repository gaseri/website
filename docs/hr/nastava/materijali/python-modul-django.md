---
author: Milan Petrović
---

# Python: Django

Primjeri web sjedišta u Djangu:

- [Instagram](https://www.instagram.com/) (velikim dijelom)
- [RiTeh](http://www.riteh.uniri.hr/) (shoutout: [Zavod za računarstvo](http://www.riteh.uniri.hr/ustroj/zavodi/zr/))

## Postavljanje razvojnog okruženja web aplikacija na operacijskim sustavima sličnim Unixu i Windowsima. Stvaranje projekta i organizacija koda. Korištenje dokumentacije

### Stvaranje prvog projekta

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

#### Projektni direktorij i njegov sadržaj

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

#### Kreiranje prve aplikacije

Započnimo s kreiranjem prve aplikacije. Za početak dodat ćemo aplikacije na naš projekt `mysite`, to činimo naredbom: `./manage.py startapp main`. Primijetit ćete da u direktoriju `mysite` je stvoren novi poddirektorij naziva `main`.

!!! zadatak
    Provjerite sadržaj direktorija `mysite/mysite` i direktorija `mysite/main` s naredbama `tree` i `ls`.

##### Lokalni poslužitelj za razvoj

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
    Provjerite ispis u terminalu prilikom osvježavanja stranice pritiskom na tipki ++control+r++ ili ++f5++.

#### Dodatak: Čitanje dokumentacije i StackOverflow

> Tko radi taj i griješi, a iz grešaka se najbolje uči.

Djangovu službenu dokumentaciju možete pronaći [ovdje](https://docs.djangoproject.com/en/3.2/).

[Stack Overflow](https://stackoverflow.com/) je mjesto za pitanja i odgovore za profesionalne programere i entuzijaste. Sadrži pitanja i odgovore na širok raspon tema u računalnom programiranju. Više na [Wikipediji](https://en.wikipedia.org/wiki/Stack_Overflow).

Primjeri pitanja postavljenih na Stack Overflowu vezanih za probleme u Pythonu i Djangu:

- [Primjer 1](https://stackoverflow.com/q/3500371)
- [Primjer 2](https://stackoverflow.com/q/28842901)
- [Primjer 3](https://stackoverflow.com/q/755857)

## Korištenje baze podataka. Stvaranje modela i objektno-relacijsko preslikavanje

### Paradigma model-view-controller (MVC)

Django koristi paradigmu [model-view-controller](https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93controller) (MVC) koju naziva model-template-view (MTV). Svaka stranica kreirana pomoću Django frameworka vjerojatno koristi ove tri stvari kako bi vam prikazala podatke.

Iako ga nazivamo MVC (model-view-controller), način na koji radi ide u obrnutom slijedu. Korisnik će posjetiti URL, vaš kontroler (`/urls.py`) ukazat će na određeni prikaz (`/views.py`). Taj se prikaz tada može (ili ne mora) povezati s vašim modelima. Drugim riječima, prvo ide Controller, zatim View i na posljetku Model po potrebi.

Unutar Djanga koristi se djelomično izmijenjena terminologija. Stoga od sad nadalje koristit ćemo Django terminologiju (model, template i view). U tablici u nastavku detaljnije su objašnjeni pojmovi i termini koji se koriste.

| Korišteni naziv | Django naziv | Značenje |
| -------------- | ----------- | ------------------------------------------------------------------------------------------------ |
| Model | Model | Sloj modela u Djangu odnosi se na bazu podataka plus Python kôd koji je izravno koristi. Modelira stvarnost. Pohranjuje sve potrebne vrijednosti unutar baze podataka koje su potrebne web aplikaciji. Django vam omogućuje pisanje Python klasa koje nazivamo modeli, koje se vezuju za tablice baze podataka. Stvoreni modeli nisu trajno zadani, nego se mogu izmjenjivati i dopunjavati. Izmjene su dostupne odmah nakon primjene migracija o kojima detaljnije kasnije na ovim vježbama u poglavlju ORM. |
| View | Template | View sloj u Djangu odnosi se na korisničko sučelje (UI). Funkcija pregleda ili skraćeno prikaz (*eng. view*) je Python funkcija koja je zadužena za generiranje HTMLa i ostalih UI elemenata. Pomoću Python kôda renderiraju se pogledi. Django uzima web zahtjev i vraća web odgovor. Ovaj odgovor može biti bilo što, npr. HTML sadržaj web stranice, preusmjeravanje, [statusni kod](https://http.cat/), XML dokument ili slika... |
| Controller | View | Središnji dio sustava, sadržava logiku koja povezuje cjeline da bi se pružio odgovor korisniku na traženi zahtjev. Upravlja zahtjevima i odgovorima na njih. uspostavlja vezu s bazom podataka i učitavanjem datoteka. |

### Uvod u Django: Hello world

U nastavku ovog poglavlja prikazano je kako kreirati Django aplikaciju za teksta na početnoj stranici. Za ovo nije potreban Model nego samo View u kojemu je definiran tekst koji želimo prikazati na stranici.

!!! zadatak
    Otvorite datoteku i provjerite što je zapisano u `mysite/mysite/url.py`.

Za definiranje putanje na koju će Django primati HTTP zahtjeve iskoristit ćemo funkciju `django.urls.path()` ([dokumentacija](https://docs.djangoproject.com/en/3.2/ref/urls/#path)) i usmjerit ćemo na administratorsko sučelje ([dokumentacija](https://docs.djangoproject.com/en/3.2/ref/contrib/#admin)).

Datoteka `mysite/mysite/url.py` ima sadržaj

``` python
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
]
```

Vidimo da imamo samo jedan URL koji se tiče stranice za administraciju koji nam trenutno ne treba. Sljedeće što je potrebno učiniti je usmjeriti view na URL. Razlog tome je što Django web stranice vidi kao kolekciju aplikacija koje 'ispisuje' pomoću danih URLova koji 'pokazuju' Djangu gdje tražiti. Datoteka `urls.py` unutar naše glavne aplikacije obično samo usmjerava na aplikacije.

Pa krenimo sa kreiranjem aplikacije, i to na način da ju kreiramo pomoću naredbe `./manage.py startapp main`.

Idemo sada usmjeriti naš glavni dio Django aplikacije da provjerava, odnosno poziva našu novostvorenu aplikaciju. To radimo unutar `mysite/mysite/url.py`:

``` python
from django.contrib import admin
from django.urls import path, include #importamo include

urlpatterns = [
    path('', include('main.urls')), #dodajemo urls
    path('admin/', admin.site.urls),
]
```

Ako pogledamo u direktorij aplikacije `main` vidjet ćemo da datoteka `urls.py` ne postoji. Kreirajmo datoteku `mysite/main/urls.py` i dodajmo sljedeći sadržaj.

``` python
from django.urls import path
from . import views

app_name = 'main'  # here for namespacing of urls.

urlpatterns = [
    path('', views.homepage, name='homepage'),
]
```

Prođimo ukratko kroz logiku rada našeg programa. Prvo se posjećuje `mysite/url.py`, u URL-u nije prosljeđeno ništa, stoga odgovara `''` iz `path('', include('main.urls'))`. Program ovo tumači tako da uključuje `main.urls`.

Program zatim pronalazi i učitava `main.urls` koji se nalazi na lokaciji `mysite/main/urls.py` i čita njegov sadržaj.
Upravo smo izmjenili sadržaj tako da smo dodali uzorak `''` koji odgovara `path('', views.homepage, name='homepage')`. Ovime smo usmjerili aplikaciju da pokrene funkciju `homepage()` unutar `main/views.py` koji još nismo izmjenili stoga ćemo to sada učiniti rako da dodamo funkciju naziva `homepage()`.

Datoteka `mysite/main/views.py` ima sadržaj:

``` python
from django.shortcuts import render
from django.http import HttpResponse

## Create your views here.
def homepage(request):
    return HttpResponse('Welcome to homepage! <strong>#samoOIRI</strong>')
    # primjetiti korištenje HTML-a
```

Osvježimo stranicu `http://127.0.0.1:8000/` u browseru.

### Modeli: Uvod

Nastavak na program iz uvoda. U ovom poglavlju detaljnije je objasnjeno stvaranje modela u Djangu.

Stvoreni model se mapira na tablicu u bazi podataka. Baza poidataka je već stvorena i možete ju vidjeti na lokaciji `mysite/mysite`. Po defaultu tip baze podataka je SQLite 3 pa je ekstenzija datoteke `.sqlite3`. Tip baze podataka možete mijenjati u `settings.py` pod `DATABASES`.

#### Stvaranje modela i polja modela

Ovime stvaramo novu tablicu u bazi podataka a zadana polja postaju stupci u toj tablici. Automatski se po defaultu kreira primarni ključ iz toga, ali po želji možemo i proizvoljno odrediti da neka od zadanih vrijednosti to bude.

!!! zadatak
    Otvorite `main/models.py` i definirajte klasu imena Predmet. U klasi definirajte stupce naziva:`predmet_naslov`, `predmet_sadrzaj` i `predmet_vrijeme_objave`. Njihovi tipovi neka budu `CharField()` koji ima zadan parametar `max_length` na `100`, `TextField()` i `DateTimeField()` koji ima naziv postavljen na `'date published'`.

**Rješenje zadataka.** U datoteku `main/models.py` dodajemo:

``` python
class Predmet(models.Model):
    predmet_naslov = models.CharField(max_length=100)
    predmet_sadrzaj = models.TextField()
    predmet_vrijeme_objave = models.DateTimeField('date published')
```

!!! zadatak
    Definirajte funkciju `__str__()` unutar klase `Predmet` koja vraća naziv predmeta `predmet_naziv`.

**Rješenje zadataka.**  U datoteku `main/models.py` unutar klase `Predmet` dodajemo:

``` python
def __str__(self):
    return self.predmet_naslov
```

Službena [Django dokumentacija](https://docs.djangoproject.com/en/3.2/ref/models/fields/) o svim poljima unutar modela.

#### Objektno-relacijsko preslikavanje

Svaki novi model je nova tablica u bazi podataka. Zbog toga moramo napraviti dvije stvari. Prva je pripremiti za migraciju naredbom `makemigrations`, a zatim napraviti migraciju nredbom `migrate`.

Pokrenimo naš lokalni server naredbom `./manage.py runserver`. Primijetite u outputu konzole sljedeću poruku `You have 18 unapplied migration(s). Your project may not work properly until you apply the migrations for app(s): admin, auth, contenttypes, sessions
Run './manage.py migrate' to apply them`.

Unesite naredbu kako biste pripremili migraciju.

``` shell
$ ./manage.py makemigrations
No changes detected
```

Razlog tomu je što još nismo povezali `main` aplikaciju. Način na koji ćemo povezati aplikaciju je taj da ju pozovemo unutar `INSTALLED_APPS` unutar `mysite/settings.py`.

Možemo vidjeti da je unutar `main/apps.py` definirana kalsa `MainConfig`.

Dopunimo sada `mysite/settings.py` sa pozivom klase `MainConfig` iz aplikacije `main`. To radimo na način da pod `INSTALLED_APPS` dodamo `'main.apps.MainConfig'`.

Datoteka `mysite/mysite/settings.py` sad ima sadržaj:

``` python
## Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'main.apps.MainConfig' # ovu liniju dodajemo
]
```

Unesite ponovo naredbu `./manage.py makemigrations` da biste pripremili migraciju.

``` shell
$ ./manage.py makemigrations
Migrations for 'main':
  main\migrations\0001_initial.py
    - Create model Predmet
```

!!! zadatak
    Pročitajte sadržaj unutar datoteke `main\migrations\0001_initial.py`.

Zatim unesimo naredbu `./manage.py migrate` da bi se migracija izvršila.

Nakon svakog rada sa modelima, bile to izmjene ili stvaranje novih modela potrebno je napraviti migraciju na načn `makemigrations` a zatim `migrate`.

Pogledajmo output u terminalu koji nam vraća naredba `./manage.py migrate`:

``` shell
$ ./manage.py migrate
Operations to perform:
  Apply all migrations: admin, auth, contenttypes, sessions
Running migrations:
  Applying contenttypes.0001_initial... OK
  Applying auth.0001_initial... OK
  Applying admin.0001_initial... OK
  Applying admin.0002_logentry_remove_auto_add... OK
  Applying admin.0003_logentry_add_action_flag_choices... OK
  Applying contenttypes.0002_remove_content_type_name... OK
  Applying auth.0002_alter_permission_name_max_length... OK
  Applying auth.0003_alter_user_email_max_length... OK
  Applying auth.0004_alter_user_username_opts... OK
  Applying auth.0005_alter_user_last_login_null... OK
  Applying auth.0006_require_contenttypes_0002... OK
  Applying auth.0007_alter_validators_add_error_messages... OK
  Applying auth.0008_alter_user_username_max_length... OK
  Applying auth.0009_alter_user_last_name_max_length... OK
  Applying auth.0010_alter_group_name_max_length... OK
  Applying auth.0011_update_proxy_permissions... OK
  Applying auth.0012_alter_user_first_name_max_length... OK
  Applying sessions.0001_initial... OK
```

#### Interakcija sa modelom

Za interakciju sa modelom koristimo naredbu `./manage.py shell` koja pokreće Python shell.

Za početak uvezimo potrebne modele i pakete naredbama:

``` python
from main.models import Predmet
from django.utils import timezone
```

Za dohvaćanje svih objekata koristimo `Predmet.objects.all()` što vraća `QuerrySet []` ondosno praznu listu.

Dodajmo vrijednosti u novu instancu klase Predmet nazvia `novi_predmet` pomoću:

``` python
novi_predmet = Predmet()
novi_predmet.predmet_naslov = 'DWA2'
novi_predmet.predmet_sadrzaj = 'ovo je opis predmeta'
novi_predmet.predmet_vrijeme_objave = timezone.now()
```

Pohranimo promjene:

``` python
novi_predmet.save()
```

Isprobajmo ponovno naredbu `Predmet.objects.all()` i uočimo novi predmet u listi.

Kroz predmete se može i iterirati, primjerice korištenjem petlje `for`:

``` python
for p in Predmet.objects.all():
    print(p.predmet_naslov)
```

### Kreiranje superusera i povezivanje modela

Administratora kreiramo naredbom:

``` shell
$ ./manage.py createsuperuser
Username (leave blank to use 'korisnik'):
Email address:
Password:
Password (again):
```

Dodjelite proizvoljno ime i lozinku, mail adresu trenutno možemo ostaviti praznu.

!!! zadatak
    Posjetite na lokalnom serveru adresu `http://127.0.0.1:8000/admin/`.

Idemo sada nadopuniti kod i povezati stvoreni model pomoću `mysite/main/admin.py`.

Sadržaj datoteke `mysite/main/admin.py` je oblika:

``` python
from django.contrib import admin
from .models import Predmet

## Register your models here.
admin.site.register(Predmet)
```

!!! zadatak
    Osvježite stranicu na adresi `http://127.0.0.1:8000/admin/`.

## Relacije među modelima. Upiti

### Modeli: Relacije među modelima

Relacije su detaljno opisane u [Djangovoj dokumnetaciji](https://docs.djangoproject.com/en/3.2/topics/db/examples/):

- [Many-to-many](https://docs.djangoproject.com/en/3.2/topics/db/examples/many_to_many/)
- [One-to-one](https://docs.djangoproject.com/en/3.2/topics/db/examples/one_to_one/)
- [Many-to-one](https://docs.djangoproject.com/en/3.2/topics/db/examples/many_to_one/)

#### Many-to-many

Na prethodnim vježbama stvorili smo model koji sadržava klasu Predmet, dodajmo sada klasu Student koja će biti povezana s Predmetom relacijom ManyToMany.

!!! zadatak
    - Model iz Vježbi2 `main/models.py` nadopunite tako da stvorite novu klasu `Student`. Klasa student sadrži stupce naziva `student_ime`, `student_prezime`, `student_broj_xice` i `student_prvi_upis_predmeta`. Tipovi podataka neka budu `CharField()` za `student_ime` sa `max_length` na `25`, `student_prezime` sa `max_length` na `50`. Vrijednost `student_broj_xice` postavite na `CharField()` za `student_ime` sa `max_length` na `10`.
    - Dodajte vrijednost `student_predmeti` koja će biti povezan s klasom `Predmet`, tip veze neka bude `ManyToMany`. Unutar klase `Predmet` izmijenite vrijednost `predmet_vrijeme_objave` tako da joj postavite zadanu vrijednost na `timezone.now`. Nakon kreirane klase pokrenite naredbe `makemigrations` i `migrate`.

**Rješenje zadatka.** Uredit ćemo datoteku `main/models.py` tako da bude oblika:

``` python
from django.db import models
from django.utils import timezone


class Predmet(models.Model):
    predmet_naslov = models.CharField(max_length=100)
    predmet_sadrzaj = models.TextField()
    predmet_vrijeme_objave = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.predmet_naslov


class Student(models.Model):
    student_ime = models.CharField(max_length=25)
    student_prezime = models.CharField(max_length=50)
    student_broj_xice = models.CharField(max_length=10)
    student_predmeti = models.ManyToManyField(Predmet)
```

!!! zadatak
    Definirajte funkciju `__str__()` unutar klase Student koja vraća `student_broj_xice`. Zatim dodajte klasu `Student` unutar `main/admin.py` tako da ona postane vidljiva u admin panelu.

**Rješenje zadatka.** U datoteci `main/models.py` unutar klase `Student` dodajemo:

``` python
def __str__(self):
    return self.student_broj_xice
```

U datoteci `main/admin.py`:

``` python
from django.contrib import admin
from .models import *

model_list = [Predmet, Student]
admin.site.register(model_list)
```

Nakon što ste nadopunili `main/models.py` primjenite pripremu i nakon toga migraciju s naredbama `makemigrations` i `migrate`. Provjerite radi li vam sve tako da posjetite `http://127.0.0.1:8000/admin/`.

#### Many-to-one

!!! zadatak
    Definirajte klasu `Profesor` koja sadrži vrijednosti, `prof_ime` i `prof_prezime` koji su `CharField` duljine 30, zatim definirajte `prof_email` koji je tipa `EmailField`.

Unutar klase zadajte funkciju `__str__()` koja vraća email adresu od profesora.

Nakon kreirane klase pokrenite naredbe `makemigrations` i `migrate`.

**Rješenje zadatka.**  U datoteci `main/models.py`:

``` python
class Profesor(models.Model):
    prof_ime = models.CharField(max_length=30)
    prof_prezime = models.CharField(max_length=30)
    prof_email = models.EmailField()

    def __str__(self):
        return self.prof_email
```

!!! zadatak
    Izmijenite klasu `Predmet` tako da joj dodate nositelja, vrijednost `nositelj` neka bude tip veze One to many. Za definiranje veze koristite `ForeignKey`.

Nakon kreirane klase pokrenite naredbe `makemigrations` i `migrate`.

**Rješenje zadatka.** U datoteci `main/models.py`:

``` python
class Predmet(models.Model):
    predmet_naslov = models.CharField(max_length=100)
    predmet_sadrzaj = models.TextField()
    predmet_vrijeme_objave = models.DateTimeField(default=timezone.now)
    predmet_nositelj = models.ForeignKey(Profesor, on_delete=models.CASCADE)

    def __str__(self):
        return self.predmet_naslov
```

#### One-to-one

Student i Profesor povezat ćemo u klasi ZavrsniRad, na kojem radi student, dok mu je profesor mentor. Svaki završni rad ima Studenta koji ga piše i profesora koji mu je mentor. Ovo ćemo ostvariti vezama *one-to-one*.

!!! zadatak
    - Definirajte klasu `ZavrsniRad`. `ZavrsniRad` neka ima zadanog nositelja `mentor` koji je povezan sa `Profesor` pomoću `OneToOne` veze, dodatni parametar koje zadajete u definiciji veze je `primary_key=True`.
    - Klasu `ZavrsniRad` zatim povežite sa `Student`, tip veze neka bude `One-to-one`, dodatni parametri koje zadajete su `on_delete=models.CASCADE` i `primary_key=True`.
    - Zatim dodajte vrijednosti `rad_naslov` i `rad_zadatak` koji su CharField duljine 25 i 50 i bool vrijednost `rad_prvi_upis` koja po defaultu ima vrijednost `True`.
    - Nakon kreirane klase pokrenite naredbe `makemigrations` i `migrate`.

**Rješenje zadatka.** U datoteci `main/models.py`:

``` python
class ZavrsniRad(models.Model):
    mentor = models.OneToOneField(
        Profesor,
        on_delete=models.CASCADE,
    )

    student = models.OneToOneField(
        Student,
        on_delete=models.CASCADE,
        primary_key=True,
    )

    rad_naslov = models.CharField(max_length=50)
    rad_zadatak = models.CharField(max_length=50)
    rad_prvi_upis = models.BooleanField(default=True)
```

!!! zadatak
    Definirajte funkciju `__str__()` unutar klase `ZavrsniRad` koja vraća `student_broj_xice`. Zatim dodajte klasu `ZavrsniRad` unutar `main/admin.py` tako da ona postane vidljiva u admin panelu.

**Rješenje zadatka.** Unutar klase `ZavrsniRad` u `main/models.py` dopunjavamo:

``` python
def __str__(self):
    return 'Završni rad studenta s brojem X-ice {}'.format(self.student.student_broj_xice)
```

U `main/admin.py` samo dopunjavamo `model_list` na način:

``` python
model_list = [Predmet, Student, ZavrsniRad]
```

Korištenje naredbe `./manage.py`:

- Nakon svake novo stvorene klase u modelu pokrenite naredbe `./manage.py makemigrations` i `./manage.py migrate`.
- Naredbu `./manage.py flush` koristite za očistiti bazu padatak od prethodno unešenih vrijednosti.
- Naredba `./manage.py dbshell` omogućuje unos SQL naredbi, primjerice `ALTER TABLE main_predmet ADD COLUMN "predmet_nositelj_id" integer;`.

#### Upiti

Naredba za pokretanje Pythonove ljuske specifične za Django:

``` shell
$ ./manage.py shell
>>>
```

Za definiranje instance klase:

``` python
>>> profesor = Profesor()
>>> predmet = Predmet()
```

Povezivanje instanci `Profesor` i `Predmet` pomoću vanjskog ključa, odnosno dodavanje nositelja na predmet:

``` python
>>> predmet.predmet_nositelj = profesor
```

Instanci klase `Student` možemo dodati *n* predmeta:

``` python
>>> student.student_predmeti.add(predmet1, predmet2)
```

Kreiranje instance klase `ZavrsniRad`:

``` python
>>> zr = ZavrsniRad()
```

Dodavanje veze tipa *one-to-one*, na instancu klase `ZavrsniRad`:

``` python
>>> zr.mentor=profesor
>>> zr.student=student
```

Upit za koji vraća sve instance tražene klase:

``` python
profesors = Profesor.objects.all()
```

Pretraga po zadanoj vrijednosti:

``` python
email = Profesor.objects.get(prof_email='prof_mail@uniri.hr')
```

Pretraga svih instanci koji imaju traženo ime:

``` python
prof_peros = Profesor.objects.filter(prof_ime__contains='Pero')
```

Uzimanje prvih 5 zapisa:

``` python
Profesor.objects.all()[:5]
```

Sortiranje i dohvaćanje prvog u listi:

``` python
Profesor.objects.order_by('prof_ime')[0]
```

### Cjelovit kod današnjih vježbi

``` python
from django.db import models
from django.utils import timezone

## Create your models here.


class Profesor(models.Model):
    prof_ime = models.CharField(max_length=30)
    prof_prezime = models.CharField(max_length=30)
    prof_email = models.EmailField()

    def __str__(self):
        return self.prof_email


class Predmet(models.Model):
    predmet_naslov = models.CharField(max_length=100)
    predmet_sadrzaj = models.TextField()
    predmet_vrijeme_objave = models.DateTimeField(default=timezone.now)
    predmet_nositelj = models.ForeignKey(Profesor, default=1, on_delete=models.CASCADE)

    def __str__(self):
        return self.predmet_naslov


class Student(models.Model):
    student_ime = models.CharField(max_length=25)
    student_prezime = models.CharField(max_length=50)
    student_broj_xice = models.CharField(max_length=10)
    student_predmeti = models.ManyToManyField(Predmet)

    def __str__(self):
        return self.student_broj_xice


class ZavrsniRad(models.Model):
    mentor = models.OneToOneField(
        Profesor,
        on_delete=models.CASCADE,
    )
    student = models.OneToOneField(
        Student,
        on_delete=models.CASCADE,
        primary_key=True
    )
    rad_naslov = models.CharField(max_length=50)
    rad_zadatak = models.CharField(max_length=150)
    rad_prvi_upis = models.BooleanField(default=True)

    def __str__(self):
        return 'Završni rad studenta s brojem X-ice {}'.format(self.student.student_broj_xice)
```

## Usmjeravanje i URL-i. Stvaranje pogleda kao odgovora na HTTP zahtjeve

Na današnjim vježbama čeka nas gradivo vezano za usmjeravanje pomoću URL-ova. Zatim ćemo vidjeti par primjera kako se izgleda odgovor na poslani HTTP zahtjev.

### Usmjeravanje pomoću `urls.py`

Potrebno je prvo stvoriti novi projekt i unutar njega aplikaciju koju ćemo povezati.

Za kreiranje projekta koristi se naredba:

``` shell
$ django-admin startproject <project_name>
(...)
```

Za kreiranje aplikacije unutar projekta koristi se naredba:

``` shell
$ django-admin startapp <app_name>
(...)
```

Nakon što su projekt i aplikacija unutar njega kreirani, potrebno ih je povezati. Ovo se radi unutar datoteka `urls.py` koja se nalazi u projektnom direktoriju.

Na ovim vježbama kreirati će se projekt naziva `vj4` unutar kojeg je stvorena aplikacija naziva `main`.

!!! zadatak
    Povežite kreiranu aplikaciju `main` s glavnim djelom aplikacije unutar `main/urls.py`.

**Rješenje zadatka.** Datoteka `urls.py`:

``` python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('main/', include('main.urls')),
]
```

!!! warning
    Potrebno je povezati novostvorenu aplikaciju `main` i unutar `setting.py`. Unutar`INSTALLED_APPS` potrebno je dodati `MainConfig` iz `apps.py` koja se nalazi unutar aplikacije `main`.

Program smo usmjerili na `main/urls.py` koji trenutno ne postoji. Iz toga razloga, potrebno ga je stvoriti.

!!! zadatak
    Stvorite datoteku `main/urls.py`. Odmah importajte sve iz datoteke `main/views.py` i neka ime aplikacije bude zadano na `app_name = 'main'`.
    Zatim definirajte uzorak URL-a neka upućuje na `homepage`, odnosno na funkciju unutar `main/views.py` koja se zove `homepage`.

**Rješenje zadatka.** Datoteka `main/urls.py`:

``` python
from django.urls import path
from . import views

app_name = 'main'

urlpatterns = [
    path('homepage', views.homepage, name='homepage'),
]
```

Definirali smo poveznice unutar datoteka `main/urls.py`. Sada je potrebno kreirati funkciju `homepage()` unutar `main/views.py` koju smo pozvali unutar `main/urls.py`.

### Slanje zahtjeva

!!! zadatak
    Definirajte funkciju `homepage()` unutar `main/views.py` koja će vraćati HTTP odgovor na zahtjev. Za vraćanje HTTP odgovora koristite funkciju `HttpResponse` koju uvozite kodom `from django.http import HttpResponse`.

**Rješenje zadatka.** U datoteci `main/views.py`:

``` python
from django.shortcuts import render
from django.http import HttpResponse
## Create your views here.

def homepage(request):
    return HttpResponse('<html><body><strong>Homepage</strong> i još neki tekst na homepage.</body></html>')
```

Pohranite sve promjene i pokrenite server.

!!! zadatak
    Definirajte funkciju `current_datetime()` unutar datoteke `main/views.py` koja će vraćati HTTP odgovor na zahtjev. Neka vrijednost koju funkcija vraća budu datum i trenutno vrijeme.

**Rješenje zadatka.** U datoteci `main/views.py`:

``` python
from django.shortcuts import render
from django.http import HttpResponse
import datetime
## Create your views here.

def current_datetime(request):
    now = datetime.datetime.now()
    html = '<html><body>Trenutno vrijeme: {}.</body></html>'.format(now)
    return HttpResponse(html)
```

#### Vraćajne grešaka u odgovorima na zahtjeve

!!! zadatak
    Definirajte funkciju `not_found()` unutar `main/views.py`. Funkcija neka vraća `HttpResponseNotFound`. Vratite proizvoljni sadržaj odgovora.

**Rješenje zadatka.** U datoteci `main/views.py`:

``` python
from django.http import HttpResponse, HttpResponseNotFound

def not_found(request):
    return HttpResponseNotFound('<h1>Page not found</h1>')
```

### Vraćanje zapisa iz baze

U nastavku je prikazano kako se mogu dohvaćati vrijednosti iz baze podataka i kako ih možemo prikazivati na stranici.

!!! zadatak
    Kreirajte klasu `Student`, neka sadrži, ime prezime i broj xice kao atribute. Dodajte ju zatim unutar `admin.py` da bi se mogle unositi vrijednosti. Za kraj pokrenite naredbe za migraciju da se kreira baza.

**Rješenje zadatka.** U datoteci `models.py`:

``` python
class Student(models.Model):
    ime = models.CharField(max_length=25)
    prezime = models.CharField(max_length=50)
    broj_xice = models.CharField(max_length=10)

    def __str__(self):
        return str(self.broj_xice)
```

U datoteci `admin.py`:

``` python
from django.contrib import admin
from main.models import *

## Register your models here.
admin.site.register(Student)
```

Nakon što je baza pomoću modela kreirana, potrebno je unijeti u nju vrijednosti da se može izvršiti tražene upite.

!!! zadatak
    Kreirajte administratora i dodajte u bazu podataka 5 studenata. Od 5 studenata, 3 neka imaju isto ime, primjerice: Marko, Marko, Marko, Ivan i Ana. Prezime i broj X-ice zadajte proizvoljno.

Kada smo popunili bazu, idemo kreirati i upite.

!!! zadatak
    Definirajte funkciju koja u bazi pronalazi sve studente zadanog imena, listu pronađenih imena proslijedite funkciji `render`. Proslijeđena rješenja neka se prikazuju unutar `students.html`. Za početak, neka vaša funkcija vraća `render(request, 'students.html', context=context)`, a datoteku `students.html` ćemo stvoriti u nastavku rješavanja ovoga zadatka.

U datoteci `views.py`:

``` python
def all_peros(request):
    peros = Student.objects.filter(ime__contains='pero')

    context = {'peros': peros}

    return render(request, 'students.html', context=context)
```

Za prikaz rješenja prethodnog zadatka, potrebna nam je HTML datoteka, koji će prikazati rezultate upita nad bazom. Kreirajte unutar direktorija `main` datoteku `main/templates`, unutar koje pohranjujete `students.html`.

Datoteka `students.html` ima sadržaj:

``` html
<ul>
    {% for p in peros %}
        <li>
            Ime: {{ p.ime }}<br>
            Prezime: {{ p.prezime }}<br>
            Broj xice:{{ p.broj_xice }}
        </li>
    {% endfor %}
</ul>
```

!!! zadatak
    Dodajte unutar datoteke `main/urls.py` putanju koja nas vodi na prethodno kreiranu funkciju i zatim provjerite prikaz rezultata na serveru.

Idemo još prikazati ukupan broj studenata u našoj bazi. Ovaj broj ćemo zatim proslijediti funkciji `render()` koja će ispisati ukupan broj studenata u za to kreiranom HTML-u.

!!! zadatak
    Definirajte funkciju koja u bazi pronalazi ukupan broj studenata, broj studenata funkciji `render`. Proslijeđena rješenja neka se prikazuju unutar datoteke `index.html`. Dodajte grešku u slučaju da `Student` u bazi nije pronađen.

**Rješenje zadatka.** U datoteci `views.py`:

``` python
def detail(request):
    try:
        num_students = Student.objects.all().count()

        context = {'num_students': num_students}

    except Student.DoesNotExist:
        raise Http404('Student does not exist')

    return render(request, 'detail.html', context=context)
```

``` html
{% block content %}
<h1>Dobrodosli na UNIRI</h1>

<p>Na faxu je upisano:</p>
<ul>
    <li><strong>Studenata:</strong> {{ num_students }}</li>
</ul>
{% endblock %}
```

## Izrada generičkih pogleda

Na današnjim vježbama radit će se generički pregledi.

### Priprema i postavljanje projekta

Prije početka rada potrebno je kreirati novi Django projekt `vj5` unutar kojeg kreirate aplikaciju  `main`.

Povežite projekt i aplikaciju:

- Dodati `main` aplikaciju pod `INSTALLED_APPS` unutar `vj5/settings.py`.
- Unutar `vj5/urls.py` dodati usmjeravanje na `main/urls.py`, `main/urls.py` još nije stvoren, stoga ga je potrebno kreirati.

Datoteka `vj5/main/urls.py` je oblika:

``` python
from django.urls import path

urlpatterns = [
]
```

Za potrebe ovih vježbi koristit će se gotov model koji je zadan u nastavku.

Datoteka `vj5/main/models.py` je oblika:

``` python
from django.db import models

## Create your models here.

class Publisher(models.Model):
    name = models.CharField(max_length=30)
    address = models.CharField(max_length=50)
    city = models.CharField(max_length=60)
    state_province = models.CharField(max_length=30)
    country = models.CharField(max_length=50)
    website = models.URLField()

    class Meta:
        ordering = ['-name']

    def __str__(self):
        return self.name

class Author(models.Model):
    salutation = models.CharField(max_length=10)
    name = models.CharField(max_length=200)
    email = models.EmailField()
    headshot = models.ImageField(upload_to='author_headshots')

    def __str__(self):
        return self.name

class Book(models.Model):
    title = models.CharField(max_length=100)
    authors = models.ManyToManyField('Author')
    publisher = models.ForeignKey(Publisher, on_delete=models.CASCADE)
    publication_date = models.DateField()
```

Nakon što je model kreiran unutar `vj5/main/models.py` potrebno je provesti migraciju. Naredbe za migraciju su:

``` shell
$ ./manage.py makemigrations
(...)
$ ./manage.py migrate
(...)
```

Napravite migraciju i zatim pokrenite server.

### Generički pogledi

Kreirajte prvi generički pogled nad stvorenim modelom.

Datoteka `vj5/main/views.py` ima sadržaj:

``` python
from django.views.generic import ListView
from main.models import Publisher

class PublisherList(ListView):
    model = Publisher
```

A zatim ga povežite unutar `main/urls.py` na način:

``` python
from django.urls import path
from main.views import PublisherList

urlpatterns = [
    path('publishers/', PublisherList.as_view()),
]
```

Kada smo kreirali pogled i pozvali ga unutar `urls.py` potreban nam je predložak unutar kojeg će se prikazati odgovor.

Sve predloške koje ćemo koristiti organizirat ćemo tako da se nalaze u zajedničkom direktoriju `templates`, koji se nalazi u korijenskom direktoriju.

Kreirajte direktorij `./templates`, unutar kojeg kreirate direktorij `main`, dakle `./templates/main`, a unutar njega kreirajte datoteku `publisher_list.html`.

Datoteka `./templates/main/publisher_list.html` ima sadržaj:

``` html
{% block content %}
    <h2>Publishers</h2>
    <ul>
        {% for publisher in object_list %}
            <li>
                Name: {{ publisher.name }}<br>
                City: {{ publisher.city }}
            </li>
        {% endfor %}
    </ul>
{% endblock %}
```

Potrebno je još zadati putanju za predloške unutar `settings.py`.

Za dodavanje putanje, pod `TEMPLATES` dodajte putanju do `templates` direktorija (`./templates`), odnosno `'DIRS': ['./templates'],`.

!!! zadatak
    Kreirajte administratora i dodajte u bazu podataka 3 izdavača. Sve vrijednosti proizvoljno zadajte. Provjerite ispis izdavača koji su dodani u bazu na adresi `http://127.0.0.1/main/publishers/`.

#### Dinamičko filtriranje

U nastavku je prikazan način na koji se omogućava dinamička pretraga pomoću URL-a. Za zadani naziv izdavača vraćat će se sve knjige koje je taj izdavač objavio. U zadanom URL uzorku u aplikaciji neće statično biti definirati naziv, nego će se on dinamično generirati.

Za početak potrebno je definirati prikaz unutar `./main/views.py` koji će vraćati sve knjige od zadanog izdavača.

Datoteka `vj5/main/views.py`:

``` python
from django.shortcuts import get_object_or_404
from django.views.generic import ListView
from main.models import Book, Publisher

class PublisherBookList(ListView):
    template_name = 'main/books_by_publisher.html'

    def get_queryset(self):
        self.publisher = get_object_or_404(Publisher, name=self.kwargs['publisher'])
        return Book.objects.filter(publisher=self.publisher)
```

Zatim unutar `./main/urls.py` povezujemo s traženim pogledom. U ovom slučaju ne koristi se statično zadani uzorak Umjesto da svakog pojedinog izdavača zadajemo pojedinačno, koristimo `<publisher>`.

Datoteka `vj5/main/urls.py`:

``` python
from django.urls import path
from main.views import PublisherList, PublisherBookList

urlpatterns = [
    path('publishers/', PublisherList.as_view()),
    path('<publisher>/', PublisherBookList.as_view()),
]
```

I za zadnji dio potrebno je kreirati prikaz unutar `./templates` koji će nam prikazivati rezultate pretrage za zadanog izdavača.

!!! zadatak
    Kreirajte `books_by_publisher.html` unutar `./templates/main` koji će ispisati sve knjige od traženog izdavača. Neka se ispisuje samo naslov svake knjige.

**Rješenje zadatka.**

``` html
{% block content %}
    <h2>Books list: </h2>
    <ul>
        {% for book in object_list %}
            <li>Book title: {{ book.title }}</li>
        {% endfor %}
    </ul>
{% endblock %}
```

Pokrenite server i provjerite pretraživanje po izdavaču.

## Predaja obrazaca HTTP metodama GET i POST. Provjera unosa i prikaz poruka o greškama

Na današnjim vježbama radit će se generičko popunjavanje baze i obrasci.

### Postavljanje projekta

!!! zadatak
    Kreirajte projekt naziva `vj6` i unutar njega aplikaciju naziva `main`. Povežite aplikaciju sa projektom: dodajte aplikaciju unutar `settings.py` i putanju `main/urls.py` unutar `urls.py`, a zatim kreirajte `main/urls.py`.

### Generičko popunjavanje baze podataka

Model koji se koristi sadrži dvije klase, `Author` i `Book`.

Sadržaj datoteke `vj6/main/models.py`:

``` python
from django.db import models

## Create your models here.


class Author(models.Model):
    name = models.CharField(max_length=30)
    address = models.CharField(max_length=50)
    city = models.CharField(max_length=60)
    country = models.CharField(max_length=50)

    def __str__(self):
        return self.name


class Book(models.Model):
    title = models.CharField(max_length=100)
    abstract = models.TextField()
    author = models.ForeignKey(Author, on_delete=models.CASCADE)
    publication_date = models.DateField()

    def __str__(self):
        return self.title
```

Kreirani model potrebno je popuniti podacima, za to će se koristiti naredba `./manage.py setup_test_data.py`. Prilikom pokretanja naredbe, program vraća grešku jer naredba još nije kreirana.

Instalacija potrebnih Python paketa:

``` shell
$ pip3 install factory_boy
(...)
```

Kada je instaliran [paket factory_boy](https://pypi.org/project/factory-boy/), potrebno je kreirati klase koje će automatski popunjavati bazu sa tzv. *dummy data*, odnostno nasumično generiranim podacima koji će nam pojednostaviti proces popunjavanja baze nad kojom želimo izvršavati upite. Detaljnije o njegovoj funkcionalnosti možete pronaći u [službenoj dokumentaciji](https://factoryboy.readthedocs.io/).

!!! zadatak
    Unutar `vj6/main` stvorite datoteku `factory.py`

Stvorena datoteka `vj6/main/factory.py` koristit će se kao predložak za popunjavanje modela definiranog unutar`vj6/main/models.py`. Primjetit ćete sličnost u stilu pisanja klasa. Dakle, potrebno je definirati klase, sukladno klasama koje su definirane unutar `vj6/main/models.py`.

Datoteka  `vj6/main/factory.py`:

``` python
## factories.py
import factory
from factory.django import DjangoModelFactory

from main.models import *

## Defining a factory
class AuthorFactory(DjangoModelFactory):
    class Meta:
        model = Author

    name = factory.Faker("first_name")
    address = factory.Faker("address")
    city = factory.Faker("city")
    country = factory.Faker("country")


class BookFactory(DjangoModelFactory):
    class Meta:
        model = Book

    title = factory.Faker("sentence", nb_words=4)
    abstract = factory.Faker("sentence", nb_words=50)
    author = factory.Iterator(Author.objects.all())
    publication_date = factory.Faker("date_time")
```

!!! zadatak
    Nakon što su klase definirane unutar `factory.py`, isprobajte njihovu funkcionalnost. Prije pokretanja ljuske primjenite migraciju na bazu.

``` shell
$ ./manage.py shell
(...)
```

``` python
>>> from main.factories import *
```

``` python
>>> a = AuthorFactory()
>>> b = BookFactory()
>>> a
>>> b.title
>>> b.author
```

!!! zadatak
    Kreirajte administratnora, zatim unutar `admin.py` registrirajte modele `Book` i `Author`. Provjerite ako su podaci generirani sa `factory.py` uneseni u bazu.

**Rješenje zadatka.** U datoteci `admin.py`:

``` python
from django.contrib import admin

from main.models import *

models_list = [Author, Book]

## Register your models here.
admin.site.register(models_list)
```

#### Kreiranje naredbe u `manage.py`

Kada je kreiran i testiran `factory.py`, slijedi kreiranje naredbe koja će se prosljeđivati `./manage.py`.

Za početak porenite naredbu:

``` shell
$ ./manage.py
(...)
```

Izlistao nam se trenutni popis opcija koje možemo izvršavati.

Kreirajte direktorij `commands`, unutar kojeg će se nalaziti skripta. Zatim se pozicionirajte u njega.

``` shell
$ mkdir main/management/commands
$ cd main/management/commands
(...)
```

A zatim, unutar direktorija `commands` kreirajte `setup_test_data.py`.

``` shell
$ touch setup_test_data.py
(...)
```

Otvorite kreirani `setup_test_data.py` unutar kojeg će se kreirati vlastita upravljačka naredba ([detaljnije o upravljačkim naredbama koje su kreirane od strane korisnika](https://simpleisbetterthancomplex.com/tutorial/2018/08/27/how-to-create-custom-django-management-commands.html)).

Sadržaj datoteke `main/management/commands/setup_test_data.py`:

``` python
import random

from django.db import transaction
from django.core.management.base import BaseCommand

from main.models import Author, Book
from main.factories import (
    AuthorFactory,
    BookFactory
)

NUM_AUTHORS = 10
NUM_BOOKS = 100

class Command(BaseCommand):
    help = "Generates test data"

    @transaction.atomic
    def handle(self, *args, **kwargs):
        self.stdout.write("Deleting old data...")
        models = [Author, Book]
        for m in models:
            m.objects.all().delete()

        self.stdout.write("Creating new data...")

        for _ in range(NUM_AUTHORS):
            author = AuthorFactory()

        for _ in range(NUM_BOOKS):
            book = BookFactory()
```

Svojstvo funkcije `handle()` je postavljeno na  `transaction.atomic`, što označava da ako je blok koda uspješno izvršen, promjene se pohranjuju u bazu podataka ([detaljnije objašnjenje o modulu transaction korištenom u prethodnom primjeru](https://docs.djangoproject.com/en/3.2/topics/db/transactions/)).

!!! zadatak
    Isprobajte funkcionalnost kreirane naredbe, a zatim provjerite ako su uneseni podaci unutar admin sučelja.

``` shell
$ ./manage.py setup_test_data
(...)
```

## Predlošci obrazaca. Stvaranje obrazaca iz modela

### Kreiranje predložaka

Nakon što je baza kreirana i popunjena teestim podacima, sljedeći korak je definiranje predložaka pomoću kojih će se ti podaci prikazivati.

#### Sintaksa

##### Varijable

Pomoću varijabli ispisujemo sadržaj koji je prosljeđen iz konteksta. Objekt je sličan Python riječniku (engl. *dictionary*) gdje je sadržaj mapiran u odnosu *key-value*.

Varijable pišemo unutar vitičastih zagrada `{{ varijabla }}`:

``` html
<p>Ime: {{ first_name }}, Prezime: {{ last_name }}.</p>
```

Što će iz konteksta `{'first_name': 'Ivo', 'last_name': 'Ivić'}` biti prikazano kao:

``` html
<p>Ime: Ivo, Prezime: Ivić.</p>
```

##### Oznake

Oznake se pišu unutar `{% oznaka %}`. Označavaju proizvoljnu logiku unutar prikaza. Oznaka može biti ispis sadržaja, ili logička cjelina ili pak pristup drugim oznakama iz predloška, Primjerice: `{% tag %} ... sadržaj ... {% endtag %}` ili `{% block title %}{{ object.title }}{% endblock %}`.

##### Filteri

Filtere koristimo da bi transformirali vrijednosti varijabli. Neki od primjera korištenja filtera mogu biti za:

- Pretvaranje u mala slova: `{{ name|lower }}`
- Uzimanje prvih 30 riječi: `{{ bio|truncatewords:30 }}`
- Dohvaćanje duljine varijable: `{{ value|length }}`

##### Komentari

Za komentiranje dijelova koristimo `#` što unutar predloška izgleda: `{# Ovo neće biti prikazano #}`.

##### Petlje

Petlje se pišu unutar `{% %}` odnosno definiraju se njihove oznake. Primjer korištenja petlje `for`:

``` html
<ul>
{% for author in author_list %}
    <li>{{ author.name }}</li>
{% endfor %}
</ul>
```

Primjer provjere autentifikacije korisnika naredbom `if`:

``` html
{% if user.is_authenticated %}
    <p>Hello, {{ user.username }}.</p>
{% endif %}
```

Primjer naredbi `if` i `else`:

``` html
{% if author_list %}
    Number of athletes: {{ athlete_list|length }}
{% else %}
    No authors.
{% endif %}
```

!!! zadatak
    Kreirajte direktorij `./templates` unutar kojeg će biti pohranjeni predlošci koji se koriste, npr. `base_generic.html`. Ne zaboravite definirati putanju unutar `settings.py`.

**Rješenje zadatka.** Datoteka `./templates/base_generic.html`:

``` html
<!DOCTYPE html>
<html lang="en">

<head>
</head>

<body>
    <h1>My library</h1>
    <div id="sidebar">
        {% block sidebar %}
        <ul>
            <li><a href="/main">Home</a></li>
            <li><a href="/main/books">Books</a></li>
            <li><a href="/main/authors">Authors</a></li>
        </ul>
        {% endblock %}
    </div>

    <div class="content">
        {% block content %}
        {% endblock %}
    </div>
</body>
</html>
```

#### Nasljeđivanje u predlošcima

Datoteka `./template/main/book_list.html`:

``` html
{% extends "base_generic.html" %}
{% block content %}
<br>
<h2> Books </h2>
<br>
{% for book in book_list %}
    <div class="book">
        <h4>{{ book.title }}</h4>
        Author: {{book.author}}
        <p>{{ book.abstract }}</p>
    </div>
{% endfor %}
{% endblock %}
```

Unutar `main/urls.py` dodajte:

``` python
path('books', BookList.as_view())
```

Datoteka `main/views.py`:

``` python
from django.shortcuts import get_object_or_404
from django.views.generic import ListView
from main.models import Author, Book

class BookList(ListView):
    model = Book
```

!!! zadatak
    Sukladno prethodnom primjeru kreirajte prikaz za sve autore unutar baze.

### CSS

Dodavanje CSS-a u HTML predložak.

#### Uvoz

``` html
<link rel="stylesheet" href="https://www.w3schools.com/html/styles.css">
```

``` html
<head>
    <link rel="stylesheet" href="https://www.w3schools.com/html/styles.css">
    <title>{% block title %}Knjižnica{% endblock %}</title>
</head>
```

#### Direktorij za statičke datoteke

Unutar aplikacije `main` potrebno je stvoriti direktorij `static` a unutar njega `style.css`.

Referenciranje na `style.css` unutar aplikacije:

``` html
{% load static %}
<link rel="stylesheet" type="text/css" href="{% static 'style.css' %}">
```

Prikaz unutar HTML templatea:

``` html
<head>
    {% load static %}
    <link rel="stylesheet" type="text/css" href="{% static 'style.css' %}">
    <title>{% block title %}Knjiznica{% endblock %}</title>
</head>
```

Sadržaj datoteke `style.css` možete proizvoljno zadati i prilagođavati vlastitim željama. Primjerice, datoteka `main/static/style.css` može biti oblika:

``` css
h1 {
    color: blue;
    text-align: center;
}

h2 {
    text-align: center;
}

li {
    list-style-type: none;
    float: left;
}

li a {
    padding: 16px;
}
```

!!! zadatak
    Dopunite prikaz na autora tako da njegovo ime bude link. Link neka vodi na prikaz svih knjiga od odabranog autora. Za tu svrhu prvo je potrebno kreirati pretraživanje po traženom autoru, a zatim vratiti sve knjige koje su pronađene za traženog autora.

**Rješenje zadatka.** Uredit ćemo datoteku `main/views.py` da bude oblika:

``` python
from django.shortcuts import get_object_or_404
from django.views.generic import ListView
from main.models import Author, Book

class AuthorBookList(ListView):
    template_name = 'main/book_list.html'

    def get_queryset(self):
        self.author = get_object_or_404(Author, name=self.kwargs['author'])
        return Book.objects.filter(author=self.author)
```

Potrebno je dodati putanju unutar `views.py`. Datoteka `main/views.py` je oblika:

``` python
path('<author>', AuthorBookList.as_view())
```

Zatim je potrebno izmjeniti predložak, odnosno dodati linkove koji vode na autore. Datoteka `author_list.html`:

``` html
{% extends "base_generic.html" %}
{% block content %}
<h2>Books</h2>
{% for book in book_list %}
    <div class="book">
        <h4>{{ book.title }}</h4>
        <p>Author: <a href="{{book.author}}"> {{book.author}}</a></p>
        <p>{{ book.abstract }}</p>
    </div>
{% endfor %}
{% endblock %}
```

Izmjenimo i predložak za prikaz knjiga. Datoteka `book_list.html`:

``` html
{% extends "base_generic.html" %}
{% block content %}
<h2>Authors</h2>
{% for author in author_list %}
    <div class="author">
        <h4>{{author.name}}</h4>
        <p>City: {{author.city}}</p>
        <p>Country: {{author.country}}</p>
        <p><a href="{{author.name}}"> All books by {{author.name}}</a></p>
    </div>
{% endfor %}
{% endblock %}
```

!!! zadatak
    Dopunite `style.css` tako da dodate obrub na elemente Knjige i Autora.

**Rješenje zadatka.** U datoteku `style.css` dodajemo:

``` css
.book {
    border-color: cyan;
    border-style: double;
    padding-left: 20px;
    padding-right: 20px;
    padding-bottom: 20px;
}

.author {
    border-color: olive;
    border-style: double;
    padding-left: 20px;
    padding-right: 20px;
    padding-bottom: 20px;
}
```

## Sijanje i migracije

### Sijanje (seeding)

!!! zadatak
    - Kreirajte projekt `vj9` i unutar njega aplikaciju `main`. Unutar modela u aplikaciji `main` kreirajte klasu `Student`. Klasa `Student` neka sadrži vrijednost `first_name`.
    - Provedite potrebne naredbe za migraciju.
    - Pokrenite `./manage.py shell` i kreirajte jednog studenta.

Naredbom `dumpdata` izvozimo vrijednosti iz baze ([detaljnije o naredbi dumpdata](https://docs.djangoproject.com/en/3.2/ref/django-admin/#dumpdata)). Pokrenimo je na način:

``` shell
$ ./manage.py dumpdata main.Student --pk 1 --indent 4 > 0001_student.json
(...)
```

!!! zadatak
    Izbrišite iz baze zapis studenta kojeg ste prethodno unjeli.

**Rješenje zadatka.**

``` python
>>> Student.objects.filter(pk=1).delete()
```

Za uvoz podataka u bazu koristimo naredbu `loaddata`. Detaljnije o naredbi [loaddata](https://docs.djangoproject.com/en/3.2/ref/django-admin/#loaddata).

!!! zadatak
    Uvezite prethodno kreirani `0001_student.json` u bazu.

#### Fixture

Fixture je zbirka podataka koje Django uvozi u bazu podataka. Najjednostavniji način rada sa podacima je pomoću naredbi `dumpdata` i `loaddata`.

#### Paket django-seed

``` shell
$ pip3 install django-seed
(...)
```

Python modul pomoću kojeg se mogu generirati podaci za bazu podataka. U pozadini koristi biblioteku [faker](https://github.com/joke2k/faker) ([dokumentacija](https://faker.readthedocs.io/)) za generiranje testnih podataka. Detaljnije o django-seed možete pronaći u [dokumentaciji](https://github.com/mstdokumaci/django-seed).

#### Brisanje podataka sijanja

U nastavku je generirana skripta `revert_seed.py` pomoću koje brišemo vrijednosti iz baze koje smo prethodno stvorili i unosili sijanjem.

``` python
import json
import glob

g = globals()
has_access = {}
fixtures = glob.glob("*.json")
fixtures.sort(reverse=True)

def get_access(model):
    import importlib

    mod = importlib.import_module(model)
    names = getattr(mod, '__all__', [n for n in dir(mod) if not n.startswith('_')])

    global g
    for name in names:
        g[name.lower()] = {
            'var': getattr(mod, name),
            'name': name
        }

for fixture in fixtures:
    msg = 'Reverting '+fixture+'\n'
    with open(fixture) as json_file:
        datas = json.load(json_file)
        for data in datas:
            app_name = data['model'].split('.')[0]
            class_name = data['model'].split('.')[1]

            if app_name not in has_access.keys():
                get_access(app_name+'.models')
                has_access[app_name] = True

            class_model = g[class_name]['var']
            class_model_name = g[class_name]['name']
            pk = data['pk']

            msg += '{}(pk={}): '.format(class_model_name, pk)
            try:
                class_model.objects.get(pk=pk).delete()
                msg += 'deleted\n'
            except:
                msg += 'not deleted\n'
    print(msg)
```

Skriptu pokrenite naredbom:

``` shell
$ manage.py shell < revert_seed.py
(...)
```

### Učitavanje podataka sijanja u testiranje

Testirajmo rad tako da dohvatimo podatke o studentu koji ima primarni ključ 1 i čije ime je Ivo.

``` python
## test1.py
from django.test import TestCase
from main.models import Student

class MyTest(TestCase):
    # fixtures = ["0001_student.json"]

    def test_should_create_group(self):
        s = Student.objects.get(pk=1)
        self.assertEqual(s.first_name, 'Ivo')
```

Kreirani test pokrenite u terminalu s naredbom:

``` shell
$ ./manage.py test test1
(...)
```

``` python
>>> from django.contrib.auth.models import User, Group
>>> Group.objects.create(name='usergroup')
>>> usergroup = Group.objects.get(name='usergroup')
>>> ivo = User.objects.create_user('Ivo')
>>> ivo.pk
>>> ivo.groups.add(usergroup)
```

``` python
>>> python manage.py dumpdata auth.User --pk 1 --indent 4
>>> python manage.py dumpdata auth.User --pk 1 --indent 4 --natural-foreign
```

## Django vježba 10: Autentifikacija. Autorizacija

### Stvaranje projekta

**Priprema za rad**: stvorite projekt naziva `vj10`, unutar njega aplikaciju naziva `main`. Provedite migraciju. Zatim kreirajte administratora, za stvaranje korisnika sa administratorskim ovlastima koristite naredbu `./manage.py createsuperuser`.

#### Povezivanje projekta i aplikacije

Datoteka `vj10/urls.py`:

``` python
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('main.urls')),
    path('accounts/', include('django.contrib.auth.urls'))
]
```

#### Homepage

Datoteka `main/urls.py`:

``` python
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
]
```

Stvaranje pogleda za `index`:

``` python
def index(request):
    return render(request, 'main/index.html')
```

Unutar aplikacije `main` stvorite si direktorij `templates`, unutar kojeg kreirate `index.html`. HTML predložak:

``` html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title></title>
</head>
<body>

</body>
</html>
```

### Kreiranje korisnika

Posjetite `http://127.0.0.1:8000/accounts/` i `http://127.0.0.1:8000/accounts/login/`. Prilikom posjete `/accounts/login/` javila se greška `TemplateDoesNotExist at /accounts/login/`, gdje možemo vidjeti iz poruke `Exception Value: registration/login.html` da Django ne može pronaći traženi predložak.

!!! zadatak
    Unutar `templates/registration` stvorite `login.html`.

#### Login

``` html
{% if form.errors %}
    <h3>Unos nije ispravan.</h3>
{% endif %}
```

``` html
{% if next %}
    {% if user.is_authenticated %}
        <p>Your account doesn't have access to this page. To proceed,
        please login with an account that has access.</p>
    {% else %}
        <p>Please login to see this page.</p>
    {% endif %}
{% endif %}
```

``` html
<form method="post" action="{% url 'login' %}">
    {% csrf_token %}
    <table>
      <tr>
        <td>{{ form.username.label_tag }}</td>
        <td>{{ form.username }}</td>
      </tr>
      <tr>
        <td>{{ form.password.label_tag }}</td>
        <td>{{ form.password }}</td>
      </tr>
    </table>
    <input type="submit" value="login" />
    <input type="hidden" name="next" value="{{ next }}" />
  </form>
```

Detaljnije o [CSRF tokenu](https://docs.djangoproject.com/en/3.2/ref/csrf/)

Postavljanje lokacije gdje želimo da korisnik bude usmjeren nakon uspješnog logina radimo unutar `settings.py`, tako da dodamo npr. `LOGIN_REDIRECT_URL = '/'` za usmjeravanje na `index.html`.

#### Registracija

Za registraciju koristimo gotovu formu sa:

``` python
from django.contrib.auth.forms import UserCreationForm
```

I kreiramo funkciju `register()`:

``` python
def register(request):
    form = UserCreationForm()
    context = {'form': form}

    return render(request, 'registration/register.html', context)
```

Kreirajmo `register.html`:

``` html
<form method="post" action="{% url 'register' %}">
    {% csrf_token %}

    {% if form.errors %}
        <p>Greška.</p>
    {% endif %}

    {{ form }}

    <input type="submit" value="Register" />
</form>
```

Izmjenimo funkciju  `register()`:

``` python
from django.contrib.auth import authenticate, login

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)

        if form.is_valid():
            form.save()
            username = form.cleaned_data['username']
            password = form.cleaned_data['password1']

            user = authenticate(username=username, password=password)
            login(request, user)
            return redirect('index')

    else:
        form = UserCreationForm()

    context = {'form': form}

    return render(request, 'registration/register.html', context)
```

Izmjene na `index.html` ako je korisnik ulogiran.

``` html
<h1>This is our homepage</h1>

{% if user.is_authenticated %}
    <p>Vaše ime: {{ user.username }}</p>
{% else %}
    <p>Niste prijavljeni.</p>
{% endif %}
```

## Django vježba 11: Testiranje

Ovaj dio je sastavljen prema [Django Tutorialu s MDN-a](https://developer.mozilla.org/en-US/docs/Learn/Server-side/Django/Testing).

**Test jedinke** (engl. *unit test*) je najbrži za izvedbu, testiraju dio koda neovisno o drugim djelovima.

``` python
def zbroji(prvi, drugi):
    return prvi + drugi

def test_zbroji():
    assert zbroji(3, 6) == 9
```

**Test integracije** testira više dijelova zajedno kako bi se osiguralo da međusobno dobro surađuju/rade.

**Funkcijsko testiranje** je test koji radi na principu da osigurava funkcionalnos iz perspektive krajnjeg korisnika. Najsporiji za izvođenje.

**Regresijsko testiranje** reproducira greške koje su se prethodno događale u programskom kodu. Svaki se test u početku pokreće kako bi se provjerilo ako je greška u kodu ispravljena, a zatim se ponovno pokreće kako bi se osiguralo da nije ponovno uveden nakon kasnijih promjena koda.

Za testiranje projekta i programskog koda unutar Django-a koristiti ćemo osnovnu klasu za testiranje koja se zove [django.test.TestCase](https://docs.djangoproject.com/en/3.2/topics/testing/tools/#testcase). Najzastupljenija je klasa za testiranje, iako neke testove ne provodi "najbrže" (svaki test ne zahtjeva kreiranje baze podataka).

``` python
class Author(models.Model):
    name = models.CharField(max_length=30)
    address = models.CharField(max_length=50)
    city = models.CharField(max_length=60)
    country = models.CharField(max_length=50)
    date_of_birth = models.DateField(null=True, blank=True)
    date_of_death = models.DateField('Died', null=True, blank=True)

    def __str__(self):
        return self.name
```

Prije početka pisanja samih testova pokrenite si projekt koji smo radili na vježbama 6. Na Merlinu je dostupna arhiva s projektom ako ju nemate.

!!! zadatak
    Preuzmite i pokrenite projekt `vj6` zatim unutar aplikacije main kreirajte direktorij `tests`.

Unutar direktorija `tests` nalaze se testovi kojima će se testirati `urls.py`, `views.py` i `models.py`.

Naredba koju koristite za pokretanje testova je:

``` shell
$ ./manage.py test main.tests
(...)
```

!!! zadatak
    Stvorite testne `.py` datoteke unutar `tests`  koje koristite za testiranje rada `urls.py`, `views.py` i `models.py`.

### Testiranje `urls.py`

``` python
from django.test import SimpleTestCase
from django.urls import reverse, resolve
from main.views import homepage, BookList, AuthorList, AuthorBookList


class TestUrls(SimpleTestCase):

    def test_homepage_url_is_resolved(self):
        url = reverse('homepage')
        # print(resolve(url))

        self.assertEquals(resolve(url).func, homepage)

    def test_books_url_is_resolved(self):
        url = reverse('books')

        self.assertEquals(resolve(url).func.view_class, BookList)

    def test_authors_url_is_resolved(self):
        url = reverse('authors')

        self.assertEquals(resolve(url).func.view_class, AuthorList)

    def test_authors_url_is_resolved(self):
        url = reverse('author_q', args=['some-author'])

        self.assertEquals(resolve(url).func.view_class, AuthorBookList)
```

### Testiranje `views.py`

``` python
from django.test import TestCase, Client
from django.urls import reverse
from main.models import Author, Book


class TestViews(TestCase):

    def setUp(self):
        self.client = Client()
        self.homepage_url = reverse('homepage')
        self.authors_q_url = reverse('author_q', args=['some-author'])

        self.author1 = Author.objects.create(
            name = 'some-author',
            address = 'TestAdress',
            city = 'TestCity',
            country = 'TestCountry'
        )

    def test_project_homepage_GET(self):
        client = Client()

        response = client.get(self.homepage_url)

        self.assertEquals(response.status_code, 200)
        self.assertTemplateUsed(response, 'base_generic.html')

    def test_project_authors_GET(self):
        client = Client()

        response = client.get(self.authors_q_url)

        self.assertEquals(response.status_code, 200)
        self.assertTemplateUsed(response, 'main/book_list.html')
```

### Testiranje `models.py`

``` python
from django.test import TestCase
from main.models import Author, Book


class Testmodels(TestCase):

    def setUp(self):
        self.author1 = Author.objects.create(
            name = "some-author",
            address = "TestAdress",
            city = "TestCity",
            country = "TestCountry"
        )

    def test_author(self):
        self.assertEquals(self.author1.name, "some-author")
```
