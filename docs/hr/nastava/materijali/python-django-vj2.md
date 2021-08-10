---
author: Milan Petrović
---

# Django vježba 2: Korištenje baze podataka. Stvaranje modela i objektno-relacijsko preslikavanje

## Paradigma model-view-controller (MVC)

Django koristi paradigmu [model-view-controller](https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93controller) (MVC) koju naziva model-template-view (MTV). Svaka stranica kreirana pomoću Django frameworka vjerojatno koristi ove tri stvari kako bi vam prikazala podatke.

Iako ga nazivamo MVC (model-view-controller), način na koji radi ide u obrnutom slijedu. Korisnik će posjetiti URL, vaš kontroler (`/urls.py`) ukazat će na određeni prikaz (`/views.py`). Taj se prikaz tada može (ili ne mora) povezati s vašim modelima. Drugim riječima, prvo ide Controller, zatim View i na posljetku Model po potrebi.

Unutar Djanga koristi se djelomično izmijenjena terminologija. Stoga od sad nadalje koristit ćemo Django terminologiju (model, template i view). U tablici u nastavku detaljnije su objašnjeni pojmovi i termini koji se koriste.

| Korišteni naziv | Django naziv | Značenje |
| -------------- | ----------- | ------------------------------------------------------------------------------------------------ |
| Model | Model | Sloj modela u Djangu odnosi se na bazu podataka plus Python kôd koji je izravno koristi. Modelira stvarnost. Pohranjuje sve potrebne vrijednosti unutar baze podataka koje su potrebne web aplikaciji. Django vam omogućuje pisanje Python klasa koje nazivamo modeli, koje se vezuju za tablice baze podataka. Stvoreni modeli nisu trajno zadani, nego se mogu izmjenjivati i dopunjavati. Izmjene su dostupne odmah nakon primjene migracija o kojima detaljnije kasnije na ovim vježbama u poglavlju ORM. |
| View | Template | View sloj u Djangu odnosi se na korisničko sučelje (UI). Funkcija pregleda ili skraćeno prikaz (*eng. view*) je Python funkcija koja je zadužena za generiranje HTMLa i ostalih UI elemenata. Pomoću Python kôda renderiraju se pogledi. Django uzima web zahtjev i vraća web odgovor. Ovaj odgovor može biti bilo što, npr. HTML sadržaj web stranice, preusmjeravanje, [statusni kod](https://http.cat/), XML dokument ili slika... |
| Controller | View | Središnji dio sustava, sadržava logiku koja povezuje cjeline da bi se pružio odgovor korisniku na traženi zahtjev. Upravlja zahtjevima i odgovorima na njih. uspostavlja vezu s bazom podataka i učitavanjem datoteka. |

## Uvod u Django: Hello world

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

# Create your views here.
def homepage(request):
    return HttpResponse('Welcome to homepage! <strong>#samoOIRI</strong>')
    # primjetiti korištenje HTML-a
```

Osvježimo stranicu `http://127.0.0.1:8000/` u browseru.

## Modeli: Uvod

Nastavak na program iz uvoda. U ovom poglavlju detaljnije je objasnjeno stvaranje modela u Djangu.

Stvoreni model se mapira na tablicu u bazi podataka. Baza poidataka je već stvorena i možete ju vidjeti na lokaciji `mysite/mysite`. Po defaultu tip baze podataka je SQLite 3 pa je ekstenzija datoteke `.sqlite3`. Tip baze podataka možete mijenjati u `settings.py` pod `DATABASES`.

### Stvaranje modela i polja modela

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

### Objektno-relacijsko preslikavanje

Svaki novi model je nova tablica u bazi podataka. Zbog toga moramo napraviti dvije stvari. Prva je pripremiti za migraciju naredbom `makemigrations`, a zatim napraviti migraciju nredbom `migrate`.

Pokrenimo naš lokalni server naredbom `./manage.py runserver`. Primijetite u outputu konzole sljedeću poruku `You have 18 unapplied migration(s). Your project may not work properly until you apply the migrations for app(s): admin, auth, contenttypes, sessions
Run './manage.py migrate' to apply them`.

Unesite naredbu  da bi pripremili migraciju.

``` shell
$ ./manage.py makemigrations
No changes detected
```

Razlog tomu je što još nismo povezali `main` aplikaciju. Način na koji ćemo povezati aplikaciju je taj da ju pozovemo unutar `INSTALLED_APPS` unutar `mysite/settings.py`.

Možemo vidjeti da je unutar `main/apps.py` definirana kalsa `MainConfig`.

Dopunimo sada `mysite/settings.py` sa pozivom klase `MainConfig` iz aplikacije `main`. To radimo na način da pod `INSTALLED_APPS` dodamo `'main.apps.MainConfig'`.

Datoteka `mysite/mysite/settings.py` sad ima sadržaj:

``` python
# Application definition

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

### Interakcija sa modelom

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

## Kreiranje superusera i povezivanje modela

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

# Register your models here.
admin.site.register(Predmet)
```

!!! zadatak
    Osvježite stranicu na adresi `http://127.0.0.1:8000/admin/`.
