# Vježbe 2: Korištenje baze podataka. Stvaranje modela i objektno-relacijsko preslikavanje

###### tags: `DWA2 Vjezbe` `Django` `Django Modeli` `Pyhton` 

# Modeli


## Paradigma model-view-controller (MVC)

Django koristi paradigmu [model-view-controller](https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93controller) (MVC) koju naziva model-template-view (MTV). Svaka stranica kreirana pomoću Django frameworka vjerojatno koristi ove tri stvari kako bi vam prikazala podatke.

Iako ga nazivamo MVC (model-view-controller), način na koji radi ide u obrnutom slijedu. Korisnik će posjetiti URL, vaš kontroler (/urls.py) ukazat će na određeni prikaz (/views.py). Taj se prikaz tada može (ili ne mora) povezati s vašim modelima. Drugim riječima, prvo ide Controller, zatim View i na posljetku Model po potrebi.

Unutar Djanga koristi se djelomično izmijenjena terminologija. Stoga od sad nadalje koristit ćemo Django terminologiju (model, template i view). U tablici u nastavku detaljnije su objašnjeni pojmovi i termini koji se koriste. 

| Korišteni naziv | Django naziv | Značenje                                                                                          |
| -------------- | ----------- | ------------------------------------------------------------------------------------------------ |
| Model          | Model       | Sloj modela u Djangu odnosi se na bazu podataka plus Python kôd koji je izravno koristi. Modelira stvarnost. Pohranjuje sve potrebne vrijednosti unutar baze podataka koje su potrebne web aplikaciji. Django vam omogućuje pisanje Python klasa koje nazivamo modeli, koje se vezuju za tablice baze podataka. Stvoreni modeli nisu trajno zadani, nego se mogu izmjenjivati i dopunjavati. Izmjene su dostupne odmah nakon primjene migracija o kojima detaljnije kasnije na ovim vježbama u poglavlju ORM.              |
| View           | Template    | View sloj u Djangu odnosi se na korisničko sučelje (UI). Funkcija pregleda ili skraćeno prikaz (*eng. view*) je Python funkcija koja je zadužena za generiranje HTMLa i ostalih UI elemenata. Pomoću Python kôda renderiraju se pogledi. Django uzima web zahtjev i vraća web odgovor. Ovaj odgovor može biti bilo što, npr. HTML sadržaj web stranice, preusmjeravanje, [statusni kod](https://http.cat/), XML dokument ili slika...                                         |
| Controller     | View        | Središnji dio sustava, sadržava logiku koja povezuje cjeline da bi se pružio odgovor korisniku na traženi zahtjev. Upravlja zahtjevima i odgovorima na njih. uspostavlja vezu s bazom podataka i učitavanjem datoteka. |
  



## Uvod u Django: Hello world 

U nastavku ovog poglavlja prikazano je kako kreirati Django aplikaciju za teksta na početnoj stranici. Za ovo nije potreban Model nego samo View u kojemu je definiran tekst koji želimo prikazati na stranici.

:::success
**Zadatak**
Otvorite datoteku i provjerite što je zapisano u `mysite/mysite/url.py`.
:::


Za definiranje putanje na koju će Django primati HTTP zahtjeve iskoristit ćemo funkciju `django.urls.path()` ([dokumentacija](https://docs.djangoproject.com/en/3.1/ref/urls/#path)) i usmjerit ćemo na administratorsko sučelje ([dokumentacija](https://docs.djangoproject.com/en/3.1/ref/contrib/#admin))

:::info

`mysite/mysite/url.py`

```python
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
]
```
:::

Vidimo da imamo samo jedan URL koji se tiče stranice za administraciju koji nam trenutno ne treba. Sljedeće što je potrebno učiniti je usmjeriti view na URL. Razlog tome je što Django web stranice vidi kao kolekciju aplikacija koje 'ispisuje' pomoću danih URLova koji 'pokazuju' Djangu gdje tražiti. Datoteka `urls.py` unutar naše "primarne" aplikacije obično samo usmjerava na aplikacije. 

Pa krenimo sa kreiranjem aplikacije, i to na način da ju kreiramo pomoću naredbe `./manage.py startapp main`.

Idemo sada usmjeriti naš glavni dio Django aplikacije da provjerava, odnosno poziva našu novostvorenu aplikaciju. To radimo unutar `mysite/mysite/url.py`:


:::info

`mysite/mysite/url.py`

```python
from django.contrib import admin
from django.urls import path, include #importamo include

urlpatterns = [
    path("", include('main.urls')), #dodajemo urls
    path('admin/', admin.site.urls),
]
```
:::


Ako pogledamo u folder aplikacije `main` vidjet ćemo da file `urls.py` ne postoji. Kreirajomo ga  i dodajmo sljedeći sadržaj.

:::info

`mysite/main/urls.py`

```python
from django.urls import path
from . import views

app_name = 'main'  # here for namespacing of urls.

urlpatterns = [
    path("", views.homepage, name="homepage"),
]
```
:::


Prođimo ukratko kroz logiku rada našeg programa. Prvo se posjećuje `mysite/url.py`, u URL-u nije prosljeđeno ništa, stoga odgovara `""` iz `path("", include('main.urls'))`. Program ovo tumači tako da ukljuuje `main.urls`. 

Program zatim pronalazi i učitava `main.urls` koji se nalazi na lokaciji `mysite/main/urls.py` i čita njegov sadržaj.
Upravo smo izmjenili sadržaj tako da smo dodali uzorak `""` koji odgovara `path("", views.homepage, name="homepage")`. Ovime smo usmjerili aplikaciju da pokrene funkciju `homepage` unutar `main/views.py` koji još nismo izmjenili stoga ćemo to sada učiniti rako da dodamo funkciju naziva `homepage()`.


:::info

`mysite/main/views.py`

```python
from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def homepage(request):
    return HttpResponse("Welcome to homepage! <strong>#samoOIRI</strong>") 
    # primjetiti korištenje HTMLa
```
:::
Osvježimo stranicu `127.0.0.1:8000` u browseru.



## Modeli: Uvod

Nastavak na program iz uvoda. U ovom poglavlju detaljnije je objasnjeno stvaranje modela u djangu.


Stvoreni model se mapira na tablicu u bazi podataka. Baza poidataka je već stvorena i možete ju vidjeti na lokaciji `mysite/mysite`. Po defaultu tip baze podataka je `.sqlite3`. Tip baze podataka možete mjenjati u `settings.py` pod `DATABASES`.

### Stvaranje modela / Model fields 

Ovime stvaramo novu tablicu u bazi podataka a zadana polja postaju stupci u toj tablici. Automatski se po defaultu kreira primarni ključ iz toga, ali po želji možemo i proizvoljno odrediti da neka od zadanih vrijednosti to bude.

:::success
**Zadatak**
Otvorite `main/models.py` i definirajte klasu imena Predmet. U klasi definirajte stupce naziva:`predmet_naslov`, `predmet_sadrzaj` i `predmet_vrijeme_objave`. Njihovi tipovi neka budu `CharField()` koji ima zadan parametar `max_length` na `100`, `TextField()` i `DateTimeField()` koji ima naziv postavljen na `"date published"`.

:::spoiler Rješenje zadataka
```python
class Predmet(models.Model):
    predmet_naslov = models.CharField(max_length=100)
    predmet_sadrzaj = models.TextField()
    predmet_vrijeme_objave = models.DateTimeField("date published")
```
:::


:::success
**Zadatak**

Definirajte funkciju `__str__()` unutar klase `Predmet` koja vraća naziv predmeta `predmet_naziv`.
:::spoiler Rješenje zadataka
```python
def __str__(self):
    return self.predmet_naslov
```
:::



Službena [Django dokumentacija](https://docs.djangoproject.com/en/3.1/ref/models/fields/) o svim poljima unutar modela.

### ORM

Svaki novi model je nova tablica u bazi podataka. Zbog toga moramo napraviti dvije stvari. Prva je pripremiti za migraciju naredbom `makemigrations`, a zatim napraviti migraciju nredbom `migrate`. 

Pokrenimo naš lokalni server sa naredbom `python/python3 manage.py runserver`. Primjetite u outputu konzole sljedeću poruku `You have 18 unapplied migration(s). Your project may not work properly until you apply the migrations for app(s): admin, auth, contenttypes, sessions
Run 'python manage.py migrate' to apply them`.


Unesite naredbu  da bi pripremili migraciju.

```
python manage.py makemigrations
```
Output u terminalu 
```
No changes detected
```

Razlog tomu je što još nismo povezali `main` aplikaciju. Način na koji ćemo povezati aplikaciju je taj da ju pozovome unutar INSTALLED_APPS unutar `mysite/settings.py`.

Možemo vidjeti da je unutar `main/apps.py` definirana kalsa `MainConfig`.

Dopunimo sada `mysite/settings.py` sa pozivom klase `MainConfig` iz aplikacije `main`. To radimo na način da pod `INSTALLED_APPS` dodamo `'main.apps.MainConfig'`.

:::info

`mysite/mysite/settings.py`

```python
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
:::

Unesite ponovo naredbu `python manage.py makemigrations` da bi pripremili migraciju.

```
Migrations for 'main':
  main\migrations\0001_initial.py
    - Create model Predmet
```

:::success
**Zadatak**
Prčitajte sadržaj unutar `main\migrations\0001_initial.py` datoteke. 
:::


Zatim unesimo naredbu `python manage.py migrate` da bi se migracija izvršila.

Nakon svakog rada sa modelima, bile to izmjene ili stvaranje novih modela potrebno je napraviti migraciju na načn `makemigrations` a zatim `migrate`.

Pogledajmo output u terminalu koji nam vraća naredba 
``` 
$ python manage.py migrate
```

Output naredbe `python manage.py migrate` u terminalu:

```
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

Za interakciju sa modelom koristimo naredbu `python manage.py shell` koja pokreće Python shell. 

Za početak uvezimo potrebne modele i pakete naredbama:
```from main.models import Predmet```
```from django.utils import timezone```


Za dohvaćanje svih objekata koristimo `Predmet.objects.all()` što vraća `QuerrySet []` ondosno praznu listu. 


Dodajmo vrijednosti u novu instancu klase Predmet nazvia `novi_predmet` pomoću:

```novi_predmet = Predmet()```
```novi_predmet.predmet_naslov="DWA2"```
```novi_predmet_sadrzaj="ovo je opis predmeta"```
```novi_predmet.predmet_vrijeme_objave=timezone.now()```

Pohranimo promjene sa `novi_predmet.save()`

Isprobajmo ponovno naredbu `Predmet.objects.all()` 

Kroz predmete se može i iterirati, primjerice korištenjem `for` petlje:
```python
for p in Predmet.objects.all():
    print(p.predmet_naslov)
```

## Kreiranje superusera i povezivanje modela



Administratora kreiramo naredbom:
```./manage.py createsuperuser ```

Dodjelite proizvoljno ime i lozinku, mail adresu trenutno možemo ostaviti praznu.

:::success
**Zadatak**

Posjetite na lokalnom serveru adresu http://127.0.0.1:8000/admin
:::

Idemo sada nadopuniti kod i povezati stvoreni model pomoću `mysite/main/admin.py`. 

:::info

`mysite/main/admin.py`

```python
from django.contrib import admin
from .models import Predmet

# Register your models here.
admin.site.register(Predmet)
```
:::

:::success
**Zadatak**

Osvježite stranicu na adresi http://127.0.0.1:8000/admin
:::
