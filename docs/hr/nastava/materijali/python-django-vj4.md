---
author: Milan Petrović
---

# Django vježba 4: Usmjeravanje i URL-i. Stvaranje pogleda kao odgovora na HTTP zahtjeve

Na današnjim vježbama čeka nas gradivo vezano za usmjeravanje pomoću URL-ova. Zatim ćemo vidjeti par primjera kako se izgleda odgovor na poslani HTTP zahtjev.

## Usmjeravanje pomoću `urls.py`

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

## Slanje zahtjeva

!!! zadatak
    Definirajte funkciju `homepage()` unutar `main/views.py` koja će vraćati HTTP odgovor na zahtjev. Za vraćanje HTTP odgovora koristite funkciju `HttpResponse` koju uvozite kodom `from django.http import HttpResponse`.

**Rješenje zadatka.** U datoteci `main/views.py`:

``` python
from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

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
# Create your views here.

def current_datetime(request):
    now = datetime.datetime.now()
    html = '<html><body>Trenutno vrijeme: {}.</body></html>'.format(now)
    return HttpResponse(html)
```

### Vraćajne grešaka u odgovorima na zahtjeve

!!! zadatak
    Definirajte funkciju `not_found()` unutar `main/views.py`. Funkcija neka vraća `HttpResponseNotFound`. Vratite proizvoljni sadržaj odgovora.

**Rješenje zadatka.** U datoteci `main/views.py`:

``` python
from django.http import HttpResponse, HttpResponseNotFound

def not_found(request):
    return HttpResponseNotFound('<h1>Page not found</h1>')
```

## Vraćanje zapisa iz baze

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

# Register your models here.
admin.site.register(Student)
```

Nakon što je baza pomoću modela kreirana, potrebno je unijeti u nju vrijednosti da se može izvršiti tražene upite.

!!! zadatak
    Kreirajte administratora i dodajte u bazu podataka 5 studenata. Od 5 studenata, 3 neka imaju isto ime, primjerice: Marko, Marko, Marko, Ivan i Ana. Prezime i broj X-ice zadajte proizvoljno.

Kada smo popunili bazu, idemo kreirati i upite.

!!! zadatak
    Definirajte funkciju koja u bazi pronalazi sve studente zadanog imena, listu pronađenih imena proslijedite funkciji `render`. Proslijeđena rješenja neka se prikazuju unutar `students.html`.  
    Za početak, neka vaša funkcija vraća `render(request, 'students.html', context=context)`, a datoteku `students.html` ćemo stvoriti u nastavku rješavanja ovoga zadatka.

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
    Definirajte funkciju koja u bazi pronalazi ukupan broj studenata, broj studenata funkciji `render`.  
    Proslijeđena rješenja neka se prikazuju unutar datoteke `index.html`.  
    Dodajte grešku u slučaju da `Student` u bazi nije pronađen.

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
