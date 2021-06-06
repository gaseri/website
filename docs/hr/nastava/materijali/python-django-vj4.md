# Vježbe 4: Usmjeravanje i URL-i. Stvaranje pogleda kao odgovora na HTTP zahtjeve

###### tags: `DWA2 Vjezbe` `Django` `Pyhton` `HTTP zahtjevi` `URL`

Na današnjim vježbama čeka nas gradivo vezano za usmjeravanje pomoću URL'ova. Zatim ćemo vidjeti par primjera kako se izgleda odgovor na poslani HTTP zahtjev


## Usmjeravanje pomoću */urls.py*

Potrebno je prvo stvoriti novi projekt i unutar njega aplikaciju koju ćemo povezati.

Za kreiranje projekta koristi se naredba:
```
django-admin startproject <prject_name>
```

Za kreiranje aplikacije unutar projekta koristi se naredba:
```
django-admin startapp <app_name>
```

Nakon što su projekt i aplikacija unutar njega kreirani, potrebno ih je povezati. Ovo se radi unutar datoteka `./urls.py` koja se nalazi u projektnom folderu.

Na ovim vježbama kreirati će se projekt naziva `vj4` unutar kojeg je stvorena aplikacija naziva `main`.

:::success
**Zadatak**
Povežite kreiranu aplikaciju `main` s glavnim djelom aplikacije unutar `./urls.py`.

:::spoiler Rješenje
```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path("main/",include('main.urls')),
]
```
:::




:::warning
**Napomena**
Potrebno je povezati novostvorenu aplikaciju `main` i unutar `setting.py`. Unutar`INSTALLED_APPS` potrebno je dodati `MainConfig` iz `apps.py` koja se nalazi unutar `main` aplikacije.

:::


Program smo usmjerili na `./main/urls.py` koji trenutno ne postoji. Iz toga razloga, potrebno ga je stvoriti.

:::success
**Zadatak**
Stvorite datoteku `./main/urls.py`. Odmah importajte sve iz filea `views.py` i neka ime aplikacije bude zadano na `app_name = 'main'`.

Zatim definirajte uzorak URL-a neka upućuje na `homepage`, odnosno na funkciju unutar `./main/views.py` koja se zove `homepage`.

:::spoiler Rješenje
```python
from django.urls import path
from . import views

app_name = 'main'

urlpatterns = [
    path("homepage", views.homepage, name="homepage"),
]
```
:::


Definirali smo poveznice unutar `url` fileova. Sada je potrebno kreirati funkciju`homepage` unutar `./main/views.py` koju smo pozvali unutar `./main/urls.py`.

## Slanje zahtjeva

:::success
**Zadatak**
Definirajte funkciju `homepage` unutar`./main/views.py` koja će vraćati Http odgovor na zahtjev.
Za vraćanje Http odgovora koristite funkciju `HttpResponse` koju uvozite sa `from django.http import HttpResponse`.


:::spoiler Rješenje
```python
from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

def homepage(request):
    return HttpResponse("<strong> Homepage </strong> jos neki tekst na homepage")
```
:::


:::info
Pohranite sve promjene i pokrenite server.
:::



:::success
**Zadatak**
Difinirajte funkciju `current_datetime` unutar`./main/views.py` koja će vraćati Http odgovor na zahtjev.
Neka vrijednost koju funkcija vraća budu datum i trenutno vrijeme.

:::spoiler Rješenje
```python
from django.shortcuts import render
from django.http import HttpResponse
import datetime
# Create your views here.

def current_datetime(request):
    now = datetime.datetime.now()
    html = "<html><body>Trenutno vrijeme: %s.</body></html>" % now
    return HttpResponse(html)
```
:::

### Vraćajne grešaka na zahtjeve

:::success
**Zadatak**
Definirajte funkciju unutar `views.py`, funkcija `not_found` neka vraća `HttpResponseNotFound`. Vratite proizvoljni odgovor.

:::spoiler Rješenje
```python
from django.http import HttpResponse, HttpResponseNotFound

def not_found(request):
    return HttpResponseNotFound('<h1>Page not found</h1>')
```
:::

## Vraćanje zapisa iz baze

U nastavku je prikazano kako se mogu dohvaćati vrijednosti iz baze podataka i kako ih možemo prikazivati na stranici.

:::success
**Zadatak**
Kreirajte klasu `Student`, neka sadrži, ime prezime i broj xice kao atribute. Dodajte ju zatim unutar `admin.py` da bi se mogle unositi vrijednosti. 
Za kraj pokrenite naredbe za migraciju da se kreira baza.

:::spoiler Rješenje
```python
#unutar models.py

class Student(models.Model):
    ime = models.CharField(max_length=25)
    prezime = models.CharField(max_length=50)
    broj_xice = models.CharField(max_length=10)

    def __str__(self):
        return str(self.broj_xice)


### unutar admin.py
from django.contrib import admin
from main.models import *

# Register your models here.
admin.site.register(Student)
```
:::

Nakon što je baza pomoću modela kreirana, potrebno je unijeti u nju vrijednosti da se može izvršiti tražene upite.

:::success
**Zadatak**
Kreirajte administratora i dodajte u bazu podataka 5 studenata. Od 5 studenata, 3 neka imaju isto ime. 
*Primjerice: Marko, Marko, Marko, Ivan i Ana.*

Prezime i broj xice, proizvoljno zadajte.
:::



Kada smo popunili bazu, idemo kreirati i upite.



:::success
**Zadatak**
Definirajte funkciju koja u bazi pronalazi sve studente zadanog imena, listu pronađenih imena proslijedite funkciji `render`. Proslijeđena rješenja neka se prikazuju unutar `students.html`. 
Za početak, neka vaša funkcija vraća `render(request, 'students.html', context=context)`. A `students.html` ćemo definirati u nastavku rješavanja ovoga zadatka.


:::spoiler Rješenje
```python
def all_peros(request):
    peros = Student.objects.filter(ime__contains="pero")
    
    context = {'peros': peros}

    return render(request, 'students.html', context=context)
```
:::

Za prikaz rješenja prethodnog zadatka, potreban nam je html file, koji će prikazati rezultate upita nad bazom.
Kreirajte unutar `main` foldera datoteku `./main/templates`, unutar koje pohranjujete `students.html`.

:::info
**students.html**
```html
<ul>
    {% for p in peros %}
        <li>Ime: {{ p.ime }} <br> Prezime: {{ p.prezime }} <br> Broj xice:{{ p.broj_xice }}</li>
        <br>
    {% endfor %}
</ul>
```
:::

:::success
Dodajte unutar `./main/urls.py` putanju koja nas vodi na prethodno kreiranu funkciju i zatim provjerite prikaz rezultata na serveru.
:::


Idemo još prikazati ukupan broj studenata u našoj bazi. Ovaj broj ćemo zatim proslijediti funkciji `render` koja će ispisati ukupan broj studenata u za to kreiranom html-u.


:::success
**Zadatak**
Definirajte funkciju koja u bazi pronalazi ukupan broj studenata, broj studenata funkciji `render`. 

Proslijeđena rješenja neka se prikazuju unutar `index.html`. 

Dodajte grešku u slučaju da Student u bazi nije pronađen.


:::spoiler Rješenje python
```python
def detail(request):
    try:
        num_students = Student.objects.all().count()
        
        context = {'num_students': num_students}

    except Student.DoesNotExist:
        raise Http404("Student does not exist")

    return render(request, 'detail.html', context=context)
```
```html
{% block content %}
  <h1>Dobrodosli na UNIRI</h1>

  <p>Na faxu je upisano:</p>
  <ul>
    <li><strong>Studenata:</strong> {{ num_students }}</li>
  </ul>
{% endblock %}
```
:::
