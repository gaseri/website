---
author: Milan Petrović
---

# Python: Django REST framework

Na današnjim vježbama raditi ćemo sa bibliotekom `djangorestframework` ([službeno web sjedište](https://www.django-rest-framework.org/)).

## Početak rada

Za početak rada potrebno je stvoriti novi projekt i unutar njega aplikaciju.

``` shell
$ django-admin startproject vj11
(...)
$ cd vj11
$ django-admin startapp main
(...)
```

Nakon što ste kreirali aplikaciju, povežite ju sa projektom i kreirajte administratora.

``` python
INSTALLED_APPS = [
    ...
    'main.apps.MainConfig',
    ...
]
```

``` shell
$ ./manage.py createsuperuser --username admin
(...)
```

## Kreiranje modela

Unutar `main/models.py` potrebno je kreirati model. Podatke koji će biti uneseni u bazu biti će naknadno vraćani na zahtjev.

Kreiranje modela:

``` python
from django.db import models

class Korisnik(models.Model):
    name = models.CharField(max_length=60)
    surname = models.CharField(max_length=60)

    def __str__(self):
    return self.name
```

Nakon kreiranog modela potrebno je izvršiti migracije:

``` shell
$ ./manage.py makemigrations
(...)
$ ./manage.py migrate
(...)
```

!!! zadatak
    Registrirajte kreirani model `Korisnik` unutar `admin.py`. Pokrenite server i provjerite prikaz unutar admin sučelja a zatim unesite podatke za 3 korisnika.

**Rješenje zadatka.**

``` python
from django.contrib import admin
from .models import Korisnik

admin.site.register(Korisnik)
```

## Postavljanje Django REST frameworka

Instalacija biblioteke `djangorestframework`:

``` shell
$ pip3 install djangorestframework
(...)
```

Pod instalirane aplikacije potrebno je dodati i `rest_framework`.

``` python
INSTALLED_APPS = [
    ...
    'rest_framework',
    ...
]
```

## Serijalizacija

Prvi korak u kreiranju Web API-ja je pružanje načina srijalizacije instanci u obliku reprezentacije JSON. To možemo učiniti deklariranjem serijalizatora, princip rada sličan je formama u Djangu.

Sljedeći korak je kreiranje datoteke `serializers.py` unutar aplikacije `main`. Tu datoteku ćemo koristiti za prikaz podataka.

``` shell
$ touch ./main/serializers.py
(...)
```

``` python
# main/serializers.py
from rest_framework import serializers

from main.models import Korisnik

class KorisnikSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Korisnik
        fields = ('name', 'surname')
```

## Pogledi

Idemo sada kreirati prikaz unutar `views.py`.

Ono što želimo da naši pogledi rade je upit nad svim korisnicima u bazi. A zatim taj upit prosljediti serializeru za korisnika kojeg smo prethodno kreirali.

``` python
# main/views.py
from rest_framework import viewsets

from main.serializers import KorisnikSerializer
from main.models import Korisnik

class KorisnikViewSet(viewsets.ModelViewSet):
    queryset = Korisnik.objects.all().order_by('name')
    serializer_class = KorisnikSerializer
```

## URL-ovi

Idemo sada sve to zajedno povezati unutar `views.py`.

Because we're using viewsets instead of views, we can automatically generate the URL conf for our API, by simply registering the viewsets with a router class.

Again, if we need more control over the API URLs we can simply drop down to using regular class-based views, and writing the URL conf explicitly.

Finally, we're including default login and logout views for use with the browsable API. That's optional, but useful if your API requires authentication and you want to use the browsable API.

``` python
# vj11/urls.py

from djagno.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('main.urls'))
]
```

``` python
# main/urls.py

from django.urls import include, path
from rest_framework import routers
from . import views

router = routers.DefaultRouter()
router.register(r'korisnici', views.KorisnikViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]
```

## Testiranje

``` shell
$ ./manage.py runserver
(...)
```

``` shell
$ pip3 install httpie
(...)
```

``` shell
$ http -a admin:admin http://127.0.0.1:8000/korisnici/
(...)
```

!!! zadatak
    Kreirajte model Vozilo koje će sadržavati polja *model* i *godina_proizvodnje*. A zatim kreirajte serializer koji će vraćati podatke o modelima vozila koja su upisana u bazu podataka.
