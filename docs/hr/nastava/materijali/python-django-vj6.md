---
author: Milan Petrović
---

# Django vježba 6: Predaja obrazaca HTTP metodama GET i POST. Provjera unosa i prikaz poruka o greškama

Na današnjim vježbama radit će se generičko popunjavanje baze i obrasci.

## Postavljanje projekta

!!! zadatak
    Kreirajte projekt naziva `vj6` i unutar njega aplikaciju naziva `main`. Povežite aplikaciju sa projektom: dodajte aplikaciju unutar `settings.py` i putanju `main/urls.py` unutar `urls.py`, a zatim kreirajte `main/urls.py`.

## Generičko popunjavanje baze podataka

Model koji se koristi sadrži dvije klase, `Author` i `Book`.

Sadržaj datoteke `vj6/main/models.py`:

``` python
from django.db import models

# Create your models here.


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

Kada je instaliran [paket factory_boy](https://pypi.org/project/factory-boy/), potrebno je kreirati klase koje će automatski popunjavati bazu sa tzv. *dummy data*, odnostno nasumično generiranim podacima koji će nam pojednostaviti proces popunjavanja baze nad kojom želimo izvršavati upite. Detaljnije o  njegovoj funkcionalnosti možete pronaći u [službenoj dokumentaciji](https://factoryboy.readthedocs.io/).

!!! zadatak
    Unutar `vj6/main` stvorite datoteku `factory.py`

Stvorena datoteka `vj6/main/factory.py` koristit će se kao predložak za popunjavanje modela definiranog unutar`vj6/main/models.py`. Primjetit ćete sličnost u stilu pisanja klasa. Dakle, potrebno je definirati klase, sukladno klasama koje su definirane unutar `vj6/main/models.py`.

Datoteka  `vj6/main/factory.py`:

``` python
# factories.py
import factory
from factory.django import DjangoModelFactory

from main.models import *

# Defining a factory
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

# Register your models here.
admin.site.register(models_list)
```

### Kreiranje naredbe u `manage.py`

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
