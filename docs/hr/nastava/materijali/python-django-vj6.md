# Vježbe 6: Predaja obrazaca HTTP metodama GET i POST. Provjera unosa i prikaz poruka o greškama

Na današnjim vježbama raditi će se generičko popunjavanje baze i obrasci.

## Postavljanje projekta
:::success
**Zadatak: Postavljanje projekta**
Kreirajte projekt naziva `vj6` i unutar njega aplikaciju naziva `main`. Povežite aplikaciju sa projektom.

- Dodati uplikaciju unutar `settings.py`
- Dodati putanju `main/urls.py` unutar `urls.py`, a zatim kreirati `main/urls.py`. 
:::

## Generičko popunjavanje baze podataka
Model koji se koristi sadrži dvije klase, `Author` i `Book`. 

:::info
`vj6/main/models.py`
```python
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
:::

Kreirani model potrebno je popuniti podacima, za to će se koristiti naredba `./manage.py setup_test_data.py`. Prilikom pokretanja naredbe, program vraća grešku jer naredba još nije kreirana. 

Instalacija potrebnih Python paketa:
```
pip3 install factory_boy
```

Kada je instaliran modul, potrebno je kreirati klase koje će automatski popunjavati bazu sa tzv. *dummy data*, odnostno nasumično generiranim podacima koji će nam pojednostaviti proces popunjavanja baze nad kojom želimo izvršavati upite. Detaljnije o funkcionalnosti možete pronaći u [*factory_boy* dokumnetaciji](https://factoryboy.readthedocs.io/en/stable/).

:::success
**Zadatak**
Unutar `./vj6/main` kreirajte `factory.py`
:::

Kreirani `vj6/main/factory.py` koristiti će kao predložak za popunjavanje modela definiranog unutar`vj6/main/models.py`. Primjetit ćete sličnost u stilu pisanja klasa.
Dakle, potrebno je definirati klase, sukladno klasama koje su definirane unutar `vj6/main/models.py`.

:::info
`vj6/main/factory.py`
```python
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
:::

:::success
**Zadatak**
Nakon što su klase definirane unutar `factory.py`, isprobajte njihovu funkcionalnost. Prije pokretanja *shell-a* primjenite migraciju na bazu.
:::


```
$ ./manage.py shell 
```

```
>>> from main.factories import *
```

```
>>> a = AuthorFactory()
>>> b = BookFactory()
>>> a
>>> b.title
>>> b.author
```

:::success
**Zadatak**
Kreirajte administratnora, zatim unutar `admin.py` registrirajte modele `Book` i `Author`.
Provjerite ako su podaci generirani sa `factory.py` uneseni u bazu.
:::spoiler
```python
#admin.py
from django.contrib import admin

from main.models import *

models_list = [Author, Book]

# Register your models here.
admin.site.register(models_list)
```
:::


### `manage.py`: kreiranje naredbe
Kada je kreiran i testiran `factory.py`, slijedi kreiranje naredbe koja će se prosljeđivati `./manage.py`.

Za početak porenite naredbu:
```
./manage.py 
```
Izlistao nam se trenutni popis opcija koje možemo izvršavati.

Kreirajte direktorij `commands`, unutar kojeg će se nalaziti skripta. Zatim se pozicionirajte u njega.
```
$ mkdir main/management/commands
$ cd main/management/commands
```
A zatim, unutar direktorija `commands` kreirajte `setup_test_data.py`.

```
$ touch setup_test_data.py
```

Otvorite kreirani `setup_test_data.py` unutar kojeg će se kreirati vlastita upravljačka naredba. Detaljnije o [upravljačkim komandama](https://simpleisbetterthancomplex.com/tutorial/2018/08/27/how-to-create-custom-django-management-commands.html) koje su kreirane od strane korisnika.

:::info
```main/management/commands/setup_test_data.py```

```python
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
:::


Svojstvo funkcije `handle` je postavljeno na  `transaction.atomic`, što označava da ako je blok koda uspješno izvršen, promjene se pohranjuju u bazu podataka. 
Detaljnije objašnjenje o [transaction](https://docs.djangoproject.com/en/3.1/topics/db/transactions/) korištenom u prethodnom primjeru.



:::success
**Zadatak**
Isprobajte funkcionalnost kreirane naredbe, a zatim provjerite ako su uneseni podaci unutar admin sučelja..
:::

```
$ ./manage.py setup_test_data
```


## Predlošci 

Nakon što je baza kreirana i popunjena teestim podacima, sljedeći korak je definiranje predložaka pomoću kojih će se ti podaci prikazivati.

### Sintaksa 

#### Varijable

Pomoću varijabli ispisujemo sadržaj koji je prosljeđen iz konteksta. Objekt je sličan Python riječniku (*eng. dictionary*) gdje je sadržaj mapiran u odnosu *key-value*.

Varijable pišemo unutar vitičastih zagrada `{{ varijabla }}`:
```
Ime: {{ first_name }}, Prezime: {{ last_name }}.
```
Što će iz konteksta `{'first_name': 'Ivo', 'last_name': 'Ivić'}` biti prikazano kao:
```
Ime: Ivo, Prezime: Ivić.
```

#### Oznake

Oznake se pišu unutar `{% oznaka %}`. Označavaju proizvoljnu logiku unutar prikaza. Oznaka može biti ispis sadržaja, ili logička cjelina ili pak pristup drugim oznakama iz predloška, Primjerice: `{% tag %} ... sadržaj ... {% endtag %}` ili `{% block title %}{{ object.title }}{% endblock %}`


#### Filteri

Filtere koristimo da bi transformirali vrijednosti varijabli.
Neki od primjera korištenja filtera mogu biti za:
- Pretvaranje u mala slova: `{{ name|lower }}`
- Uzimanje prvih 30 riječi: `{{ bio|truncatewords:30 }}`
- Dohvaćanje duljine varijable: `{{ value|length }}`


#### Komentari

Za komentiranje djelova koristimo `#` što unutar predloška izgleda: `{# Ovo neće biti prikazano #}`


#### Petlje

Petlje se pišu unutar `{% %}` odnosno definiraju se njihove oznake.
Primjer korištenja for petlje:
```
<ul>
{% for author in author_list %}
    <li>{{ author.name }}</li>
{% endfor %}
</ul>
```

Primjer autentifikacije korisnika sa if petljom:
`{% if user.is_authenticated %}Hello, {{ user.username }}.{% endif %}`

Primjer if else petlje:
```
{% if author_list %}
    Number of athletes: {{ athlete_list|length }}
{% else %}
    No authors.
{% endif %}
```



:::success
Kreirajte `./templates` unutar kojeg će biti pohranjeni predlošci koji se koriste. Ne zaboravite definirati putanju unutar `settings.py`. 
base_generic.html
:::



:::info
./templates/base_generic.html

```html
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
::: 

### Nasljeđivanje u predlošcima

:::info
./template/main/book_list.html
```html
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
:::


Unutar `main/urls.py`.

```
path("books", BookList.as_view())
```

:::info
main/views.py
```python
from django.shortcuts import get_object_or_404
from django.views.generic import ListView
from main.models import Author, Book

class BookList(ListView):
    model = Book
```
:::

:::success
**Zadatak**
Sukladno prethodnom primjeru kreirajte prikaz za sve autore unutar baze.
:::


## CSS
Dodavanje css-a u html template. 

### Uvoz 
```html
<link rel="stylesheet" href="https://www.w3schools.com/html/styles.css"> 
```
```html
<head>
    <link rel="stylesheet" href="https://www.w3schools.com/html/styles.css"> 
    <title>{% block title %}Knjiznica{% endblock %}</title>
</head>
```



### Static css
Unutar main aplikacije potrebno je stvoriti direktorij `static` a unutar njega `style.css`.

Referenciranje na `style.css` unutar uplikacije:
```html
{% load static %}
<link rel="stylesheet" type="text/css" href="{% static 'style.css' %}">
```

Prikaz unutar HTML templatea:
```html
<head>
    {% load static %}
    <link rel="stylesheet" type="text/css" href="{% static 'style.css' %}">
    <title>{% block title %}Knjiznica{% endblock %}</title>
</head>
```

:::info
`style.css` možete proizvoljno zadati i prilagođavati vlastitim željama.
:::

:::info
main/static/style.css
```css
h1 {
    color: blue;
    text-align: center;
}

h2{
    text-align: center;
}

li{
    list-style-type: none;
    float: left;
}

li a{
    padding: 16px;
}
```
:::


:::success
**Zadatak**
Dopunite prikaz na autora tako da njegovo ime bude link. Link neka vodi na prikaz svih knjiga od odabranog autora.
:::

Prvo je potrebno kreirati pretraživanje po traženom autoru, a zatim vratiti sve knjige koje su pronađene za trraženog autora. 

:::info
main/views.py
```python
from django.shortcuts import get_object_or_404
from django.views.generic import ListView
from main.models import Author, Book

class AuthorBookList(ListView):
    
    template_name = 'main/book_list.html'

    def get_queryset(self):
        self.author = get_object_or_404(Author, name=self.kwargs['author'])
        return Book.objects.filter(author=self.author)
```
:::

Potrebno je dodati putanju unutar `views.py`.

:::info
main/views.py
```python
path("<author>", AuthorBookList.as_view())
```
:::

Zatim je potrebno izmjeniti predložak, odnosno dodati linkvoe koji vode na autore.

:::info
author_list.html
```html
{% extends "base_generic.html" %}
{% block content %}
<br>
<h2> Books </h2>
<br>
{% for book in book_list %}
    <div class="book">
        <h4>{{ book.title }}</h4>
        Author: <a href="{{book.author}}"> {{book.author}} </a>
        <p>{{ book.abstract }}</p>
    </div>
{% endfor %}
{% endblock %}
```
:::

Izmjenimo i predložak za prikaz knjiga.

:::info
book_list.html

```html
{% extends "base_generic.html" %}
{% block content %}
<br>
<h2> Authors </h2>
<br>
{% for author in author_list %}
    <div class="author">
        <h4> {{author.name}} </h4>

        City: {{author.city}}<br>
        Country: {{author.country}} <br>

        <a href="{{author.name}}"> All books by  {{author.name}} </a> 
    </div>
{% endfor %}
{% endblock %}
```
:::


Dopunite `style.css` tako da dodate obrub na elemente Knjige i Autora.

```css
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
