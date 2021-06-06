# Vježbe 5: Generički pogledi

###### tags: `DWA2 Vjezbe` `Django` `Pyhton` `Generički pregledi`


Na današnjim vježbama radit će se generički pregledi.

## Priprema i postavljanje projekta
Prije početka rada potrebno je kreirati novi Django projekt `vj5` unutar kojeg kreirate aplikaciju  `main`. 

Povežite projekt i aplikaciju. :
- Dodati `main` aplikaciju pod `INSTALLED_APPS` unutar `vj5/settings.py`.
- Unutar `vj5/urls.py` dodati usmjeravanje na `main/urls.py`, `main/urls.py` još nije stvoren, stoga ga je potrebno kreirati. 

:::info
`vj5/main/urls.py`

```python
from django.urls import path

urlpatterns = [
]
```
:::


Za potrebe ovih vježbi koristit će se gotov model koji je zadan u nastavku.

:::info
`vj5/main/models.py`
```python
from django.db import models

# Create your models here.

class Publisher(models.Model):
    name = models.CharField(max_length=30)
    address = models.CharField(max_length=50)
    city = models.CharField(max_length=60)
    state_province = models.CharField(max_length=30)
    country = models.CharField(max_length=50)
    website = models.URLField()

    class Meta:
        ordering = ["-name"]

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
:::


Nakon što je model kreiran unutar `vj5/main/models.py` potrebno je provesti migraciju. 
Naredbe za migraciju su:

``` 
$ ./manage.py makemigrations

$ ./manage.py migrate 
```

:::info
Napravite migraciju i zatim pokrenite server
:::

## Generički pogledi 

Kreirajte prvi generički pogled nad stvorenim modelom.

:::info
`vj5/main/views.py`
```python
from django.views.generic import ListView
from main.models import Publisher

class PublisherList(ListView):
    model = Publisher
```
:::

A zatim ga povežite unutar `main/urls.py`

:::info
`vj5/main/urls.py`
```python
from django.urls import path
from main.views import PublisherList

urlpatterns = [
    path('publishers/', PublisherList.as_view()),
]
```
:::

Kada smo kreirali pogled i pozvali ga unutar `urls.py` potreban nam je predložak unutar kojeg će se prikazati odgovor. 

Sve predloške koje ćemo koristiti organizirat ćemo tako da se nalaze u zajedničkom direktoriju `tempaltes`, koji se nalazi u korijenskom direktoriju.

Kreirajte `./templates` direktorij, unutar kojeg kreirate `main` direktorij, dakle `./templates/main`, a unutar njega  kreirajte `publisher_list.html`, koji sadržava sljedeći sadržaj:

:::info
`./templates/main/publisher_list.html` 
```html
{% block content %}
    <h2>Publishers</h2>
    <ul>
        {% for publisher in object_list %}
            <li>Name: {{ publisher.name }} <br> City: {{ publisher.city }}</li>
        {% endfor %}
    </ul>
{% endblock %}
```
:::

Potrebno je još zadati putanju za predloške unutar `settings.py`. 

Za dodavanje putanje, pod `TEMPLATES` dodajte putanju do `templates` direktorija (`./templates`), odnosno `'DIRS': ['./templates'],`.


:::success
**Zadatak**
Kreirajte administratora i dodajte u bazu podataka 3 izdavača. 

Sve vrijednosti proizvoljno zadajte.
:::

:::info
Provjerite ispis izdavača koji su dodani u bazu na `127.0.0.1/main/publishers`
:::


### Dinamičko filtriranje
U nastavku je prikazan način na koji se omogućava dinamička pretraga pomoću URL-a. Za zadani naziv izdavača vraćat će se sve knjige koje je taj izdavač objavio. U zadanom URL uzorku u aplikaciji neće statično biti definirati naziv, nego će se on dinamično generirati.

Za početak potrebno je definirati prikaz unutar `./main/views.py` koji će vraćati sve knjige od zadanog izdavača.

:::info
`vj5/main/views.py`
```python
from django.shortcuts import get_object_or_404
from django.views.generic import ListView
from main.models import Book, Publisher

class PublisherBookList(ListView):

    template_name = 'main/books_by_publisher.html'

    def get_queryset(self):
        self.publisher = get_object_or_404(Publisher, name=self.kwargs['publisher'])
        return Book.objects.filter(publisher=self.publisher)
```
:::

Zatim unutar `./main/urls.py` povezujemo s traženim pogledom. U ovom slučaju ne koristi se statično zadani uzorak Umjesto da svakog pojedinog izdavača zadajemo pojedinačno, koristimo `<publisher>`.

:::info
`vj5/main/urls.py`
```python
from django.urls import path
from main.views import PublisherList, PublisherBookList

urlpatterns = [
    path('publishers/', PublisherList.as_view()),
    path('<publisher>/', PublisherBookList.as_view()),
]

```
:::

I za zadnji dio potrebno je kreirati prikaz unutar `./templates` koji će nam prikazivati rezultate pretrage za zadanog izdavača.


:::success
**Zadatak**
Kreirajte `books_by_publisher.html` unutar `./templates/main
` koji će ispisati sve knjige od traženog izdavača. Neka se ispisuje samo naslov svake knjige. 

:::spoiler Rješenje
```html
{% block content %}
    <h2>Books list: </h2>
    <ul>
        {% for book in object_list %}
            <li>Book title: {{ book.title }}</li><br>
        {% endfor %}
    </ul>
{% endblock %}
```
:::


:::info
Pokrenite server i provjerite pretraživanje po izdavaču.
:::
