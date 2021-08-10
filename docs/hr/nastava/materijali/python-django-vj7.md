---
author: Milan Petrović
---

# Django vježba 7: Predlošci obrazaca. Stvaranje obrazaca iz modela

## Kreiranje predložaka

Nakon što je baza kreirana i popunjena teestim podacima, sljedeći korak je definiranje predložaka pomoću kojih će se ti podaci prikazivati.

### Sintaksa

#### Varijable

Pomoću varijabli ispisujemo sadržaj koji je prosljeđen iz konteksta. Objekt je sličan Python riječniku (engl. *dictionary*) gdje je sadržaj mapiran u odnosu *key-value*.

Varijable pišemo unutar vitičastih zagrada `{{ varijabla }}`:

``` html
<p>Ime: {{ first_name }}, Prezime: {{ last_name }}.</p>
```

Što će iz konteksta `{'first_name': 'Ivo', 'last_name': 'Ivić'}` biti prikazano kao:

``` html
<p>Ime: Ivo, Prezime: Ivić.</p>
```

#### Oznake

Oznake se pišu unutar `{% oznaka %}`. Označavaju proizvoljnu logiku unutar prikaza. Oznaka može biti ispis sadržaja, ili logička cjelina ili pak pristup drugim oznakama iz predloška, Primjerice: `{% tag %} ... sadržaj ... {% endtag %}` ili `{% block title %}{{ object.title }}{% endblock %}`.

#### Filteri

Filtere koristimo da bi transformirali vrijednosti varijabli. Neki od primjera korištenja filtera mogu biti za:

- Pretvaranje u mala slova: `{{ name|lower }}`
- Uzimanje prvih 30 riječi: `{{ bio|truncatewords:30 }}`
- Dohvaćanje duljine varijable: `{{ value|length }}`

#### Komentari

Za komentiranje dijelova koristimo `#` što unutar predloška izgleda: `{# Ovo neće biti prikazano #}`.

#### Petlje

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

### Nasljeđivanje u predlošcima

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

## CSS

Dodavanje CSS-a u HTML predložak.

### Uvoz

``` html
<link rel="stylesheet" href="https://www.w3schools.com/html/styles.css"> 
```

``` html
<head>
    <link rel="stylesheet" href="https://www.w3schools.com/html/styles.css"> 
    <title>{% block title %}Knjižnica{% endblock %}</title>
</head>
```

### Direktorij za statičke datoteke

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
