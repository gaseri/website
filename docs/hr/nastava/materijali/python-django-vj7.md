# Vježbe 7: Predlošci obrazaca. Stvaranje obrazaca iz modela

## Kreiranje prodložaka

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



## Nasljedivanje u predlosku

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

Unutar main/urls.py

```
path("books", BookList.as_view())
```

main/views.py
```python
from django.shortcuts import get_object_or_404
from django.views.generic import ListView
from main.models import Author, Book

class BookList(ListView):
    model = Book
```


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
