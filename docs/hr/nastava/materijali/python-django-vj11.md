# Vjezbe 11: Testiranje

https://developer.mozilla.org/en-US/docs/Learn/Server-side/Django/Testing

**Unit tests:**

Najbrži za izvedbu, testiraju dio koda neovisno o drugim djelovima. 

```python
def zbroji(prvi, drugi):
    return prvi + drugi

def test_zbroji():
    assert zbroji(3,6) == 9
```

**Test integracije:**

Testirajte više dijelova zajedno kako bi se osiguralo da međusobno dobro surađuju/rade. 

**Funkcijsko testiranje:**

Test koji radi na principu da osigurava funkcionalnos iz perspektive krajnjeg korisnika. Najsporiji za izvođenje.

**Regresijsko testiranje:**
Ovaj test reproducira greške koje su se prethodno događale u programoskom kodu.
Svaki se test u početku pokreće kako bi se provjerilo ako je greška u kodu ispravljena, a zatim se ponovno pokreće kako bi se osiguralo da nije ponovno uveden nakon kasnijih promjena koda.


Za testiranje projekta i programskog koda unutar Django-a koristiti ćemo osnovnu klasu za testiranje koja se zove [django.test.TestCase](https://docs.djangoproject.com/en/3.1/topics/testing/tools/#testcase). Najzastupljenija je klasa za testiranje, iako neke testove ne provodi "najbrže" (svaki test ne zahtjeva kreiranje baze podataka).

```python
class Author(models.Model):
    name = models.CharField(max_length=30)
    address = models.CharField(max_length=50)
    city = models.CharField(max_length=60)
    country = models.CharField(max_length=50)
    date_of_birth = models.DateField(null=True, blank=True)
    date_of_death = models.DateField('Died', null=True, blank=True)

    def __str__(self):
        return self.name
```


Prije početka pisanja samih testova pokrenite si projekt koji smo radili na vježbama 6. 
Na merlinu je dostupna .rar datoteka ako ju nemate.

:::info
Preuzmite i pokrenite projekt `vj6` zatim unutar aplikacije main kreirajte direktorij `tests`.
:::

Unutar direktorija `tests` nalaze se testovi 
kojima će se testirati `urls.py`, `views.py` i `models.py`.

Naredba koju koristite za pokretanje testova je: 
```
./manage.py test main.tests
```

:::success
**Zadatak**

Stvorite testne `.py` datoteke unutar `tests`  koje koristite za testiranje rada `urls.py`, `views.py` i `models.py`.
:::


### Testiranje urls.py

```python
from django.test import SimpleTestCase
from django.urls import reverse, resolve
from main.views import homepage, BookList, AuthorList, AuthorBookList
```

```python

class TestUrls(SimpleTestCase):

    def test_homepage_url_is_resolved(self):
        url = reverse("homepage")
        # print(resolve(url))

        self.assertEquals(resolve(url).func, homepage)
```


```python
def test_books_url_is_resolved(self):
    url = reverse("books")

    self.assertEquals(resolve(url).func.view_class, BookList)


def test_authors_url_is_resolved(self):
    url = reverse("authors")

    self.assertEquals(resolve(url).func.view_class, AuthorList)
```

```python    
def test_authors_url_is_resolved(self):
    url = reverse("author_q", args=['some-author'])

    self.assertEquals(resolve(url).func.view_class, AuthorBookList)


```


## Testiranje views.py


```python
from django.test import TestCase, Client
from django.urls import reverse
from main.models import Author, Book
```

```python
class TestViews(TestCase):

def setUp(self):
    self.client = Client()
    self.homepage_url = reverse("homepage")
    self.authors_q_url = reverse("author_q", args=['some-author'])

    self.author1 = Author.objects.create(
        name = "some-author",
        address = "TestAdress",
        city = "TestCity",
        country = "TestCountry"
    )
        
```


```python
def test_project_homepage_GET(self):
    client = Client()

    response = client.get(self.homepage_url)

    self.assertEquals(response.status_code, 200)
    self.assertTemplateUsed(response, "base_generic.html")
```


```python
def test_project_authors_GET(self):
    client = Client()

    response = client.get(self.authors_q_url)

    self.assertEquals(response.status_code, 200)
    self.assertTemplateUsed(response, "main/book_list.html")

```


## Testiranje models.py

```python
from django.test import TestCase
from main.models import Author, Book


class Testmodels(TestCase):

    def setUp(self):
        self.author1 = Author.objects.create(
            name = "some-author",
            address = "TestAdress",
            city = "TestCity",
            country = "TestCountry"
        )

    def test_author(self):
        self.assertEquals(self.author1.name, "some-author")
```
