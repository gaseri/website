# Vježbe 9: Sijanje i migracije 


## Sijanje (seeding) 
:::info
**Zadatak**

Kreirajte projek `vj9` i unutar njega aplikaciju `main`.

Unutar modela u aplikaciji `main` kreirajte klasu Student. Klasa Student neka sadrži vrijednost `first_name`.

Provedite potrebne naredbe za migraciju.

Pokrenite `./manage.py shell` i kreirajte jednog studenta
:::

S naredbom `dumpdata` izvozimo vrijednosti iz baze. Detaljnije o naredbi [dumpdata](https://docs.djangoproject.com/en/3.2/ref/django-admin/#dumpdata).
```
./manage.py dumpdata main.Student --pk 1 --indent 4 > 0001_student.json
```


:::info
**Zadatak**

Izbrišite iz baze zapis studenta kojeg ste prethodno unjeli.
:::spoiler
```python
>>> Student.objects.filter(pk=1).delete()
```
:::


Za uvoz podataka u bazu koristimo naredbu `loaddata`. Detaljnije o naredbi [loaddata](https://docs.djangoproject.com/en/3.2/ref/django-admin/#loaddata).

:::info
**Zadatak**
Uvezite prethodno kreirani `0001_student.json` u bazu.
:::


### Django Fixture
Fixture je zbirka podataka koje Django uvozi u bazu podataka. 
Najjednostavniji na;in rada sa podacima je pomoću naredbi `dumpdata` i `loaddata`. 


### django-seed

```
pip3 install django-seed
```
Python modul pomoću kojeg se mogu generirati podaci za bazu podataka. U pozadini koristi biblioteku [faker](https://github.com/joke2k/faker/) za generiranje testnih podataka. Detaljnije o django-seed možete pronaći u [dokumentaciji](https://github.com/mstdokumaci/django-seed).


### Brisanje podataka sijanja

U nastavku je generirana skripta `revert_seed.py` pomoću koje brišemo vrijednosti iz baze koje smo prethodno stvorili i unosili sijanjem.

```python
import json
import glob

g = globals()
has_access = {}
fixtures = glob.glob("*.json")
fixtures.sort(reverse=True)

def get_access(model):
    import importlib

    mod = importlib.import_module(model)
    names = getattr(mod, '__all__', [n for n in dir(mod) if not n.startswith('_')])

    global g
    for name in names:
        g[name.lower()] = {
            'var': getattr(mod, name),
            'name': name
        }

for fixture in fixtures:
    msg = 'Reverting '+fixture+'\n'
    with open(fixture) as json_file:
        datas = json.load(json_file)
        for data in datas:
            app_name = data['model'].split('.')[0]
            class_name = data['model'].split('.')[1]

            if app_name not in has_access.keys():
                get_access(app_name+'.models')
                has_access[app_name] = True

            class_model = g[class_name]['var']
            class_model_name = g[class_name]['name']
            pk = data['pk']

            msg += '{}(pk={}): '.format(class_model_name, pk)
            try:
                class_model.objects.get(pk=pk).delete()
                msg += 'deleted\n'
            except:
                msg += 'not deleted\n'
    print(msg)

```
Skriptu pokrenite sa naredbom:

```
manage.py shell < revert_seed.py 
```


## Učitavanje podataka sijanja u Testiranje

Testirajmo rad tako da dohvatimo podatke o studentu koji ima primarni ključ 1 i čije ime je Ivo.

```python
#test.py
from django.test import TestCase
from main.models import Student

class MyTest(TestCase): 
    
    # fixtures = ["0001_student.json"]  
    
    def test_should_create_group(self):
        s = Student.objects.get(pk=1)
        self.assertEqual(s.first_name, "ivo")

```

Kreirani test pokrenite u terminalu s naredbom:
```
./manage.py test test
```

```python

>>> from django.contrib.auth.models import User, Group
>>> Group.objects.create(name="useregroup")
>>> usergroup = Group.objects.get(name="useregroup")
>>> ivo = User.objects.create_user("ivo")
>>> ivo.pk

>>> ivo.groups.add(usergroup)

```

```
>>> python manage.py dumpdata auth.User --pk 1 --indent 4
>>> python manage.py dumpdata auth.User --pk 1 --indent 4 --natural-foreign
```
