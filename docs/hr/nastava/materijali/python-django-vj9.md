---
author: Milan Petrović
---

# Django vježba 9: Sijanje i migracije

## Sijanje (seeding)

!!! zadatak
    - Kreirajte projekt `vj9` i unutar njega aplikaciju `main`. Unutar modela u aplikaciji `main` kreirajte klasu `Student`. Klasa `Student` neka sadrži vrijednost `first_name`.  
    - Provedite potrebne naredbe za migraciju.  
    - Pokrenite `./manage.py shell` i kreirajte jednog studenta.

Naredbom `dumpdata` izvozimo vrijednosti iz baze ([detaljnije o naredbi dumpdata](https://docs.djangoproject.com/en/3.2/ref/django-admin/#dumpdata)). Pokrenimo je na način:

``` shell
$ ./manage.py dumpdata main.Student --pk 1 --indent 4 > 0001_student.json
(...)
```

!!! zadatak
    Izbrišite iz baze zapis studenta kojeg ste prethodno unjeli.

**Rješenje zadatka.**

``` python
>>> Student.objects.filter(pk=1).delete()
```

Za uvoz podataka u bazu koristimo naredbu `loaddata`. Detaljnije o naredbi [loaddata](https://docs.djangoproject.com/en/3.2/ref/django-admin/#loaddata).

!!! zadatak
    Uvezite prethodno kreirani `0001_student.json` u bazu.

### Fixture

Fixture je zbirka podataka koje Django uvozi u bazu podataka. Najjednostavniji način rada sa podacima je pomoću naredbi `dumpdata` i `loaddata`.

### Paket django-seed

``` shell
$ pip3 install django-seed
(...)
```

Python modul pomoću kojeg se mogu generirati podaci za bazu podataka. U pozadini koristi biblioteku [faker](https://github.com/joke2k/faker) ([dokumentacija](https://faker.readthedocs.io/)) za generiranje testnih podataka. Detaljnije o django-seed možete pronaći u [dokumentaciji](https://github.com/mstdokumaci/django-seed).

### Brisanje podataka sijanja

U nastavku je generirana skripta `revert_seed.py` pomoću koje brišemo vrijednosti iz baze koje smo prethodno stvorili i unosili sijanjem.

``` python
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

Skriptu pokrenite naredbom:

``` shell
$ manage.py shell < revert_seed.py
(...) 
```

## Učitavanje podataka sijanja u testiranje

Testirajmo rad tako da dohvatimo podatke o studentu koji ima primarni ključ 1 i čije ime je Ivo.

``` python
# test1.py
from django.test import TestCase
from main.models import Student

class MyTest(TestCase): 
    # fixtures = ["0001_student.json"]  
    
    def test_should_create_group(self):
        s = Student.objects.get(pk=1)
        self.assertEqual(s.first_name, 'Ivo')
```

Kreirani test pokrenite u terminalu s naredbom:

``` shell
$ ./manage.py test test1
(...)
```

``` python
>>> from django.contrib.auth.models import User, Group
>>> Group.objects.create(name='usergroup')
>>> usergroup = Group.objects.get(name='usergroup')
>>> ivo = User.objects.create_user('Ivo')
>>> ivo.pk
>>> ivo.groups.add(usergroup)
```

``` python
>>> python manage.py dumpdata auth.User --pk 1 --indent 4
>>> python manage.py dumpdata auth.User --pk 1 --indent 4 --natural-foreign
```
